"""Core audio injection functionality for Magenta RT JAM CLI."""

import os
import sys
import dataclasses
import functools
import concurrent.futures
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import librosa
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.prompt import Confirm

from .config import Config

# Import Magenta RT components (these would need to be installed)
try:
    import jax
    from magenta_rt import audio as audio_lib
    from magenta_rt import musiccoca
    from magenta_rt import spectrostream
    from magenta_rt import system
    from magenta_rt import utils
    MAGENTA_RT_AVAILABLE = True
except ImportError:
    MAGENTA_RT_AVAILABLE = False
    # Mock classes for development without Magenta RT
    class MockSystem:
        pass


def load_audio(audio_filename: Path, sample_rate: int) -> np.ndarray:
    """Load an audio file.
    
    Args:
        audio_filename: File path to load.
        sample_rate: The number of samples per second at which the audio will be
            returned. Resampling will be performed if necessary.
    
    Returns:
        A numpy array of audio samples, sampled at the specified rate, in float32
        format.
    """
    y, unused_sr = librosa.load(audio_filename, sr=sample_rate, mono=False)
    return y


def get_metronome_audio(
    loop_samples: int,
    beats_per_loop: int,
    sample_rate: int,
    chunk_samples: int
) -> np.ndarray:
    """Generate metronome audio.
    
    Args:
        loop_samples: The number of samples in a loop.
        beats_per_loop: The number of beats in a loop.
        sample_rate: The sample rate of the audio.
        chunk_samples: The number of samples in a chunk.
    
    Returns:
        A numpy array of metronome audio samples.
    """
    metronome_audio = np.zeros((loop_samples,))
    BEEP_SECONDS = 0.04
    BEEP_VOLUME = 0.25
    beeps = []
    
    for freq in (880, 440):
        beeps.append(BEEP_VOLUME * np.sin(np.linspace(
            0,
            2 * np.pi * freq * BEEP_SECONDS,
            int(BEEP_SECONDS * sample_rate))))
    
    ramp_samples = 100
    beep_envelope = np.concatenate([
        np.linspace(0, 1, ramp_samples),
        np.ones((int(BEEP_SECONDS * sample_rate) - 2 * ramp_samples,)),
        np.linspace(1, 0, ramp_samples)
    ])
    
    for i in range(len(beeps)):
        beeps[i] *= beep_envelope
    
    beat_length = loop_samples // beats_per_loop
    for i in range(beats_per_loop):
        beep = beeps[0 if i == 0 else 1]
        metronome_audio[i * beat_length:i * beat_length + len(beep)] = beep
    
    # Add an extra buffer to the metronome audio to make slicing easier later.
    return np.concatenate([metronome_audio, metronome_audio[:chunk_samples]])


class AudioFade:
    """Handle cross fade between audio chunks.
    
    Args:
        chunk_size: Number of audio samples per predicted frame.
        num_chunks: Number of audio chunks to fade between.
        stereo: Whether the predicted audio is stereo or mono.
    """
    
    def __init__(self, chunk_size: int, num_chunks: int, stereo: bool):
        fade_size = chunk_size * num_chunks
        self.fade_size = fade_size
        self.num_chunks = num_chunks
        
        self.previous_chunk = np.zeros(fade_size)
        self.ramp = np.sin(np.linspace(0, np.pi / 2, fade_size)) ** 2
        
        if stereo:
            self.previous_chunk = self.previous_chunk[:, np.newaxis]
            self.ramp = self.ramp[:, np.newaxis]
    
    def reset(self):
        self.previous_chunk = np.zeros_like(self.previous_chunk)
    
    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        chunk[:self.fade_size] *= self.ramp
        chunk[:self.fade_size] += self.previous_chunk
        self.previous_chunk = chunk[-self.fade_size:] * np.flip(self.ramp)
        return chunk[:-self.fade_size]


@dataclasses.dataclass
class AudioInjectionState:
    """State management for Audio Injection."""
    # The most recent context window (10s) of audio tokens corresponding to the
    # model's predicted output.
    context_tokens_orig: np.ndarray
    # Stores all audio input (mono for live input, stereo for prerecorded input)
    all_inputs: np.ndarray
    # Stores all audio output (stereo)
    all_outputs: np.ndarray
    # How many chunks of audio have been generated
    step: int


class AudioInjectionApp:
    """Main application class for audio injection."""
    
    def __init__(self, config: Config, console: Console):
        self.config = config
        self.console = console
        self.system = None
        self.streamer = None
        self.input_audio = None
        self.metronome_audio = None
        
    def run(self):
        """Run the main application."""
        try:
            self._initialize()
            self._load_audio()
            self._setup_streaming()
            self._run_session()
        except Exception as e:
            self.console.print(f"[red]Application error: {e}[/red]")
            raise
    
    def _initialize(self):
        """Initialize the Magenta RT system."""
        if not MAGENTA_RT_AVAILABLE:
            self.console.print("[red]Magenta RT is not available. Please install magenta-realtime.[/red]")
            sys.exit(1)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Initializing Magenta RT model...", total=None)
            
            # Initialize SpectroStream model
            spectrostream_model = spectrostream.SpectroStreamJAX(lazy=False)
            
            # Initialize Magenta RT system
            device_str = f"{self.config.device}:v2-8" if self.config.device == "tpu" else self.config.device
            self.system = MagentaRTCFGTied(
                tag=self.config.model_tag,
                device=device_str,
                lazy=False
            )
            
            progress.update(task, completed=True)
        
        self.console.print("[green]✓ Magenta RT model initialized[/green]")
    
    def _load_audio(self):
        """Load input audio if using file input."""
        if self.config.use_prerecorded_input and self.config.audio_file:
            if not self.config.audio_file.exists():
                raise FileNotFoundError(f"Audio file not found: {self.config.audio_file}")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Loading audio file...", total=None)
                
                audio_samples = load_audio(self.config.audio_file, self.config.audio.sample_rate)
                
                # Postprocess audio
                if audio_samples.ndim == 2:
                    audio_samples = audio_samples.T
                else:
                    audio_samples = np.tile(audio_samples[:, None], 2)
                
                # Add one buffer of looped audio to make slicing easier later
                chunk_samples = self.config.audio.chunk_samples
                self.input_audio = np.concatenate([audio_samples, audio_samples[:chunk_samples]])
                
                progress.update(task, completed=True)
            
            self.console.print(f"[green]✓ Audio loaded: {self.config.audio_file.name}[/green]")
            self.console.print(f"  Duration: {len(audio_samples) / self.config.audio.sample_rate:.2f} seconds")
        
        # Generate metronome audio
        self.metronome_audio = get_metronome_audio(
            self.config.loop_samples,
            self.config.beats_per_loop,
            self.config.audio.sample_rate,
            self.config.audio.chunk_samples
        )
    
    def _setup_streaming(self):
        """Set up the audio streaming components."""
        # This would integrate with the AudioInjectionStreamer from the original code
        # For now, we'll create a placeholder
        self.console.print("[yellow]Audio streaming setup (placeholder)[/yellow]")
    
    def _run_session(self):
        """Run the main audio generation session."""
        self.console.print(Panel.fit(
            "[bold green]Starting Audio Injection Session[/bold green]\n"
            f"[dim]Model will join after {self.config.intro_loops * self.config.loop_seconds:.2f} seconds[/dim]",
            border_style="green"
        ))
        
        if not Confirm.ask("Ready to start?"):
            self.console.print("[yellow]Session cancelled[/yellow]")
            return
        
        # Main generation loop would go here
        # This is where the original AudioInjectionStreamer.generate method logic would be implemented
        self.console.print("[yellow]Session loop (placeholder)[/yellow]")
        
        # For now, just show a mock progress
        with Progress(console=self.console) as progress:
            task = progress.add_task("Generating audio...", total=100)
            for i in range(100):
                # Simulate processing
                import time
                time.sleep(0.1)
                progress.update(task, advance=1)
        
        self.console.print("[green]✓ Session completed[/green]")


# Mock implementation of MagentaRTCFGTied for development
class MagentaRTCFGTied:
    """Mock implementation of the custom Magenta RT system."""
    
    def __init__(self, tag: str, device: str, lazy: bool = False):
        self.tag = tag
        self.device = device
        self.config = MockConfig()
    
    def generate_chunk(self, **kwargs):
        # Mock implementation
        return None, None


class MockConfig:
    """Mock configuration for development."""
    context_length = 10.0
    codec_frame_rate = 25.0
    crossfade_length = 0.04
    frame_length_samples = 1920
    decoder_codec_rvq_depth = 16