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
from .model_manager import ModelManager
from .audio_stream import AudioStreamer

# Import Magenta RT components (these would need to be installed)
try:
    import jax
    import jax.numpy as jnp
    from magenta_rt import audio as audio_lib
    from magenta_rt import musiccoca
    from magenta_rt import spectrostream
    from magenta_rt import system
    from magenta_rt import utils
    MAGENTA_RT_AVAILABLE = True
except ImportError:
    MAGENTA_RT_AVAILABLE = False
    jax = None
    jnp = None


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
        self.spectrostream_model = None
        self.model_manager = ModelManager(console)
        self.audio_streamer = None
        
        # Audio injection state
        self.injection_state = None
        self.fade = None
        
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
            self.console.print("[red]Magenta RT is not available.[/red]")
            self.console.print("Please install it with:")
            self.console.print("  git clone https://github.com/magenta/magenta-realtime.git")
            self.console.print("  pip install -e magenta-realtime/[tpu]  # or [gpu] for GPU")
            sys.exit(1)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            # Download model if needed
            task1 = progress.add_task("Checking model availability...", total=None)
            model_path = self.model_manager.download_model(self.config.model_tag)
            progress.update(task1, completed=True, description="✓ Model ready")
            
            # Initialize SpectroStream model
            task2 = progress.add_task("Loading SpectroStream codec...", total=None)
            self.spectrostream_model = spectrostream.SpectroStreamJAX(lazy=False)
            progress.update(task2, completed=True, description="✓ Codec loaded")
            
            # Initialize Magenta RT system
            task3 = progress.add_task("Initializing MagentaRT system...", total=None)
            device_str = f"{self.config.device}:v2-8" if self.config.device == "tpu" else self.config.device
            self.system = MagentaRTCFGTied(
                tag=self.config.model_tag,
                device=device_str,
                lazy=False
            )
            progress.update(task3, completed=True, description="✓ System initialized")
            
            # Initialize audio processing components
            task4 = progress.add_task("Setting up audio processing...", total=None)
            self._setup_audio_processing()
            progress.update(task4, completed=True, description="✓ Audio ready")
        
        self.console.print("[green]✓ Magenta RT system fully initialized[/green]")
    
    def _setup_audio_processing(self):
        """Set up audio processing components."""
        config = self.system.config
        
        # Initialize crossfade processor
        self.fade = AudioFade(
            chunk_size=int(config.codec_sample_rate * config.crossfade_length),
            num_chunks=1,
            stereo=True
        )
        
        # Initialize injection state
        context_seconds = config.context_length
        context_frames = int(context_seconds * config.codec_frame_rate)
        context_samples = int(context_seconds * self.config.audio.sample_rate)
        
        self.injection_state = AudioInjectionState(
            context_tokens_orig=np.zeros(
                (context_frames, config.decoder_codec_rvq_depth),
                dtype=np.int32
            ),
            all_inputs=np.zeros(
                (context_samples, 2) if self.config.use_prerecorded_input 
                else (context_samples,),
                dtype=np.float32
            ),
            all_outputs=np.zeros((context_samples, 2), dtype=np.float32),
            step=-1,  # Will be 0 after warmup
        )
    
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
        # Create audio streamer with our generation callback
        self.audio_streamer = AudioStreamer(
            config=self.config,
            console=self.console,
            audio_callback=self._audio_generation_callback
        )
        
        # Test audio devices
        if not self.config.use_prerecorded_input:
            device_table = self.audio_streamer.list_audio_devices()
            self.console.print(device_table)
            
            if not self.audio_streamer.test_audio_devices():
                raise RuntimeError("Audio device test failed")
    
    def _audio_generation_callback(self, input_chunk: Optional[np.ndarray]) -> np.ndarray:
        """Audio generation callback for real-time processing."""
        try:
            return self._generate_audio_chunk(input_chunk)
        except Exception as e:
            self.console.print(f"[red]Generation error: {e}[/red]")
            # Return silence on error
            return np.zeros((self.config.audio.chunk_samples, 2), dtype=np.float32)
    
    def _generate_audio_chunk(self, input_chunk: Optional[np.ndarray]) -> np.ndarray:
        """Generate a single audio chunk using the Magenta RT model."""
        chunk_samples = self.config.audio.chunk_samples
        
        # Handle input audio
        if self.config.use_prerecorded_input:
            # Get next chunk from pre-recorded audio
            if self.input_audio is not None:
                start = (self.injection_state.step * chunk_samples) % (
                    len(self.input_audio) - chunk_samples)
                end = start + chunk_samples
                inputs = self.input_audio[start:end]
                inputs_mono = np.mean(inputs, axis=-1)
            else:
                inputs = np.zeros((chunk_samples, 2), dtype=np.float32)
                inputs_mono = np.zeros(chunk_samples, dtype=np.float32)
        else:
            # Use microphone input
            inputs_mono = input_chunk if input_chunk is not None else np.zeros(chunk_samples)
            inputs = np.column_stack([inputs_mono, inputs_mono])  # Convert to stereo
        
        # Add input to injection state
        self.injection_state.all_inputs = np.concatenate(
            [self.injection_state.all_inputs, inputs], axis=0
        )
        
        # Calculate mixing parameters
        mix_samples = (
            chunk_samples + self.config.mix_prefix_frames
            * self.system.config.frame_length_samples
        )
        
        # Calculate I/O offset for audio delay
        io_offset = chunk_samples - int(
            self.system.config.crossfade_length * self.config.audio.sample_rate)
        if not self.config.use_prerecorded_input:
            io_offset += self.config.loop_samples
        
        if io_offset < 0:
            io_offset = 0
        
        # Select windows for mixing
        inputs_to_mix = self.injection_state.all_inputs[
            -(io_offset + mix_samples):-io_offset if io_offset > 0 else None
        ]
        outputs_to_mix = self.injection_state.all_outputs[-mix_samples:]
        outputs_to_mix *= self.config.model.model_feedback
        
        # Apply input gap (silence recent input to discourage copying)
        input_gap_samples = int(self.config.audio.sample_rate * self.config.model.input_gap / 1000)
        if input_gap_samples > 0:
            ramp_samples = min(100, input_gap_samples)
            ramp = np.linspace(1, 0, ramp_samples)
            if self.config.use_prerecorded_input:
                ramp = np.column_stack([ramp, ramp])
            
            envelope = np.concatenate([
                np.ones_like(inputs_to_mix[input_gap_samples:]),
                ramp,
                np.zeros_like(inputs_to_mix[:max(0, input_gap_samples - ramp_samples)])
            ])
            inputs_to_mix = inputs_to_mix * envelope
        
        # Mix input and output audio
        if not self.config.use_prerecorded_input:
            inputs_to_mix = inputs_to_mix[:, None]  # Add channel dimension for mono input
        
        mix_audio = audio_lib.Waveform(
            inputs_to_mix + outputs_to_mix,
            sample_rate=self.config.audio.sample_rate
        )
        
        # Encode mixed audio to tokens
        mix_tokens = self.spectrostream_model.encode(mix_audio)[
            self.config.left_edge_frames_to_remove:
        ]
        
        # Update model state with mixed tokens
        state = self.system.init_state() if self.injection_state.step < 0 else getattr(self, '_model_state', None)
        if state is not None:
            self.injection_state.context_tokens_orig = state.context_tokens.copy()
            state.context_tokens[-len(mix_tokens):] = mix_tokens[
                :, :self.system.config.decoder_codec_rvq_depth]
        
        # Generate new audio chunk
        max_decode_frames = round(
            self.config.audio.chunk_seconds * self.system.config.codec_frame_rate)
        
        # Get style embedding (for now use default)
        style_embedding = None  # Could be implemented with prompts
        
        chunk_waveform, new_state = self.system.generate_chunk(
            state=state,
            style=style_embedding,
            seed=None,
            max_decode_frames=max_decode_frames,
            context_tokens_orig=self.injection_state.context_tokens_orig,
            **{
                'temperature': self.config.model.temperature,
                'topk': self.config.model.topk,
                'guidance_weight': self.config.model.guidance_weight,
            }
        )
        
        # Store updated state
        self._model_state = new_state
        
        # Add chunk to outputs (before crossfading)
        chunk_samples_raw = chunk_waveform.samples[self.fade.fade_size:]
        self.injection_state.all_outputs = np.concatenate([
            self.injection_state.all_outputs,
            chunk_samples_raw
        ])
        
        # Apply crossfading
        chunk_faded = self.fade(chunk_waveform.samples)
        chunk_faded *= self.config.model.model_volume
        
        # Add metronome if enabled and we're in intro phase
        intro_chunks = int(self.config.intro_loops * self.config.loop_samples / chunk_samples)
        if (hasattr(self.config.model, 'metronome') and 
            getattr(self.config.model, 'metronome', False) and 
            self.injection_state.step < intro_chunks):
            
            start = (self.injection_state.step * chunk_samples) % self.config.loop_samples
            end = start + chunk_samples
            metronome_chunk = self.metronome_audio[start:end]
            chunk_faded += metronome_chunk[:, None]
        
        # Add input audio for playthrough (if using file input)
        if self.config.use_prerecorded_input:
            chunk_faded += inputs * self.config.model.input_volume
        
        # Update step counter
        self.injection_state.step += 1
        
        return chunk_faded.astype(np.float32)
    
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
        
        try:
            with self.audio_streamer:
                # Start streaming
                self.audio_streamer.start_streaming()
                
                # Create live statistics display
                with self.audio_streamer.create_live_display():
                    self.console.print("[green]🎵 Audio session running! Press Ctrl+C to stop[/green]")
                    
                    # Keep running until interrupted
                    try:
                        while True:
                            time.sleep(0.1)
                    except KeyboardInterrupt:
                        self.console.print("\n[yellow]Stopping session...[/yellow]")
                        
        except Exception as e:
            self.console.print(f"[red]Session error: {e}[/red]")
            raise
        finally:
            self.console.print("[green]✓ Session completed[/green]")
            self._save_session_output()
    
    def _save_session_output(self):
        """Save the generated audio session to files."""
        if self.injection_state is None or self.injection_state.step <= 0:
            self.console.print("[yellow]No audio generated to save[/yellow]")
            return
        
        try:
            import soundfile as sf
            from datetime import datetime
            
            # Create output directory
            output_dir = Path("jamcli_output")
            output_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Trim context samples (keep only generated audio)
            context_seconds = self.system.config.context_length if MAGENTA_RT_AVAILABLE else 10.0
            context_samples = int(context_seconds * self.config.audio.sample_rate)
            
            generated_inputs = self.injection_state.all_inputs[context_samples:]
            generated_outputs = self.injection_state.all_outputs[context_samples:]
            
            if len(generated_outputs) > 0:
                # Save output audio (stereo)
                output_file = output_dir / f"jamcli_output_{timestamp}.wav"
                sf.write(output_file, generated_outputs, self.config.audio.sample_rate)
                self.console.print(f"[green]✓ Output saved: {output_file}[/green]")
                
                # Save input audio for comparison
                if len(generated_inputs) > 0:
                    if self.config.use_prerecorded_input and generated_inputs.ndim == 2:
                        input_file = output_dir / f"jamcli_input_{timestamp}.wav" 
                        sf.write(input_file, generated_inputs, self.config.audio.sample_rate)
                    else:
                        # Mono microphone input
                        input_file = output_dir / f"jamcli_input_{timestamp}.wav"
                        sf.write(input_file, generated_inputs, self.config.audio.sample_rate)
                    
                    self.console.print(f"[green]✓ Input saved: {input_file}[/green]")
                
                # Save mixed output (input + model output)
                if self.config.use_prerecorded_input:
                    # Delay input to align with output
                    delay_samples = int(self.system.config.crossfade_length * self.config.audio.sample_rate) if MAGENTA_RT_AVAILABLE else 1920
                    delayed_inputs = np.concatenate([
                        generated_inputs[delay_samples:],
                        np.zeros((delay_samples, generated_inputs.shape[1]))
                    ])
                    
                    # Mix for stereo output
                    mixed_audio = delayed_inputs * 0.5 + generated_outputs * 0.5
                    mixed_file = output_dir / f"jamcli_mixed_{timestamp}.wav"
                    sf.write(mixed_file, mixed_audio, self.config.audio.sample_rate)
                    self.console.print(f"[green]✓ Mixed audio saved: {mixed_file}[/green]")
                
        except ImportError:
            self.console.print("[yellow]soundfile not available. Install with: pip install soundfile[/yellow]")
        except Exception as e:
            self.console.print(f"[red]Error saving audio: {e}[/red]")


if MAGENTA_RT_AVAILABLE:
    class MagentaRTCFGTied(system.MagentaRTT5X):
        """Magenta RT T5X system with "tied" CFG controlling input and style."""

        def generate_chunk(
            self,
            state: Optional[system.MagentaRTState] = None,
            style: Optional[musiccoca.StyleEmbedding] = None,
            seed: Optional[int] = None,
            **kwargs,
        ) -> tuple[audio_lib.Waveform, system.MagentaRTState]:
            """Generate a chunk of audio and return updated state.

            Args:
              state: The current state of the system.
              style: The style embedding to use for the generation.
              seed: The seed to use for the generation.
              **kwargs: Additional keyword arguments for sampling params, e.g.
                temperature, topk, guidance_weight, max_decode_frames.

            Returns:
              A tuple of the generated audio and the updated state.
            """
            # Init state, style, and seed (if not provided)
            if state is None:
                state = self.init_state()
            if seed is None:
                seed = np.random.randint(0, 2**31)

            context_tokens = {
                "orig": kwargs.get("context_tokens_orig", state.context_tokens),
                "mix": state.context_tokens,
            }
            codec_tokens_lm = {}
            for key, tokens in context_tokens.items():
                # Prepare codec tokens for LLM
                codec_tokens_lm[key] = np.where(
                    tokens >= 0,
                    utils.rvq_to_llm(
                        np.maximum(tokens, 0),
                        self.config.codec_rvq_codebook_size,
                        self.config.vocab_codec_offset,
                    ),
                    np.full_like(tokens, self.config.vocab_mask_token),
                )
                assert (
                    codec_tokens_lm[key].shape == self.config.context_tokens_shape
                )  # (250, 16)
                assert (
                    codec_tokens_lm[key].min() >= self.config.vocab_mask_token
                    and codec_tokens_lm[key].max()
                    < (self.config.vocab_codec_offset + self.config.vocab_codec_size)
                )  # check range [1, 16386)

            # Prepare style tokens for LLM
            if style is None:
                style_tokens_lm = np.full(
                    (self.config.encoder_style_rvq_depth,),
                    self.config.vocab_mask_token,
                    dtype=np.int32,
                )
            else:
                if style.shape != (self.style_model.config.embedding_dim,):
                    raise ValueError(f"Invalid style shape: {style.shape}")
                style_tokens = self.style_model.tokenize(style)
                assert style_tokens.shape == (self.style_model.config.rvq_depth,)
                style_tokens = style_tokens[: self.config.encoder_style_rvq_depth]
                style_tokens_lm = utils.rvq_to_llm(
                    style_tokens,
                    self.config.style_rvq_codebook_size,
                    self.config.vocab_style_offset,
                )
                assert (
                    style_tokens_lm.min() >= self.config.vocab_style_offset
                    and style_tokens_lm.max()
                    < (self.config.vocab_style_offset + self.config.vocab_style_size)
                )  # check range [17140, 23554)
            assert style_tokens_lm.shape == (
                self.config.encoder_style_rvq_depth,
            )  # (6,)

            # Prepare encoder input
            batch_size, _, _ = self._device_params
            encoder_inputs_pos = np.concatenate(
                [codec_tokens_lm["mix"][
                    :, :self.config.encoder_codec_rvq_depth].reshape(-1),
                 style_tokens_lm
                ],
                axis=0,
            )
            assert encoder_inputs_pos.shape == (1006,)

            # Construct negative using original context tokens, and masking style.
            encoder_inputs_neg = np.concatenate(
                [codec_tokens_lm["orig"][
                    :, :self.config.encoder_codec_rvq_depth].reshape(-1),
                 style_tokens_lm
                ],
                axis=0,
            )
            encoder_inputs_neg[-self.config.encoder_style_rvq_depth:] = (
                self.config.vocab_mask_token)
            assert encoder_inputs_neg.shape == (1006,)

            encoder_inputs = np.stack([encoder_inputs_pos, encoder_inputs_neg], axis=0)
            assert encoder_inputs.shape == (2, 1006)

            # Generate tokens / NLL scores.
            max_decode_frames = kwargs.get(
                "max_decode_frames", self.config.chunk_length_frames
            )
            generated_tokens, _ = self._llm(
                {
                    "encoder_input_tokens": encoder_inputs,
                    "decoder_input_tokens": np.zeros(
                        (
                            batch_size,
                            self.config.chunk_length_frames
                            * self.config.decoder_codec_rvq_depth,
                        ),
                        dtype=np.int32,
                    ),
                },
                {
                    "max_decode_steps": np.array(
                        max_decode_frames * self.config.decoder_codec_rvq_depth,
                        dtype=np.int32,
                    ),
                    "guidance_weight": kwargs.get(
                        "guidance_weight", self._guidance_weight
                    ),
                    "temperature": kwargs.get("temperature", self._temperature),
                    "topk": kwargs.get("topk", self._topk),
                },
                jax.random.PRNGKey(seed + state.chunk_index),
            )

            # Process generated tokens
            generated_tokens = np.array(generated_tokens)
            assert generated_tokens.shape == (
                batch_size,
                self.config.chunk_length_frames * self.config.decoder_codec_rvq_depth,
            )
            generated_tokens = generated_tokens[:1]  # larger batch sizes unsupported
            generated_tokens = generated_tokens.reshape(
                self.config.chunk_length_frames, self.config.decoder_codec_rvq_depth
            )  # (50, 16)
            generated_tokens = generated_tokens[:max_decode_frames]  # (N, 16)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                generated_rvq_tokens = utils.llm_to_rvq(
                    generated_tokens,
                    self.config.codec_rvq_codebook_size,
                    self.config.vocab_codec_offset,
                    safe=False,
                )

            # Decode via SpectroStream using additional frame of samples for crossfading
            xfade_frames = state.context_tokens[-self.config.crossfade_length_frames :]
            if state.chunk_index == 0:
                # NOTE: This will create 40ms of gibberish at the beginning but it's OK.
                xfade_frames = np.zeros_like(xfade_frames)
            assert xfade_frames.min() >= 0
            xfade_tokens = np.concatenate([xfade_frames, generated_rvq_tokens], axis=0)
            assert xfade_tokens.shape == (
                self.config.crossfade_length_frames + max_decode_frames,
                self.config.decoder_codec_rvq_depth,
            )  # (N+1, 16)
            waveform = self.codec.decode(xfade_tokens)
            assert isinstance(waveform, audio_lib.Waveform)
            assert waveform.samples.shape == (
                self.config.crossfade_length_samples
                + max_decode_frames * self.config.frame_length_samples,
                self.num_channels,
            )  # ((N+1)*1920, 2)

            # Update state
            state.update(generated_rvq_tokens)

            return (waveform, state)

else:
    # Mock implementation when Magenta RT is not available
    class MagentaRTCFGTied:
        """Mock implementation of MagentaRTCFGTied for development."""
        
        def __init__(self, tag: str, device: str, lazy: bool = False):
            self.tag = tag
            self.device = device
            self.config = MockConfig()
        
        def generate_chunk(self, **kwargs):
            raise RuntimeError("Magenta RT is not available")


# Mock classes for when Magenta RT is not available
class MockSystem:
    """Mock system for development without Magenta RT."""
    def __init__(self, tag: str, device: str, lazy: bool = False):
        self.tag = tag
        self.device = device
        self.config = MockConfig()
    
    def generate_chunk(self, **kwargs):
        return None, None


class MockConfig:
    """Mock configuration for development."""
    context_length = 10.0
    codec_frame_rate = 25.0
    crossfade_length = 0.04
    frame_length_samples = 1920
    decoder_codec_rvq_depth = 16
