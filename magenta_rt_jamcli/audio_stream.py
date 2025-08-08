"""Real-time audio streaming implementation for Magenta RT JAM CLI."""

import time
import threading
from queue import Queue, Empty
from typing import Optional, Callable, Any
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.table import Table

from .config import Config


@dataclass
class AudioStreamStats:
    """Statistics for audio streaming."""
    chunks_processed: int = 0
    total_samples: int = 0
    underruns: int = 0
    overruns: int = 0
    latency_ms: float = 0.0
    cpu_load: float = 0.0


class AudioStreamer:
    """Real-time audio streaming with PortAudio via sounddevice."""
    
    def __init__(
        self, 
        config: Config, 
        console: Console,
        audio_callback: Callable[[np.ndarray], np.ndarray]
    ):
        self.config = config
        self.console = console
        self.audio_callback = audio_callback
        
        # Audio parameters
        self.sample_rate = config.audio.sample_rate
        self.chunk_samples = config.audio.chunk_samples
        self.channels = 2  # Always stereo output
        
        # Streaming state
        self.is_streaming = False
        self.stream = None
        self.input_queue = Queue(maxsize=10)
        self.output_queue = Queue(maxsize=10)
        self.processing_thread = None
        
        # Statistics
        self.stats = AudioStreamStats()
        
        # Audio buffers
        self.silence = np.zeros((self.chunk_samples, self.channels), dtype=np.float32)
        
    def list_audio_devices(self) -> Table:
        """List available audio devices in a Rich table."""
        devices = sd.query_devices()
        
        table = Table(title="Available Audio Devices", show_header=True)
        table.add_column("ID", style="dim", width=4)
        table.add_column("Name", style="bold")
        table.add_column("Channels", justify="center", width=8)
        table.add_column("Type", width=8)
        table.add_column("Sample Rate", justify="right", width=12)
        
        for i, device in enumerate(devices):
            device_type = []
            if device['max_input_channels'] > 0:
                device_type.append("Input")
            if device['max_output_channels'] > 0:
                device_type.append("Output")
            
            table.add_row(
                str(i),
                device['name'][:40],
                f"I:{device['max_input_channels']} O:{device['max_output_channels']}",
                " & ".join(device_type),
                f"{device['default_samplerate']:.0f} Hz"
            )
        
        return table
    
    def test_audio_devices(self) -> bool:
        """Test audio device configuration."""
        try:
            # Test input device if using microphone
            if not self.config.use_prerecorded_input:
                self.console.print("[blue]Testing microphone input...[/blue]")
                with sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    blocksize=self.chunk_samples,
                    dtype=np.float32
                ) as stream:
                    # Record a brief test
                    test_data, _ = stream.read(self.chunk_samples)
                    rms = np.sqrt(np.mean(test_data ** 2))
                    
                    if rms < 1e-6:
                        self.console.print("[yellow]⚠ Very low input level detected. Check microphone.[/yellow]")
                    else:
                        self.console.print(f"[green]✓ Microphone working (RMS: {rms:.6f})[/green]")
            
            # Test output device
            self.console.print("[blue]Testing audio output...[/blue]")
            with sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.chunk_samples,
                dtype=np.float32
            ) as stream:
                # Play a brief test tone
                t = np.linspace(0, 0.1, int(0.1 * self.sample_rate))
                test_tone = 0.1 * np.sin(2 * np.pi * 440 * t)
                test_stereo = np.column_stack([test_tone, test_tone])
                stream.write(test_stereo)
            
            self.console.print("[green]✓ Audio output working[/green]")
            return True
            
        except Exception as e:
            self.console.print(f"[red]Audio device test failed: {e}[/red]")
            return False
    
    def _audio_stream_callback(self, indata, outdata, frames, time, status):
        """PortAudio callback for real-time audio processing."""
        if status:
            if status.input_underflow:
                self.stats.underruns += 1
            if status.output_underflow:
                self.stats.underruns += 1
            if status.input_overflow:
                self.stats.overruns += 1
            if status.output_overflow:  
                self.stats.overruns += 1
        
        # Calculate latency
        self.stats.latency_ms = time.outputBufferDacTime - time.inputAdcTime if time.inputAdcTime else 0
        
        try:
            # Put input data in queue for processing
            if not self.config.use_prerecorded_input and indata is not None:
                input_mono = np.mean(indata, axis=1) if indata.shape[1] > 1 else indata[:, 0]
                self.input_queue.put_nowait(input_mono.copy())
            
            # Get processed output from queue
            try:
                processed_audio = self.output_queue.get_nowait()
                outdata[:] = processed_audio
                self.stats.chunks_processed += 1
                self.stats.total_samples += frames
            except Empty:
                # No processed audio available, output silence
                outdata[:] = self.silence
                self.stats.underruns += 1
        
        except Exception as e:
            # Log error but don't crash the audio thread
            outdata[:] = self.silence
    
    def _processing_thread_func(self):
        """Background thread for audio processing."""
        while self.is_streaming:
            try:
                if self.config.use_prerecorded_input:
                    # For file input, generate chunks at regular intervals
                    input_chunk = None  # Will be handled by the callback
                    time.sleep(self.chunk_samples / self.sample_rate)
                else:
                    # Wait for microphone input
                    try:
                        input_chunk = self.input_queue.get(timeout=0.1)
                    except Empty:
                        continue
                
                # Process audio through the model
                try:
                    processed_chunk = self.audio_callback(input_chunk)
                    
                    # Ensure stereo output
                    if processed_chunk.ndim == 1:
                        processed_chunk = np.column_stack([processed_chunk, processed_chunk])
                    
                    # Put processed audio in output queue
                    self.output_queue.put_nowait(processed_chunk)
                    
                except Exception as e:
                    self.console.print(f"[red]Processing error: {e}[/red]")
                    # Output silence on error
                    self.output_queue.put_nowait(self.silence.copy())
            
            except Exception as e:
                self.console.print(f"[red]Processing thread error: {e}[/red]")
                time.sleep(0.01)  # Prevent busy loop
    
    def start_streaming(self):
        """Start real-time audio streaming."""
        if self.is_streaming:
            return
        
        self.console.print("[blue]Starting audio stream...[/blue]")
        
        # Reset statistics
        self.stats = AudioStreamStats()
        
        # Start processing thread
        self.is_streaming = True
        self.processing_thread = threading.Thread(
            target=self._processing_thread_func, 
            daemon=True
        )
        self.processing_thread.start()
        
        # Configure stream parameters
        stream_kwargs = {
            'samplerate': self.sample_rate,
            'blocksize': self.chunk_samples,
            'dtype': np.float32,
            'callback': self._audio_stream_callback,
            'channels': [1 if not self.config.use_prerecorded_input else 0, self.channels],
            'latency': 'low'  # Request low latency
        }
        
        # Start audio stream
        try:
            self.stream = sd.Stream(**stream_kwargs)
            self.stream.start()
            self.console.print("[green]✓ Audio streaming started[/green]")
        except Exception as e:
            self.is_streaming = False
            self.console.print(f"[red]Failed to start audio stream: {e}[/red]")
            raise
    
    def stop_streaming(self):
        """Stop audio streaming."""
        if not self.is_streaming:
            return
        
        self.console.print("[blue]Stopping audio stream...[/blue]")
        
        # Stop streaming
        self.is_streaming = False
        
        # Stop and close audio stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # Wait for processing thread to finish
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        
        self.console.print("[green]✓ Audio streaming stopped[/green]")
    
    def create_live_display(self) -> Live:
        """Create a Rich Live display for streaming statistics."""
        def make_stats_panel():
            stats_table = Table.grid(padding=(0, 2))
            stats_table.add_column("Label", style="dim")
            stats_table.add_column("Value", style="bold")
            
            stats_table.add_row("Chunks Processed:", f"{self.stats.chunks_processed:,}")
            stats_table.add_row("Total Samples:", f"{self.stats.total_samples:,}")
            stats_table.add_row("Underruns:", f"{self.stats.underruns:,}")
            stats_table.add_row("Overruns:", f"{self.stats.overruns:,}")
            stats_table.add_row("Latency:", f"{self.stats.latency_ms*1000:.1f} ms")
            
            # Calculate streaming duration
            if self.stats.total_samples > 0:
                duration = self.stats.total_samples / self.sample_rate
                stats_table.add_row("Duration:", f"{duration:.1f}s")
            
            return Panel(
                stats_table,
                title="[bold blue]Audio Stream Statistics[/bold blue]",
                border_style="blue"
            )
        
        return Live(make_stats_panel(), refresh_per_second=4, console=self.console)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_streaming()