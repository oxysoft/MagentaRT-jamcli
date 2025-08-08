"""Terminal User Interface for Magenta RT JAM CLI using prompt_toolkit."""

import asyncio
import time
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from prompt_toolkit import Application
from prompt_toolkit.application import get_app
from prompt_toolkit.layout import Layout, HSplit, VSplit, Window
from prompt_toolkit.layout.containers import Float, FloatContainer
from prompt_toolkit.widgets import Frame, Box, ProgressBar
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console

from .config import Config


@dataclass
class AudioStats:
    """Audio statistics for display."""
    runtime: float = 0.0
    chunks_processed: int = 0
    input_rms: float = 0.0
    output_rms: float = 0.0
    latency_ms: float = 0.0
    processing_efficiency: float = 0.0
    gpu_usage: float = 0.0
    memory_usage: float = 0.0
    generation_rate: float = 0.0
    buffer_health: float = 100.0
    
    # Volume levels (0-100 for display)
    input_volume: float = 0.0
    output_volume: float = 0.0
    mixed_volume: float = 0.0


@dataclass
class TUIState:
    """State management for TUI."""
    is_running: bool = False
    is_recording: bool = False
    session_start_time: Optional[datetime] = None
    last_update: datetime = field(default_factory=datetime.now)
    stats: AudioStats = field(default_factory=AudioStats)
    status_message: str = "Ready to start"
    model_status: str = "Loading..."
    device_status: str = "Checking..."


class LiveVolumeBar:
    """Custom volume bar with peak indicators."""
    
    def __init__(self, title: str, width: int = 40):
        self.title = title
        self.width = width
        self.current_level = 0.0
        self.peak_level = 0.0
        self.peak_time = time.time()
        self.peak_hold_duration = 1.0  # seconds
    
    def update(self, level: float):
        """Update volume level (0-100)."""
        self.current_level = max(0, min(100, level))
        
        # Update peak hold
        current_time = time.time()
        if self.current_level > self.peak_level or (current_time - self.peak_time) > self.peak_hold_duration:
            self.peak_level = self.current_level
            self.peak_time = current_time
    
    def render(self) -> str:
        """Render the volume bar as formatted text."""
        # Calculate bar segments
        filled_chars = int((self.current_level / 100.0) * self.width)
        peak_pos = int((self.peak_level / 100.0) * self.width)
        
        # Color zones: green (0-60), yellow (60-80), red (80-100)
        bar = ""
        for i in range(self.width):
            if i < filled_chars:
                if i < self.width * 0.6:
                    bar += "█"  # Green zone
                elif i < self.width * 0.8:
                    bar += "█"  # Yellow zone  
                else:
                    bar += "█"  # Red zone
            elif i == peak_pos and peak_pos > filled_chars:
                bar += "▎"  # Peak indicator
            else:
                bar += "░"  # Empty
        
        # Add volume percentage
        level_str = f"{self.current_level:5.1f}%"
        
        return f"{self.title:12} │{bar}│ {level_str}"


class MagentaTUI:
    """Main TUI application class."""
    
    def __init__(self, config: Config, console: Console):
        self.config = config
        self.console = console
        self.state = TUIState()
        
        # Volume bars
        self.input_bar = LiveVolumeBar("Input", width=30)
        self.output_bar = LiveVolumeBar("AI Output", width=30)
        self.mixed_bar = LiveVolumeBar("Mixed", width=30)
        
        # Audio callback reference
        self.audio_callback = None
        
        # Setup TUI components
        self._setup_key_bindings()
        self._setup_style()
        self._setup_layout()
        
        # Setup application
        self.app = Application(
            layout=self.layout,
            key_bindings=self.key_bindings,
            style=self.style,
            full_screen=True,
            mouse_support=True,
        )
    
    def _setup_key_bindings(self):
        """Setup keyboard shortcuts."""
        self.key_bindings = KeyBindings()
        
        @self.key_bindings.add('q')
        @self.key_bindings.add('c-c')
        def quit_app(event):
            """Quit the application."""
            self.state.is_running = False
            event.app.exit()
        
        @self.key_bindings.add('space')
        def toggle_recording(event):
            """Toggle recording on/off."""
            if not self.state.is_recording:
                self.start_session()
            else:
                self.stop_session()
        
        @self.key_bindings.add('r')
        def reset_stats(event):
            """Reset statistics."""
            self.reset_statistics()
    
    def _setup_style(self):
        """Setup TUI styling."""
        self.style = Style([
            ('title', 'bold cyan'),
            ('subtitle', 'bold yellow'),
            ('info', 'cyan'),
            ('success', 'green'),
            ('warning', 'yellow'),
            ('error', 'red'),
            ('highlight', 'bold white'),
            ('dim', 'white'),
            ('volume-low', 'green'),
            ('volume-mid', 'yellow'),
            ('volume-high', 'red'),
            ('border', 'cyan'),
            ('status-bar', 'bold white bg:blue'),
        ])
    
    def _setup_layout(self):
        """Setup TUI layout."""
        # Header
        header = Window(
            content=FormattedTextControl(self._get_header_text),
            height=3,
            style='class:title'
        )
        
        # Configuration panel
        config_panel = Frame(
            body=Window(
                content=FormattedTextControl(self._get_config_text),
                height=10,
            ),
            title="Configuration"
        )
        
        # Volume meters
        volume_panel = Frame(
            body=Window(
                content=FormattedTextControl(self._get_volume_text),
                height=6,
            ),
            title="Live Volume Levels"
        )
        
        # Statistics panel  
        stats_panel = Frame(
            body=Window(
                content=FormattedTextControl(self._get_stats_text),
                height=8,
            ),
            title="Session Statistics"
        )
        
        # Status panel
        status_panel = Frame(
            body=Window(
                content=FormattedTextControl(self._get_status_text),
                height=4,
            ),
            title="System Status"
        )
        
        # Help panel
        help_panel = Frame(
            body=Window(
                content=FormattedTextControl(self._get_help_text),
                height=4,
            ),
            title="Controls"
        )
        
        # Main layout
        self.layout = Layout(
            HSplit([
                header,
                HSplit([
                    VSplit([
                        config_panel,
                        volume_panel,
                    ], padding=1),
                    VSplit([
                        stats_panel,
                        status_panel,
                    ], padding=1),
                ]),
                help_panel,
            ], padding=1)
        )
    
    def _get_header_text(self):
        """Get header text."""
        return HTML(
            '<title>🎵 Magenta RT JAM CLI - Terminal Interface</title>\n'
            '<subtitle>Real-time AI Audio Generation</subtitle>\n'
            '<dim>Press SPACE to start/stop, Q to quit</dim>'
        )
    
    def _get_config_text(self):
        """Get configuration panel text."""
        config_lines = [
            f"<info>Model:</info> <highlight>{self.config.model_tag}</highlight>",
            f"<info>Device:</info> <highlight>{self.config.device}</highlight>",
            f"<info>Input:</info> <highlight>{self.config.input_source}</highlight>",
            f"<info>Sample Rate:</info> <highlight>{self.config.audio.sample_rate} Hz</highlight>",
            f"<info>BPM:</info> <highlight>{self.config.bpm}</highlight>",
            f"<info>Beats/Loop:</info> <highlight>{self.config.beats_per_loop}</highlight>",
            f"<info>Temperature:</info> <highlight>{self.config.model.temperature}</highlight>",
            f"<info>Chunk Size:</info> <highlight>{self.config.audio.chunk_seconds}s</highlight>",
        ]
        return HTML('\n'.join(config_lines))
    
    def _get_volume_text(self):
        """Get volume meters text."""
        lines = [
            self.input_bar.render(),
            self.output_bar.render(),
            self.mixed_bar.render(),
            "",
            "<dim>Volume levels update in real-time during session</dim>",
        ]
        return HTML('\n'.join(lines))
    
    def _get_stats_text(self):
        """Get statistics panel text."""
        stats = self.state.stats
        runtime_str = str(timedelta(seconds=int(stats.runtime)))
        
        stats_lines = [
            f"<info>Runtime:</info> <highlight>{runtime_str}</highlight>",
            f"<info>Chunks Processed:</info> <highlight>{stats.chunks_processed}</highlight>",
            f"<info>Generation Rate:</info> <highlight>{stats.generation_rate:.1f} chunks/s</highlight>",
            f"<info>Latency:</info> <highlight>{stats.latency_ms:.1f}ms</highlight>",
            f"<info>Processing:</info> <highlight>{stats.processing_efficiency:.1f}%</highlight>",
            f"<info>Buffer Health:</info> <highlight>{stats.buffer_health:.1f}%</highlight>",
        ]
        return HTML('\n'.join(stats_lines))
    
    def _get_status_text(self):
        """Get status panel text."""
        recording_status = "<success>🔴 RECORDING</success>" if self.state.is_recording else "<dim>⏸ STOPPED</dim>"
        
        status_lines = [
            f"<info>Status:</info> {recording_status}",
            f"<info>Model:</info> <highlight>{self.state.model_status}</highlight>", 
            f"<info>Audio:</info> <highlight>{self.state.device_status}</highlight>",
        ]
        return HTML('\n'.join(status_lines))
    
    def _get_help_text(self):
        """Get help panel text."""
        help_lines = [
            "<info>SPACE:</info> Start/Stop Recording  <info>R:</info> Reset Stats  <info>Q/Ctrl+C:</info> Quit",
            f"<dim>Last Update: {self.state.last_update.strftime('%H:%M:%S')}</dim>",
        ]
        return HTML('\n'.join(help_lines))
    
    def update_volume_levels(self, input_rms: float, output_rms: float, mixed_rms: float):
        """Update volume levels from audio processing."""
        # Convert RMS to percentage (0-100) with reasonable scaling
        def rms_to_percentage(rms_val: float) -> float:
            if rms_val <= 0:
                return 0.0
            # Scale logarithmically: -60dB to 0dB maps to 0-100%
            db_val = 20 * np.log10(max(rms_val, 1e-6))
            percentage = max(0, min(100, (db_val + 60) / 60 * 100))
            return percentage
        
        input_pct = rms_to_percentage(input_rms)
        output_pct = rms_to_percentage(output_rms) 
        mixed_pct = rms_to_percentage(mixed_rms)
        
        self.input_bar.update(input_pct)
        self.output_bar.update(output_pct)
        self.mixed_bar.update(mixed_pct)
        
        # Update stats
        self.state.stats.input_rms = input_rms
        self.state.stats.output_rms = output_rms
        self.state.stats.input_volume = input_pct
        self.state.stats.output_volume = output_pct
        self.state.stats.mixed_volume = mixed_pct
        
        self._update_timestamp()
    
    def update_stats(self, **kwargs):
        """Update statistics."""
        for key, value in kwargs.items():
            if hasattr(self.state.stats, key):
                setattr(self.state.stats, key, value)
        
        self._update_timestamp()
    
    def set_status(self, model_status: str = None, device_status: str = None):
        """Set status messages."""
        if model_status:
            self.state.model_status = model_status
        if device_status:
            self.state.device_status = device_status
        
        self._update_timestamp()
    
    def start_session(self):
        """Start recording session."""
        if not self.state.is_recording:
            self.state.is_recording = True
            self.state.session_start_time = datetime.now()
            self.state.status_message = "Recording started"
    
    def stop_session(self):
        """Stop recording session.""" 
        if self.state.is_recording:
            self.state.is_recording = False
            self.state.status_message = "Recording stopped"
    
    def reset_statistics(self):
        """Reset all statistics."""
        self.state.stats = AudioStats()
        self.state.session_start_time = datetime.now() if self.state.is_recording else None
        self.state.status_message = "Statistics reset"
    
    def _update_timestamp(self):
        """Update last update timestamp."""
        self.state.last_update = datetime.now()
        
        # Update runtime if recording
        if self.state.is_recording and self.state.session_start_time:
            self.state.stats.runtime = (datetime.now() - self.state.session_start_time).total_seconds()
    
    async def run_async(self):
        """Run the TUI application asynchronously."""
        self.state.is_running = True
        
        # Start the refresh task
        refresh_task = asyncio.create_task(self._refresh_loop())
        
        try:
            # Run the application
            await self.app.run_async()
        finally:
            self.state.is_running = False
            refresh_task.cancel()
            try:
                await refresh_task
            except asyncio.CancelledError:
                pass
    
    async def _refresh_loop(self):
        """Refresh the TUI at regular intervals."""
        while self.state.is_running:
            try:
                # Trigger a refresh
                if self.app.is_running:
                    self.app.invalidate()
                await asyncio.sleep(0.1)  # 10 FPS refresh rate
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue
                pass
    
    def run(self):
        """Run the TUI application (blocking)."""
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            pass