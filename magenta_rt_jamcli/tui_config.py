"""Full-screen TUI configuration flow for Magenta RT JAM CLI."""

import asyncio
import threading
import time
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
from prompt_toolkit import Application
from prompt_toolkit.layout import Layout, HSplit, VSplit, Window
from prompt_toolkit.widgets import Frame
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console

from .config import Config
from .model_manager import ModelManager
from .tui import LiveVolumeBar


@dataclass
class ConfigOption:
    """Configuration option with display information."""
    key: str
    label: str
    value: Any
    options: Optional[List[Any]] = None
    description: str = ""


@dataclass
class TUIPage:
    """Represents a TUI configuration page."""
    title: str
    subtitle: str
    options: List[ConfigOption]
    selected_index: int = 0
    show_volume_preview: bool = False


class AudioMonitor:
    """Real-time audio monitoring for device selection."""
    
    def __init__(self):
        self.is_monitoring = False
        self.current_volume = 0.0
        self.volume_bar = LiveVolumeBar("Live Input", width=40)
        self.monitor_thread = None
        self.device_id = None
    
    def start_monitoring(self, device_id: int):
        """Start monitoring audio from the specified device."""
        self.device_id = device_id
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop audio monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """Audio monitoring loop."""
        try:
            def audio_callback(indata, frames, time, status):
                if self.is_monitoring and len(indata) > 0:
                    # Calculate RMS and convert to percentage
                    rms = np.sqrt(np.mean(indata ** 2))
                    if rms > 0:
                        db_val = 20 * np.log10(max(rms, 1e-6))
                        percentage = max(0, min(100, (db_val + 60) / 60 * 100))
                        self.volume_bar.update(percentage)
                        self.current_volume = percentage
            
            with sd.InputStream(
                device=self.device_id,
                channels=1,
                callback=audio_callback,
                blocksize=1024,
                samplerate=48000
            ):
                while self.is_monitoring:
                    time.sleep(0.1)
                    
        except Exception:
            # Device not available or other error
            self.current_volume = 0.0


class ConfigurationFlow:
    """Manages the full TUI configuration flow."""
    
    def __init__(self, config: Config, console: Console):
        self.config = config
        self.console = console
        self.model_manager = ModelManager(console)
        self.audio_monitor = AudioMonitor()
        
        # Current state
        self.current_page_index = 0
        self.pages: List[TUIPage] = []
        self.result_config: Optional[Config] = None
        self.app: Optional[Application] = None
        
        # Setup pages
        self._setup_pages()
        
        # Setup TUI
        self._setup_key_bindings()
        self._setup_style()
        self._setup_layout()
    
    def _setup_pages(self):
        """Setup configuration pages."""
        # Page 1: Input Source Selection
        self.pages.append(TUIPage(
            title="🎵 Audio Input Source",
            subtitle="Choose your audio input method",
            options=[
                ConfigOption("mic", "🎤 Microphone", "mic", description="Live microphone input for real-time jamming"),
                ConfigOption("file", "📁 Audio File", "file", description="Process pre-recorded audio files"),
            ]
        ))
        
        # Page 2: Device Selection (if mic selected)
        if self.config.input_source == "mic":
            devices = self._get_audio_devices()
            device_options = []
            for i, device in enumerate(devices):
                device_options.append(ConfigOption(
                    f"device_{i}", 
                    f"🔊 {device['name']}", 
                    i,
                    description=f"{device['inputs']} inputs, {device['rate']} Hz"
                ))
            
            self.pages.append(TUIPage(
                title="🎛️ Audio Device Selection",
                subtitle="Select your audio input device - live volume preview below",
                options=device_options,
                show_volume_preview=True
            ))
        
        # Page 3: Model Selection
        model_options = []
        for tag, info in self.model_manager.MODEL_CONFIGS.items():
            model_info = self.model_manager.get_model_info(tag)
            status = "✓" if model_info["available"] else "⬇"
            size = model_info.get("size", "Unknown")
            model_options.append(ConfigOption(
                f"model_{tag}",
                f"{status} {tag.title()} Model ({size})",
                tag,
                description=info["description"]
            ))
        
        self.pages.append(TUIPage(
            title="🧠 AI Model Selection",
            subtitle="Choose your AI model - larger models provide better quality",
            options=model_options
        ))
        
        # Page 4: Advanced Settings
        self.pages.append(TUIPage(
            title="⚙️ Advanced Settings",
            subtitle="Fine-tune your session parameters",
            options=[
                ConfigOption("bpm", "🥁 BPM", self.config.bpm, 
                           options=list(range(60, 201, 10)), 
                           description="Beats per minute for timing"),
                ConfigOption("device", "⚡ Device", self.config.device,
                           options=["cpu", "gpu", "mps"],
                           description="Processing device (gpu recommended)"),
                ConfigOption("temperature", "🌡️ Temperature", self.config.model.temperature,
                           options=[0.8, 1.0, 1.2, 1.5, 2.0],
                           description="AI creativity level (higher = more creative)"),
            ]
        ))
        
        # Page 5: Final Confirmation
        self.pages.append(TUIPage(
            title="🚀 Ready to Jam!",
            subtitle="Review your configuration and start the session",
            options=[
                ConfigOption("start", "▶️ Start Jamming Session", "start", 
                           description="Begin real-time AI audio generation"),
                ConfigOption("back", "◀️ Review Settings", "back",
                           description="Go back to modify configuration"),
            ]
        ))
    
    def _get_audio_devices(self) -> List[Dict]:
        """Get available audio input devices."""
        devices = []
        try:
            device_list = sd.query_devices()
            for i, device in enumerate(device_list):
                if device['max_input_channels'] > 0:
                    devices.append({
                        'id': i,
                        'name': device['name'][:40],  # Truncate long names
                        'inputs': device['max_input_channels'],
                        'rate': int(device['default_samplerate'])
                    })
        except Exception:
            # Fallback if audio devices can't be enumerated
            devices.append({
                'id': 0,
                'name': 'Default Audio Device',
                'inputs': 2,
                'rate': 48000
            })
        return devices
    
    def _setup_key_bindings(self):
        """Setup keyboard shortcuts."""
        self.key_bindings = KeyBindings()
        
        @self.key_bindings.add('up')
        def move_up(event):
            """Move selection up."""
            current_page = self.pages[self.current_page_index]
            if current_page.selected_index > 0:
                current_page.selected_index -= 1
                self._handle_selection_change()
        
        @self.key_bindings.add('down')
        def move_down(event):
            """Move selection down."""
            current_page = self.pages[self.current_page_index]
            if current_page.selected_index < len(current_page.options) - 1:
                current_page.selected_index += 1
                self._handle_selection_change()
        
        @self.key_bindings.add('left')
        def previous_option(event):
            """Previous option value (for lists)."""
            self._cycle_option_value(-1)
        
        @self.key_bindings.add('right')  
        def next_option(event):
            """Next option value (for lists)."""
            self._cycle_option_value(1)
        
        @self.key_bindings.add('enter')
        def select_option(event):
            """Select current option."""
            self._handle_enter_pressed()
        
        @self.key_bindings.add('escape')
        @self.key_bindings.add('q')
        def quit_app(event):
            """Quit the application."""
            self.audio_monitor.stop_monitoring()
            event.app.exit()
        
        @self.key_bindings.add('backspace')
        def go_back(event):
            """Go to previous page."""
            if self.current_page_index > 0:
                self.audio_monitor.stop_monitoring()
                self.current_page_index -= 1
                self._handle_page_change()
    
    def _cycle_option_value(self, direction: int):
        """Cycle through option values for the current selection."""
        current_page = self.pages[self.current_page_index]
        option = current_page.options[current_page.selected_index]
        
        if option.options:
            current_idx = 0
            if option.value in option.options:
                current_idx = option.options.index(option.value)
            
            new_idx = (current_idx + direction) % len(option.options)
            option.value = option.options[new_idx]
    
    def _handle_selection_change(self):
        """Handle selection change (for volume monitoring)."""
        current_page = self.pages[self.current_page_index]
        
        if current_page.show_volume_preview:
            # Update audio monitoring for device selection
            option = current_page.options[current_page.selected_index]
            device_id = option.value
            self.audio_monitor.stop_monitoring()
            time.sleep(0.1)  # Brief pause
            self.audio_monitor.start_monitoring(device_id)
    
    def _handle_enter_pressed(self):
        """Handle enter key press."""
        current_page = self.pages[self.current_page_index]
        option = current_page.options[current_page.selected_index]
        
        # Apply the selected configuration
        self._apply_option(option)
        
        # Handle special cases
        if option.key == "start":
            self._finalize_config()
            self.app.exit()
        elif option.key == "back":
            if self.current_page_index > 0:
                self.current_page_index -= 1
                self._handle_page_change()
        else:
            # Move to next page
            if self.current_page_index < len(self.pages) - 1:
                self.audio_monitor.stop_monitoring()
                self.current_page_index += 1
                self._handle_page_change()
    
    def _apply_option(self, option: ConfigOption):
        """Apply the selected option to configuration."""
        if option.key == "mic":
            self.config.input_source = "mic"
        elif option.key == "file":
            self.config.input_source = "file"
        elif option.key.startswith("device_"):
            self.config.audio_device_id = option.value
        elif option.key.startswith("model_"):
            self.config.model_tag = option.value
        elif option.key == "bpm":
            self.config.bpm = option.value
        elif option.key == "device":
            self.config.device = option.value
        elif option.key == "temperature":
            self.config.model.temperature = option.value
    
    def _handle_page_change(self):
        """Handle page change."""
        current_page = self.pages[self.current_page_index]
        
        if current_page.show_volume_preview and current_page.options:
            # Start monitoring for the initially selected device
            option = current_page.options[current_page.selected_index]
            self.audio_monitor.start_monitoring(option.value)
    
    def _finalize_config(self):
        """Finalize the configuration."""
        self.result_config = self.config
    
    def _setup_style(self):
        """Setup TUI styling."""
        self.style = Style([
            ('title', 'bold cyan'),
            ('subtitle', 'italic yellow'),
            ('selected', 'reverse bold'),
            ('unselected', 'white'),
            ('description', '#888888'),  # Gray instead of 'dim'
            ('key-hint', 'bold blue'),
            ('volume-bar', 'green'),
            ('border', 'cyan'),
        ])
    
    def _setup_layout(self):
        """Setup TUI layout."""
        # Header
        self.header = Window(
            content=FormattedTextControl(self._get_header_text),
            height=4,
            style='class:title'
        )
        
        # Main content
        self.content = Window(
            content=FormattedTextControl(self._get_content_text),
            wrap_lines=True,
        )
        
        # Volume preview (shown when needed)
        self.volume_preview = Window(
            content=FormattedTextControl(self._get_volume_text),
            height=4,
        )
        
        # Footer
        self.footer = Window(
            content=FormattedTextControl(self._get_footer_text),
            height=3,
            style='class:key-hint'
        )
        
        # Layout
        self.layout = Layout(
            HSplit([
                self.header,
                Frame(self.content, title="Configuration"),
                self.volume_preview,
                self.footer,
            ], padding=1)
        )
        
        # Application
        self.app = Application(
            layout=self.layout,
            key_bindings=self.key_bindings,
            style=self.style,
            full_screen=True,
            mouse_support=False,
        )
    
    def _get_header_text(self):
        """Get header text."""
        current_page = self.pages[self.current_page_index]
        progress = f"Step {self.current_page_index + 1} of {len(self.pages)}"
        
        return HTML(
            f'<title>{current_page.title}</title>\n'
            f'<subtitle>{current_page.subtitle}</subtitle>\n'
            f'<description>{progress}</description>\n'
        )
    
    def _get_content_text(self):
        """Get main content text."""
        current_page = self.pages[self.current_page_index]
        lines = []
        
        for i, option in enumerate(current_page.options):
            if i == current_page.selected_index:
                lines.append(f'<selected>▶ {option.label}</selected>')
                lines.append(f'  <description>{option.description}</description>')
            else:
                lines.append(f'<unselected>  {option.label}</unselected>')
            
            # Show current value for options with choices
            if option.options and hasattr(option, 'value'):
                lines.append(f'    <description>Current: {option.value}</description>')
            
            lines.append('')
        
        return HTML('\n'.join(lines))
    
    def _get_volume_text(self):
        """Get volume preview text."""
        current_page = self.pages[self.current_page_index]
        
        if current_page.show_volume_preview:
            volume_display = self.audio_monitor.volume_bar.render()
            return HTML(
                f'<volume-bar>🎤 Live Audio Preview</volume-bar>\n'
                f'<description>{volume_display}</description>\n'
                f'<description>Speak or play something to test the input level</description>'
            )
        else:
            return HTML('')
    
    def _get_footer_text(self):
        """Get footer text."""
        current_page = self.pages[self.current_page_index]
        
        if current_page.show_volume_preview:
            hints = "↑↓ Select Device  ENTER Next  BACKSPACE Back  Q Quit"
        elif any(opt.options for opt in current_page.options):
            hints = "↑↓ Navigate  ←→ Change Value  ENTER Next  BACKSPACE Back  Q Quit"
        else:
            hints = "↑↓ Navigate  ENTER Select  BACKSPACE Back  Q Quit"
        
        return HTML(
            f'<key-hint>Controls: {hints}</key-hint>\n'
            f'<description>Use keyboard to navigate - no mouse needed</description>'
        )
    
    async def run_async(self) -> Optional[Config]:
        """Run the configuration flow asynchronously."""
        try:
            # Start with first page
            self._handle_page_change()
            
            # Run the application
            await self.app.run_async()
            
            return self.result_config
        finally:
            self.audio_monitor.stop_monitoring()
    
    def run(self) -> Optional[Config]:
        """Run the configuration flow (blocking)."""
        try:
            return asyncio.run(self.run_async())
        except KeyboardInterrupt:
            return None
        finally:
            self.audio_monitor.stop_monitoring()