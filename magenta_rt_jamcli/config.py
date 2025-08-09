"""Configuration management for Magenta RT JAM CLI."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import toml
from rich.console import Console


@dataclass
class AudioConfig:
    """Audio-related configuration."""
    sample_rate: int = 48000
    chunk_seconds: float = 2.0
    
    @property
    def chunk_samples(self) -> int:
        """Calculate chunk samples from chunk seconds and sample rate."""
        return int(self.chunk_seconds * self.sample_rate)


@dataclass
class ModelConfig:
    """Model-related configuration."""
    temperature: float = 1.2
    topk: int = 30
    guidance_weight: float = 1.5
    model_volume: float = 0.6
    input_volume: float = 1.0
    model_feedback: float = 0.95
    input_gap: int = 400  # milliseconds


@dataclass
class PromptConfig:
    """Prompt-related configuration."""
    text_prompts: List[str] = field(default_factory=lambda: [
        "lofi hip hop beat",
        "funk jam", 
        "acid house"
    ])
    prompt_weights: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])


@dataclass
class Config:
    """Main configuration class."""
    # Basic settings
    input_source: str = "file"  # "mic" or "file"
    audio_file: Optional[Path] = None
    audio_device_id: Optional[int] = None  # Audio input device ID
    bpm: int = 120
    beats_per_loop: int = 8
    intro_loops: int = 4
    device: str = "cpu"  # "cpu", "gpu", "mps"
    model_tag: str = "large"
    
    # Sub-configurations
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    
    # Advanced settings
    mix_prefix_frames: int = 16
    left_edge_frames_to_remove: int = 8
    
    @property
    def loop_seconds(self) -> float:
        """Calculate loop duration in seconds."""
        return self.beats_per_loop * 60 / self.bpm
    
    @property
    def loop_samples(self) -> int:
        """Calculate loop duration in samples."""
        return int(self.loop_seconds * self.audio.sample_rate)
    
    @property
    def use_prerecorded_input(self) -> bool:
        """Whether using prerecorded input."""
        return self.input_source == "file"


def load_config(config_path: Path) -> Config:
    """Load configuration from TOML file."""
    try:
        with open(config_path, 'r') as f:
            data = toml.load(f)
        
        # Parse nested configurations
        audio_config = AudioConfig(**data.get('audio', {}))
        model_config = ModelConfig(**data.get('model', {}))
        prompt_config = PromptConfig(**data.get('prompts', {}))
        
        # Remove nested configs from main data
        data.pop('audio', None)
        data.pop('model', None) 
        data.pop('prompts', None)
        
        # Handle Path conversion
        if 'audio_file' in data and data['audio_file']:
            data['audio_file'] = Path(data['audio_file'])
        
        return Config(
            audio=audio_config,
            model=model_config,
            prompts=prompt_config,
            **data
        )
    except Exception as e:
        console = Console()
        console.print(f"[red]Error loading config from {config_path}: {e}[/red]")
        raise


def save_config(config: Config, config_path: Path) -> None:
    """Save configuration to TOML file."""
    try:
        # Convert to dictionary
        data = {
            'input_source': config.input_source,
            'audio_file': str(config.audio_file) if config.audio_file else None,
            'bpm': config.bpm,
            'beats_per_loop': config.beats_per_loop,
            'intro_loops': config.intro_loops,
            'device': config.device,
            'model_tag': config.model_tag,
            'mix_prefix_frames': config.mix_prefix_frames,
            'left_edge_frames_to_remove': config.left_edge_frames_to_remove,
            
            'audio': {
                'sample_rate': config.audio.sample_rate,
                'chunk_seconds': config.audio.chunk_seconds,
            },
            
            'model': {
                'temperature': config.model.temperature,
                'topk': config.model.topk,
                'guidance_weight': config.model.guidance_weight,
                'model_volume': config.model.model_volume,
                'input_volume': config.model.input_volume,
                'model_feedback': config.model.model_feedback,
                'input_gap': config.model.input_gap,
            },
            
            'prompts': {
                'text_prompts': config.prompts.text_prompts,
                'prompt_weights': config.prompts.prompt_weights,
            }
        }
        
        with open(config_path, 'w') as f:
            toml.dump(data, f)
            
    except Exception as e:
        console = Console()
        console.print(f"[red]Error saving config to {config_path}: {e}[/red]")
        raise