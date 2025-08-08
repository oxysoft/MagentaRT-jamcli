#!/usr/bin/env python3
"""Main CLI interface for Magenta RT Audio Injection."""

import os
import sys
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.progress import track
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .audio_injection import AudioInjectionApp
from .config import Config, load_config, save_config
from .model_manager import ModelManager

console = Console()


@click.group()
@click.version_option()
@click.pass_context
def cli(ctx):
    """Magenta RT JAM CLI - Audio Injection with Magenta RealTime."""
    ctx.ensure_object(dict)


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file path")
@click.option("--input-source", type=click.Choice(["mic", "file"]), default="file", 
              help="Input source: microphone or audio file")
@click.option("--audio-file", "-f", type=click.Path(exists=True), 
              help="Audio file path (for file input)")
@click.option("--bpm", type=int, default=120, help="Beats per minute for metronome")
@click.option("--beats-per-loop", type=int, default=8, help="Number of beats per loop")
@click.option("--intro-loops", type=int, default=4, 
              help="Number of intro loops before model joins")
@click.option("--device", type=click.Choice(["cpu", "gpu", "mps"]), default="cpu", 
              help="Device to run the model on (gpu=CUDA, mps=Apple Silicon)")
@click.option("--model-tag", default="large", help="Model tag to use")
@click.option("--tui", is_flag=True, help="Use Terminal User Interface with live volume meters")
@click.pass_context
def run(ctx, config, input_source, audio_file, bpm, beats_per_loop, intro_loops, device, model_tag, tui):
    """Run the audio injection session."""
    
    # Load configuration
    if config:
        cfg = load_config(config)
    else:
        cfg = Config()
    
    # Override config with CLI arguments
    cfg.input_source = input_source
    if audio_file:
        cfg.audio_file = Path(audio_file)
    cfg.bpm = bpm
    cfg.beats_per_loop = beats_per_loop
    cfg.intro_loops = intro_loops
    cfg.device = device
    cfg.model_tag = model_tag
    
    # Display welcome message
    console.print(Panel.fit(
        "[bold blue]Magenta RT JAM CLI[/bold blue]\n"
        "[dim]Audio Injection with Magenta RealTime[/dim]",
        border_style="blue"
    ))
    
    # Display configuration
    display_config(cfg)
    
    # Confirm settings
    if not Confirm.ask("Continue with these settings?"):
        console.print("[yellow]Cancelled by user[/yellow]")
        return
    
    try:
        # Initialize and run the audio injection app
        app = AudioInjectionApp(cfg, console, use_tui=tui)
        
        if tui:
            console.print("[bold cyan]🎛️  Starting TUI Mode[/bold cyan]")
            console.print("[dim]Use SPACE to start/stop, Q to quit[/dim]\n")
        
        app.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Session interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--output", "-o", type=click.Path(), default="jamcli-config.toml",
              help="Output configuration file path")
def init_config(output):
    """Initialize a new configuration file."""
    config_path = Path(output)
    
    if config_path.exists():
        if not Confirm.ask(f"Configuration file {config_path} exists. Overwrite?"):
            console.print("[yellow]Cancelled by user[/yellow]")
            return
    
    # Interactive configuration setup
    console.print("[bold]Setting up Magenta RT JAM CLI configuration[/bold]")
    
    cfg = Config()
    
    # Basic settings
    cfg.input_source = Prompt.ask(
        "Input source", 
        choices=["mic", "file"], 
        default="file"
    )
    
    if cfg.input_source == "file":
        audio_file = Prompt.ask("Audio file path (optional)")
        if audio_file:
            cfg.audio_file = Path(audio_file)
    
    cfg.bpm = int(Prompt.ask("BPM", default="120"))
    cfg.beats_per_loop = int(Prompt.ask("Beats per loop", default="8"))
    cfg.intro_loops = int(Prompt.ask("Intro loops", default="4"))
    
    cfg.device = Prompt.ask(
        "Device", 
        choices=["cpu", "gpu", "mps"], 
        default="cpu"
    )
    
    # Audio settings
    cfg.audio.sample_rate = int(Prompt.ask("Sample rate", default="48000"))
    cfg.audio.chunk_seconds = float(Prompt.ask("Chunk seconds", default="2.0"))
    
    # Model settings
    cfg.model.temperature = float(Prompt.ask("Temperature", default="1.2"))
    cfg.model.topk = int(Prompt.ask("Top-k", default="30"))
    cfg.model.guidance_weight = float(Prompt.ask("Guidance weight", default="1.5"))
    
    # Save configuration
    save_config(cfg, config_path)
    console.print(f"[green]Configuration saved to {config_path}[/green]")


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
def show_config(config_file):
    """Show configuration file contents."""
    cfg = load_config(config_file)
    display_config(cfg)


@cli.group()
def models():
    """Model management commands."""
    pass


@models.command("list")
def list_models():
    """List available models and their download status."""
    model_manager = ModelManager(console)
    
    table = Table(title="Available PyTorch Audio Models", show_header=True, header_style="bold magenta")
    table.add_column("Model", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Size", justify="right")
    table.add_column("Description")
    table.add_column("Location", style="dim")
    
    for model_tag in model_manager.MODEL_CONFIGS.keys():
        info = model_manager.get_model_info(model_tag)
        config = info["config"]
        
        if info["available"]:
            status_text = "[green]✓ Available[/green]"
            size_text = info["size"]
            location = str(info["path"])
        else:
            status_text = "[red]✗ Not Available[/red]"
            size_text = "[dim]--[/dim]"
            location = "[dim]Not downloaded[/dim]"
        
        # Build description with parameters
        params = config["parameters"]
        description = f"{config['description']}\n[dim]{params['hidden_dim']}d, {params['num_layers']}L, {params['num_heads']}H[/dim]"
        
        table.add_row(
            model_tag, 
            status_text, 
            size_text,
            description,
            location
        )
    
    console.print(table)


@models.command("download")
@click.argument("model_tag", type=click.Choice(["large", "medium", "small"]))
@click.option("--force", "-f", is_flag=True, help="Force re-download even if cached")
def download_model(model_tag, force):
    """Download a Magenta RT model."""
    model_manager = ModelManager(console)
    
    try:
        model_path = model_manager.download_model(model_tag, force=force)
        console.print(f"[green]✓ Model '{model_tag}' ready at {model_path}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to download model '{model_tag}': {e}[/red]")
        sys.exit(1)


@models.command("clear")
@click.argument("model_tag", type=click.Choice(["large", "medium", "small", "all"]), required=False)
@click.option("--confirm", "-y", is_flag=True, help="Skip confirmation prompt")
def clear_models(model_tag, confirm):
    """Clear cached models."""
    model_manager = ModelManager(console)
    
    if model_tag == "all":
        if not confirm and not Confirm.ask("Clear all cached models?"):
            console.print("[yellow]Cancelled by user[/yellow]")
            return
        model_manager.clear_cache()
    elif model_tag:
        if not confirm and not Confirm.ask(f"Clear cached model '{model_tag}'?"):
            console.print("[yellow]Cancelled by user[/yellow]")
            return
        model_manager.clear_cache(model_tag)
    else:
        console.print("[red]Please specify a model tag or 'all'[/red]")
        sys.exit(1)


@cli.command()
def devices():
    """List available audio devices."""
    try:
        import sounddevice as sd
        from .audio_stream import AudioStreamer
        from .config import Config
        
        # Create a temporary config for device listing
        config = Config()
        streamer = AudioStreamer(config, console, lambda x: x)
        
        device_table = streamer.list_audio_devices()
        console.print(device_table)
        
    except ImportError:
        console.print("[red]sounddevice not available. Install with: pip install sounddevice[/red]")
    except Exception as e:
        console.print(f"[red]Error listing audio devices: {e}[/red]")


@cli.command()
@click.option("--output-dir", "-o", type=click.Path(), default="jamcli_output",
              help="Output directory to check")
def sessions(output_dir):
    """List saved audio sessions."""
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        console.print(f"[yellow]Output directory {output_dir} does not exist[/yellow]")
        return
    
    # Find audio files
    audio_files = []
    for pattern in ["*.wav", "*.mp3", "*.flac"]:
        audio_files.extend(output_dir.glob(pattern))
    
    if not audio_files:
        console.print(f"[yellow]No audio sessions found in {output_dir}[/yellow]")
        return
    
    table = Table(title="Saved Audio Sessions", show_header=True, header_style="bold green")
    table.add_column("Filename", style="bold")
    table.add_column("Size", justify="right")
    table.add_column("Modified", justify="right")
    
    for file_path in sorted(audio_files, key=lambda f: f.stat().st_mtime, reverse=True):
        stat = file_path.stat()
        size_mb = stat.st_size / (1024 * 1024)
        modified = time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime))
        
        table.add_row(
            file_path.name,
            f"{size_mb:.1f} MB",
            modified
        )
    
    console.print(table)


def display_config(cfg: Config) -> None:
    """Display configuration in a nice table."""
    table = Table(title="Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="dim")
    table.add_column("Value")
    
    # Basic settings
    table.add_row("Input Source", cfg.input_source)
    if cfg.audio_file:
        table.add_row("Audio File", str(cfg.audio_file))
    table.add_row("BPM", str(cfg.bpm))
    table.add_row("Beats per Loop", str(cfg.beats_per_loop))
    table.add_row("Intro Loops", str(cfg.intro_loops))
    table.add_row("Device", cfg.device)
    table.add_row("Model Tag", cfg.model_tag)
    
    # Audio settings
    table.add_section()
    table.add_row("Sample Rate", str(cfg.audio.sample_rate))
    table.add_row("Chunk Seconds", str(cfg.audio.chunk_seconds))
    
    # Model settings
    table.add_section()
    table.add_row("Temperature", str(cfg.model.temperature))
    table.add_row("Top-k", str(cfg.model.topk))
    table.add_row("Guidance Weight", str(cfg.model.guidance_weight))
    
    console.print(table)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()