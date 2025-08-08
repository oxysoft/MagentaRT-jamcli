"""Model management and downloading for Magenta RT."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse

import requests
from rich.console import Console
from rich.progress import Progress, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
from google.cloud import storage

from .config import Config


class ModelManager:
    """Manages Magenta RT model downloads and initialization."""
    
    # Known model configurations
    MODEL_CONFIGS = {
        "large": {
            "checkpoint_dir": "magenta_rt_large_2024",
            "gs_path": "gs://magenta-rt-public/checkpoints/large/",
            "files": [
                "checkpoint_1000000",
                "checkpoint_1000000.data-00000-of-00001", 
                "checkpoint_1000000.index",
                "config.json"
            ]
        },
        "medium": {
            "checkpoint_dir": "magenta_rt_medium_2024",
            "gs_path": "gs://magenta-rt-public/checkpoints/medium/",
            "files": [
                "checkpoint_800000",
                "checkpoint_800000.data-00000-of-00001",
                "checkpoint_800000.index", 
                "config.json"
            ]
        }
    }
    
    def __init__(self, console: Console, cache_dir: Optional[Path] = None):
        self.console = console
        self.cache_dir = cache_dir or Path.home() / ".cache" / "magenta-rt-jamcli"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_model_path(self, model_tag: str) -> Path:
        """Get the local path for a model."""
        if model_tag not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model tag: {model_tag}. Available: {list(self.MODEL_CONFIGS.keys())}")
        
        config = self.MODEL_CONFIGS[model_tag]
        return self.cache_dir / config["checkpoint_dir"]
    
    def is_model_available(self, model_tag: str) -> bool:
        """Check if a model is already downloaded."""
        model_path = self.get_model_path(model_tag)
        if not model_path.exists():
            return False
        
        config = self.MODEL_CONFIGS[model_tag]
        for filename in config["files"]:
            if not (model_path / filename).exists():
                return False
        
        return True
    
    def download_model(self, model_tag: str, force: bool = False) -> Path:
        """Download a Magenta RT model if not already cached.
        
        Args:
            model_tag: The model tag to download
            force: Force re-download even if cached
            
        Returns:
            Path to the downloaded model directory
        """
        if not force and self.is_model_available(model_tag):
            model_path = self.get_model_path(model_tag)
            self.console.print(f"[green]✓ Model '{model_tag}' already available at {model_path}[/green]")
            return model_path
        
        if model_tag not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model tag: {model_tag}")
        
        config = self.MODEL_CONFIGS[model_tag]
        model_path = self.get_model_path(model_tag)
        model_path.mkdir(parents=True, exist_ok=True)
        
        self.console.print(f"[blue]Downloading Magenta RT model '{model_tag}'...[/blue]")
        
        # Try gsutil first (faster), fall back to Python API
        if self._has_gsutil():
            success = self._download_with_gsutil(config["gs_path"], model_path)
        else:
            success = self._download_with_api(config, model_path)
        
        if success:
            self.console.print(f"[green]✓ Model '{model_tag}' downloaded to {model_path}[/green]")
            return model_path
        else:
            raise RuntimeError(f"Failed to download model '{model_tag}'")
    
    def _has_gsutil(self) -> bool:
        """Check if gsutil is available."""
        try:
            subprocess.run(["gsutil", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _download_with_gsutil(self, gs_path: str, local_path: Path) -> bool:
        """Download using gsutil (faster for large files)."""
        try:
            with Progress(
                *Progress.get_default_columns(),
                DownloadColumn(),
                TransferSpeedColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task("Downloading with gsutil...", total=None)
                
                cmd = ["gsutil", "-m", "cp", "-r", gs_path + "*", str(local_path)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                progress.update(task, completed=True)
                
                if result.returncode == 0:
                    return True
                else:
                    self.console.print(f"[yellow]gsutil failed: {result.stderr}[/yellow]")
                    return False
        except Exception as e:
            self.console.print(f"[yellow]gsutil error: {e}[/yellow]")
            return False
    
    def _download_with_api(self, config: Dict[str, Any], local_path: Path) -> bool:
        """Download using Google Cloud Storage API."""
        try:
            # Parse bucket and prefix from gs:// URL
            gs_path = config["gs_path"]
            if not gs_path.startswith("gs://"):
                return False
            
            bucket_name = gs_path[5:].split("/")[0]
            prefix = "/".join(gs_path[5:].split("/")[1:])
            
            client = storage.Client.create_anonymous_client()
            bucket = client.bucket(bucket_name)
            
            with Progress(
                *Progress.get_default_columns(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=self.console,
            ) as progress:
                
                blobs = list(bucket.list_blobs(prefix=prefix))
                total_size = sum(blob.size for blob in blobs if blob.size)
                
                task = progress.add_task(
                    f"Downloading {len(blobs)} files...", 
                    total=total_size
                )
                
                for blob in blobs:
                    if blob.name.endswith("/"):
                        continue  # Skip directory markers
                    
                    # Get filename relative to prefix
                    rel_name = blob.name[len(prefix):]
                    local_file = local_path / rel_name
                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Download file
                    blob.download_to_filename(local_file)
                    
                    if blob.size:
                        progress.update(task, advance=blob.size)
                
                progress.update(task, completed=True)
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]Download failed: {e}[/red]")
            return False
    
    def list_available_models(self) -> Dict[str, bool]:
        """List all available models and their download status."""
        status = {}
        for model_tag in self.MODEL_CONFIGS:
            status[model_tag] = self.is_model_available(model_tag)
        return status
    
    def clear_cache(self, model_tag: Optional[str] = None):
        """Clear cached models."""
        if model_tag:
            if model_tag not in self.MODEL_CONFIGS:
                raise ValueError(f"Unknown model tag: {model_tag}")
            
            model_path = self.get_model_path(model_tag)
            if model_path.exists():
                shutil.rmtree(model_path)
                self.console.print(f"[green]✓ Cleared cache for model '{model_tag}'[/green]")
            else:
                self.console.print(f"[yellow]Model '{model_tag}' not cached[/yellow]")
        else:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self.console.print("[green]✓ Cleared all model cache[/green]")


def install_magenta_rt(console: Console) -> bool:
    """Install Magenta RT from the official repository."""
    try:
        console.print("[blue]Installing Magenta RealTime...[/blue]")
        
        with Progress(console=console) as progress:
            task = progress.add_task("Cloning repository...", total=None)
            
            # Clone the repository
            result = subprocess.run([
                "git", "clone", "https://github.com/magenta/magenta-realtime.git"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                console.print(f"[red]Failed to clone repository: {result.stderr}[/red]")
                return False
            
            progress.update(task, description="Installing package...")
            
            # Install the package
            result = subprocess.run([
                "pip", "install", "-e", "magenta-realtime/[tpu]"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                progress.update(task, completed=True)
                console.print("[green]✓ Magenta RealTime installed successfully[/green]")
                return True
            else:
                console.print(f"[red]Failed to install: {result.stderr}[/red]")
                return False
                
    except Exception as e:
        console.print(f"[red]Installation error: {e}[/red]")
        return False