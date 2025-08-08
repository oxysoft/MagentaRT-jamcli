# Magenta RT JAM CLI

A command-line interface for Magenta RT Audio Injection, converted from the original Colab notebook.

## Features

- 🎵 Real-time audio injection with Magenta RT models
- 🎤 Support for both live microphone and pre-recorded audio input
- ⚙️ Configurable parameters via TOML files or CLI arguments
- 🎨 Rich terminal UI with progress indicators and interactive prompts
- 🔧 Multiple device support (CPU, GPU, TPU)

## Installation

This project uses `uv` for dependency management. Install it first if you haven't:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project:

```bash
uv pip install -e .
```

## Quick Start

1. **Initialize a configuration file:**
   ```bash
   jamcli init-config
   ```

2. **Run with an audio file:**
   ```bash
   jamcli run --audio-file path/to/your/audio.wav
   ```

3. **Run with live microphone input:**
   ```bash
   jamcli run --input-source mic
   ```

## CLI Commands

### `jamcli run`

Run the audio injection session.

**Options:**
- `--config, -c`: Configuration file path
- `--input-source`: Input source (`mic` or `file`)
- `--audio-file, -f`: Audio file path (for file input)
- `--bpm`: Beats per minute for metronome (default: 120)
- `--beats-per-loop`: Number of beats per loop (default: 8)
- `--intro-loops`: Number of intro loops before model joins (default: 4)
- `--device`: Device to run the model on (`cpu`, `gpu`, `tpu`)
- `--model-tag`: Model tag to use (default: `large`)

**Examples:**
```bash
# Basic usage with audio file
jamcli run --audio-file music.wav

# Custom settings
jamcli run --audio-file music.wav --bpm 140 --beats-per-loop 16

# Use configuration file
jamcli run --config my-config.toml

# Live microphone input
jamcli run --input-source mic --device gpu
```

### `jamcli init-config`

Create a new configuration file interactively.

**Options:**
- `--output, -o`: Output configuration file path (default: `jamcli-config.toml`)

### `jamcli show-config`

Display the contents of a configuration file.

**Usage:**
```bash
jamcli show-config config.toml
```

## Configuration

The CLI supports TOML configuration files for persistent settings. Use `jamcli init-config` to create one interactively, or create one manually:

```toml
input_source = "file"
audio_file = "path/to/audio.wav"
bpm = 120
beats_per_loop = 8
intro_loops = 4
device = "cpu"
model_tag = "large"

[audio]
sample_rate = 48000
chunk_seconds = 2.0

[model]
temperature = 1.2
topk = 30
guidance_weight = 1.5
model_volume = 0.6
input_volume = 1.0
model_feedback = 0.95
input_gap = 400

[prompts]
text_prompts = ["lofi hip hop beat", "funk jam", "acid house"]
prompt_weights = [1.0, 0.0, 0.0]
```

## Requirements

- Python 3.10+
- Magenta RealTime library
- JAX (with appropriate backend for your device)
- Audio libraries (librosa, etc.)

## Development

The project structure:
```
magenta_rt_jamcli/
├── __init__.py
├── cli.py              # Main CLI interface
├── config.py           # Configuration management
└── audio_injection.py  # Core audio injection functionality
```

## Original Notebook

This CLI is based on the [Magenta RT Audio Injection notebook](https://colab.research.google.com/github/magenta/magenta-realtime/blob/main/notebooks/Magenta_RT_Audio_Injection.ipynb).

## License

This project follows the same licensing as Magenta RealTime:
- Code: Apache 2.0
- Model weights: Creative Commons Attribution 4.0 International