# Magenta RT JAM CLI

A command-line interface for Magenta RT Audio Injection, converted from the original Colab notebook.

## Features

- 🎵 **Real-time audio injection** with Magenta RT models (full notebook implementation)
- 🎤 **Live microphone & file input** with automatic audio device detection
- 📡 **Automatic model downloading** from Google Cloud Storage with progress tracking
- 💾 **Session recording** with automatic export of input/output/mixed audio files
- ⚙️ **TOML configuration** with interactive setup and command-line overrides
- 🎨 **Rich terminal UI** with live streaming statistics and progress indicators
- 🔧 **Multi-device support** (CPU, GPU, TPU) with proper JAX integration
- 📊 **Audio streaming stats** including latency monitoring and performance metrics
- 🎛️ **Professional audio handling** with crossfading, metronome, and real-time processing

## Installation

This project uses `uv` for dependency management. Install it first if you haven't:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project:

```bash
uv pip install -e .

# For TPU support (recommended, works on free Colab TPUs):
uv pip install -e ".[tpu]"

# For GPU support (requires compatible GPU):
uv pip install -e ".[gpu]"
```

**Install Magenta RealTime (required for actual model functionality):**

```bash
# Clone and install the official Magenta RT repository
git clone https://github.com/magenta/magenta-realtime.git
uv pip install -e magenta-realtime/[tpu]  # or [gpu] for GPU support
```

## Quick Start

1. **Download a model (first time only):**
   ```bash
   jamcli models download large
   ```

2. **Check available audio devices:**
   ```bash
   jamcli devices
   ```

3. **Initialize a configuration file:**
   ```bash
   jamcli init-config
   ```

4. **Run with an audio file:**
   ```bash
   jamcli run --audio-file path/to/your/audio.wav
   ```

5. **Run with live microphone input:**
   ```bash
   jamcli run --input-source mic
   ```

6. **View saved sessions:**
   ```bash
   jamcli sessions
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

### `jamcli models`

Manage Magenta RT models.

**Sub-commands:**
- `jamcli models list` - List available models and download status
- `jamcli models download <model>` - Download a model (large, medium)
- `jamcli models clear <model|all>` - Clear cached models

**Examples:**
```bash
# List available models
jamcli models list

# Download the large model
jamcli models download large

# Clear all cached models
jamcli models clear all
```

### `jamcli devices`

List and test available audio devices.

### `jamcli sessions`

List saved audio sessions with file sizes and timestamps.

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

## Complete Workflow Example

Here's a full workflow from setup to audio generation:

```bash
# 1. Install Magenta RT JAM CLI
uv pip install -e .

# 2. Install Magenta RealTime (required)
git clone https://github.com/magenta/magenta-realtime.git
uv pip install -e magenta-realtime/[tpu]

# 3. Check available models
jamcli models list

# 4. Download a model (takes ~5 minutes first time)
jamcli models download large

# 5. Check your audio devices (optional)
jamcli devices

# 6. Create a configuration file
jamcli init-config

# 7. Run with your audio file
jamcli run --audio-file your-music.wav

# 8. Or run with microphone input
jamcli run --input-source mic

# 9. Check your generated sessions
jamcli sessions
```

**What happens during a session:**
1. 🔄 Model initializes (loads from cache after first time)
2. 🎵 Audio processing begins with your input
3. 📊 Live statistics show latency, processing rate, etc.
4. 🎤 Model "joins in" after the configured intro loops
5. 💾 Session automatically saves to `jamcli_output/` when stopped
6. 🎧 Three files are saved: input, output, and mixed audio

## Requirements

- Python 3.10+
- Magenta RealTime library (installed separately)
- JAX with TPU/GPU support (for model inference)
- Audio system (PortAudio) for real-time streaming
- ~2GB disk space for model checkpoints

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