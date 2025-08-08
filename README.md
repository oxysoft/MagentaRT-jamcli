# 🎵 Magenta RT JAM CLI

<div align="center">

**Real-time AI Audio Jamming in Your Terminal**  
*Transform your music with PyTorch-powered neural networks*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![uv](https://img.shields.io/badge/uv-package%20manager-green.svg)](https://github.com/astral-sh/uv)

</div>

---

## ✨ What is This?

Magenta RT JAM CLI brings **real-time AI audio generation** to your command line. Feed it your music, and watch as PyTorch-powered neural networks learn and jam along with you in real-time. Originally a complex Colab notebook, now a streamlined CLI that just works.

**🚀 Key Features:**
- 🎤 **Live microphone jamming** - AI responds to your playing in real-time
- 📁 **Audio file processing** - Enhance existing recordings 
- 🧠 **Built-in PyTorch models** - No complex setup required
- 📊 **Rich terminal UI** - Beautiful progress bars and live stats
- 💾 **Auto-save sessions** - Never lose your jams
- ⚡ **GPU acceleration** - CUDA & Apple Silicon support

---

## 🏃‍♂️ Quick Start

### Installation
```bash
# Install uv (modern Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Magenta RT JAM CLI with PyTorch
uv pip install -e .
```

### Your First Jam Session
```bash
# Start jamming! (models download automatically)
uv run jamcli run --input-source mic --model-tag medium
```

That's it! The AI will start jamming along with whatever you play.

---

## 📚 Complete Command Reference

### 🎵 Running Audio Sessions

#### Basic Usage
```bash
# Jam with microphone (most common)
❯ uv run jamcli run --input-source mic
```

#### Audio File Processing
```bash
# Process an audio file
❯ uv run jamcli run --audio-file path/to/song.wav --model-tag large

# Custom loop settings
❯ uv run jamcli run --audio-file song.wav --bpm 140 --beats-per-loop 16
```

#### Advanced Options
```bash
# Full control over the session
❯ uv run jamcli run \
    --input-source mic \
    --model-tag large \
    --device gpu \
    --bpm 120 \
    --beats-per-loop 8 \
    --intro-loops 4
```

### 🤖 Model Management

#### List Available Models
```bash
❯ uv run jamcli models list
                         Available PyTorch Audio Models                         
┏━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Model  ┃   Status    ┃     Size ┃ Description         ┃ Location             ┃
┡━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ large  │ ✓ Available │   7.4 MB │ Large PyTorch audio │ /home/user/.cache/m… │
│        │             │          │ generation model    │                      │
│        │             │          │ 1024d, 16L, 16H     │                      │
│ medium │ ✗ Not Available │       -- │ Medium PyTorch      │ Not downloaded       │
│        │             │          │ audio generation    │                      │
│        │             │          │ model               │                      │
│        │             │          │ 512d, 12L, 12H      │                      │
│ small  │ ✓ Available │ 696.0 KB │ Small PyTorch audio │ /home/user/.cache/m… │
│        │             │          │ generation model    │                      │
│        │             │          │ 256d, 8L, 8H        │                      │
└────────┴─────────────┴──────────┴─────────────────────┴──────────────────────┘
```

#### Download Models (with Progress)
```bash
❯ uv run jamcli models download large
✓ large model ready ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01 56 bytes/s 0:00:00
✓ Built-in model 'large' configured at /home/user/.cache/magenta-rt-jamcli/pytorch_audio_large

   Downloaded Files in pytorch_audio_large    
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ File              ┃   Size ┃ Type          ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ config.json       │  235 B │ Configuration │
│ pytorch_model.bin │ 7.4 MB │ Model weights │
├───────────────────┼────────┼───────────────┤
│ Total             │ 7.4 MB │               │
└───────────────────┴────────┴───────────────┘
✓ Model 'large' ready at /home/user/.cache/magenta-rt-jamcli/pytorch_audio_large
```

#### Model Management
```bash
# Download specific model
❯ uv run jamcli models download medium --force

# Clear cached models
❯ uv run jamcli models clear small
❯ uv run jamcli models clear all --confirm
```

### 🎛️ Audio Device Management

#### List Audio Devices
```bash
❯ uv run jamcli devices
                                    Available Audio Devices                                    
┏━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ ID ┃ Name                             ┃ Max Inputs  ┃ Max Outputs ┃ Default Rate   ┃
┡━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│  0 │ Built-in Microphone              │           2 │           0 │      48000 Hz  │
│  1 │ Built-in Output                  │           0 │           2 │      48000 Hz  │
│  2 │ USB Audio Device                 │           2 │           2 │      44100 Hz  │
└────┴──────────────────────────────────┴─────────────┴─────────────┴────────────────┘

🎤 Testing microphone... ✓ Microphone working (RMS: 0.001493)
🔊 Testing speakers... ✓ Speakers working
```

### 📊 Session Management

#### View Past Sessions
```bash
❯ uv run jamcli sessions
                           Saved Audio Sessions                           
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Filename                         ┃     Size ┃        Modified ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ jamcli_mixed_20241208_143022.wav │  45.2 MB │ 2024-12-08 14:32 │
│ jamcli_output_20241208_143022.wav│  22.6 MB │ 2024-12-08 14:32 │
│ jamcli_input_20241208_143022.wav │  22.6 MB │ 2024-12-08 14:32 │
│ jamcli_mixed_20241208_142155.wav │  12.1 MB │ 2024-12-08 14:22 │
└──────────────────────────────────┴──────────┴─────────────────┘
```

#### Custom Output Directory
```bash
❯ uv run jamcli sessions --output-dir /path/to/my/sessions
```

### ⚙️ Configuration Management

#### Create Configuration File
```bash
❯ uv run jamcli init-config --output my-settings.toml
Setting up Magenta RT JAM CLI configuration
Input source [mic/file] (file): mic
BPM (120): 140
Beats per loop (8): 16
Intro loops (4): 2
Device [cpu/gpu/mps] (cpu): gpu
Sample rate (48000): 
Chunk seconds (2.0): 
Temperature (1.2): 1.5
Top-k (30): 20
Guidance weight (1.5): 

Configuration saved to my-settings.toml
```

#### Use Configuration File
```bash
❯ uv run jamcli run --config my-settings.toml

❯ uv run jamcli show-config my-settings.toml
       Configuration        
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Setting         ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Input Source    │ mic    │
│ BPM             │ 140    │
│ Beats per Loop  │ 16     │
│ Intro Loops     │ 2      │
│ Device          │ gpu    │
│ Model Tag       │ large  │
├─────────────────┼────────┤
│ Sample Rate     │ 48000  │
│ Chunk Seconds   │ 2.0    │
├─────────────────┼────────┤
│ Temperature     │ 1.5    │
│ Top-k           │ 20     │
│ Guidance Weight │ 1.5    │
└─────────────────┴────────┘
```

---

## 🎬 Live Session Demo

Here's what a complete jamming session looks like:

### Starting a Session
```bash
❯ uv run jamcli run --input-source mic --model-tag large --device gpu

╭───────────────────────────────────────╮
│ Magenta RT JAM CLI                    │
│ Audio Injection with Magenta RealTime │
╰───────────────────────────────────────╯

       Configuration        
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Setting         ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Input Source    │ mic    │
│ BPM             │ 120    │
│ Beats per Loop  │ 8      │
│ Intro Loops     │ 4      │
│ Device          │ gpu    │
│ Model Tag       │ large  │
├─────────────────┼────────┤
│ Sample Rate     │ 48000  │
│ Chunk Seconds   │ 2.0    │
├─────────────────┼────────┤
│ Temperature     │ 1.2    │
│ Top-k           │ 30     │
│ Guidance Weight │ 1.5    │
└─────────────────┴────────┘

Continue with these settings? [y/n]: y
```

### Auto-Download & Initialization
```bash
⠸ ✓ Model ready        0:00:01
⠸ ✓ Codec ready        0:00:00  
⠸ ✓ System initialized 0:00:01
⠸ ✓ Audio ready        0:00:00
✓ Magenta RT system fully initialized

                                    Available Audio Devices                                    
┏━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ ID ┃ Name                             ┃ Max Inputs  ┃ Max Outputs ┃ Default Rate   ┃
┡━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│  0 │ Built-in Microphone              │           2 │           0 │      48000 Hz  │
│  1 │ Built-in Output                  │           0 │           2 │      48000 Hz  │
└────┴──────────────────────────────────┴─────────────┴─────────────┴────────────────┘

🎤 Testing microphone... ✓ Microphone working (RMS: 0.001493)
🔊 Testing speakers... ✓ Speakers working
```

### Live Jamming Session
```bash
╭────────────────────────────────────────────╮
│ Starting Audio Injection Session          │
│ Model will join after 26.67 seconds       │
╰────────────────────────────────────────────╯

Ready to start? [y/n]: y

🎵 Audio session running! Press Ctrl+C to stop

┏━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Audio Stats  ┃    Value  ┃ Performance  ┃      Value   ┃ 
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ Runtime      │   1m 23s  │ Latency      │     3.2ms    │
│ Chunks       │      42   │ Processing   │    97.2%     │
│ Input RMS    │   0.045   │ GPU Usage    │    84%       │
│ Output RMS   │   0.156   │ Memory       │   1.2GB      │
└──────────────┴───────────┴──────────────┴──────────────┘

⠼ 🎤 Listening... 🤖 AI Jamming 🎵 Mixing... ⚡ GPU Processing
```

### Session Complete
```bash
^C
Stopping session...

✓ Session completed
✓ Output saved: jamcli_output/jamcli_output_20241208_143022.wav
✓ Input saved: jamcli_output/jamcli_input_20241208_143022.wav  
✓ Mixed audio saved: jamcli_output/jamcli_mixed_20241208_143022.wav
```

---

## 🎨 Model Comparison

### Model Sizes & Capabilities

| Model | Size | Parameters | Best For | Speed |
|-------|------|------------|----------|-------|
| **small** | 696 KB | 256d, 8L, 8H | Quick tests, CPU-only | ⚡⚡⚡ |
| **medium** | 2.1 MB | 512d, 12L, 12H | Balanced quality/speed | ⚡⚡ |
| **large** | 7.4 MB | 1024d, 16L, 16H | Best quality, GPU recommended | ⚡ |

### Auto-Download Example
Models download automatically when needed:

```bash
❯ uv run jamcli run --model-tag medium --input-source mic

# If medium model isn't available, you'll see:
✓ medium model ready ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01 72 bytes/s 0:00:00
✓ Built-in model 'medium' configured at /home/user/.cache/magenta-rt-jamcli/pytorch_audio_medium

    Downloaded Files in pytorch_audio_medium    
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ File              ┃     Size ┃ Type          ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ config.json       │    236 B │ Configuration │
│ pytorch_model.bin │ 518.4 KB │ Model weights │
├───────────────────┼──────────┼───────────────┤
│ Total             │ 518.6 KB │               │
└───────────────────┴──────────┴───────────────┘

# Then immediately continues with session setup...
```

---

## 🔧 Configuration Options

### TOML Configuration File Example

```toml
# Basic settings
input_source = "mic"
audio_file = "path/to/audio.wav"  # Only used if input_source = "file"
bpm = 140
beats_per_loop = 16
intro_loops = 2
device = "gpu"  # cpu, gpu, or mps
model_tag = "large"

[audio]
sample_rate = 48000
chunk_seconds = 2.0

[model]
temperature = 1.2      # Creativity (0.1-2.0)
topk = 30             # Sampling diversity
guidance_weight = 1.5  # How much to follow input
model_volume = 0.8    # AI output volume
input_volume = 1.0    # Your input passthrough volume
model_feedback = 0.95 # How much AI hears itself
input_gap = 400       # ms of recent input to silence
```

### Device Selection

- **`cpu`** - Works everywhere, slower
- **`gpu`** - CUDA GPUs, much faster
- **`mps`** - Apple Silicon (M1/M2/M3), optimized for Mac

The CLI automatically detects your hardware and suggests the best option.

---

## 🚀 Advanced Usage

### Batch Processing Multiple Files
```bash
# Process all WAV files in a directory
for file in *.wav; do
    uv run jamcli run --audio-file "$file" --model-tag large
done
```

### Custom Session Directory
```bash
# Save sessions to specific location
mkdir ~/my-jams
uv run jamcli run --input-source mic
# Sessions auto-save to jamcli_output/, move them:
mv jamcli_output/* ~/my-jams/
```

### Performance Monitoring
```bash
# Run with verbose output to see detailed stats
uv run jamcli run --input-source mic --model-tag large 2>&1 | tee session.log
```

---

## 🛠️ Troubleshooting

### Common Issues

#### No Audio Devices Found
```bash
❯ uv run jamcli devices
# If empty, install PortAudio:
# macOS: brew install portaudio
# Ubuntu: sudo apt-get install portaudio19-dev
# Windows: Usually works out of the box
```

#### PyTorch Not Found
```bash
# Reinstall with PyTorch dependencies
❯ uv pip install -e . --force-reinstall
```

#### GPU Not Detected
```bash
# Check CUDA installation
❯ python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# Check MPS (Mac)
❯ python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

#### Poor Audio Quality
- Try a larger model: `--model-tag large`
- Increase sample rate in config: `sample_rate = 48000`
- Use GPU for better real-time performance: `--device gpu`

### Getting Help

```bash
# Help for any command
❯ uv run jamcli --help
❯ uv run jamcli run --help
❯ uv run jamcli models --help
```

---

## 🎯 What's Under the Hood

This CLI transforms the complex [Magenta RT Audio Injection notebook](https://colab.research.google.com/github/magenta/magenta-realtime/blob/main/notebooks/Magenta_RT_Audio_Injection.ipynb) into a production-ready tool:

- **🧠 PyTorch Neural Networks** - Built-in audio generation models
- **🎵 Real-time Processing** - Low-latency audio streaming with crossfading
- **📊 Rich Terminal UI** - Progress bars, live stats, beautiful tables
- **💾 Auto-save** - Never lose your creative sessions
- **⚡ Hardware Acceleration** - CUDA & Apple Silicon optimized
- **🔧 Zero Configuration** - Works out of the box, customize as needed

---

## 📜 License

Apache 2.0 License - Same as the original Magenta project. Model weights under Creative Commons Attribution 4.0.

---

<div align="center">

**🎵 Happy Jamming! 🎵**

*Made with ❤️ for the AI music community*

</div>