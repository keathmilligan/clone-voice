# Clone Voice

Local voice cloning solution using XTTS v2 for zero-shot text-to-speech generation.

## Features

- **Zero-shot voice cloning**: Clone any voice with just 5-30 seconds of reference audio
- **Batch processing**: Generate speech from text files efficiently
- **Audio preprocessing**: Denoise, normalize, and trim silence from recordings
- **Cross-platform**: Works on Windows, Linux, and macOS
- **GPU accelerated**: CUDA support for fast generation

## Requirements

- Python 3.10 or 3.11
- NVIDIA GPU with CUDA (recommended, 8GB+ VRAM)
- ~10GB disk space for models

## Installation

### Using uv (recommended)

```bash
cd clone-voice

# Create venv and install PyTorch with CUDA
uv venv
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install the package
uv pip install -e .

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate
```

### Using pip

```bash
cd clone-voice
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

## Quick Start

### 1. Record a voice sample

```bash
# Interactive recording (recommended)
clone-voice record -n my_voice

# Or with custom duration
clone-voice record -n my_voice -d 15
```

### 2. Generate speech

```bash
# Single text
clone-voice generate "Hello, this is my cloned voice!" -v samples/my_voice.wav

# From a text file (batch processing)
clone-voice batch input.txt -v samples/my_voice.wav -o outputs/
```

### 3. Process audio (optional)

```bash
# Clean up a recording
clone-voice process samples/my_voice.wav
```

## Commands

| Command | Description |
|---------|-------------|
| `record` | Record a voice sample from microphone |
| `process` | Process audio (denoise, normalize, trim) |
| `generate` | Generate speech from text |
| `batch` | Generate speech from a text file |
| `list` | List recorded voice samples |
| `devices` | List available audio input devices |
| `info` | Show information about an audio file |
| `models` | List available TTS models |

## Tips for Best Results

### Recording Voice Samples

- Use a quiet environment
- Keep microphone 6-12 inches from mouth
- Speak naturally at normal volume
- Record 10-30 seconds of varied speech
- Avoid background noise and echo

### Text Generation

- Keep sentences reasonably short (< 200 characters)
- Use punctuation for natural pauses
- The model handles most text naturally

## Configuration

Settings can be customized via environment variables (prefix: `CLONE_VOICE_`):

```bash
# Example .env file
CLONE_VOICE_DEVICE=cuda
CLONE_VOICE_LANGUAGE=en
CLONE_VOICE_SAMPLE_RATE=22050
```

## Project Structure

```
clone-voice/
├── src/clone_voice/
│   ├── cli.py          # Command-line interface
│   ├── config.py       # Configuration management
│   ├── recorder.py     # Audio recording
│   ├── preprocessor.py # Audio preprocessing
│   └── synthesizer.py  # TTS synthesis
├── samples/            # Voice samples
├── models/             # Cached models
├── outputs/            # Generated audio
└── pyproject.toml
```

## License

MIT
