-> In this project, you can easily convert your Audio to a text file easily.

# Whisper Audio Transcriber

A Python script for fast transcription of audio files (including Hindi) using OpenAI's Whisper model, with GPU acceleration support.

## Features

- 🎙️ Supports Hindi and English transcription
- ⚡ GPU acceleration for faster processing
- 🔄 Optional Hindi-to-English translation
- 📊 Progress tracking during transcription
- 📝 Clean output with timestamps
- 🎚️ Multiple model sizes for speed/accuracy tradeoff

## Supported Languages

- Hindi (`hi`)
- English (`en`)
- *(Whisper supports [many others](https://github.com/openai/whisper#available-models-and-languages))*

## Installation

### Prerequisites

- Python 3.8+
- FFmpeg (for audio processing)
- NVIDIA GPU (optional, for acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/LiGerNikhil/transcribe_script.git
cd whisper-transcriber

# Create virtual environment
python -m venv venv
source venv/bin/activate 
# On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

python transcribe.py --input english_audio.wav --language en

python transcribe.py --input hindi_audio.mp3 --model medium --language hi

python transcribe.py --input hindi_audio.ogg --language hi --translate

python transcribe.py --input long_audio.mp3 --device cuda --fp16 --model small