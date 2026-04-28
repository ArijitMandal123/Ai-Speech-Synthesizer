# 🎬 AI Speech Synthesis — YouTube Shorts Voice Mixer

A blazing fast, locally-hosted AI voice generator specifically tuned for creating YouTube Shorts, Anime Fan Theories, and dynamic narrations. 

Powered by the highly-efficient **Kokoro-82M** TTS engine and **Vosk** for word-level captioning, this app provides a premium Gradio web interface to turn your scripts into polished, professional audio tracks.

## ✨ Features

- **Kokoro-82M Engine:** Incredibly fast and high-quality Text-to-Speech optimized for CPU and GPU (CUDA) inference.
- **18 Voice Profiles:** Includes American and British voices, both male and female (e.g., `am_eric`, `af_bella`).
- **Word-Level Captions:** Automatically generates precise timestamped JSON captions for your scripts, perfect for syncing on-screen text in video editors.
- **Audacity-Grade Audio Effects:** Automatically applies a 6-stage audio processing pipeline (Normalization, High-Pass Filter, Bass Boost, Compression) to give voices that deep, professional "YouTuber" resonance.
- **Premium UI:** A sleek, glassmorphism-styled Gradio web interface.

## 🚀 Quick Start

This project uses [uv](https://github.com/astral-sh/uv), the ultrafast Python package manager written in Rust, for dependency management.

### 1. Prerequisites
- Python 3.10 or higher.
- Install `uv`. (On Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`)

### 2. Installation
Clone the repository and navigate into the folder:
```bash
git clone <your-repo-url>
cd <your-folder>
```

Sync the environment to install all dependencies instantly:
```bash
uv sync
```

### 3. Run the App
Launch the local Gradio interface:
```bash
uv run app.py
```
Then open your browser to `http://127.0.0.1:7860`.

## ✍️ Script Formatting
The app allows you to pass "Style Tags" in your script to visually segment your text:
- `[Shock]` - High energy, disbelief
- `[Fast]` - Rapid-fire explainer
- `[Deep]` - Slow, dramatic

**Example:**
```text
[Shock] Breaking news! [Fast] Here is the story you missed. [Deep] And it changes everything.
```

## 📂 Project Structure
- `app.py`: The main Gradio application and TTS generation loop.
- `audio_effect.py`: The 6-stage audio enhancement pipeline.
- `caption_generator.py`: The Vosk-powered speech-to-text module for generating word-level timestamps.
- `pyproject.toml`: The `uv` dependency configuration file.
- `output/`: Directory where generated `.wav` files are saved.

## 📄 License
MIT
