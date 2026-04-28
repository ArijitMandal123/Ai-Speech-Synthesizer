# 🎬 AI Speech Synthesis Automation

A comprehensive, automated Text-to-Speech (TTS) pipeline tailored for YouTube Shorts, Anime Fan Theories, and dynamic video content. This repository contains two specialized environments for running the **Kokoro-82M** TTS engine and **Vosk** caption generator.

## 📂 Repository Structure

This repository is split into two main environments:

### 1. `local/` (Premium UI & Local Execution)
The `local` directory contains a modernized, `uv`-managed Gradio application designed to run on your local machine with full GPU (CUDA) support.
- **Features:** Glassmorphism UI, 18 Voice Profiles, Word-Level Captions, and a 6-stage audio effects chain (Normalization, High-Pass Filter, Bass Boost, Compression).
- **Setup:** Uses `uv` for lightning-fast dependency management.
- **Usage:** Run `uv run app.py` to launch the local web interface.
- *See `local/README.md` for detailed instructions.*

### 2. `Deploy/` (Hugging Face Spaces API)
The `Deploy` directory contains a lightweight FastAPI backend designed to be deployed to the cloud (specifically Hugging Face Spaces Free Tier).
- **Features:** Optimized for 16GB RAM/2 vCPU environments. Exposes a `/synthesize` POST endpoint.
- **Usage:** Intended to be hit by webhooks or automation platforms like **n8n** to generate audio and JSON captions programmatically without a UI.

## ✨ Core Technologies

- **Kokoro-82M Engine:** Incredibly fast and high-quality Text-to-Speech.
- **Vosk:** Offline, lightweight Speech-to-Text for generating precise word-level JSON captions.
- **Librosa & Soundfile:** For advanced audio manipulation, silence trimming, and effect processing.
- **FastAPI & Gradio:** Powering the API and web interfaces.
- **uv:** Astral's ultrafast Python package manager.

## 🚀 How to Use

Depending on your goal, navigate into one of the directories to get started:

- **Want to generate audio manually with a nice UI?**
  Navigate to `local/` and follow its README.
- **Want to deploy an API for n8n automation?**
  Navigate to `Deploy/` and deploy it to a Hugging Face Space.

## 📄 License
MIT
