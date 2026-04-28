#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Speech Synthesis API — Kokoro-82M
HuggingFace Space deployment with synchronous /synthesize endpoint
Uses hexgrad/Kokoro-82M for lightweight, high-quality TTS on CPU.
"""

import os
import time
import gc
import threading
import re
import soundfile as sf
import numpy as np
import librosa
from typing import Optional
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────
OUTPUT_DIR = "output"
CLEANUP_INTERVAL_SECONDS = 600  # Clean up every 10 minutes
MAX_FILE_AGE_SECONDS = 9000     # Delete files older than 2.5 hours
SAMPLE_RATE = 24000              # Kokoro outputs 24 kHz audio

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global pipeline instance
_pipeline = None

# ── Model Loading ──────────────────────────────────────────────────────────
def _load_model():
    """Lazy-load the Kokoro pipeline on first request."""
    global _pipeline
    if _pipeline is not None:
        return

    from kokoro import KPipeline

    print("[INFO] Loading Kokoro-82M pipeline (lang_code='a') …")
    _pipeline = KPipeline(lang_code='a')  # American English
    gc.collect()
    print("[INFO] Kokoro pipeline loaded and ready.")

# ── Auto-cleanup Background Task ───────────────────────────────────────────
def _cleanup_old_files():
    """Delete audio files older than MAX_FILE_AGE_SECONDS."""
    while True:
        try:
            now = time.time()
            for filename in os.listdir(OUTPUT_DIR):
                filepath = os.path.join(OUTPUT_DIR, filename)
                if os.path.isfile(filepath):
                    age = now - os.path.getmtime(filepath)
                    if age > MAX_FILE_AGE_SECONDS:
                        os.remove(filepath)
                        print(f"[CLEANUP] Deleted: {filename} (age: {age/60:.1f} min)")
        except Exception as e:
            print(f"[CLEANUP ERROR] {e}")

        time.sleep(CLEANUP_INTERVAL_SECONDS)

# Start cleanup thread
_cleanup_thread = threading.Thread(target=_cleanup_old_files, daemon=True)
_cleanup_thread.start()

# ── Voice ──────────────────────────────────────────────────────────────────
# Single voice for the entire audio. Change this variable to switch voice.
# Options: am_adam, am_michael, am_echo, am_eric, am_fenrir, am_liam, am_onyx
#          af_heart, af_bella, af_nicole, af_sarah, af_sky, af_nova
#          bm_george, bm_lewis, bf_emma, bf_isabella
VOICE = "am_eric"

# ── Import Audio Processing Functions ─────────────────────────────────────
from audio_effect import enhance_audio
from caption_generator import generate_captions

# ── Text Parsing ───────────────────────────────────────────────────────────
def parse_text(text: str):
    """
    Parse script into chunks with tags.
    Example: "[Shock] Breaking news! [Fast] Here is the story."
    """
    pattern = r'\[(\w+)\]\s*([^\[]+)'
    matches = re.findall(pattern, text)

    chunks = []
    for tag, content in matches:
        chunks.append({
            'tag': tag.upper(),
            'text': content.strip()
        })

    return chunks

# ── Audio Processing Helpers ───────────────────────────────────────────────
def trim_silence(audio: np.ndarray, sr: int, threshold_db: float = -40.0, pad_ms: int = 50) -> np.ndarray:
    """
    Aggressively trim silence from start/end using librosa.
    Leaves a small padding (default 50ms) to avoid cutting content abruptly.
    """
    trimmed, _ = librosa.effects.trim(audio, top_db=-threshold_db)

    pad_samples = int(sr * pad_ms / 1000)
    if pad_samples > 0:
        silence = np.zeros(pad_samples, dtype=audio.dtype)
        trimmed = np.concatenate([silence, trimmed, silence])

    return trimmed

# ── Main Synthesis Function ────────────────────────────────────────────────
def synthesize(text: str, seed: int = 42, speed: float = 1.2, voice: str = "") -> dict:
    """
    Generate audio from text script and return public URL.

    Args:
        text: Script with tags, e.g. "[Shock] Breaking news!"
        seed: Random seed for reproducibility
        speed: Speed multiplier (1.0 = normal, 1.2 = slightly faster)
        voice: Voice preset to use (defaults to VOICE constant if empty)

    Returns:
        Dict with 'audio_url' and 'words'
    """
    t0 = time.time()

    # Resolve voice: API param > script-level default
    use_voice = voice.strip() if voice.strip() else VOICE

    # Log received text for debugging
    print(f"[INPUT] Received text ({len(text)} chars), voice={use_voice}: {text[:200]}...")

    # Load model if not loaded
    _load_model()

    # Parse chunks
    chunks = parse_text(text)
    if not chunks:
        raise ValueError("No valid chunks found in text")

    print("=" * 60)
    print(f"[PIPELINE] {len(chunks)} chunk(s) | seed={seed} | speed={speed}")
    print("=" * 60)
    print()

    # Generate audio for each chunk
    audio_segments = []
    sr = SAMPLE_RATE

    np.random.seed(seed)

    for i, chunk in enumerate(chunks):
        tag = chunk['tag']
        content = chunk['text']

        print(f"  ▸ [{i+1}/{len(chunks)}] [{tag}] voice={use_voice} speed={speed}")
        print(f"    \"{content}\"")
        print()

        t_gen = time.time()

        # Kokoro returns a generator of (graphemes, phonemes, audio) tuples.
        # audio may be a PyTorch tensor or numpy array depending on version.
        segment_audios = []
        generator = _pipeline(content, voice=use_voice, speed=speed)

        for _gs, _ps, audio_segment in generator:
            if audio_segment is None:
                continue
            # Convert PyTorch tensor to numpy if needed
            if hasattr(audio_segment, 'cpu'):
                audio_segment = audio_segment.cpu().numpy()
            audio_segment = np.asarray(audio_segment, dtype=np.float32).flatten()
            if len(audio_segment) > 0:
                segment_audios.append(audio_segment)

        if not segment_audios:
            print(f"    [WARN] No audio generated for chunk {i+1}, skipping")
            continue

        # Concatenate all sub-segments for this chunk
        audio = np.concatenate(segment_audios) if len(segment_audios) > 1 else segment_audios[0]

        # Post-Processing: Trim silence
        audio = trim_silence(audio, sr, threshold_db=-40.0, pad_ms=50)

        duration = len(audio) / sr
        elapsed = time.time() - t_gen
        print(f"    → {duration:.1f}s audio generated in {elapsed:.1f}s")
        print()

        audio_segments.append(audio)

    if not audio_segments:
        raise ValueError("No audio was generated for any chunk")

    # ── Generate captions from audio (before effects) ──
    t_stt = time.time()
    trimmed_stt_chunks = []
    valid_chunk_idx = 0
    for i, chunk in enumerate(chunks):
        if valid_chunk_idx < len(audio_segments):
            trimmed_stt_chunks.append({
                'audio': audio_segments[valid_chunk_idx],
                'text': chunk['text'],
                'tag': chunk['tag']
            })
            valid_chunk_idx += 1
    captions = generate_captions(trimmed_stt_chunks, sr)
    del trimmed_stt_chunks
    elapsed_stt = time.time() - t_stt
    print(f"[STT] Captions generated in {elapsed_stt:.1f}s")
    print()

    # Apply audio enhancement
    t_enhance = time.time()
    enhanced_segments = []
    for i, audio_chunk in enumerate(audio_segments):
        enhanced = enhance_audio(audio_chunk, sr)
        enhanced_segments.append(enhanced)
        print(f"  [CHUNK {i+1}] | {len(enhanced)/sr:.1f}s")

    elapsed_enhance = time.time() - t_enhance
    print(f"[ENHANCE] Done in {elapsed_enhance:.1f}s")
    print()

    # Concatenate audio chunks
    t_concat = time.time()
    final_audio = np.concatenate(enhanced_segments, axis=0)

    # Normalize volume to -1.0 dBFS
    peak = np.max(np.abs(final_audio))
    if peak > 1e-6:
        target = 10 ** (-1.0 / 20.0)
        final_audio = final_audio * (target / peak)
    final_audio = np.clip(final_audio, -1.0, 1.0)

    elapsed_concat = time.time() - t_concat
    print(f"[CONCATENATION] Done in {elapsed_concat:.1f}s")
    print()

    # Resample to 48kHz Mono
    print("[OUTPUT] Resampling to 48kHz...")
    final_sr = 48000
    if sr != final_sr:
        final_audio = librosa.resample(final_audio, orig_sr=sr, target_sr=final_sr)

    # Save file as 24-bit PCM
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"final_short_{timestamp}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)
    sf.write(filepath, final_audio, final_sr, subtype='PCM_24')

    total = time.time() - t0
    final_duration = len(final_audio) / final_sr

    # Generate public URL
    audio_url = f"https://ari123er-ai-speech-synthesis.hf.space/output/{filename}"

    print(f"[DONE] {filename} | {final_duration:.1f}s audio | took {total:.1f}s")
    print(f"[API] Audio URL: {audio_url}")
    print(f"[INFO] File will auto-delete after {MAX_FILE_AGE_SECONDS//60} min")
    print("=" * 60)
    print()

    return {
        "audio_url": audio_url,
        "words": captions
    }


# ─────────────────────────────────────────────────────────────────────────────
# PURE FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import gradio as gr

# Request model for validation
class SynthesizeRequest(BaseModel):
    text: str
    seed: int = 42
    speed: float = 1.2
    voice: str = ""  # Voice preset (leave empty to use default VOICE)

# Create main FastAPI app
app = FastAPI()

# Mount output directory for audio file access
app.mount("/output", StaticFiles(directory="output"), name="output")

@app.get("/")
def home():
    return {
        "status": "Running",
        "model": "Kokoro-82M",
        "voice": VOICE,
        "message": "AI Speech Synthesis API",
        "endpoint": "POST /synthesize",
        "ui": "/ui (Gradio interface)"
    }

@app.post("/synthesize")
async def api_synthesize(req: SynthesizeRequest):
    """
    Async endpoint that runs blocking synthesize() in thread pool.
    Uses explicit Content-Length so n8n knows when response is complete.
    """
    import asyncio
    import json
    from starlette.responses import Response

    try:
        # Run blocking synthesize() in thread pool
        result = await asyncio.to_thread(synthesize, req.text, req.seed, req.speed, req.voice)

        # Pre-serialize JSON and set Content-Length explicitly
        body = json.dumps([{
            'success': True,
            'audio_url': result['audio_url'],
            'words': result['words']
        }])

        return Response(
            content=body,
            status_code=200,
            media_type="application/json",
            headers={
                "Content-Length": str(len(body)),
                "Connection": "close"
            }
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR] /synthesize failed: {e}")
        print(tb)
        err_body = json.dumps({
            'success': False,
            'error': str(e),
            'traceback': tb
        })
        return Response(
            content=err_body,
            status_code=500,
            media_type="application/json",
            headers={
                "Content-Length": str(len(err_body)),
                "Connection": "close"
            }
        )

print("[INFO] FastAPI endpoint registered: POST /synthesize")

# ── Optional Gradio UI (mounted under /ui) ───────────────────────────────────
demo = gr.Blocks(title="AI Speech Synthesis — Kokoro-82M")

with demo:
    gr.Markdown("# AI Speech Synthesis — Kokoro-82M\n\n**API Endpoint:** `POST /synthesize`")

    with gr.Row():
        text_input = gr.Textbox(
            label="Script",
            placeholder="[Shock] Breaking news! [Fast] Here is the story.",
            lines=3
        )
    with gr.Row():
        seed_input = gr.Number(label="Seed", value=42, precision=0)
        speed_input = gr.Number(label="Speed", value=1.2, precision=2)

    output_url = gr.Textbox(label="Audio URL", interactive=False)
    submit_btn = gr.Button("Generate")

    def _gradio_synthesize(text, seed, speed):
        """Wrapper for Gradio — extracts audio_url from dict return."""
        result = synthesize(text, int(seed), float(speed))
        return result["audio_url"]

    submit_btn.click(
        fn=_gradio_synthesize,
        inputs=[text_input, seed_input, speed_input],
        outputs=output_url
    )

# Mount Gradio UI at /ui path
app = gr.mount_gradio_app(app, demo, path="/ui")

print("[INFO] Gradio UI mounted at /ui")
print("[INFO] Model will load on first request.")
print(f"[INFO] Auto-cleanup active: files expire after {MAX_FILE_AGE_SECONDS//60} min")
print(f"[INFO] Voice: {VOICE}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)