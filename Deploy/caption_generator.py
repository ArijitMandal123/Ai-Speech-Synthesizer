"""
Caption Generator — Vosk Speech-to-Text
========================================
Generates word-level captions (JSON) from audio chunks using Vosk.
Optimized for HuggingFace Free Tier (16GB RAM, 2 vCPU).

Usage (imported by app.py):
    from caption_generator import generate_captions
    captions = generate_captions(trimmed_chunks, sr)
"""

import os
import gc
import json
import wave
import struct
import tempfile
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Vosk model — smallest English model (~40MB download, ~300MB RAM)
VOSK_MODEL_NAME = "vosk-model-small-en-us-0.15"
VOSK_MODEL_DIR = os.path.join(tempfile.gettempdir(), "vosk-models")
VOSK_SR = 16000  # Vosk requires 16kHz mono 16-bit PCM

# Singleton model instance
_vosk_model = None


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

def _load_vosk_model():
    """
    Lazy-load Vosk model. Downloads on first call, caches on disk.
    Returns the Model instance.
    """
    global _vosk_model
    if _vosk_model is not None:
        return _vosk_model

    from vosk import Model, SetLogLevel

    # Suppress Vosk's verbose logging
    SetLogLevel(-1)

    model_path = os.path.join(VOSK_MODEL_DIR, VOSK_MODEL_NAME)

    if not os.path.isdir(model_path):
        print(f"[VOSK] Downloading model '{VOSK_MODEL_NAME}'...")
        os.makedirs(VOSK_MODEL_DIR, exist_ok=True)

        # vosk.Model auto-downloads if given just the model name
        _vosk_model = Model(model_name=VOSK_MODEL_NAME)
        print(f"[VOSK] Model downloaded and loaded.")
    else:
        print(f"[VOSK] Loading cached model from {model_path}")
        _vosk_model = Model(model_path=model_path)
        print(f"[VOSK] Model loaded.")

    return _vosk_model


# ═══════════════════════════════════════════════════════════════════════════
# AUDIO CONVERSION
# ═══════════════════════════════════════════════════════════════════════════

def _audio_to_pcm16(audio: np.ndarray, sr: int) -> bytes:
    """
    Convert float32 numpy audio to 16kHz mono 16-bit PCM bytes.
    Resamples if sr != 16000. This is what Vosk expects.
    """
    # Resample to 16kHz if needed
    if sr != VOSK_SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=VOSK_SR)

    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Clip and convert float32 → int16
    audio = np.clip(audio, -1.0, 1.0)
    pcm_data = (audio * 32767).astype(np.int16)

    return pcm_data.tobytes()


# ═══════════════════════════════════════════════════════════════════════════
# CAPTION GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_captions(trimmed_chunks: list, sr: int) -> list:
    """
    Generate word-level captions from trimmed audio chunks using Vosk.

    Args:
        trimmed_chunks: List of dicts with keys:
            - 'audio': np.ndarray (trimmed float32 audio at original sr)
            - 'text':  str (original script text for this chunk)
            - 'tag':   str (style tag, e.g. 'SHOCK', 'FAST')
        sr: Sample rate of the audio (e.g. 24000)

    Returns:
        Flat list of word dicts:
        [
            {"word": "Dragon", "start": 0.0, "end": 0.32, "conf": 0.98},
            {"word": "Ball", "start": 0.32, "end": 0.55, "conf": 0.99},
            ...
        ]
    """
    from vosk import KaldiRecognizer

    model = _load_vosk_model()

    print(f"\n{'='*60}")
    print(f"  CAPTION GENERATOR — Vosk STT")
    print(f"{'='*60}")

    all_words = []
    time_offset = 0.0  # Running offset for multi-chunk positioning

    for i, chunk in enumerate(trimmed_chunks):
        audio = chunk['audio']
        script_text = chunk['text']
        tag = chunk['tag']

        chunk_duration = len(audio) / sr
        print(f"  ▸ [{i+1}/{len(trimmed_chunks)}] [{tag}] {chunk_duration:.2f}s — \"{script_text[:50]}...\"")

        # Convert to 16kHz PCM
        pcm_bytes = _audio_to_pcm16(audio, sr)

        # Create recognizer with word-level timestamps
        rec = KaldiRecognizer(model, VOSK_SR)
        rec.SetWords(True)

        # Feed audio in small chunks (4000 bytes ≈ 125ms at 16kHz 16-bit)
        FEED_SIZE = 4000
        for pos in range(0, len(pcm_bytes), FEED_SIZE):
            rec.AcceptWaveform(pcm_bytes[pos:pos + FEED_SIZE])

        # Get final result
        result = json.loads(rec.FinalResult())

        # Extract words with offset-adjusted timestamps
        if 'result' in result:
            for w in result['result']:
                all_words.append({
                    "word": w['word'],
                    "start": round(w['start'] + time_offset, 3),
                    "end": round(w['end'] + time_offset, 3),
                    "conf": round(w.get('conf', 0.0), 3)
                })

        time_offset += chunk_duration

        # Free recognizer memory
        del rec, pcm_bytes
        gc.collect()

        print(f"    → {len(all_words)} total words so far")

    print(f"{'='*60}")
    print(f"  DONE — {len(all_words)} total words across {len(trimmed_chunks)} chunk(s)")
    print(f"{'='*60}\n")

    return all_words
"""
Standalone module for Vosk STT captions.
Uses the smallest English model for efficient CPU inference.
"""