#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Speech Synthesis API — Kokoro-82M
Local Gradio App
"""

import os
import time
import gc
import threading
import re
import soundfile as sf
import numpy as np
import librosa
import torch
import gradio as gr
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────
OUTPUT_DIR = "output"
CLEANUP_INTERVAL_SECONDS = 600  # Clean up every 10 minutes
MAX_FILE_AGE_SECONDS = 9000     # Delete files older than 2.5 hours
SAMPLE_RATE = 24000              # Kokoro outputs 24 kHz audio

os.makedirs(OUTPUT_DIR, exist_ok=True)

_pipeline = None

def _load_model():
    """Lazy-load the Kokoro pipeline on first request."""
    global _pipeline
    if _pipeline is not None:
        return

    from kokoro import KPipeline
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Loading Kokoro-82M pipeline (lang_code='a') on {device} …")
    
    try:
        _pipeline = KPipeline(lang_code='a', device=device)
    except TypeError:
        _pipeline = KPipeline(lang_code='a')

    gc.collect()
    print("[INFO] Kokoro pipeline loaded and ready.")

# ── Auto-cleanup Background Task ───────────────────────────────────────────
def _cleanup_old_files():
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

_cleanup_thread = threading.Thread(target=_cleanup_old_files, daemon=True)
_cleanup_thread.start()

# ── Voice Options ──────────────────────────────────────────────────────────
VOICE_CHOICES = [
    "am_adam", "am_michael", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_onyx",
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "af_alloy", "af_nova",
    "bm_george", "bm_lewis", "bf_emma", "bf_isabella"
]
DEFAULT_VOICE = "am_eric"

from audio_effect import enhance_audio
from caption_generator import generate_captions

def parse_text(text: str):
    pattern = r'\[(\w+)\]\s*([^\[]+)'
    matches = re.findall(pattern, text)
    if not matches:
        # If no tags, assume default
        return [{'tag': 'FAST', 'text': text}]
    
    chunks = []
    for tag, content in matches:
        chunks.append({
            'tag': tag.upper(),
            'text': content.strip()
        })
    return chunks

def trim_silence(audio: np.ndarray, sr: int, threshold_db: float = -40.0, pad_ms: int = 50) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(audio, top_db=-threshold_db)
    pad_samples = int(sr * pad_ms / 1000)
    if pad_samples > 0:
        silence = np.zeros(pad_samples, dtype=audio.dtype)
        trimmed = np.concatenate([silence, trimmed, silence])
    return trimmed

def synthesize(text: str, seed: int = 42, speed: float = 1.2, voice: str = "") -> dict:
    """Full pipeline: parse → generate → caption → enhance → normalize → save."""
    t0 = time.time()
    use_voice = voice.strip() if voice.strip() else DEFAULT_VOICE
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log(f"🎤 Voice: {use_voice}  |  Seed: {seed}  |  Speed: {speed}x")
    _load_model()

    chunks = parse_text(text)
    log(f"📝 Parsed {len(chunks)} chunk(s) from script")

    audio_segments = []
    sr = SAMPLE_RATE

    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Step 1: TTS Generation ──
    for i, chunk in enumerate(chunks):
        tag = chunk['tag']
        content = chunk['text']
        t_gen = time.time()

        segment_audios = []
        generator = _pipeline(content, voice=use_voice, speed=speed)

        for _gs, _ps, audio_segment in generator:
            if audio_segment is None:
                continue
            if hasattr(audio_segment, 'cpu'):
                audio_segment = audio_segment.cpu().numpy()
            audio_segment = np.asarray(audio_segment, dtype=np.float32).flatten()
            if len(audio_segment) > 0:
                segment_audios.append(audio_segment)

        if not segment_audios:
            log(f"  ⚠️ Chunk {i+1} [{tag}] — no audio generated, skipped")
            continue

        audio = np.concatenate(segment_audios) if len(segment_audios) > 1 else segment_audios[0]
        audio = trim_silence(audio, sr, threshold_db=-40.0, pad_ms=50)
        dur = len(audio) / sr
        elapsed = time.time() - t_gen
        log(f"  ✅ Chunk {i+1}/{len(chunks)} [{tag}] — {dur:.1f}s audio in {elapsed:.1f}s")
        audio_segments.append(audio)

    if not audio_segments:
        raise gr.Error("No audio was generated. Check your script format.")

    # ── Step 2: Caption Generation (Vosk STT) ──
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
    log(f"💬 Captions: {len(captions)} words detected in {time.time()-t_stt:.1f}s")

    # ── Step 3: Audio Enhancement (6-stage Audacity chain) ──
    t_fx = time.time()
    enhanced_segments = []
    for i, audio_chunk in enumerate(audio_segments):
        enhanced = enhance_audio(audio_chunk, sr)
        enhanced_segments.append(enhanced)
    log(f"🎛️ Audio Effects applied in {time.time()-t_fx:.1f}s")

    # ── Step 4: Final Mix ──
    final_audio = np.concatenate(enhanced_segments, axis=0)

    peak = np.max(np.abs(final_audio))
    if peak > 1e-6:
        target = 10 ** (-1.0 / 20.0)
        final_audio = final_audio * (target / peak)
    final_audio = np.clip(final_audio, -1.0, 1.0)

    # ── Step 5: Resample to 48kHz 24-bit ──
    final_sr = 48000
    if sr != final_sr:
        final_audio = librosa.resample(final_audio, orig_sr=sr, target_sr=final_sr)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"final_short_{timestamp}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)
    sf.write(filepath, final_audio, final_sr, subtype='PCM_24')

    total_dur = len(final_audio) / final_sr
    total_time = time.time() - t0
    peak_db = 20.0 * np.log10(max(np.max(np.abs(final_audio)), 1e-10))
    log(f"📦 Output: {total_dur:.1f}s @ 48kHz 24-bit  |  Peak: {peak_db:.1f} dBFS")
    log(f"⏱️ Total pipeline time: {total_time:.1f}s")

    # Build a pipeline summary string
    pipeline_md = f"""### ✅ Generation Complete

| Metric | Value |
|---|---|
| **Voice** | `{use_voice}` |
| **Chunks** | {len(chunks)} |
| **Words Detected** | {len(captions)} |
| **Duration** | {total_dur:.1f}s |
| **Sample Rate** | 48,000 Hz / 24-bit PCM |
| **Peak Level** | {peak_db:.1f} dBFS |
| **Pipeline Time** | {total_time:.1f}s |

#### 🎛️ Audio Effects Applied
1. ✅ Normalize (DC offset removal, peak → -1.0 dB)
2. ✅ Low Rolloff (HP filter, -27 dB @ 20 Hz)
3. ✅ Bass Boost (+9 dB sub-bass 20–100 Hz)
4. ✅ Bass & Treble Pass 1 (+5 dB each)
5. ✅ Bass & Treble Pass 2 (+5 dB each)
6. ✅ Compressor (-10 dB threshold, 10:1 ratio)
"""

    return {
        "filepath": filepath,
        "words": captions,
        "pipeline_md": pipeline_md,
        "log": "\n".join(log_lines),
    }


# ═══════════════════════════════════════════════════════════════════════════
# PREMIUM UI
# ═══════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
/* ── Google Font Import ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary: #0b1120;
    --bg-card: rgba(17, 24, 43, 0.85);
    --bg-card-hover: rgba(25, 35, 60, 0.9);
    --border-subtle: rgba(99, 102, 241, 0.15);
    --border-glow: rgba(99, 102, 241, 0.4);
    --accent-blue: #6366f1;
    --accent-cyan: #22d3ee;
    --accent-pink: #ec4899;
    --accent-green: #34d399;
    --accent-amber: #fbbf24;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --radius: 14px;
    --shadow-card: 0 4px 24px rgba(0, 0, 0, 0.4), 0 0 0 1px var(--border-subtle);
    --shadow-glow: 0 0 20px rgba(99, 102, 241, 0.15);
}

/* ── Global ── */
body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}

/* ── Header ── */
.hero-header {
    text-align: center;
    padding: 2rem 1rem 1rem;
}
.hero-header h1 {
    font-size: 2.8rem;
    font-weight: 900;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 30%, #ec4899 60%, #22d3ee 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.3rem;
    line-height: 1.1;
}
.hero-header p {
    color: var(--text-secondary);
    font-size: 1rem;
    font-weight: 400;
    margin: 0;
}

/* ── Tag Pills ── */
.tag-guide {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin: 1rem 0;
}
.tag-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    border: 1px solid;
    transition: all 0.2s ease;
}
.tag-pill:hover { transform: translateY(-2px); }
.tag-shock  { background: rgba(239, 68, 68, 0.12);  border-color: rgba(239, 68, 68, 0.3);  color: #fca5a5; }
.tag-fast   { background: rgba(59, 130, 246, 0.12); border-color: rgba(59, 130, 246, 0.3); color: #93c5fd; }
.tag-deep   { background: rgba(139, 92, 246, 0.12); border-color: rgba(139, 92, 246, 0.3); color: #c4b5fd; }
.tag-sarc   { background: rgba(251, 191, 36, 0.12); border-color: rgba(251, 191, 36, 0.3); color: #fde68a; }
.tag-quest  { background: rgba(34, 211, 238, 0.12); border-color: rgba(34, 211, 238, 0.3); color: #a5f3fc; }

/* ── Cards ── */
.card {
    background: var(--bg-card) !important;
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow-card) !important;
    padding: 1.25rem !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {
    border-color: var(--border-glow) !important;
    box-shadow: var(--shadow-card), var(--shadow-glow) !important;
}

/* ── Section Headers inside cards ── */
.section-title {
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin: 0 0 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-subtle);
}

/* ── Textbox / Input Overrides ── */
.gradio-container textarea, .gradio-container input[type="text"] {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', monospace !important;
    font-size: 0.92rem !important;
    transition: border-color 0.2s ease !important;
}
.gradio-container textarea:focus, .gradio-container input[type="text"]:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}

/* ── Primary Button ── */
.gr-button-primary {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.02em;
    padding: 12px 24px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}
.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(99, 102, 241, 0.5) !important;
}

/* ── Tabs ── */
.gradio-container .tabs .tab-nav button {
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    border-radius: 10px 10px 0 0 !important;
}

/* ── Pipeline Markdown ── */
.pipeline-output table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}
.pipeline-output th, .pipeline-output td {
    padding: 6px 12px;
    border-bottom: 1px solid var(--border-subtle);
    text-align: left;
}
.pipeline-output code {
    background: rgba(99, 102, 241, 0.15);
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.82rem;
    color: var(--accent-cyan);
}

/* ── Log Output ── */
.log-box textarea {
    font-family: 'Cascadia Code', 'Fira Code', 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    line-height: 1.6 !important;
    background: rgba(2, 6, 23, 0.8) !important;
    color: var(--accent-green) !important;
    border-radius: 10px !important;
}

/* ── Examples ── */
.gradio-container .examples-table {
    border-radius: var(--radius) !important;
    overflow: hidden;
}
"""

# ── Example Scripts ─────────────────────────────────────────────────────────
EXAMPLES = [
    [
        "[Shock] You will NOT believe what just leaked! "
        "[Fast] So basically, the creator just confirmed that the villain "
        "was actually the hero's father the entire time. "
        "[Deep] And when you go back and rewatch the series, "
        "every single clue was right there in front of us."
    ],
    [
        "[Fast] What's up everyone! Today we're breaking down the craziest "
        "fan theory about Attack on Titan. "
        "[Deep] And I'm warning you right now, this changes everything. "
        "[Question] But what if Eren planned it all from episode one? "
        "[Shock] That's right, he knew everything!"
    ],
    [
        "[Fast] Alright let's get right into it. "
        "[Deep] There's a hidden detail in Elden Ring that nobody noticed. "
        "[Sarcastic] Oh and the Two Fingers? They had no idea what was happening. "
        "[Shock] Marika played everyone!"
    ],
    [
        "[Question] Have you ever wondered why Gojo was sealed? "
        "[Fast] Well here's the thing, it wasn't just about power. "
        "[Deep] It was about control. "
        "[Shock] And when you realize what Kenjaku actually wanted, "
        "it changes the entire story!"
    ],
]

# ── Build the App ──────────────────────────────────────────────────────────

device_label = "CUDA GPU" if torch.cuda.is_available() else "CPU"

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
)

demo = gr.Blocks()

with demo:

    # ── Hero Header ──
    gr.HTML(f"""
    <div class="hero-header">
        <h1>🎬 AI Speech Synthesizer</h1>
        <p>Kokoro-82M Engine  ·  Vosk Captions  ·  6-Stage Audio FX  ·  Running on <strong>{device_label}</strong></p>
    </div>
    <div class="tag-guide">
        <span class="tag-pill tag-shock">💥 [Shock] Disbelief</span>
        <span class="tag-pill tag-fast">💨 [Fast] Explainer</span>
        <span class="tag-pill tag-deep">🔊 [Deep] Cinematic</span>
        <span class="tag-pill tag-sarc">😏 [Sarcastic] Dry Wit</span>
        <span class="tag-pill tag-quest">❓ [Question] Engaging</span>
    </div>
    """)

    # ── Main Layout ──
    with gr.Row(equal_height=False):

        # ──── Left Column: Input ────
        with gr.Column(scale=5):
            with gr.Group(elem_classes="card"):
                gr.HTML("<div class='section-title'>📝 Script Editor</div>")
                text_input = gr.Textbox(
                    placeholder=(
                        "[Shock] Breaking news just dropped!\n"
                        "[Fast] Here's everything you need to know about the latest reveal.\n"
                        "[Deep] And trust me, it changes the entire story.\n"
                        "[Question] But what do you think? Let me know!"
                    ),
                    lines=10,
                    show_label=False,
                    max_lines=20,
                )

            with gr.Group(elem_classes="card"):
                gr.HTML("<div class='section-title'>⚙️ Generation Settings</div>")
                with gr.Row():
                    voice_input = gr.Dropdown(
                        label="Voice Profile",
                        choices=VOICE_CHOICES,
                        value=DEFAULT_VOICE,
                        interactive=True,
                    )
                    speed_input = gr.Slider(
                        label="Speed",
                        minimum=0.5, maximum=2.0,
                        value=1.2, step=0.05,
                    )
                    seed_input = gr.Number(
                        label="Seed",
                        value=42, precision=0,
                    )

            submit_btn = gr.Button(
                "🚀  Generate Audio & Captions",
                variant="primary", size="lg",
            )

        # ──── Right Column: Output ────
        with gr.Column(scale=5):
            with gr.Group(elem_classes="card"):
                gr.HTML("<div class='section-title'>🎧 Audio Output</div>")
                output_audio = gr.Audio(
                    label="Final Enhanced Audio (48 kHz / 24-bit)",
                    type="filepath", interactive=False,
                )

            with gr.Tabs():
                with gr.TabItem("📊 Pipeline"):
                    with gr.Group(elem_classes="card pipeline-output"):
                        output_pipeline = gr.Markdown(
                            value="*Generate audio to see the pipeline summary here.*"
                        )
                with gr.TabItem("💬 Captions"):
                    with gr.Group(elem_classes="card"):
                        output_captions = gr.JSON(label="Word-Level Timestamps")
                with gr.TabItem("📋 Log"):
                    with gr.Group(elem_classes="card log-box"):
                        output_log = gr.Textbox(
                            label="Pipeline Log",
                            lines=12, max_lines=20,
                            interactive=False,
                        )

    # ── Examples ──
    gr.HTML("<div style='text-align:center; margin: 1.5rem 0 0.5rem;'><span style='font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#64748b;'>📂 Example Scripts — Click to load</span></div>")
    gr.Examples(
        examples=EXAMPLES,
        inputs=text_input,
        label="",
    )

    # ── Event Handler ──
    def _gradio_synthesize(text, seed, speed, voice):
        if not text or not text.strip():
            raise gr.Error("Please enter a script to generate.")
        result = synthesize(text, int(seed), float(speed), voice)
        return (
            result["filepath"],
            result["pipeline_md"],
            result["words"],
            result["log"],
        )

    submit_btn.click(
        fn=_gradio_synthesize,
        inputs=[text_input, seed_input, speed_input, voice_input],
        outputs=[output_audio, output_pipeline, output_captions, output_log],
    )


# ═══════════════════════════════════════════════════════════════════════════
# LAUNCH
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  AI Speech Synthesizer - Local Edition")
    print(f"  Device: {'CUDA GPU' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        theme=THEME,
        css=CUSTOM_CSS,
    )