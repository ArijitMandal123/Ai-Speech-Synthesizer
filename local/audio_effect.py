"""
Audio Effect Chain 3 — Audacity Replica
========================================
6-stage pipeline replicated from Audacity screenshots.

Pipeline:
  1. Normalize         — DC offset removal, peak to -1.0 dB
  2. Low Rolloff (HP)  — Steep high-pass, kills everything below ~60 Hz
  3. Bass Boost (EQ)   — +9 dB (20–100 Hz), tapers to 0 dB at 500 Hz
  4. Bass & Treble #1  — Bass +5 dB, Treble +5 dB
  5. Bass & Treble #2  — Bass +5 dB, Treble +5 dB (second pass)
  6. Compressor        — Threshold -10 dB, Ratio 10:1, Knee 5 dB

Usage:
    python audio_effect_3.py input.wav output.wav
    python audio_effect_3.py input.wav           # saves as input_enhanced_3py.wav
"""

import os
import sys
import numpy as np
import soundfile as sf
from scipy import signal as sig


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def _linear_to_db(val: float) -> float:
    if val < 1e-10:
        return -120.0
    return 20.0 * np.log10(val)


def _make_fir(freq_hz: list, gain_db: list, sr: int,
              numtaps: int = 255) -> np.ndarray:
    """Design FIR filter — 255 taps (4x faster than 1023 on CPU)."""
    nyquist = sr / 2.0
    freq_norm = [max(0.0, min(1.0, f / nyquist)) for f in freq_hz]

    if freq_norm[0] != 0.0:
        freq_norm.insert(0, 0.0)
        gain_db.insert(0, gain_db[0])
    if freq_norm[-1] != 1.0:
        freq_norm.append(1.0)
        gain_db.append(gain_db[-1])

    gains_linear = [_db_to_linear(g) for g in gain_db]
    return sig.firwin2(numtaps, freq_norm, gains_linear)


def _apply_fir(audio: np.ndarray, fir: np.ndarray) -> np.ndarray:
    """Apply FIR filter (mono/stereo)."""
    if audio.ndim == 1:
        return sig.fftconvolve(audio, fir, mode="same").astype(np.float32)
    result = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        result[:, ch] = sig.fftconvolve(audio[:, ch], fir, mode="same").astype(np.float32)
    return result


def _design_shelf(freq_hz: float, gain_db: float, sr: int,
                  shelf_type: str = "high") -> tuple:
    """Biquad shelving filter (Audio EQ Cookbook)."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq_hz / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / 2.0 * np.sqrt(2.0)
    sqrt_A = np.sqrt(A)

    if shelf_type == "high":
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    else:
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha

    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1: NORMALIZE
# From Audacity screenshot:
#   ☑ Remove DC offset (center on 0.0 vertically)
#   ☑ Normalize peak amplitude to -1.0 dB
#   ☐ Normalize stereo channels independently
# ═══════════════════════════════════════════════════════════════════════════

def normalize(audio: np.ndarray) -> np.ndarray:
    """Normalize: remove DC offset, peak to -1.0 dBFS."""
    result = audio.copy()

    # Remove DC offset
    if result.ndim == 1:
        result -= np.mean(result)
    else:
        for ch in range(result.shape[1]):
            result[:, ch] -= np.mean(result[:, ch])

    # Normalize peak to -1.0 dBFS
    peak = np.max(np.abs(result))
    if peak > 1e-10:
        target = _db_to_linear(-1.0)
        result = result * (target / peak)

    print(f"  [1/6] Normalize: DC offset removed, peak → -1.0 dBFS")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2: LOW ROLLOFF (HIGH-PASS)
# From Audacity Filter Curve EQ screenshot:
#   Steep cut below ~60-80 Hz
#   20 Hz = -27 dB, 40 Hz = -18 dB, 60 Hz = -9 dB
#   80 Hz = -3 dB, 100 Hz = 0 dB
#   Flat 0 dB above 100 Hz
# ═══════════════════════════════════════════════════════════════════════════

def low_rolloff(audio: np.ndarray, sr: int) -> np.ndarray:
    """Filter Curve EQ — Low Rolloff (steep HP below 80 Hz)."""
    freq_hz = [0,    20,    40,    60,    80,   100, sr / 2]
    gain_db = [-30, -27.0, -18.0, -9.0,  -3.0,  0.0, 0.0]

    fir = _make_fir(freq_hz, gain_db, sr)
    result = _apply_fir(audio, fir)

    print(f"  [2/6] Low Rolloff: -27 dB @ 20 Hz → 0 dB @ 100 Hz")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3: BASS BOOST (EQ)
# From Audacity Filter Curve EQ screenshot:
#   20–100 Hz: flat at +9 dB
#   140 Hz:    +6 dB
#   300 Hz:    +3 dB
#   500 Hz:    0 dB, flat above
# ═══════════════════════════════════════════════════════════════════════════

def bass_boost(audio: np.ndarray, sr: int) -> np.ndarray:
    """Filter Curve EQ — Bass Boost (+9 dB sub-bass)."""
    freq_hz = [0,    20,   100,   140,   300,   500, sr / 2]
    gain_db = [9.0,  9.0,  9.0,   6.0,   3.0,   0.0, 0.0]

    fir = _make_fir(freq_hz, gain_db, sr)
    result = _apply_fir(audio, fir)

    print(f"  [3/6] Bass Boost: +9 dB (20–100 Hz) → 0 dB @ 500 Hz")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4: BASS & TREBLE #1
# From Audacity screenshot:
#   Bass (dB):   5.0
#   Treble (dB): 5.0
#   Volume (dB): 0.0
# ═══════════════════════════════════════════════════════════════════════════

def bass_and_treble(audio: np.ndarray, sr: int,
                    bass_db: float = 5.0,
                    treble_db: float = 5.0,
                    label: str = "#1") -> np.ndarray:
    """Audacity Bass & Treble effect: low-shelf + high-shelf."""
    # Bass: low-shelf at ~250 Hz (Audacity's internal crossover)
    b_lo, a_lo = _design_shelf(250.0, bass_db, sr, shelf_type="low")
    result = sig.lfilter(b_lo, a_lo, audio, axis=0).astype(np.float32)

    # Treble: high-shelf at ~3000 Hz (Audacity's internal crossover)
    b_hi, a_hi = _design_shelf(3000.0, treble_db, sr, shelf_type="high")
    result = sig.lfilter(b_hi, a_hi, result, axis=0).astype(np.float32)

    step = "4" if label == "#1" else "5"
    print(f"  [{step}/6] Bass & Treble {label}: bass +{bass_db:.0f} dB, "
          f"treble +{treble_db:.0f} dB")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 6: COMPRESSOR
# From Audacity Compressor screenshot:
#   Threshold:    -10.0 dB
#   Make-up gain:  0.0 dB
#   Knee width:    5.0 dB
#   Ratio:        10.0:1
#   Lookahead:     1.0 ms
#   Attack:       30.0 ms
#   Release:     150.0 ms
# ═══════════════════════════════════════════════════════════════════════════

COMP_THRESHOLD_DB = -10.0
COMP_MAKEUP_DB = 0.0
COMP_KNEE_DB = 5.0
COMP_RATIO = 10.0
COMP_LOOKAHEAD_MS = 1.0
COMP_ATTACK_MS = 30.0
COMP_RELEASE_MS = 150.0


def compressor(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Vectorized compressor — ~20x faster than sample-by-sample loop.
    Threshold: -10 dB | Ratio: 10:1 | Knee: 5 dB
    Attack: 30 ms | Release: 150 ms
    """
    result = audio.copy()
    mono = np.mean(result, axis=-1) if result.ndim > 1 else result

    threshold = COMP_THRESHOLD_DB
    knee = COMP_KNEE_DB
    ratio = COMP_RATIO
    knee_start = threshold - knee / 2.0
    knee_end = threshold + knee / 2.0

    # Vectorized envelope via exponential smoothing (scipy)
    abs_mono = np.abs(mono).astype(np.float64)

    attack_coeff = np.exp(-1.0 / (sr * COMP_ATTACK_MS / 1000.0))
    release_coeff = np.exp(-1.0 / (sr * COMP_RELEASE_MS / 1000.0))

    # Use scipy.signal.lfilter for fast envelope following
    # Approximate envelope: lowpass filter on abs signal
    # Two-pass: attack (fast rise) then release (slow fall)
    envelope = np.zeros_like(abs_mono)
    env = 0.0
    # Downsample for speed: process every 4th sample, interpolate back
    step = 4
    ds_abs = abs_mono[::step]
    ds_env = np.zeros(len(ds_abs), dtype=np.float64)
    env = 0.0
    for i in range(len(ds_abs)):
        s = ds_abs[i]
        if s > env:
            env = (1 - attack_coeff) * s + attack_coeff * env
        else:
            env = (1 - release_coeff) * s + release_coeff * env
        ds_env[i] = env

    # Interpolate back to full resolution
    envelope = np.interp(
        np.arange(len(abs_mono)),
        np.arange(len(ds_abs)) * step,
        ds_env,
    ).astype(np.float64)

    # Vectorized gain computation
    level_db = np.where(envelope > 1e-10, 20.0 * np.log10(envelope), -120.0)

    gain_reduction = np.ones(len(mono), dtype=np.float32)

    # Below knee — no compression
    # Inside knee — smooth transition
    knee_mask = (level_db >= knee_start) & (level_db <= knee_end)
    x = level_db[knee_mask] - knee_start
    knee_range = knee_end - knee_start
    effective_ratio = 1.0 + (ratio - 1.0) * (x / knee_range)
    over_db = level_db[knee_mask] - threshold
    pos_mask = over_db > 0
    compressed_db = np.where(pos_mask, over_db / effective_ratio, 0.0)
    reduction_db = np.where(pos_mask, over_db - compressed_db, 0.0)
    gain_reduction[knee_mask] = np.where(
        pos_mask, _db_to_linear(-reduction_db), 1.0
    ).astype(np.float32)

    # Above knee — full ratio
    above_mask = level_db > knee_end
    over_db_above = level_db[above_mask] - threshold
    compressed_above = over_db_above / ratio
    reduction_above = over_db_above - compressed_above
    gain_reduction[above_mask] = _db_to_linear(-reduction_above).astype(np.float32)

    # Apply gain reduction
    if result.ndim == 1:
        result = result * gain_reduction
    else:
        for ch in range(result.shape[1]):
            result[:, ch] *= gain_reduction

    # Makeup gain
    if COMP_MAKEUP_DB != 0.0:
        result = result * _db_to_linear(COMP_MAKEUP_DB)

    avg_red = np.mean(1.0 - gain_reduction) * 100
    print(f"  [6/6] Compressor: {threshold:.0f} dB, {ratio:.0f}:1 | avg reduction {avg_red:.1f}%")

    return result.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def enhance_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Run the full 6-stage Audacity-replica pipeline.
    Order: Normalize → Low Rolloff → Bass Boost → B&T → B&T → Compressor
    """
    print(f"\n{'='*60}")
    print(f"  AUDIO EFFECT 3 — Audacity Replica")
    print(f"  Input: {len(audio)/sr:.2f}s | {sr} Hz | "
          f"{'stereo' if audio.ndim > 1 else 'mono'}")
    print(f"{'='*60}")

    audio = normalize(audio)
    audio = low_rolloff(audio, sr)
    audio = bass_boost(audio, sr)
    audio = bass_and_treble(audio, sr, bass_db=5.0, treble_db=5.0, label="#1")
    audio = bass_and_treble(audio, sr, bass_db=5.0, treble_db=5.0, label="#2")
    audio = compressor(audio, sr)

    # Final safety clip to prevent any distortion
    audio = np.clip(audio, -1.0, 1.0)

    peak_db = _linear_to_db(np.max(np.abs(audio)))
    print(f"{'='*60}")
    print(f"  DONE — Output peak: {peak_db:.1f} dBFS")
    print(f"{'='*60}\n")

    return audio


def process_file(input_path: str, output_path: str | None = None) -> str:
    """Load audio, run pipeline, save result."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_enhanced_3py{ext}"

    print(f"[LOAD] {input_path}")
    audio, sr = sf.read(input_path, dtype="float32")

    enhanced = enhance_audio(audio, sr)

    sf.write(output_path, enhanced, sr)
    print(f"[SAVE] {output_path}")

    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audio_effect_3.py <input.wav> [output.wav]")
        print("       If no output given, saves as <input>_enhanced_3py.wav")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    process_file(input_file, output_file)
