"""Onset-preserving audio augmentation for the param-predictor corpus.

Every transform keeps the waveform's length and its onset *times* unchanged, so
one labeled song yields many training rows: the ground-truth onsets stay valid
and the oracle simply re-sweeps on the new activation curve (design spec
§augmentation). Transforms that add latency (reverb tail, lossy-codec
encoder/decoder delay) are length-truncated and delay-compensated by
cross-correlation back to the original, so peaks land where the labels say.

Excluded by design: time-stretch / pitch-shift (they move onsets). Pure
numpy/scipy, except `apply_codec`, which shells out to ffmpeg.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, lfilter

#: Max codec/reverb latency we search for when re-aligning (s).
_MAX_LAG_S = 0.1


def apply_gain(y: np.ndarray, db: float) -> np.ndarray:
    """Scale by `db` decibels (the residual gain variance median-normalize misses)."""
    return (np.asarray(y, dtype=np.float32) * float(10.0 ** (db / 20.0))).astype(np.float32)


def apply_eq_tilt(y: np.ndarray, sr: int, tilt_db: float) -> np.ndarray:
    """First-order spectral tilt: +`tilt_db` brightens (kit gets brighter), - darkens.
    A one-pole high/low shelf via a simple pre-emphasis-style filter, length-preserving."""
    y = np.asarray(y, dtype=np.float32)
    # pre-emphasis coefficient mapped from the requested tilt; clamped to stable range.
    a = float(np.clip(tilt_db / 24.0, -0.95, 0.95))
    # a >= 0 brightens (emphasize highs); a < 0 darkens (one-pole low-pass)
    out = lfilter([1.0, -a], [1.0], y) if a >= 0 else lfilter([1.0 + a], [1.0, a], y)
    out = out.astype(np.float32)
    peak = float(np.max(np.abs(out)) or 1.0)
    return (out / peak * float(np.max(np.abs(y)) or peak)).astype(np.float32)


def apply_reverb(y: np.ndarray, sr: int, decay_s: float = 0.3, wet: float = 0.3) -> np.ndarray:
    """Convolve with a synthetic exponential-decay impulse response (a causal IR
    whose direct path is at lag 0, so onsets don't move), mix `wet`, then truncate
    the tail back to the original length."""
    y = np.asarray(y, dtype=np.float32)
    n_ir = max(1, int(decay_s * sr))
    t = np.arange(n_ir) / sr
    rng = np.random.default_rng(0)
    ir = (np.exp(-t / (decay_s / 4.0)) * rng.standard_normal(n_ir)).astype(np.float32)
    ir[0] = 1.0  # direct path at lag 0 -> onset position preserved
    wetsig = fftconvolve(y, ir)[: len(y)].astype(np.float32)
    wetsig *= float(np.max(np.abs(y)) or 1.0) / float(np.max(np.abs(wetsig)) or 1.0)
    return ((1.0 - wet) * y + wet * wetsig).astype(np.float32)


def apply_compression(y: np.ndarray, sr: int, threshold_db: float = -20.0, ratio: float = 4.0) -> np.ndarray:
    """Feed-forward dynamic-range compression on a smoothed envelope - reshapes
    the peak-height distribution without shifting transients."""
    y = np.asarray(y, dtype=np.float32)
    eps = 1e-9
    env = np.abs(y)
    # ~10ms attack/release smoothing so the gain curve doesn't add its own transient
    win = max(1, int(0.01 * sr))
    kernel = np.ones(win, dtype=np.float32) / win
    # FFT-based convolution of a non-negative signal can dip slightly negative
    # from roundoff (largest near silence next to loud hits); clamp so log10 is
    # always defined.
    env = np.maximum(fftconvolve(env, kernel, mode="same"), 0.0).astype(np.float32)
    env_db = 20.0 * np.log10(env + eps)
    over = np.maximum(0.0, env_db - threshold_db)
    gain_db = -over * (1.0 - 1.0 / ratio)
    gain = (10.0 ** (gain_db / 20.0)).astype(np.float32)
    out = y * gain
    out *= float(np.max(np.abs(y)) or 1.0) / float(np.max(np.abs(out)) or 1.0)  # make-up to original peak
    return out.astype(np.float32)


def apply_noise(y: np.ndarray, snr_db: float = 20.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Add white noise at the requested signal-to-noise ratio (separation /
    bleed artifacts)."""
    y = np.asarray(y, dtype=np.float32)
    rng = rng or np.random.default_rng()
    sig_p = float(np.mean(y.astype(np.float64) ** 2)) + 1e-12
    noise_p = sig_p / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(len(y)).astype(np.float32) * float(np.sqrt(noise_p))
    return (y + noise).astype(np.float32)


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _align_to(reference: np.ndarray, y: np.ndarray, sr: int) -> np.ndarray:
    """Shift `y` to best match `reference` (removing codec delay) and crop/pad to
    the reference length. Lag searched within +/-`_MAX_LAG_S`."""
    n = len(reference)
    max_lag = int(_MAX_LAG_S * sr)
    a = reference[: min(len(reference), 4 * sr)].astype(np.float64)
    b = y[: min(len(y), 4 * sr + max_lag)].astype(np.float64)
    corr = fftconvolve(b, a[::-1], mode="full")
    mid = len(a) - 1
    window = corr[mid - max_lag : mid + max_lag + 1]
    lag = int(np.argmax(window)) - max_lag  # >0 => y is delayed
    shifted = y[lag:] if lag > 0 else np.concatenate([np.zeros(-lag, dtype=y.dtype), y])
    out = np.zeros(n, dtype=np.float32)
    out[: min(n, len(shifted))] = shifted[:n]
    return out


def apply_codec(y: np.ndarray, sr: int, bitrate_kbps: int = 128, codec: str = "libmp3lame") -> np.ndarray:
    """Lossy-codec round-trip (default 128 kbps MP3) - band-limits and adds codec
    pre-echo. The decoder delay is removed by `_align_to`, so onsets stay put."""
    import soundfile as sf

    y = np.asarray(y, dtype=np.float32)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "in.wav"
        enc = Path(td) / f"c.{'mp3' if codec == 'libmp3lame' else 'opus'}"
        dec = Path(td) / "out.wav"
        sf.write(str(src), y, sr)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-c:a", codec, "-b:a", f"{bitrate_kbps}k", str(enc)], check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(enc), "-ar", str(sr), str(dec)], check=True,
        )
        decoded, _ = sf.read(str(dec), dtype="float32")
    if decoded.ndim > 1:
        decoded = decoded.mean(axis=1)
    return _align_to(y, decoded.astype(np.float32), sr)


def random_chain(
    y: np.ndarray, sr: int, rng: np.random.Generator, *, use_codec: bool = True,
    codec_bitrates: tuple[int, ...] = (128, 256),
) -> tuple[np.ndarray, str]:
    """Apply a random subset of the onset-preserving transforms and return
    `(augmented, description)`. The description records the chain for provenance."""
    out = np.asarray(y, dtype=np.float32)
    steps: list[str] = []
    if rng.random() < 0.8:
        db = float(rng.uniform(-9.0, 9.0))
        out = apply_gain(out, db)
        steps.append(f"gain{db:+.1f}dB")
    if rng.random() < 0.6:
        tilt = float(rng.uniform(-12.0, 12.0))
        out = apply_eq_tilt(out, sr, tilt)
        steps.append(f"tilt{tilt:+.1f}dB")
    if rng.random() < 0.4:
        decay = float(rng.uniform(0.15, 0.6))
        out = apply_reverb(out, sr, decay_s=decay, wet=float(rng.uniform(0.15, 0.45)))
        steps.append(f"reverb{decay:.2f}s")
    if rng.random() < 0.5:
        ratio = float(rng.uniform(2.0, 8.0))
        out = apply_compression(out, sr, threshold_db=float(rng.uniform(-30.0, -12.0)), ratio=ratio)
        steps.append(f"comp{ratio:.1f}:1")
    if rng.random() < 0.5:
        snr = float(rng.uniform(12.0, 30.0))
        out = apply_noise(out, snr_db=snr, rng=rng)
        steps.append(f"noise{snr:.0f}dB")
    if use_codec and has_ffmpeg() and rng.random() < 0.5:
        br = int(rng.choice(codec_bitrates))
        out = apply_codec(out, sr, bitrate_kbps=br)
        steps.append(f"mp3{br}k")
    # contract: always return finite audio so a corpus build never dies on one
    # pathological sample (backstop; transforms should already stay finite).
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return out, "+".join(steps) if steps else "identity"
