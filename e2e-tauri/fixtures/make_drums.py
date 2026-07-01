"""Synthesize a simple, known drum loop for the end-to-end transcription e2e.

A canonical rock beat at a fixed BPM: kick 4-on-the-floor (every beat), snare on
the backbeats (2 & 4), closed hi-hat on every 8th note. The pattern is
deterministic so the e2e can check the transcription against it (roughly -- real
separation + onset detection isn't exact). The synthesis is deliberately
drum-like (transient + body + noise) rather than pure tones so the separator /
onset model behave closer to how they do on real audio.

Usage: python make_drums.py <out.wav> [bpm] [bars]
"""
import sys

import numpy as np
import soundfile as sf

SR = 22050


def _kick(n: int) -> np.ndarray:
    t = np.arange(n) / SR
    # Pitch-dropping sine (80 -> 45 Hz) + a short click transient, fast decay.
    freq = 80.0 * np.exp(-t * 30.0) + 45.0
    body = np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t * 12.0)
    click = np.exp(-t * 400.0) * np.random.default_rng(1).standard_normal(n) * 0.3
    return (body + click).astype(np.float32)


def _snare(n: int) -> np.ndarray:
    t = np.arange(n) / SR
    rng = np.random.default_rng(2)
    noise = rng.standard_normal(n) * np.exp(-t * 22.0)
    tone = 0.4 * np.sin(2 * np.pi * 190.0 * t) * np.exp(-t * 18.0)
    return (0.9 * noise + tone).astype(np.float32)


def _hat(n: int) -> np.ndarray:
    t = np.arange(n) / SR
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(n)
    # Crude high-pass: subtract a smoothed copy so low freqs cancel.
    hp = noise - np.convolve(noise, np.ones(8) / 8, mode="same")
    return (hp * np.exp(-t * 60.0) * 0.5).astype(np.float32)


def _place(track: np.ndarray, hit: np.ndarray, at_sec: float, gain: float) -> None:
    i = int(at_sec * SR)
    end = min(i + len(hit), len(track))
    track[i:end] += gain * hit[: end - i]


def main() -> None:
    out = sys.argv[1]
    bpm = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
    bars = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    beat = 60.0 / bpm
    total = bars * 4 * beat
    n = int(total * SR) + SR  # +1s tail
    y = np.zeros(n, np.float32)

    kick, snare, hat = _kick(int(0.25 * SR)), _snare(int(0.2 * SR)), _hat(int(0.08 * SR))
    for b in range(bars):
        for beat_idx in range(4):
            t0 = (b * 4 + beat_idx) * beat
            _place(y, kick, t0, 0.9)  # 4-on-the-floor
            if beat_idx in (1, 3):
                _place(y, snare, t0, 0.8)  # backbeat
            _place(y, hat, t0, 0.35)
            _place(y, hat, t0 + beat / 2, 0.3)  # off-beat 8ths

    sf.write(out, (y / np.abs(y).max() * 0.9).astype(np.float32), SR)


if __name__ == "__main__":
    main()
