"""Write a metronomic, lightly-accented click track Beat This! can lock onto.

Usage: python make_click.py <out.wav> [bpm] [seconds]

Mirrors the generator in transcriber/tests/test_onnx_model_e2e.py (a known-good
signal for the ONNX Beat This! model). Used by the desktop beat-detection e2e to
stage a small, deterministic input for the sidecar.
"""
import sys

import numpy as np
import soundfile as sf


def main() -> None:
    out = sys.argv[1]
    bpm = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
    sr = 22050
    n = int(sr * dur)
    t = np.arange(n) / sr
    rng = np.random.default_rng(0)
    y = 0.05 * rng.standard_normal(n) + 0.2 * np.sin(2 * np.pi * 220 * t)
    period = 60.0 / bpm
    for k in range(int(dur / period)):
        i = int(k * period * sr)
        env = np.exp(-np.arange(min(3000, n - i)) / 300.0)
        amp = 1.0 if k % 4 == 0 else 0.6  # accent the downbeats
        y[i : i + len(env)] += amp * env * rng.standard_normal(len(env))
    sf.write(out, (y / np.abs(y).max() * 0.9).astype(np.float32), sr)


if __name__ == "__main__":
    main()
