"""Deterministic analytic peak-pick params from an activation curve.

Closed-form rules that adapt to a single song with no learning: the threshold
sits at the Otsu knee of the candidate-peak-height distribution (the valley
between the decay-wobble cluster and the real-onset cluster), and the prominence
scales with the curve's noise floor. The non-derivable params (min-distance,
decay-reset) fall back to the seed.

Two roles (design spec §baseline): the baseline the learned regressor must beat,
and input features for that regressor (a good deterministic guess is a strong
feature). Pure numpy/scipy, no torch.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.signal import find_peaks

PARAM_NAMES: tuple[str, ...] = (
    "threshold", "prominence", "min_distance_s", "decay_reset_frac", "decay_reset_floor",
)

#: prominence = `_PROM_K` * noise-floor std (rejects decay wobble that rides just
#: above the floor), clamped into a sane band.
_PROM_K = 4.0
_PROM_MIN, _PROM_MAX = 0.02, 0.40


def candidate_peak_heights(
    activation: np.ndarray, fps: float, min_distance_s: float, probe_threshold: float = 0.01
) -> np.ndarray:
    """Heights of all local maxima (>= `probe_threshold`, min-distance apart) -
    the empirical peak-height distribution the knee is fit to."""
    if activation.size == 0:
        return np.empty(0, dtype=np.float64)
    distance = max(1, round(min_distance_s * fps))
    peaks, _ = find_peaks(activation, height=probe_threshold, distance=distance)
    return activation[peaks].astype(np.float64)


def otsu_threshold(values: np.ndarray, bins: int = 64) -> float:
    """Otsu's method: the value that maximizes between-class variance of a
    (assumed bimodal) 1-D set - the valley between a low cluster and a high one."""
    v = np.asarray(values, dtype=np.float64)
    if v.size < 2 or float(v.max() - v.min()) < 1e-9:
        return float(v.mean()) if v.size else 0.0
    hist, edges = np.histogram(v, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = hist.astype(np.float64)
    total = w.sum()
    w0 = np.cumsum(w)
    w1 = total - w0
    sum_all = np.cumsum(w * centers)
    grand = sum_all[-1]
    # guard empty classes
    valid = (w0 > 0) & (w1 > 0)
    mu0 = np.where(w0 > 0, sum_all / np.where(w0 > 0, w0, 1), 0.0)
    mu1 = np.where(w1 > 0, (grand - sum_all) / np.where(w1 > 0, w1, 1), 0.0)
    between = w0 * w1 * (mu0 - mu1) ** 2
    between = np.where(valid, between, -1.0)
    return float(centers[int(np.argmax(between))])


def noise_std(activation: np.ndarray, pct: float = 50.0) -> float:
    """Std of the curve's lower `pct`% of samples - an estimate of the
    between-onset noise floor's spread."""
    if activation.size == 0:
        return 0.0
    cutoff = float(np.percentile(activation, pct))
    floor = activation[activation <= cutoff]
    return float(floor.std()) if floor.size else 0.0


def knee_threshold(
    activation: np.ndarray,
    fps: float,
    min_distance_s: float,
    *,
    floor: float = 0.05,
    ceil: float = 0.9,
    min_candidates: int = 2,
) -> float:
    """Deterministic per-song threshold: the Otsu knee of the candidate-peak
    heights, clamped to [`floor`, `ceil`]. Falls back to `floor` when there are
    too few candidate peaks to fit a meaningful split."""
    heights = candidate_peak_heights(activation, fps, min_distance_s)
    if heights.size < min_candidates:
        return floor
    return float(np.clip(otsu_threshold(heights), floor, ceil))


def deterministic_params(
    activation: np.ndarray, fps: float, seed: Mapping[str, float]
) -> dict[str, float]:
    """Full param dict for this curve: threshold from the knee, prominence from
    the noise floor, and the remaining (min-distance / decay-reset) params from
    `seed`."""
    md = float(seed["min_distance_s"])
    prom = float(np.clip(_PROM_K * noise_std(activation), _PROM_MIN, _PROM_MAX))
    return {
        "threshold": knee_threshold(activation, fps, md),
        "prominence": prom,
        "min_distance_s": md,
        "decay_reset_frac": float(seed["decay_reset_frac"]),
        "decay_reset_floor": float(seed["decay_reset_floor"]),
    }
