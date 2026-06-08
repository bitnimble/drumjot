"""Onset peak-picking and onset-F1 scoring.

`pick_onsets` turns a per-frame activation curve into onset times; `onset_f1`
wraps `mir_eval.onset.f_measure` at the field-standard +/-50 ms tolerance.
Used both to score the model (design spec §4 metric) and to sanity-check
target round-trips.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.signal import find_peaks


def pick_onsets(
    activation: np.ndarray,
    fps: float,
    threshold: float,
    min_distance_s: float,
) -> np.ndarray:
    """Peak-pick `activation` into onset times (seconds, ascending).

    Height `threshold` and a `min_distance_s` minimum spacing mirror the
    transcriber's deterministic peak-pick (`adtof_onsets.py`); prominence
    is left to callers that need it.
    """
    if activation.size == 0:
        return np.empty(0, dtype=np.float64)
    distance = max(1, round(min_distance_s * fps))
    peaks, _ = find_peaks(activation, height=threshold, distance=distance)
    return peaks.astype(np.float64) / fps


def onset_f1(
    ref_times: Sequence[float],
    est_times: Sequence[float],
    tolerance: float = 0.05,
) -> dict[str, float]:
    """Onset F-measure / precision / recall at +/-`tolerance` seconds.

    Thin wrapper over `mir_eval.onset.f_measure`; inputs are sorted first
    (mir_eval requires ascending arrays). Returns {"f", "p", "r"}.
    """
    from mir_eval.onset import f_measure

    ref = np.sort(np.asarray(ref_times, dtype=np.float64))
    est = np.sort(np.asarray(est_times, dtype=np.float64))
    f, p, r = f_measure(ref, est, window=tolerance)
    return {"f": float(f), "p": float(p), "r": float(r)}
