"""Onset peak-picking and onset-F1 scoring.

`pick_onsets` turns a per-frame activation curve into onset times; `onset_f1`
wraps `mir_eval.onset.f_measure` at the field-standard +/-50 ms tolerance.
Used both to score the model (design spec §4 metric) and to sanity-check
target round-trips.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from drumjot_dsp import peakpick

# Per-lane peak-pick params for the 11 training lanes (domain config; the shared
# `drumjot_dsp.peakpick` is the algorithm only). min-distance matches the
# transcriber (clean 20ms / hat 50ms / cymbal 70ms) and caps to humanly-playable
# rates; hats + cymbals add prominence + decay-reset to kill sustained-ring
# phantom streams.
_CLEAN = {"min_distance_s": 0.020, "prominence": 0.10, "decay_reset_frac": 0.0, "decay_reset_floor": 0.0}
_HAT = {"min_distance_s": 0.050, "prominence": 0.10, "decay_reset_frac": 0.6, "decay_reset_floor": 0.05}
_CYM = {"min_distance_s": 0.070, "prominence": 0.20, "decay_reset_frac": 0.6, "decay_reset_floor": 0.05}
DEFAULT_PEAK_PARAMS = dict(_CLEAN)
LANE_PEAK_PARAMS: dict[str, dict[str, float]] = {
    "k": dict(_CLEAN), "s": dict(_CLEAN), "ss": dict(_CLEAN), "t": dict(_CLEAN),
    "hc": dict(_HAT), "ho": dict(_HAT),
    "rd": dict(_CYM), "cr": dict(_CYM),
    # legacy: lets old 10/11-lane checkpoints (mc/mp lanes) still peak-pick
    "mc": dict(_CYM), "mp": dict(_CLEAN),
}


def pick_onsets(
    activation: np.ndarray,
    fps: float,
    threshold: float,
    min_distance_s: float,
    *,
    prominence: float | None = None,
    decay_reset_frac: float = 0.0,
    decay_reset_floor: float = 0.0,
) -> np.ndarray:
    """Peak-pick `activation` into onset times (seconds, ascending).

    Thin wrapper over the shared `peakpick.pick_peaks` (height + min-distance +
    optional prominence + decay-reset); see `peakpick` for the algorithm. Use
    `pick_onsets_lane` to apply the per-lane parameter table.
    """
    frames = peakpick.pick_peaks(
        activation, fps, threshold=threshold, min_distance_s=min_distance_s,
        prominence=prominence, decay_reset_frac=decay_reset_frac, decay_reset_floor=decay_reset_floor,
    )
    return frames.astype(np.float64) / fps


def pick_onsets_lane(activation: np.ndarray, fps: float, lane: str, threshold: float) -> np.ndarray:
    """Per-lane peak-pick using the shared `peakpick.LANE_PEAK_PARAMS` (the
    transcriber-grade clean/hat/cymbal post-processing: per-lane min-distance,
    prominence, and decay-reset). `threshold` is the lane's tuned peak height."""
    p = LANE_PEAK_PARAMS.get(lane, DEFAULT_PEAK_PARAMS)
    return pick_onsets(activation, fps, threshold, **p)


def onset_f1(
    ref_times: Sequence[float] | np.ndarray,
    est_times: Sequence[float] | np.ndarray,
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
