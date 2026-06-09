"""Shared deterministic peak-picker for per-frame onset-activation curves.

The single implementation behind both the transcriber's ADTOF post-processor
(`app/pipeline/adtof_onsets.py`) and the learned model's inference/eval
(`drumjot_training.metrics`), so the two can't drift. Pure numpy/scipy.

Pipeline: `find_peaks` (height + min-distance + prominence) then an optional
decay-reset pass that collapses one sustained ring (open-hat / cymbal), whose
wobble clears height+prominence, into a single onset.

Domain-specific parameter tables (per-lane min-distance etc.) live with each
consumer (`drumjot_training.metrics.LANE_PEAK_PARAMS`, the transcriber's
`settings`), not here; this module is the algorithm only.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def resolve_threshold(
    activation: np.ndarray,
    *,
    fixed: float,
    adaptive: bool = False,
    k: float = 0.5,
    pct: float = 95.0,
    floor: float = 0.0,
) -> float:
    """Peak-pick height: a `fixed` value, or an adaptive ``max(floor, k * pXX)``
    that self-calibrates to this curve's own confidence range (for noisy lanes
    whose absolute activation scale drifts out-of-distribution)."""
    if not adaptive or activation.size == 0:
        return fixed
    return max(floor, k * float(np.percentile(activation, pct)))


def decay_reset_filter(
    activation: np.ndarray, peaks: np.ndarray, reset_frac: float, reset_floor: float
) -> np.ndarray:
    """Drop peaks that re-trigger before the previous accepted peak's energy
    decayed, one sustained open-hat/cymbal ring read as a stream of hits.

    A candidate survives only if the activation somewhere between it and the
    previous *accepted* peak fell below ``max(reset_floor, reset_frac *
    prev_height)`` (the prior ring actually came back down). Continuous sustain
    never dips, so it collapses to a single onset; genuinely separate hits (the
    activation plunges between them) all survive. `peaks` must be ascending."""
    if peaks.size == 0:
        return peaks
    kept = [int(peaks[0])]
    for raw in peaks[1:]:
        cand = int(raw)
        prev = kept[-1]
        between = activation[prev + 1 : cand]
        if between.size == 0:
            continue  # adjacent (shouldn't occur post min-distance): same event
        reset_level = max(reset_floor, reset_frac * float(activation[prev]))
        if float(between.min()) < reset_level:
            kept.append(cand)
        # else: the ring never decayed -> same sustained event, drop.
    return np.asarray(kept, dtype=int)


def pick_peaks(
    activation: np.ndarray,
    fps: float,
    *,
    threshold: float,
    min_distance_s: float,
    prominence: float | None = None,
    decay_reset_frac: float = 0.0,
    decay_reset_floor: float = 0.0,
) -> np.ndarray:
    """Peak-pick a per-frame curve -> ascending frame indices.

    height >= `threshold`, peaks >= `min_distance_s` apart, optional
    `prominence` (rise above local baseline; rejects decay-tail wobble), and an
    optional decay-reset pass when `decay_reset_frac` > 0.
    """
    if activation.size == 0:
        return np.empty(0, dtype=int)
    distance = max(1, round(min_distance_s * fps))
    prom = prominence if (prominence and prominence > 0.0) else None
    peaks, _ = find_peaks(activation, height=threshold, distance=distance, prominence=prom)
    if decay_reset_frac > 0.0 and peaks.size:
        peaks = decay_reset_filter(activation, peaks, decay_reset_frac, decay_reset_floor)
    return peaks.astype(int)
