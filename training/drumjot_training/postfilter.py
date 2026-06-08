"""Deterministic envelope post-processing for model onset predictions.

The model is run as a high-recall candidate generator; these audio-level
filters (the same idea as the transcriber's deterministic stages) prune
candidates with no onset-strength support and snap survivors onto the nearest
envelope peak. Used to measure whether post-processing improves onset-F1 over
the raw model output.

The gate/align math is host-testable; the envelope itself comes from
`forced_align.onset_envelope` (lazy librosa). NOTE: here the envelope is the
full mix (val clips are mixes), so alignment is noisier than the transcriber's
per-stem alignment, this is a first-order measurement.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from drumjot_training import forced_align


def support_floor_from_env(env, percentile: float) -> float:
    """Support floor = the `percentile`-th percentile of the onset envelope, so
    the gate adapts per clip to its overall transient level."""
    return float(np.percentile(np.asarray(env, dtype=float), percentile))


def filter_lane(
    est_sec: Sequence[float],
    env: np.ndarray,
    env_fps: float,
    window_s: float,
    support_floor: float,
) -> list[float]:
    """Envelope-power gate + peak alignment for one lane's estimated onsets.

    Drops onsets with no envelope peak >= `support_floor` within +/-`window_s`
    and snaps survivors to that peak. Returns the filtered, sorted times.
    """
    aligned = forced_align.align_lane(est_sec, env, env_fps, window_s, support_floor)
    return sorted(t for t, ok in aligned if ok)
