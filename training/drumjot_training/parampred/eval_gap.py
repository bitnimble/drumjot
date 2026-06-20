"""Per-lane oracle-gap records from model probability curves.

Bridges the model output (per-lane activation curves + ground-truth onsets) to
the report layer: for each lane it scores onset-F1 at the current global params
(the seed) and at the per-song oracle params, and -- if a trained predictor is
supplied -- at the predicted params too. Returns `report.GapRecord`s.

Operates at the raw per-lane granularity (no hat/cymbal folding): the params
live per lane, so this is where "which lane / which param carries the prize" is
legible. The folded ParaDB headline stays in eval_paradb. Restrict to a stem's
own lanes (`restrict_lanes`) to mirror how predictions are scored there.
"""
from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

import numpy as np

from drumjot_training import metrics
from drumjot_training.parampred import features, oracle, report


def lane_gap_records(
    probs: np.ndarray,
    fps: float,
    lanes: Sequence[str],
    thresholds: Mapping[str, float],
    gt: Mapping[str, Sequence[float]],
    *,
    default_threshold: float = 0.5,
    tolerance: float = 0.05,
    restrict_lanes: Collection[str] | None = None,
    predictor=None,
    waveform: np.ndarray | None = None,
    sr: int | None = None,
    beat_period_s: float | None = None,
) -> list[report.GapRecord]:
    """One `GapRecord` per lane that has ground truth (and, if given, is in
    `restrict_lanes`). `probs` is `(n_lanes, T)` aligned with `lanes`."""
    records: list[report.GapRecord] = []
    for i, lane in enumerate(lanes):
        if restrict_lanes is not None and lane not in restrict_lanes:
            continue
        ref = gt.get(lane)
        if not ref:
            continue
        seed = {
            "threshold": float(thresholds.get(lane, default_threshold)),
            **metrics.LANE_PEAK_PARAMS.get(lane, metrics.DEFAULT_PEAK_PARAMS),
        }
        ores = oracle.oracle_lane_params(
            probs[i], fps, ref, seed=seed, grids=oracle.default_grids(seed), tolerance=tolerance
        )
        predicted_f1 = _predicted_f1(
            probs[i], fps, lane, ref, seed, predictor, waveform, sr, beat_period_s, tolerance
        )
        if predicted_f1 is None:
            predicted_f1 = ores.baseline_f1
        records.append(report.GapRecord(
            lane=lane,
            current_f1=ores.baseline_f1,
            predicted_f1=predicted_f1,
            oracle_f1=ores.f1,
        ))
    return records


def _predicted_f1(
    activation, fps, lane, ref, seed, predictor, waveform, sr, beat_period_s, tolerance
) -> float | None:
    """F1 at the predictor's params, or None when no usable prediction exists
    (no predictor, no waveform, or the predictor wasn't trained for this lane)."""
    if predictor is None or waveform is None or sr is None:
        return None
    try:
        x = features.feature_vector(
            activation, fps, seed["min_distance_s"], waveform, sr, beat_period_s=beat_period_s
        )
        params = predictor.predict_row(lane, x)
    except KeyError:
        return None  # lane never fit -> fall back to current
    est = metrics.pick_onsets(
        activation, fps,
        threshold=params["threshold"],
        min_distance_s=params["min_distance_s"],
        prominence=params["prominence"],
        decay_reset_frac=params["decay_reset_frac"],
        decay_reset_floor=params["decay_reset_floor"],
    )
    return metrics.onset_f1(ref, est, tolerance)["f"]
