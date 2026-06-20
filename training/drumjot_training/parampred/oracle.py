"""Per-song oracle peak-picking parameters.

For one song's per-lane activation curve and its ground-truth onsets, find the
peakpick param vector that maximizes onset-F1 *on this song* (cheating: it uses
the labels). This is the per-song ceiling the adaptive predictor chases, and the
supervised regression target the predictor is trained on (design spec §oracle).

Search is coordinate ascent over per-param grids, seeded from the current global
params. Two deliberate choices keep the labels well-conditioned:

- Monotonic: each param sweep includes the current value, so F1 never decreases;
  the returned F1 is >= the seed (`baseline`) F1 by construction.
- Flat-region tie-break: when several grid values tie for best F1 (common for
  `min_distance_s` / `decay_reset_*`, whose F1 surface is often flat), return the
  one closest to the seed rather than an arbitrary maximizer.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from drumjot_training import metrics

#: Sweep grids. Threshold is fine-grained (it carries most of the per-song
#: variance); the rest are coarse. Callers pass the subset of params to sweep via
#: `grids`; any param absent from `grids` is held fixed at its seed value.
THRESHOLD_GRID: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 19))  # 0.05..0.90
PROMINENCE_GRID: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
MIN_DISTANCE_GRID: tuple[float, ...] = (0.010, 0.020, 0.030, 0.050, 0.070, 0.100)
DECAY_FRAC_GRID: tuple[float, ...] = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8)
DECAY_FLOOR_GRID: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10)

PARAM_NAMES: tuple[str, ...] = (
    "threshold", "prominence", "min_distance_s", "decay_reset_frac", "decay_reset_floor",
)

_EPS = 1e-12


def default_grids(seed: Mapping[str, float]) -> dict[str, tuple[float, ...]]:
    """Param->grid for `oracle_lane_params`: always sweep threshold / prominence
    / min-distance; add the decay-reset grids only for sustained lanes (those
    whose seed already enables decay-reset, i.e. hats and cymbals)."""
    grids: dict[str, tuple[float, ...]] = {
        "threshold": THRESHOLD_GRID,
        "prominence": PROMINENCE_GRID,
        "min_distance_s": MIN_DISTANCE_GRID,
    }
    if float(seed.get("decay_reset_frac", 0.0)) > 0.0:
        grids["decay_reset_frac"] = DECAY_FRAC_GRID
        grids["decay_reset_floor"] = DECAY_FLOOR_GRID
    return grids


@dataclass(frozen=True)
class OracleResult:
    """Best per-song params (`params`), their onset-F1 (`f1`), and the F1 at the
    seed params (`baseline_f1`, = today's global-param score on this song)."""

    params: dict[str, float]
    f1: float
    baseline_f1: float


def _f1_for(
    activation: np.ndarray,
    fps: float,
    ref_onsets: Sequence[float],
    params: Mapping[str, float],
    tolerance: float,
) -> float:
    est = metrics.pick_onsets(
        activation, fps,
        threshold=params["threshold"],
        min_distance_s=params["min_distance_s"],
        prominence=params["prominence"],
        decay_reset_frac=params["decay_reset_frac"],
        decay_reset_floor=params["decay_reset_floor"],
    )
    return metrics.onset_f1(ref_onsets, est, tolerance)["f"]


def _sweep_param(
    activation: np.ndarray,
    fps: float,
    ref_onsets: Sequence[float],
    cur: dict[str, float],
    name: str,
    grid: Sequence[float],
    seed_val: float,
    tolerance: float,
) -> tuple[float, float]:
    """Best value of one param holding the others at `cur`. Includes the current
    value so F1 can't drop; ties break toward `seed_val`. Returns (value, f1)."""
    candidates = sorted({*grid, cur[name]})
    scored = [(v, _f1_for(activation, fps, ref_onsets, {**cur, name: v}, tolerance)) for v in candidates]
    best_f1 = max(f for _, f in scored)
    tied = [v for v, f in scored if best_f1 - f <= _EPS]
    chosen = min(tied, key=lambda v: abs(v - seed_val))
    return chosen, best_f1


def oracle_lane_params(
    activation: np.ndarray,
    fps: float,
    ref_onsets: Sequence[float],
    *,
    seed: Mapping[str, float],
    grids: Mapping[str, Sequence[float]],
    tolerance: float = 0.05,
    passes: int = 2,
) -> OracleResult:
    """Coordinate-ascent oracle: the param vector maximizing onset-F1 on this
    lane's `activation` against `ref_onsets`.

    `seed` must supply all five params (the current global values); `grids` maps
    each param to sweep -> its grid (params absent are held at their seed). The
    result's F1 is >= the seed F1 by construction.
    """
    cur = {p: float(seed[p]) for p in PARAM_NAMES}
    baseline_f1 = _f1_for(activation, fps, ref_onsets, cur, tolerance)
    best_f1 = baseline_f1
    for _ in range(passes):
        changed = False
        for name, grid in grids.items():
            value, f1 = _sweep_param(
                activation, fps, ref_onsets, cur, name, grid, float(seed[name]), tolerance
            )
            if value != cur[name]:
                cur[name] = value
                changed = True
            best_f1 = f1
        if not changed:
            break
    return OracleResult(params=cur, f1=best_f1, baseline_f1=baseline_f1)
