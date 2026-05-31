"""Pure global correction: tiers 0-1 of the warp family.

Finds a single global warp `t' = a*t + b` that best aligns the chart to the
audio, so the score reflects notation faithfulness rather than a fixable
global drift, and reports `(a, b)` so a batch run can clean training pairs.
No per-note ICP nudge and no MIDI export (research §8.2 is out of scope for
v1). Numbers in, numbers out.

  * Tier 0 (offset): cross-correlate the per-lane onset impulse trains and
    take the lag of maximum coincidence. Threshold-free; pulls the chart
    inside the match band so the soft score becomes a usable objective.
  * Tier 1 (affine tempo): with the offset-aligned DP correspondence, a
    robust (Huber) least-squares fit of `audio ~= a*chart + b`. Bounded
    `a in [0.5, 2.0]` and gated on >= 3 matched pairs; otherwise no tempo
    correction (keep the tier-0 offset). A large `|a - 1|` is a red-flag
    diagnostic, not a free pass.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from app.scoring.alignment import DEFAULT_BAND_S, DEFAULT_SIGMA_S, match_pairs

# Offset search half-width (~2 bars at 120 BPM 4/4). The cross-correlation
# only considers same-lane pairs whose time difference is within this bound.
DEFAULT_OFFSET_BOUND_S = 4.0
_OFFSET_BIN_S = 0.010
_A_BOUNDS = (0.5, 2.0)
_MIN_PAIRS = 3
# Huber transition scale (residuals beyond this are down-weighted linearly).
_HUBER_SCALE_S = 0.050


@dataclass(frozen=True)
class GlobalCorrection:
    offset_sec: float  # b in t' = a*t + b
    tempo_ratio: float  # a (1.0 = no tempo correction)
    matched_pairs: int  # pairs the affine fit used (trust signal)
    corrected_by_lane: dict[str, list[float]]  # chart times after the warp


def estimate_offset(
    chart_by_lane: Mapping[str, Sequence[float]],
    audio_by_lane: Mapping[str, Sequence[float]],
    *,
    bound_sec: float = DEFAULT_OFFSET_BOUND_S,
    bin_sec: float = _OFFSET_BIN_S,
) -> float:
    """Tier 0: the global offset `b` (seconds to ADD to chart times) that
    maximises onset coincidence with the audio. Histogram of same-lane
    `audio - chart` differences within +/-`bound_sec`, binned to `bin_sec`;
    the busiest bin's mean difference is the offset. 0.0 when no pair falls
    within the bound."""
    diffs: list[float] = []
    for lane, chart in chart_by_lane.items():
        audio = audio_by_lane.get(lane)
        if not audio or not chart:
            continue
        for c in chart:
            for a in audio:
                d = a - c
                if -bound_sec <= d <= bound_sec:
                    diffs.append(d)
    if not diffs:
        return 0.0

    # Bucket by bin index; pick the busiest bin, ties broken toward zero lag.
    buckets: dict[int, list[float]] = {}
    for d in diffs:
        buckets.setdefault(round(d / bin_sec), []).append(d)
    best_bin = max(buckets, key=lambda k: (len(buckets[k]), -abs(k)))
    winning = buckets[best_bin]
    return sum(winning) / len(winning)


def fit_affine(
    pairs: Sequence[tuple[float, float]],
    fallback_offset: float,
    *,
    a_bounds: tuple[float, float] = _A_BOUNDS,
    min_pairs: int = _MIN_PAIRS,
) -> tuple[float, float, int]:
    """Tier 1: robust fit of `audio ~= a*chart + b` over `(chart, audio)`
    matched pairs. Returns `(a, b, n_pairs)`. Falls back to `(1.0,
    fallback_offset, n)` when there are fewer than `min_pairs` pairs or the
    fitted slope lands outside `a_bounds` (an under-determined or runaway fit
    must not be allowed to inflate the corrected score)."""
    n = len(pairs)
    if n < min_pairs:
        return (1.0, fallback_offset, n)

    t = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=float)

    def residuals(params: np.ndarray) -> np.ndarray:
        a, b = params
        return a * t + b - y

    result = least_squares(
        residuals, x0=[1.0, fallback_offset], loss="huber", f_scale=_HUBER_SCALE_S
    )
    a, b = float(result.x[0]), float(result.x[1])
    lo, hi = a_bounds
    if not (lo <= a <= hi):
        return (1.0, fallback_offset, n)
    return (a, b, n)


def global_align(
    chart_by_lane: Mapping[str, Sequence[float]],
    audio_by_lane: Mapping[str, Sequence[float]],
    *,
    band: float = DEFAULT_BAND_S,
    sigma: float = DEFAULT_SIGMA_S,
    offset_bound_sec: float = DEFAULT_OFFSET_BOUND_S,
    a_bounds: tuple[float, float] = _A_BOUNDS,
    min_pairs: int = _MIN_PAIRS,
) -> GlobalCorrection:
    """Run tiers 0-1 and return the recovered warp plus the corrected chart
    times. The offset is applied first so the DP correspondence is solved
    inside the band; the affine fit then refines `(a, b)` on those pairs."""
    b0 = estimate_offset(chart_by_lane, audio_by_lane, bound_sec=offset_bound_sec)

    pairs: list[tuple[float, float]] = []
    for lane, chart in chart_by_lane.items():
        audio = audio_by_lane.get(lane)
        if not audio or not chart:
            continue
        shifted = [c + b0 for c in chart]
        for ci, aj in match_pairs(shifted, list(audio), band=band, sigma=sigma):
            pairs.append((chart[ci], audio[aj]))

    a, b, n = fit_affine(pairs, fallback_offset=b0, a_bounds=a_bounds, min_pairs=min_pairs)

    corrected_by_lane = {
        lane: [a * c + b for c in chart] for lane, chart in chart_by_lane.items()
    }
    return GlobalCorrection(
        offset_sec=b, tempo_ratio=a, matched_pairs=n, corrected_by_lane=corrected_by_lane
    )
