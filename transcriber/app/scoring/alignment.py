"""Pure soft-cost onset-alignment scorer.

Per lane, a monotonic-injective DP (bounded Needleman-Wunsch with a soft
Gaussian substitution reward and free gaps on both sides) matches the
chart's onsets to the audio's onsets, then per-lane soft precision /
recall / F1 roll up to a weighted score. Numbers in, numbers out: no
audio, no I/O; tested against synthetic onset lists. See
research/midi-audio-alignment-score.md §5.

The band `B` decides *who pairs with whom* (correspondence gate); the
kernel width `sigma` decides *how much credit* a pair gets. They are
separate knobs on purpose: the reward keeps charging for every
millisecond of error out to the band edge (no flat zone), so the score is
a smooth objective the correction stage can climb.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.scoring.lanes import LANES

# Correspondence gate: pairs further apart than this never match.
DEFAULT_BAND_S = 0.050
# Credit kernel width. B = 2*sigma -> edge reward ~= e^-2 ~= 0.14.
DEFAULT_SIGMA_S = 0.025


@dataclass(frozen=True)
class LaneScore:
    soft_f1: float
    soft_precision: float
    soft_recall: float
    n_chart: int
    n_audio: int


@dataclass(frozen=True)
class ScoreResult:
    f1_macro: float
    f1_weighted: float
    per_lane: dict[str, LaneScore]


def match_quality(
    chart: Sequence[float],
    audio: Sequence[float],
    *,
    band: float = DEFAULT_BAND_S,
    sigma: float = DEFAULT_SIGMA_S,
) -> float:
    """Total matched quality (TPQ): the max sum of soft rewards over an
    order-preserving, injective partial matching of `chart` onsets to
    `audio` onsets. Both sequences must be ascending. `reward(m, a) =
    exp(-(m-a)^2 / 2 sigma^2)` when `|m-a| <= band`, else the pair can't
    match."""
    n, m = len(chart), len(audio)
    if n == 0 or m == 0:
        return 0.0

    two_sigma_sq = 2.0 * sigma * sigma
    # Rolling rows of the DP table over audio index j (1-indexed; col 0 = 0).
    prev = [0.0] * (m + 1)
    cur = [0.0] * (m + 1)
    for i in range(1, n + 1):
        ci = chart[i - 1]
        cur[0] = 0.0
        for j in range(1, m + 1):
            # Skip chart i (insertion) or skip audio j (deletion).
            best = prev[j] if prev[j] >= cur[j - 1] else cur[j - 1]
            d = ci - audio[j - 1]
            if -band <= d <= band:
                diag = prev[j - 1] + math.exp(-(d * d) / two_sigma_sq)
                if diag > best:
                    best = diag
            cur[j] = best
        prev, cur = cur, prev
    return prev[m]


def match_pairs(
    chart: Sequence[float],
    audio: Sequence[float],
    *,
    band: float = DEFAULT_BAND_S,
    sigma: float = DEFAULT_SIGMA_S,
) -> list[tuple[int, int]]:
    """The order-preserving, injective correspondence that maximises total
    reward: a list of `(chart_index, audio_index)` pairs, ascending. Ties
    favour matching (more pairs) over a gap. Used by the correction stage to
    fit a warp on matched pairs; the scoring roll-up only needs the total
    quality (`match_quality`), not the pairs."""
    n, m = len(chart), len(audio)
    if n == 0 or m == 0:
        return []

    two_sigma_sq = 2.0 * sigma * sigma
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    # choice[i][j]: 1 = skip chart i, 2 = skip audio j, 3 = match i<->j.
    choice = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ci = chart[i - 1]
        for j in range(1, m + 1):
            best, ch = dp[i - 1][j], 1  # skip chart i
            if dp[i][j - 1] > best:
                best, ch = dp[i][j - 1], 2  # skip audio j
            d = ci - audio[j - 1]
            if -band <= d <= band:
                diag = dp[i - 1][j - 1] + math.exp(-(d * d) / two_sigma_sq)
                if diag > best:
                    best, ch = diag, 3  # match
            dp[i][j], choice[i][j] = best, ch

    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        ch = choice[i][j]
        if ch == 3:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif ch == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def score_lane(
    chart: Sequence[float],
    audio: Sequence[float],
    *,
    band: float = DEFAULT_BAND_S,
    sigma: float = DEFAULT_SIGMA_S,
) -> LaneScore:
    """Soft precision / recall / F1 for one lane. `soft_f1 = 0` (not NaN)
    whenever `precision + recall == 0`, i.e. a one-sided lane or no pair
    inside the band."""
    n_chart, n_audio = len(chart), len(audio)
    tpq = match_quality(chart, audio, band=band, sigma=sigma)
    precision = tpq / n_chart if n_chart else 0.0
    recall = tpq / n_audio if n_audio else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return LaneScore(
        soft_f1=f1,
        soft_precision=precision,
        soft_recall=recall,
        n_chart=n_chart,
        n_audio=n_audio,
    )


def score(
    chart_by_lane: Mapping[str, Sequence[float]],
    audio_by_lane: Mapping[str, Sequence[float]],
    *,
    band: float = DEFAULT_BAND_S,
    sigma: float = DEFAULT_SIGMA_S,
    lanes: Sequence[str] = LANES,
) -> ScoreResult:
    """Score every lane and roll up. A lane empty on both sides is skipped
    (absent from `per_lane` and from both roll-ups). `f1_macro` is the mean
    of scored lanes' F1; `f1_weighted` weights each lane by `max(n_chart,
    n_audio)` (busy lanes count more), so a lane the chart over-notates but the
    reference has 0 onsets in still carries weight (its low precision drags the
    headline); falls back to 0 when every scored lane is empty on both sides."""
    per_lane: dict[str, LaneScore] = {}
    for lane in lanes:
        chart = chart_by_lane.get(lane, ())
        audio = audio_by_lane.get(lane, ())
        if not chart and not audio:
            continue
        per_lane[lane] = score_lane(chart, audio, band=band, sigma=sigma)

    if not per_lane:
        return ScoreResult(f1_macro=0.0, f1_weighted=0.0, per_lane={})

    f1_macro = sum(ls.soft_f1 for ls in per_lane.values()) / len(per_lane)
    total_w = sum(max(ls.n_chart, ls.n_audio) for ls in per_lane.values())
    f1_weighted = (
        sum(ls.soft_f1 * max(ls.n_chart, ls.n_audio) for ls in per_lane.values()) / total_w
        if total_w
        else 0.0
    )
    return ScoreResult(f1_macro=f1_macro, f1_weighted=f1_weighted, per_lane=per_lane)
