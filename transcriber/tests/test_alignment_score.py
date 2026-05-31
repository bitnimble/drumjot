"""Unit tests for the pure scoring DP (`app.scoring.alignment`).

Numbers in, numbers out: per-lane onset-second lists -> soft P/R/F1.
No audio, no I/O. See research/midi-audio-alignment-score.md §5.
"""
from __future__ import annotations

import math

import pytest

from app.scoring.alignment import (
    DEFAULT_BAND_S,
    DEFAULT_SIGMA_S,
    match_pairs,
    match_quality,
    score,
    score_lane,
)


def test_match_pairs_perfect_pairs_all_indices() -> None:
    assert match_pairs([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]) == [(0, 0), (1, 1), (2, 2)]


def test_match_pairs_injective_uses_exact_onset() -> None:
    # Only one audio onset; the exact chart onset wins, the other is unmatched.
    assert match_pairs([0.0, 0.01], [0.0]) == [(0, 0)]


def test_match_pairs_out_of_band_is_empty() -> None:
    assert match_pairs([0.5], [0.0]) == []


def test_match_pairs_preserves_order_and_skips_gaps() -> None:
    # Chart onset at 5.0 has no audio neighbour; it's skipped, order kept.
    assert match_pairs([0.0, 1.0, 5.0], [0.0, 1.0]) == [(0, 0), (1, 1)]


def test_match_quality_perfect_match() -> None:
    # Identical onsets: every pair rewards exp(0) = 1, so TPQ = count.
    assert match_quality([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]) == pytest.approx(3.0)


def test_match_quality_out_of_band_is_zero() -> None:
    # 0.5 s apart, well past the 50 ms band -> no admissible match.
    assert match_quality([0.5], [0.0]) == pytest.approx(0.0)


def test_match_quality_injective_one_audio_onset_used_once() -> None:
    # Two chart onsets both near one audio onset: monotonic-injective
    # matching can use the single audio onset only once. The exact pair
    # wins, TPQ = 1.0 (not 1.0 + 0.92).
    assert match_quality([0.0, 0.01], [0.0]) == pytest.approx(1.0)


def test_score_lane_perfect_is_one() -> None:
    lane = score_lane([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    assert lane.soft_precision == pytest.approx(1.0)
    assert lane.soft_recall == pytest.approx(1.0)
    assert lane.soft_f1 == pytest.approx(1.0)
    assert lane.n_chart == 3
    assert lane.n_audio == 3


def test_score_lane_uniform_20ms_shift_high_but_below_one() -> None:
    chart = [0.020, 1.020, 2.020]
    audio = [0.000, 1.000, 2.000]
    expected = math.exp(-(0.020**2) / (2 * DEFAULT_SIGMA_S**2))  # ~0.726
    lane = score_lane(chart, audio)
    assert lane.soft_f1 == pytest.approx(expected, abs=1e-6)
    assert 0.5 < lane.soft_f1 < 0.99


def test_score_lane_extra_chart_notes_drop_precision() -> None:
    # Chart has a third onset audio doesn't -> precision < recall.
    lane = score_lane([0.0, 1.0, 5.0], [0.0, 1.0])
    assert lane.soft_precision == pytest.approx(2 / 3)
    assert lane.soft_recall == pytest.approx(1.0)


def test_score_lane_missing_chart_notes_drop_recall() -> None:
    # Chart misses one audio onset -> recall < precision.
    lane = score_lane([0.0, 1.0], [0.0, 1.0, 2.0])
    assert lane.soft_precision == pytest.approx(1.0)
    assert lane.soft_recall == pytest.approx(2 / 3)


def test_score_lane_one_sided_scores_zero() -> None:
    assert score_lane([0.0, 1.0], []).soft_f1 == pytest.approx(0.0)
    assert score_lane([], [0.0, 1.0]).soft_f1 == pytest.approx(0.0)


def test_score_skips_empty_both_lanes_and_weights_by_audio() -> None:
    chart = {"k": [0.0, 1.0, 2.0, 3.0], "s": [10.0]}
    audio = {"k": [0.0, 1.0, 2.0, 3.0], "s": [20.0]}
    # k is perfect (f1=1, n_audio=4); s is out of band (f1=0, n_audio=1);
    # t/h/cy are empty on both sides and skipped.
    result = score(chart, audio)
    assert set(result.per_lane) == {"k", "s"}
    assert result.f1_macro == pytest.approx(0.5)  # mean(1, 0)
    assert result.f1_weighted == pytest.approx(0.8)  # (1*4 + 0*1) / 5


def test_score_all_empty_is_zero_not_nan() -> None:
    result = score({}, {})
    assert result.f1_macro == 0.0
    assert result.f1_weighted == 0.0
    assert result.per_lane == {}


def test_band_and_sigma_are_separate_knobs() -> None:
    # A 40 ms shift is inside the default 50 ms band but a tighter band
    # rejects it entirely.
    assert score_lane([0.040], [0.000]).soft_f1 > 0.0
    assert score_lane([0.040], [0.000], band=0.030).soft_f1 == pytest.approx(0.0)
    # Defaults are the documented 50 ms / 25 ms.
    assert pytest.approx(0.050) == DEFAULT_BAND_S
    assert pytest.approx(0.025) == DEFAULT_SIGMA_S
