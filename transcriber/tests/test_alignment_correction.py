"""Unit tests for the pure global-correction stage (`app.scoring.correction`).

Tiers 0-1 of research/midi-audio-alignment-score.md §8: cross-correlation
offset, then bounded affine-tempo fit on matched pairs. Numbers in, numbers
out.
"""
from __future__ import annotations

import pytest

from app.scoring.alignment import score
from app.scoring.correction import estimate_offset, fit_affine, global_align


def test_estimate_offset_recovers_injected_offset() -> None:
    audio = {"k": [1.0, 2.0, 3.0, 4.0]}
    chart = {"k": [0.97, 1.97, 2.97, 3.97]}  # 30 ms early
    assert estimate_offset(chart, audio) == pytest.approx(0.030, abs=2e-3)


def test_estimate_offset_zero_when_nothing_in_window() -> None:
    audio = {"k": [0.0]}
    chart = {"k": [100.0]}
    assert estimate_offset(chart, audio, bound_sec=1.0) == 0.0


def test_fit_affine_recovers_identity_with_offset() -> None:
    pairs = [(0.0, 0.02), (1.0, 1.02), (2.0, 2.02), (3.0, 3.02)]
    a, b, n = fit_affine(pairs, fallback_offset=0.02)
    assert a == pytest.approx(1.0, abs=1e-3)
    assert b == pytest.approx(0.02, abs=1e-3)
    assert n == 4


def test_fit_affine_recovers_tempo() -> None:
    pairs = [(t, 1.02 * t) for t in (1.0, 2.0, 3.0, 4.0, 5.0)]
    a, b, n = fit_affine(pairs, fallback_offset=0.0)
    assert a == pytest.approx(1.02, abs=1e-3)
    assert b == pytest.approx(0.0, abs=2e-3)


def test_fit_affine_too_few_pairs_falls_back() -> None:
    a, b, n = fit_affine([(0.0, 0.03), (1.0, 1.03)], fallback_offset=0.03)
    assert a == 1.0
    assert b == pytest.approx(0.03)
    assert n == 2


def test_fit_affine_out_of_bounds_slope_falls_back() -> None:
    # Exact slope 3.0 is outside [0.5, 2.0] -> reject, keep the offset only.
    a, b, n = fit_affine([(0.0, 0.0), (1.0, 3.0), (2.0, 6.0)], fallback_offset=0.0)
    assert a == 1.0
    assert b == pytest.approx(0.0)
    assert n == 3


def test_global_align_offset_only() -> None:
    audio = {"k": [1.0, 2.0, 3.0, 4.0], "s": [1.5, 2.5, 3.5]}
    chart = {lane: [t - 0.030 for t in times] for lane, times in audio.items()}
    corr = global_align(chart, audio)
    assert corr.tempo_ratio == pytest.approx(1.0, abs=2e-3)
    assert corr.offset_sec == pytest.approx(0.030, abs=3e-3)
    for lane, times in audio.items():
        for got, want in zip(corr.corrected_by_lane[lane], times, strict=True):
            assert got == pytest.approx(want, abs=3e-3)
    assert score(corr.corrected_by_lane, audio).f1_weighted == pytest.approx(1.0, abs=0.05)


def test_global_align_recovers_small_tempo_error() -> None:
    a_true = 1.02
    audio = {"k": [1.0, 2.0, 3.0, 4.0, 5.0]}
    chart = {"k": [t / a_true for t in audio["k"]]}  # audio = a_true * chart
    corr = global_align(chart, audio)
    assert corr.tempo_ratio == pytest.approx(a_true, abs=5e-3)
    assert score(corr.corrected_by_lane, audio).f1_weighted > 0.95


def test_global_align_corrected_beats_raw_on_offset_chart() -> None:
    audio = {"k": [1.0, 2.0, 3.0, 4.0]}
    chart = {"k": [t - 0.045 for t in audio["k"]]}  # 45 ms early: low raw score
    corr = global_align(chart, audio)
    raw = score(chart, audio).f1_weighted
    corrected = score(corr.corrected_by_lane, audio).f1_weighted
    assert corrected > raw
