"""Tests for the pure scoring path (no transcriber service, no datasets)."""
from __future__ import annotations

import math

from benchmarks.core.classes import DrumClass
from benchmarks.core.events import OnsetEvent
from benchmarks.core.score import score_track, summarise


def _ev(t: float, cls: DrumClass) -> OnsetEvent:
    return OnsetEvent(time=t, drum_class=cls)


def test_perfect_match_scores_one() -> None:
    ref = [_ev(0.1, DrumClass.KD), _ev(0.5, DrumClass.SD), _ev(0.9, DrumClass.HH)]
    est = list(ref)
    score = score_track("perfect", ref, est, tolerance=0.05)
    assert score.f1_macro == 1.0
    assert score.f1_weighted == 1.0
    for cls in (DrumClass.KD, DrumClass.SD, DrumClass.HH):
        assert score.per_class[cls].f1 == 1.0


def test_completely_missing_class_scores_zero_for_that_class() -> None:
    # Reference has snare, prediction omits snare entirely.
    ref = [_ev(0.1, DrumClass.KD), _ev(0.5, DrumClass.SD)]
    est = [_ev(0.1, DrumClass.KD)]
    score = score_track("miss-snare", ref, est, tolerance=0.05)
    assert score.per_class[DrumClass.KD].f1 == 1.0
    assert score.per_class[DrumClass.SD].f1 == 0.0
    # Macro F1 averages KD's 1.0 with SD's 0.0 = 0.5.
    assert score.f1_macro == 0.5
    # Weighted F1 = (1.0 * 1 + 0.0 * 1) / 2 = 0.5.
    assert score.f1_weighted == 0.5


def test_classes_with_no_ref_or_est_are_skipped() -> None:
    # No HH onsets anywhere. HH should not appear in per_class.
    ref = [_ev(0.1, DrumClass.KD)]
    est = [_ev(0.1, DrumClass.KD)]
    score = score_track("no-hh", ref, est, tolerance=0.05)
    assert DrumClass.HH not in score.per_class
    assert score.f1_macro == 1.0


def test_onset_outside_tolerance_misses() -> None:
    # 60 ms off with a 50 ms window = miss.
    ref = [_ev(0.5, DrumClass.KD)]
    est = [_ev(0.56, DrumClass.KD)]
    score = score_track("miss-tolerance", ref, est, tolerance=0.05)
    assert score.per_class[DrumClass.KD].f1 == 0.0


def test_onset_within_tolerance_hits() -> None:
    ref = [_ev(0.5, DrumClass.KD)]
    est = [_ev(0.53, DrumClass.KD)]  # 30 ms off, inside 50 ms window
    score = score_track("hit-tolerance", ref, est, tolerance=0.05)
    assert score.per_class[DrumClass.KD].f1 == 1.0


def test_extra_estimated_onset_drops_precision_not_recall() -> None:
    ref = [_ev(0.1, DrumClass.KD)]
    est = [_ev(0.1, DrumClass.KD), _ev(2.0, DrumClass.KD)]  # one true, one spurious
    score = score_track("extra-est", ref, est, tolerance=0.05)
    kd = score.per_class[DrumClass.KD]
    assert kd.recall == 1.0
    assert math.isclose(kd.precision, 0.5)
    # F1 = 2 * 1.0 * 0.5 / (1.0 + 0.5) = 0.6667.
    assert math.isclose(kd.f1, 2 / 3, rel_tol=1e-9)


def test_summary_averages_across_tracks() -> None:
    t1 = score_track(
        "t1",
        [_ev(0.1, DrumClass.KD)],
        [_ev(0.1, DrumClass.KD)],
        tolerance=0.05,
    )
    t2 = score_track(
        "t2",
        [_ev(0.1, DrumClass.KD)],
        [],  # nothing predicted -> F1 = 0
        tolerance=0.05,
    )
    summary = summarise("synthetic", [t1, t2], tolerance=0.05)
    assert summary.n_tracks == 2
    # (1.0 + 0.0) / 2
    assert summary.f1_macro_mean == 0.5
    assert summary.per_class_f1_mean[DrumClass.KD] == 0.5
    assert summary.per_class_n_reference[DrumClass.KD] == 2


def test_empty_summary_returns_zeros() -> None:
    summary = summarise("empty", [], tolerance=0.05)
    assert summary.n_tracks == 0
    assert summary.f1_macro_mean == 0.0
    assert summary.per_class_f1_mean == {}
