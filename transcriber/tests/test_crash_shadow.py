"""Differential tests for the O(n) `_crash_shadow_filter` rewrite.

The production filter was rewritten from a per-candidate `candidates[:i]`
rescan (O(n^2)) to a sliding-window monotonic-deque pass (O(n)). These tests
pin that the rewrite is behaviour-preserving: for the same inputs it keeps and
drops exactly the same candidates, in the same order.

`_reference_shadow_filter` is a verbatim copy of the original O(n^2) inner
logic, sharing the real `_energy_injection` so the injection gate stays in
lockstep with production. Feeding a silent buffer makes that gate a no-op
passthrough (RMS is flat, injection == 0.0 < inject_max), which isolates the
shadow-window logic under test.
"""

from __future__ import annotations

import random

import numpy as np

from app.models import OnsetCandidate
from app.pipeline import adtof_onsets as ao


def _oc(time: float, amp: float | None) -> OnsetCandidate:
    return OnsetCandidate(time=time, strength=0.5, amplitude=amp)


def _reference_shadow_filter(
    candidates: list[OnsetCandidate],
    audio: np.ndarray,
    sample_rate: int,
    window_s: float,
    louder_mult: float,
    inject_max: float,
) -> tuple[list[OnsetCandidate], int]:
    """The pre-optimisation O(n^2) implementation, kept for differential diffing."""
    import librosa

    if louder_mult <= 0.0 or len(candidates) < 2:
        return candidates, 0
    rms = librosa.feature.rms(y=audio, hop_length=ao._SHADOW_RMS_HOP)[0]
    rms_t = librosa.times_like(rms, sr=sample_rate, hop_length=ao._SHADOW_RMS_HOP)
    kept: list[OnsetCandidate] = []
    dropped = 0
    for i, c in enumerate(candidates):
        if c.amplitude is None:
            kept.append(c)
            continue
        if ao._energy_injection(rms, rms_t, float(c.time)) >= inject_max:
            kept.append(c)
            continue
        in_shadow = any(
            0.0 < c.time - p.time <= window_s
            and p.amplitude is not None
            and p.amplitude >= louder_mult * c.amplitude
            for p in candidates[:i]
        )
        if in_shadow:
            dropped += 1
        else:
            kept.append(c)
    return kept, dropped


_SILENT = np.zeros(22050, dtype=np.float32)


def _assert_same(cands: list[OnsetCandidate], **kw) -> None:
    """Both implementations must agree on kept identities (by order) + count."""
    ref_kept, ref_dropped = _reference_shadow_filter(cands, _SILENT, 22050, **kw)
    new_kept, new_dropped = ao._crash_shadow_filter(cands, _SILENT, 22050, **kw)
    # Identity comparison: same objects, same order.
    assert [id(c) for c in new_kept] == [id(c) for c in ref_kept]
    assert new_dropped == ref_dropped
    assert len(new_kept) + new_dropped == len(cands)


_KW = dict(window_s=1.5, louder_mult=3.0, inject_max=0.85)


def test_single_shadow_within_window() -> None:
    # Loud predecessor 0.5s earlier, quiet successor: dropped.
    _assert_same([_oc(0.0, 1.0), _oc(0.5, 0.1)], **_KW)


def test_multiple_candidates_inside_one_shadow_window() -> None:
    # One loud crash at t=0; four quiet re-triggers all inside its 1.5s shadow.
    cands = [_oc(0.0, 1.0), _oc(0.3, 0.1), _oc(0.6, 0.1), _oc(0.9, 0.2), _oc(1.4, 0.05)]
    _assert_same(cands, **_KW)


def test_window_boundary_exact_kept_and_dropped() -> None:
    # Predecessor exactly window_s (1.5s) earlier is still in-window (<=).
    _assert_same([_oc(0.0, 1.0), _oc(1.5, 0.1)], **_KW)
    # A hair past the boundary falls out -> not shadowed.
    _assert_same([_oc(0.0, 1.0), _oc(1.5001, 0.1)], **_KW)


def test_equal_time_predecessor_never_shadows() -> None:
    # 0.0 < c.time - p.time is strict: a same-time earlier hit casts no shadow.
    _assert_same([_oc(0.5, 1.0), _oc(0.5, 0.1)], **_KW)


def test_tie_time_block_then_later_candidate() -> None:
    # Two loud hits share a time, then a quiet successor: both count as it.
    _assert_same([_oc(0.5, 1.0), _oc(0.5, 0.9), _oc(0.9, 0.1)], **_KW)


def test_not_loud_enough_predecessor_kept() -> None:
    # Predecessor only 2x louder (< 3x mult): no shadow.
    _assert_same([_oc(0.0, 0.2), _oc(0.5, 0.1)], **_KW)


def test_amplitude_none_passthrough() -> None:
    # None-amplitude candidates are always kept and never cast shadows.
    cands = [_oc(0.0, 1.0), _oc(0.3, None), _oc(0.6, 0.1), _oc(0.9, None)]
    _assert_same(cands, **_KW)


def test_shadow_source_expires_before_successor() -> None:
    # Loud crash, then a quiet hit inside its window (dropped), then a second
    # quiet hit AFTER the crash has aged out of the window (kept).
    cands = [_oc(0.0, 1.0), _oc(0.5, 0.1), _oc(2.0, 0.1)]
    _assert_same(cands, **_KW)


def test_max_amplitude_when_multiple_predecessors() -> None:
    # A weak then a strong predecessor: the strong one determines the shadow.
    cands = [_oc(0.0, 0.15), _oc(0.4, 1.0), _oc(0.8, 0.1)]
    _assert_same(cands, **_KW)


def test_predecessor_that_is_itself_dropped_still_shadows() -> None:
    # A quiet re-trigger (dropped) can still be the loud predecessor that
    # shadows an even-quieter later hit: the scan keys off `p.time`/`p.amplitude`
    # only, not whether `p` was kept. loud -> mid (dropped) -> tiny (dropped by mid).
    cands = [_oc(0.0, 1.0), _oc(0.4, 0.3), _oc(0.8, 0.05)]
    _assert_same(cands, **_KW)


def test_disabled_when_mult_zero() -> None:
    _assert_same([_oc(0.0, 1.0), _oc(0.5, 0.1)], window_s=1.5, louder_mult=0.0, inject_max=0.85)


def test_single_candidate_is_noop() -> None:
    _assert_same([_oc(0.0, 1.0)], **_KW)


def test_randomised_matches_reference() -> None:
    rng = random.Random(20260702)
    for _ in range(500):
        n = rng.randint(0, 14)
        # Time-ordered (the filter's documented contract), with occasional ties.
        cands: list[OnsetCandidate] = []
        t = 0.0
        for _ in range(n):
            t += rng.choice([0.0, 0.0, 0.05, 0.2, 0.6, 1.4, 2.5])
            amp = None if rng.random() < 0.12 else round(rng.uniform(0.02, 1.0), 3)
            cands.append(_oc(round(t, 4), amp))
        window_s = rng.choice([0.5, 1.0, 1.5, 2.0])
        louder_mult = rng.choice([1.5, 2.0, 3.0, 4.0])
        _assert_same(cands, window_s=window_s, louder_mult=louder_mult, inject_max=0.85)
