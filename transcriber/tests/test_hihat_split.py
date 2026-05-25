"""Unit tests for the unified ternary hi-hat classifier.

Covers the deterministic helpers that don't require librosa / anthropic
to exercise:

  * `_coerce_index_set` — out-of-range clamping and non-int ignore.
  * `_open_tail_filter` — discard exclusion, closed-in-tail and
    open-in-tail sizzle drops, tail extension off discards is a no-op.
  * `_classify_fallback` — returns `(open_set, empty discard_set)` in
    the no-API-key path so the runner contract holds.
  * `_label_for` — debug-dump label priority (tail labels > LLM
    discard > open > closed).

The `_classify_llm` and `_measure` paths require external dependencies
(anthropic and librosa respectively) and are exercised in the manual
end-to-end pass described in the plan, not here.
"""
from __future__ import annotations

from types import SimpleNamespace

from app.pipeline.hihat_split import (
    _OPEN_IN_TAIL_MAX_PRE_RMS,
    _classify_fallback,
    _coerce_index_set,
    _Feat,
    _label_for,
    _open_tail_filter,
)


def _c(time: float, bar: int = 0, beat: float = 1.0):
    """Duck-typed `OnsetCandidate` (`_open_tail_filter` only reads `.time`)."""
    return SimpleNamespace(time=time, strength=1.0, bar=bar, beat_in_bar=beat)


def _f(
    *,
    late: float = 0.0,
    pre: float = 0.0,
    tail_end_t: float = 0.0,
) -> _Feat:
    return _Feat(
        late_rms=late,
        pre_rms=pre,
        attack_s=0.0,
        flatness=0.0,
        centroid_hz=0.0,
        gap_s=0.0,
        tail_end_t=tail_end_t,
    )


def test_coerce_index_set_clamps_and_dedupes() -> None:
    # 5 valid indices; out-of-range, negative, duplicates, and non-int
    # entries all get filtered.
    out = _coerce_index_set([0, 2, 2, 4, 9, -1, "x", None, 1.5], n=5)
    # `int(1.5) == 1` — float coercion is allowed (matches _extract_rejected).
    assert out == {0, 1, 2, 4}


def test_open_tail_filter_skips_discarded_and_does_not_extend_off_them() -> None:
    # Three onsets: a "discarded" open at t=0 (LLM said sizzle artifact),
    # then a closed at t=0.5. If the discard were not skipped, the tail
    # would extend to t=1.5 and the closed would be dropped. With the
    # discard correctly excluded, the closed survives.
    onsets = [_c(0.0), _c(0.5)]
    feats = [
        _f(tail_end_t=1.5),  # would extend tail to t=1.5 if treated as open
        _f(),
    ]
    closed_dropped, open_dropped = _open_tail_filter(
        onsets, feats, open_idx={0}, discard_idx={0},
    )
    # No surviving open => no tracked tail => closed is OUTSIDE any tail.
    assert closed_dropped == set()
    assert open_dropped == set()


def test_open_tail_filter_drops_closed_in_tail() -> None:
    # Open at t=0 with a 1.5s ring; closed at t=0.5 falls inside.
    onsets = [_c(0.0), _c(0.5)]
    feats = [_f(tail_end_t=1.5), _f()]
    closed_dropped, open_dropped = _open_tail_filter(
        onsets, feats, open_idx={0}, discard_idx=set(),
    )
    assert closed_dropped == {1}
    assert open_dropped == set()


def test_open_tail_filter_drops_sizzle_open_in_tail() -> None:
    # Open at t=0 with a 1.5s ring; another open at t=0.5 has high
    # pre_rms (above the threshold), so it's a sizzle bump — drop it.
    high_pre = _OPEN_IN_TAIL_MAX_PRE_RMS + 0.1
    onsets = [_c(0.0), _c(0.5)]
    feats = [_f(tail_end_t=1.5), _f(pre=high_pre, tail_end_t=2.0)]
    closed_dropped, open_dropped = _open_tail_filter(
        onsets, feats, open_idx={0, 1}, discard_idx=set(),
    )
    assert closed_dropped == set()
    assert open_dropped == {1}


def test_open_tail_filter_keeps_repeated_open_with_fresh_attack() -> None:
    # Open at t=0, then a second open at t=0.5 with LOW pre_rms (it's a
    # genuine restrike, not a sizzle bump). Keep and extend.
    onsets = [_c(0.0), _c(0.5)]
    feats = [_f(tail_end_t=1.5), _f(pre=0.1, tail_end_t=2.0)]
    closed_dropped, open_dropped = _open_tail_filter(
        onsets, feats, open_idx={0, 1}, discard_idx=set(),
    )
    assert closed_dropped == set()
    assert open_dropped == set()


def test_open_tail_filter_no_opens_is_noop() -> None:
    onsets = [_c(0.0), _c(0.5)]
    feats = [_f(), _f()]
    closed_dropped, open_dropped = _open_tail_filter(
        onsets, feats, open_idx=set(), discard_idx=set(),
    )
    assert closed_dropped == set()
    assert open_dropped == set()


def test_classify_fallback_never_discards() -> None:
    # The no-API-key path must always return an empty discard set so the
    # runner contract holds. Mix a ringing onset (high late) and a quiet
    # one — the ringing one classifies as open, the quiet one as closed.
    feats = [_f(late=0.6, pre=0.1), _f(late=0.05, pre=0.02)]
    open_set, discard_set = _classify_fallback(
        [_c(0.0), _c(0.25)], feats,
    )
    assert open_set == {0}
    assert discard_set == set()


def test_label_for_priority() -> None:
    # tail labels beat LLM `discard`, which beats `open`, which beats
    # `closed`. Same onset in multiple sets resolves to the first label
    # it matches (the dump shouldn't lose specificity).
    assert _label_for(
        0, open_idx={0}, discard_idx={0},
        closed_in_tail=set(), open_in_tail={0},
    ) == "open_in_tail"
    assert _label_for(
        0, open_idx=set(), discard_idx={0},
        closed_in_tail={0}, open_in_tail=set(),
    ) == "closed_in_tail"
    assert _label_for(
        0, open_idx={0}, discard_idx={0},
        closed_in_tail=set(), open_in_tail=set(),
    ) == "discard"
    assert _label_for(
        0, open_idx={0}, discard_idx=set(),
        closed_in_tail=set(), open_in_tail=set(),
    ) == "open"
    assert _label_for(
        0, open_idx=set(), discard_idx=set(),
        closed_in_tail=set(), open_in_tail=set(),
    ) == "closed"
