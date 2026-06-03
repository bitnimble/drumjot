"""Unit tests for the unified ternary hi-hat classifier.

Covers the deterministic helpers that don't require librosa / anthropic
to exercise:

  * `_coerce_index_set` — out-of-range clamping and non-int ignore.
  * `_open_tail_filter`, discard exclusion, closed-in-tail and
    open-in-tail sizzle drops (now keyed on `attack_flux`, not `pre_rms`),
    tail extension off discards is a no-op.
  * `_envelope_open_verdict`, the deterministic open/closed guardrail:
    any single open signature forces open, all-low forces closed, the
    mid-range defers to the LLM, and tail is a duration not an absolute.
  * `_classify_fallback`, returns `(open_set, empty discard_set)` in
    the no-API-key path so the runner contract holds.
  * `_label_for`, debug-dump label priority (tail labels > LLM
    discard > open > closed).

The `_classify_llm` and `_measure` paths require external dependencies
(anthropic and librosa respectively) and are exercised in the manual
end-to-end pass described in the plan, not here.
"""
from __future__ import annotations

from types import SimpleNamespace

from app.pipeline.hihat_split import (
    _BLEED_LOWBAND_RATIO_MAX,
    _OPEN_IN_TAIL_MIN_FLUX,
    _RESCUE_MIN_FLUX,
    _RESCUE_STRONG_CLOSED_FLUX,
    _classify_fallback,
    _coerce_index_set,
    _envelope_open_verdict,
    _Feat,
    _label_for,
    _open_tail_filter,
    _rescue_discards,
)


def _c(time: float, bar: int = 0, beat: float = 1.0):
    """Duck-typed `OnsetCandidate` (`_open_tail_filter` only reads `.time`)."""
    return SimpleNamespace(time=time, strength=1.0, bar=bar, beat_in_bar=beat)


def _f(
    *,
    late: float = 0.0,
    pre: float = 0.0,
    tail_end_t: float = 0.0,
    gap_s: float = 0.5,
    lowband_ratio: float = 0.0,
    # Default to a clear fresh transient so an open isn't dropped as sizzle
    # unless a test deliberately sets a low flux.
    attack_flux: float = 10.0,
) -> _Feat:
    return _Feat(
        late_rms=late,
        pre_rms=pre,
        attack_s=0.0,
        flatness=0.0,
        centroid_hz=0.0,
        gap_s=gap_s,
        tail_end_t=tail_end_t,
        attack_flux=attack_flux,
        lowband_ratio=lowband_ratio,
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
    # Open at t=0 with a 1.5s ring; another open at t=0.5 has a LOW
    # attack_flux (no fresh transient, just ring wobble), so it's a
    # sizzle bump, drop it. High pre_rms alone must NOT trigger the drop.
    low_flux = _OPEN_IN_TAIL_MIN_FLUX - 0.1
    onsets = [_c(0.0), _c(0.5)]
    feats = [_f(tail_end_t=1.5), _f(pre=0.9, attack_flux=low_flux, tail_end_t=2.0)]
    closed_dropped, open_dropped = _open_tail_filter(
        onsets, feats, open_idx={0, 1}, discard_idx=set(),
    )
    assert closed_dropped == set()
    assert open_dropped == {1}


def test_open_tail_filter_keeps_repeated_open_with_fresh_attack() -> None:
    # Open at t=0, then a second open at t=0.5 riding on the ring (HIGH
    # pre_rms) but with a clear fresh attack_flux, a genuine restrike,
    # not a sizzle bump. Keep and extend. This is the regression the
    # pre_rms-based rule got wrong (it dropped real repeated open strikes).
    onsets = [_c(0.0), _c(0.5)]
    feats = [
        _f(tail_end_t=1.5),
        _f(pre=0.9, attack_flux=_OPEN_IN_TAIL_MIN_FLUX + 5.0, tail_end_t=2.0),
    ]
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
    open_set, discard_set, low_conf = _classify_fallback(
        [_c(0.0), _c(0.25)], feats,
    )
    assert open_set == {0}
    assert discard_set == set()
    assert low_conf == set()


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


def test_envelope_verdict_open_on_any_single_signature() -> None:
    # A long ring or a high late_rms is open on its own (onset_time=0 so
    # tail_s == tail_end_t).
    assert _envelope_open_verdict(_f(tail_end_t=0.5), 0.0) == "open"  # long ring
    assert _envelope_open_verdict(_f(late=0.4), 0.0) == "open"        # still ringing
    # `pre` riding-on-ring is open ONLY with sustain corroboration (see the
    # phantom-8th regression test below); pre alone with no late is not.
    assert _envelope_open_verdict(_f(pre=0.6, late=0.15), 0.0) == "open"


def test_envelope_verdict_closed_requires_all_signatures_low() -> None:
    # Short, dry, not-riding -> decisively closed.
    assert _envelope_open_verdict(
        _f(late=0.02, pre=0.05, tail_end_t=0.09), 0.0
    ) == "closed"


def test_envelope_verdict_pre_signature_requires_sustain_corroboration() -> None:
    # Phantom onset on a near-zero peak: pre_rms explodes (here 5.0) but the
    # hit has NO sustain (late=0, minimal tail). It must NOT read as open, # an open hat always rings. (Regression: Cold-Hard-Bitch phantom 8ths.)
    assert _envelope_open_verdict(
        _f(late=0.0, pre=5.0, tail_end_t=0.09), 0.0
    ) is None
    # Same high pre but WITH corroborating late ring -> genuinely open.
    assert _envelope_open_verdict(
        _f(late=0.15, pre=5.0, tail_end_t=0.09), 0.0
    ) == "open"
    # A real open still reads open on tail alone, no pre needed.
    assert _envelope_open_verdict(
        _f(late=0.0, pre=0.0, tail_end_t=0.67), 0.0
    ) == "open"


def test_envelope_verdict_ambiguous_returns_none() -> None:
    # Mid-range: not decisive either way (tail between the closed ceiling
    # and the open floor, late/pre moderate) -> defer to the LLM.
    assert _envelope_open_verdict(
        _f(late=0.15, pre=0.35, tail_end_t=0.25), 0.0
    ) is None


def test_envelope_verdict_uses_tail_duration_not_absolute_time() -> None:
    # tail_end_t is absolute; the verdict must subtract onset_time. A late
    # onset with a short ring (tail_end_t just past onset_time) is closed,
    # even though tail_end_t itself is large.
    assert _envelope_open_verdict(
        _f(late=0.02, pre=0.05, tail_end_t=10.09), onset_time=10.0
    ) == "closed"


def test_rescue_low_confidence_discard_with_hat_signature() -> None:
    # An LLM low-confidence discard that the envelope says is decisively
    # closed, looks like a hat (low lowband_ratio), has a fresh flux, and a
    # normal gap -> rescued to closed.
    onsets = [_c(0.0)]
    feats = [_f(late=0.02, pre=0.05, tail_end_t=0.09, attack_flux=8.0,
                lowband_ratio=0.01, gap_s=0.3)]
    open_idx, discard_idx = set(), {0}
    r_open, r_closed = _rescue_discards(onsets, feats, open_idx, discard_idx,
                                        low_conf_discards={0})
    assert r_closed == {0} and r_open == set()
    assert discard_idx == set()  # removed from discard


def test_rescue_blocked_by_bleed_signature() -> None:
    # Same decisive-closed envelope, but a high lowband_ratio = bleed.
    # Must NOT be rescued even though the LLM was unsure.
    onsets = [_c(0.0)]
    feats = [_f(late=0.02, pre=0.05, tail_end_t=0.09, attack_flux=8.0,
                lowband_ratio=_BLEED_LOWBAND_RATIO_MAX + 0.2, gap_s=0.3)]
    open_idx, discard_idx = set(), {0}
    r_open, r_closed = _rescue_discards(onsets, feats, open_idx, discard_idx,
                                        low_conf_discards={0})
    assert r_open == set() and r_closed == set()
    assert discard_idx == {0}  # still discarded


def test_rescue_blocked_by_sizzle_no_fresh_flux() -> None:
    onsets = [_c(0.0)]
    feats = [_f(late=0.02, pre=0.05, tail_end_t=0.09,
                attack_flux=_RESCUE_MIN_FLUX - 0.5, lowband_ratio=0.01, gap_s=0.3)]
    open_idx, discard_idx = set(), {0}
    _rescue_discards(onsets, feats, open_idx, discard_idx, low_conf_discards={0})
    assert discard_idx == {0}


def test_rescue_confident_discard_needs_overwhelming_evidence() -> None:
    # A HIGH-confidence discard (not in low_conf) with a merely-decisive
    # closed envelope is respected; but an overwhelming flux overturns it.
    onsets = [_c(0.0)]
    weak = [_f(late=0.02, pre=0.05, tail_end_t=0.09,
               attack_flux=_RESCUE_STRONG_CLOSED_FLUX - 5.0,
               lowband_ratio=0.01, gap_s=0.3)]
    oi, di = set(), {0}
    _rescue_discards(onsets, weak, oi, di, low_conf_discards=set())
    assert di == {0}  # confident discard respected

    strong = [_f(late=0.02, pre=0.05, tail_end_t=0.09,
                 attack_flux=_RESCUE_STRONG_CLOSED_FLUX + 5.0,
                 lowband_ratio=0.01, gap_s=0.3)]
    oi, di = set(), {0}
    _, r_closed = _rescue_discards(onsets, strong, oi, di, low_conf_discards=set())
    assert r_closed == {0} and di == set()  # overwhelming -> rescued


def test_rescue_ambiguous_envelope_respects_discard() -> None:
    onsets = [_c(0.0)]
    feats = [_f(late=0.15, pre=0.35, tail_end_t=0.25, attack_flux=8.0,
                lowband_ratio=0.01, gap_s=0.3)]
    oi, di = set(), {0}
    _rescue_discards(onsets, feats, oi, di, low_conf_discards={0})
    assert di == {0}  # ambiguous verdict -> not rescued
