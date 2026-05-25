"""Unit tests for the unified ternary cymbal classifier.

Same shape as `test_hihat_split.py`: covers the deterministic helpers
that don't require librosa / anthropic to exercise.

  * `_coerce_index_set`; out-of-range clamping and non-int ignore.
  * `_classify_fallback`; returns `(crash_set, empty discard_set)` in
    the no-API-key path so the runner contract holds.

The `_classify_llm` and `_measure` paths require external dependencies
(anthropic and librosa respectively) and are exercised in the manual
end-to-end pass, not here.
"""
from __future__ import annotations

from types import SimpleNamespace

from app.pipeline.cymbal_split import (
    _FALLBACK_DECAY_S,
    _FALLBACK_ISOLATION_S,
    _classify_fallback,
    _coerce_index_set,
    _Feat,
)


def _c(time: float, bar: int = 0, beat: float = 1.0):
    """Duck-typed `OnsetCandidate`."""
    return SimpleNamespace(time=time, strength=1.0, bar=bar, beat_in_bar=beat)


def _f(*, decay: float = 0.0, gap: float = 0.0) -> _Feat:
    return _Feat(
        decay_s=decay, flatness=0.0, centroid_hz=0.0, gap_s=gap,
    )


def test_coerce_index_set_clamps_and_dedupes() -> None:
    # Mirrors the hi-hat test: out-of-range, negative, duplicates, and
    # non-int entries all get filtered.
    out = _coerce_index_set([0, 2, 2, 4, 9, -1, "x", None, 1.5], n=5)
    assert out == {0, 1, 2, 4}


def test_classify_fallback_never_discards() -> None:
    # The no-API-key path must always return an empty discard set so the
    # runner contract holds. The crash signature is "rings long AND is
    # isolated"; both thresholds must be cleared.
    feats = [
        _f(decay=_FALLBACK_DECAY_S + 0.1, gap=_FALLBACK_ISOLATION_S + 0.1),
        _f(decay=0.05, gap=0.1),
    ]
    crash_set, discard_set = _classify_fallback(
        [_c(0.0), _c(0.25)], feats,
    )
    assert crash_set == {0}
    assert discard_set == set()
