"""Tests for the lint-pass segment grouping in refine.py.

`_build_lint_segments` is responsible for deciding which diagnostics
share an LLM call. The merge rule (touching context windows merge into
one segment) drives both per-iteration call count and prompt token cost,
so it's worth pinning down with explicit examples.
"""
from __future__ import annotations

from app.pipeline.lint import BarRange, LintDiagnostic, LintResult
from app.pipeline.refine import LINT_CONTEXT_BARS, _build_lint_segments


def _diag(voice: int, bar: int, rule: str = "instrument/invalid-modifier") -> LintDiagnostic:
    return LintDiagnostic(
        rule_id=rule,
        severity="error",
        kind="instrument",
        message="m",
        voice_index=voice,
        bar_index=bar,
    )


def _bars_for_voice(n: int, bar_width: int = 10, start: int = 0) -> list[BarRange]:
    """Synthesise `n` consecutive bars with `bar_width` bytes each."""
    out: list[BarRange] = []
    pos = start
    for _ in range(n):
        out.append(BarRange(start=pos, end=pos + bar_width))
        pos += bar_width
    return out


def test_single_diagnostic_produces_one_segment_with_context() -> None:
    result = LintResult(
        diagnostics=[_diag(0, 3)],
        errors=1,
        warnings=0,
        bars=[_bars_for_voice(10)],
    )
    segments = _build_lint_segments(result, "x" * 100)
    assert len(segments) == 1
    s = segments[0]
    assert s.voice_index == 0
    assert s.first_bar == 3
    assert s.last_bar == 3
    assert s.context_first == 3 - LINT_CONTEXT_BARS
    assert s.context_last == 3 + LINT_CONTEXT_BARS
    assert s.byte_start == 20  # bar 2 starts at byte 20
    assert s.byte_end == 50    # bar 4 ends at byte 50


def test_adjacent_diagnostics_merge_into_one_segment() -> None:
    # Diagnostics in bars 3 and 5 share context (windows are [2..4] and [4..6]).
    result = LintResult(
        diagnostics=[_diag(0, 3), _diag(0, 5)],
        errors=2,
        warnings=0,
        bars=[_bars_for_voice(10)],
    )
    segments = _build_lint_segments(result, "x" * 100)
    assert len(segments) == 1
    s = segments[0]
    assert s.first_bar == 3
    assert s.last_bar == 5
    assert s.context_first == 2
    assert s.context_last == 6
    assert len(s.diagnostics) == 2


def test_far_diagnostics_produce_separate_segments() -> None:
    # Diagnostics in bars 2 and 9 are far apart; their context windows
    # don't touch (bar 3 vs bar 8 — 5 bars of gap).
    result = LintResult(
        diagnostics=[_diag(0, 2), _diag(0, 9)],
        errors=2,
        warnings=0,
        bars=[_bars_for_voice(12)],
    )
    segments = _build_lint_segments(result, "x" * 200)
    assert len(segments) == 2
    # Segments are sorted right-to-left for safe right-to-left patching.
    assert segments[0].byte_start > segments[1].byte_start


def test_different_voices_never_merge() -> None:
    result = LintResult(
        diagnostics=[_diag(0, 3), _diag(1, 3)],
        errors=2,
        warnings=0,
        bars=[_bars_for_voice(8), _bars_for_voice(8, start=100)],
    )
    segments = _build_lint_segments(result, "x" * 300)
    assert len(segments) == 2
    voice_indices = {s.voice_index for s in segments}
    assert voice_indices == {0, 1}


def test_diagnostics_without_bar_info_are_skipped() -> None:
    result = LintResult(
        diagnostics=[
            _diag(0, 3),
            LintDiagnostic(
                rule_id="instrument/invalid-modifier",
                severity="error",
                kind="instrument",
                message="m",
            ),  # no bar/voice info
        ],
        errors=2,
        warnings=0,
        bars=[_bars_for_voice(10)],
    )
    segments = _build_lint_segments(result, "x" * 100)
    assert len(segments) == 1
    assert segments[0].first_bar == 3
    assert len(segments[0].diagnostics) == 1


def test_context_clamps_to_bar_count_at_start_and_end() -> None:
    # Diagnostic in the first bar: context_first must clamp to 0,
    # not go negative.
    result = LintResult(
        diagnostics=[_diag(0, 0)],
        errors=1,
        warnings=0,
        bars=[_bars_for_voice(5)],
    )
    segments = _build_lint_segments(result, "x" * 100)
    assert len(segments) == 1
    assert segments[0].context_first == 0
    assert segments[0].context_last == 1

    # Diagnostic in the last bar: context_last clamps to bars-1.
    result = LintResult(
        diagnostics=[_diag(0, 4)],
        errors=1,
        warnings=0,
        bars=[_bars_for_voice(5)],
    )
    segments = _build_lint_segments(result, "x" * 100)
    assert segments[0].context_first == 3
    assert segments[0].context_last == 4


def test_no_diagnostics_yields_no_segments() -> None:
    result = LintResult(diagnostics=[], errors=0, warnings=0, bars=[_bars_for_voice(5)])
    assert _build_lint_segments(result, "x" * 100) == []
