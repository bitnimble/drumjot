"""Pure-Python tests for the lint module's helpers.

The bun bridge itself is exercised by the TypeScript test suite under
`src/linter/__tests__/`. This file covers the Python-side parsing /
formatting so a JSON shape change in the bridge can't silently bypass
the refinement loop's expectations.
"""
from __future__ import annotations

from app.pipeline.lint import (
    LintDiagnostic,
    LintResult,
    _parse_diagnostic,
    format_for_prompt,
)


def test_parse_diagnostic_full_payload() -> None:
    raw = {
        "ruleId": "instrument/invalid-modifier",
        "severity": "error",
        "kind": "instrument",
        "message": "':o' is not valid on kick",
        "range": {"start": 12, "end": 18},
        "line": 3,
        "column": 5,
        "endLine": 3,
        "endColumn": 9,
        "snippet": "| k:o |",
        "suggestedFix": "Remove ':o'",
    }
    d = _parse_diagnostic(raw)
    assert d.rule_id == "instrument/invalid-modifier"
    assert d.severity == "error"
    assert d.kind == "instrument"
    assert d.line == 3
    assert d.snippet == "| k:o |"
    assert d.suggested_fix == "Remove ':o'"


def test_parse_diagnostic_minimal_payload() -> None:
    d = _parse_diagnostic({
        "ruleId": "performance/too-many-hands",
        "severity": "error",
        "kind": "performance",
        "message": "3 simultaneous hand strokes",
    })
    assert d.line is None
    assert d.column is None
    assert d.snippet is None
    assert d.location() == "(no position)"


def test_location_formats_compact() -> None:
    same_line = LintDiagnostic(
        rule_id="x",
        severity="error",
        kind="instrument",
        message="m",
        line=4,
        column=2,
        end_line=4,
        end_column=7,
    )
    assert same_line.location() == "4:2-7"

    multi_line = LintDiagnostic(
        rule_id="x",
        severity="error",
        kind="performance",
        message="m",
        line=4,
        column=2,
        end_line=5,
        end_column=10,
    )
    assert multi_line.location() == "4:2-5:10"


def test_format_for_prompt_orders_errors_before_warnings() -> None:
    result = LintResult(
        bars=[],
        diagnostics=[
            LintDiagnostic(
                rule_id="performance/roll-on-kick",
                severity="warning",
                kind="performance",
                message="kick roll suspicious",
                line=2,
                column=1,
            ),
            LintDiagnostic(
                rule_id="instrument/invalid-modifier",
                severity="error",
                kind="instrument",
                message="':o' invalid on kick",
                line=3,
                column=4,
            ),
        ],
        errors=1,
        warnings=1,
    )
    out = format_for_prompt(result)
    # The error should appear before the warning in the formatted text.
    err_idx = out.index("ERROR instrument/invalid-modifier")
    warn_idx = out.index("WARNING performance/roll-on-kick")
    assert err_idx < warn_idx


def test_has_errors_flag() -> None:
    result = LintResult(diagnostics=[], errors=2, warnings=5, bars=[])
    assert result.has_errors
    assert result.has_any

    clean = LintResult(diagnostics=[], errors=0, warnings=0, bars=[])
    assert not clean.has_errors
    assert not clean.has_any
