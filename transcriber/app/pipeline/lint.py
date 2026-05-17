"""Python wrapper around the Drumjot Jot linter (`tools/lint_jot.ts`).

The actual linter lives in TypeScript alongside the parser; this module is
the thin shim the refinement loop uses to call it via bun. Output is parsed
into structured `LintDiagnostic` dataclasses so callers don't deal with
JSON dicts.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

LINT_TOOL_PATH = Path(
    os.environ.get("LINT_JOT_TOOL", "/app/tools/lint_jot.ts")
)


LintSeverity = Literal["error", "warning"]
LintKind = Literal["instrument", "performance"]


@dataclass(frozen=True, slots=True)
class LintDiagnostic:
    """Structured form of one diagnostic from the bun bridge.

    Position fields are 1-indexed and copy the LSP / editor convention. They
    are populated only when the underlying AST node carried a source range
    (i.e. it came from the parser, not a hand-built Jot).

    `bar_index` / `voice_index` point at the Jot bar/voice the offending
    element sits in. The refinement loop uses these to look up the bar's
    audio time range via `BeatStructure.bars` and pull the relevant onset
    candidates back into the prompt — without that, the LLM would be
    "fixing" a lint message blind, with no view of what the source audio
    actually contains in that region.
    """

    rule_id: str
    severity: LintSeverity
    kind: LintKind
    message: str
    line: int | None = None
    column: int | None = None
    end_line: int | None = None
    end_column: int | None = None
    snippet: str | None = None
    bar_index: int | None = None
    voice_index: int | None = None
    suggested_fix: str | None = None

    def location(self, line_offset: int = 0) -> str:
        """Compact `line:col` / `line:col-line:col` representation for prompts.

        `line_offset` subtracts from both the start and end line numbers
        so positions can be reported relative to a segment rather than
        the full DSL — necessary when the LLM only sees a slice and
        otherwise has no way to map an absolute line back to what it
        was shown.
        """
        if self.line is None or self.column is None:
            return "(no position)"
        adj_line = self.line - line_offset
        if self.end_line is None or self.end_column is None:
            return f"{adj_line}:{self.column}"
        adj_end_line = self.end_line - line_offset
        if adj_end_line == adj_line:
            return f"{adj_line}:{self.column}-{self.end_column}"
        return f"{adj_line}:{self.column}-{adj_end_line}:{self.end_column}"


@dataclass(frozen=True, slots=True)
class BarRange:
    """Byte range of one bar in the source DSL.

    `start` is the position of the `|` that opens the bar; `end` is the
    position of the next `|` (or EOF for the final bar). Bars without a
    recorded range (hand-built jots) carry `start = end = 0`.
    """

    start: int
    end: int

    @property
    def is_known(self) -> bool:
        return self.end > self.start


@dataclass(frozen=True, slots=True)
class LintResult:
    diagnostics: list[LintDiagnostic]
    errors: int
    warnings: int
    # Per-voice bar ranges into the source DSL. `bars[voice_index][bar_index]`
    # = (start, end) byte offsets. Empty if the linter wasn't given a
    # source-derived Jot (no parser ⇒ no ranges).
    bars: list[list[BarRange]]

    @property
    def has_errors(self) -> bool:
        return self.errors > 0

    @property
    def has_any(self) -> bool:
        return self.errors > 0 or self.warnings > 0

    def bar_range(self, voice_index: int, bar_index: int) -> BarRange | None:
        if voice_index < 0 or voice_index >= len(self.bars):
            return None
        bars_v = self.bars[voice_index]
        if bar_index < 0 or bar_index >= len(bars_v):
            return None
        r = bars_v[bar_index]
        return r if r.is_known else None


class LintError(Exception):
    """Raised when the bun bridge failed (couldn't parse the DSL, etc.)."""


def lint_jot(dsl_text: str, timeout: float = 30.0) -> LintResult:
    """Run the linter on `dsl_text`. Raises `LintError` if the DSL doesn't parse."""
    if not LINT_TOOL_PATH.exists():
        raise FileNotFoundError(
            f"lint_jot bridge missing at {LINT_TOOL_PATH}; "
            "the Docker build didn't include /app/tools/ correctly."
        )

    try:
        result = subprocess.run(
            ["bun", "run", str(LINT_TOOL_PATH)],
            input=dsl_text.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise LintError("Bun lint bridge timed out") from exc
    except FileNotFoundError as exc:
        raise LintError(f"`bun` is not installed in PATH: {exc}") from exc

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        msg = stderr.replace("PARSE_ERROR:", "").strip() or "unknown lint error"
        raise LintError(msg)

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise LintError(
            f"lint bridge returned non-JSON: {result.stdout[:200]!r}"
        ) from exc

    diagnostics = [_parse_diagnostic(d) for d in payload.get("diagnostics", [])]
    raw_bars = payload.get("bars") or []
    bars: list[list[BarRange]] = []
    for voice_bars in raw_bars:
        voice_out: list[BarRange] = []
        for b in voice_bars:
            voice_out.append(
                BarRange(start=int(b.get("start", 0)), end=int(b.get("end", 0)))
            )
        bars.append(voice_out)
    return LintResult(
        diagnostics=diagnostics,
        errors=int(payload.get("errors", 0)),
        warnings=int(payload.get("warnings", 0)),
        bars=bars,
    )


def _parse_diagnostic(d: dict) -> LintDiagnostic:
    return LintDiagnostic(
        rule_id=str(d.get("ruleId", "")),
        severity=str(d.get("severity", "error")),  # type: ignore[arg-type]
        kind=str(d.get("kind", "instrument")),  # type: ignore[arg-type]
        message=str(d.get("message", "")),
        line=d.get("line"),
        column=d.get("column"),
        end_line=d.get("endLine"),
        end_column=d.get("endColumn"),
        snippet=d.get("snippet"),
        bar_index=d.get("barIndex"),
        voice_index=d.get("voiceIndex"),
        suggested_fix=d.get("suggestedFix"),
    )


def format_for_prompt(result: LintResult, line_offset: int = 0) -> str:
    """Render diagnostics for embedding in a refinement prompt.

    Each diagnostic gets a compact one-line header followed by its snippet
    indented two spaces, so the LLM can both read the message and see the
    exact offending text. Errors are listed before warnings.

    `line_offset` is the number of newlines that precede the segment
    being shown to the LLM. When non-zero, the rendered `line:col`
    positions are made segment-relative (so "line 3" means line 3 of
    the snippet, not line 3 of the full DSL the LLM never sees).
    """
    lines: list[str] = []
    sorted_diags = sorted(
        result.diagnostics,
        key=lambda d: (0 if d.severity == "error" else 1, d.rule_id, d.line or 0),
    )
    for d in sorted_diags:
        header = (
            f"[{d.severity.upper()} {d.rule_id} at {d.location(line_offset)}] "
            f"{d.message}"
        )
        lines.append(header)
        if d.snippet:
            for snippet_line in d.snippet.splitlines():
                lines.append(f"  {snippet_line}")
        if d.suggested_fix:
            lines.append(f"  fix: {d.suggested_fix}")
    return "\n".join(lines)
