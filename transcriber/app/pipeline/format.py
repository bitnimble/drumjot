"""Pretty-print a Jot DSL string.

Thin wrapper around the canonical TypeScript formatter
(`src/format.ts`, invoked via `tools/format_jot.ts`). Keeping the
formatter in TS — next to the parser it must round-trip through — means
DSL serialisation has a single source of truth, exactly like
`recompose.py` delegates merging and `jot_extract.py` delegates parsing.

Formatting is purely cosmetic: it never changes what the Jot *means*,
only how it reads. Accordingly, `format_dsl` is best-effort and
`fail-open` — any bridge problem (tool missing, parse error, timeout,
`bun` absent) logs a warning and returns the input string untouched.
A formatter hiccup must never corrupt or drop a transcription that the
rest of the pipeline produced successfully.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

TOOL_PATH = Path(
    os.environ.get("FORMAT_JOT_TOOL", "/app/tools/format_jot.ts")
)


def format_dsl(dsl: str, timeout: float = 30.0) -> str:
    """Return `dsl` reformatted for readability, or `dsl` unchanged if
    formatting fails for any reason (the failure is logged, never raised)."""
    if not dsl or not dsl.strip():
        return dsl
    if not TOOL_PATH.exists():
        log.warning(
            "format_dsl: tool missing at %s; returning DSL unformatted",
            TOOL_PATH,
        )
        return dsl

    try:
        result = subprocess.run(
            ["bun", "run", str(TOOL_PATH)],
            input=dsl.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        log.warning("format_dsl: bridge timed out; returning DSL unformatted")
        return dsl
    except FileNotFoundError:
        log.warning("format_dsl: `bun` not in PATH; returning DSL unformatted")
        return dsl

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        log.warning(
            "format_dsl: bridge exited %d (%s); returning DSL unformatted",
            result.returncode,
            stderr or "no stderr",
        )
        return dsl

    formatted = result.stdout.decode("utf-8", errors="replace")
    if not formatted.strip():
        log.warning("format_dsl: bridge produced empty output; returning DSL unformatted")
        return dsl
    return formatted
