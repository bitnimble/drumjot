"""Convert a Drumjot DSL string into a list of `OnsetEvent`.

Re-uses the existing bun bridge at `transcriber/tools/jot_to_onsets.ts`
rather than rewriting the parser in Python.

The bridge does `import { parse } from 'src/parser'`, which bun resolves
via the `paths` alias in a `tsconfig.json` somewhere upward from cwd:

- Inside the transcriber container: `/app/tsconfig.json` (synthesised
  by the Dockerfile). Tool lives at `/app/tools/jot_to_onsets.ts`.
- On the host: the repo root's `tsconfig.json`. Tool lives at
  `<repo>/transcriber/tools/jot_to_onsets.ts`.

The container path is set via `JOT_TO_ONSETS_TOOL` in the Dockerfile;
the host path is the default computed relative to this file.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .classes import JOT_PITCH_TO_CLASS
from .events import OnsetEvent

# transcriber/benchmarks/core/jot_events.py
#   parents[0]=core, [1]=benchmarks, [2]=transcriber, [3]=<repo root>
_DEFAULT_TOOL = (
    Path(__file__).resolve().parents[3] / "transcriber" / "tools" / "jot_to_onsets.ts"
)


class JotParseError(Exception):
    """Raised when the bun bridge couldn't parse the DSL."""


def _find_tsconfig_dir(tool_path: Path) -> Path:
    """Walk upward from `tool_path` until a directory containing tsconfig.json is found.

    That directory becomes bun's cwd, which is what makes `src/parser`
    resolve correctly via the tsconfig's `paths` alias.
    """
    for parent in [tool_path.parent, *tool_path.parents]:
        if (parent / "tsconfig.json").exists():
            return parent
    raise FileNotFoundError(
        f"no tsconfig.json found upward from {tool_path}; "
        "bun won't be able to resolve `src/parser`."
    )


def jot_dsl_to_events(dsl: str, timeout: float = 30.0) -> list[OnsetEvent]:
    """Parse a Drumjot DSL string and return its KD/SD/HH onsets.

    Pitches outside the 3-class taxonomy (`d`/`c`/`t`) are dropped.
    Returned events are sorted by time.
    """
    tool_path = Path(os.environ.get("JOT_TO_ONSETS_TOOL", str(_DEFAULT_TOOL)))
    if not tool_path.exists():
        raise FileNotFoundError(
            f"jot_to_onsets bridge missing at {tool_path}. "
            "Set JOT_TO_ONSETS_TOOL or run from inside the container."
        )

    cwd = _find_tsconfig_dir(tool_path)
    try:
        result = subprocess.run(
            ["bun", "run", str(tool_path)],
            input=dsl.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
            cwd=str(cwd),
            check=False,
        )
    except FileNotFoundError as exc:
        raise JotParseError(f"`bun` is not on PATH: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise JotParseError("bun bridge timed out") from exc

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        msg = stderr.replace("PARSE_ERROR:", "").strip() or "unknown parse error"
        raise JotParseError(msg)

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise JotParseError(f"bun bridge returned non-JSON: {result.stdout[:200]!r}") from exc

    events: list[OnsetEvent] = []
    for pitch, items in data.get("onsets", {}).items():
        drum_class = JOT_PITCH_TO_CLASS.get(pitch)
        if drum_class is None:
            continue
        for item in items:
            events.append(OnsetEvent(time=float(item["time"]), drum_class=drum_class))
    events.sort(key=lambda e: (e.time, e.drum_class.value))
    return events
