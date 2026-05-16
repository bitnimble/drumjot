"""Bridge from Drumjot DSL text -> per-pitch onset list.

Delegates parsing to the canonical TypeScript parser in `src/parser` by
shelling out to `bun run tools/jot_to_onsets.ts`. Keeping the parser in
TypeScript only avoids drift between the frontend and the refinement
pipeline.

Layout in the container:
    /app/src/                  - copy of repo's `src/` (TS parser + types)
    /app/tools/jot_to_onsets.ts - bun CLI bridge
    /app/tsconfig.json          - so `bun` resolves the `src/*` alias
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

TOOL_PATH = Path(os.environ.get("JOT_TO_ONSETS_TOOL", "/app/tools/jot_to_onsets.ts"))


@dataclass
class PredictedOnset:
    pitch: str
    time: float
    velocity: int
    modifiers: list[str]


@dataclass
class ExtractedJot:
    bpm: float
    time_signature: tuple[int, int]
    onsets_by_pitch: dict[str, list[PredictedOnset]]


class JotParseError(Exception):
    """Raised when the LLM's DSL output failed to parse via the TS parser."""

    def __init__(self, message: str, stderr: str = "") -> None:
        super().__init__(message)
        self.stderr = stderr


def extract_jot(dsl_text: str, timeout: float = 30.0) -> ExtractedJot:
    if not TOOL_PATH.exists():
        raise FileNotFoundError(
            f"jot_to_onsets tool missing at {TOOL_PATH}; "
            "the Docker build didn't include /app/tools/ correctly."
        )
    try:
        result = subprocess.run(
            ["bun", "run", str(TOOL_PATH)],
            input=dsl_text.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise JotParseError("Bun extraction timed out", "") from exc
    except FileNotFoundError as exc:
        raise JotParseError(f"`bun` is not installed in PATH: {exc}", "") from exc

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        msg = stderr.replace("PARSE_ERROR:", "").strip() or "unknown parse error"
        raise JotParseError(msg, stderr)

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise JotParseError(
            f"Bun tool returned non-JSON: {result.stdout[:200]!r}"
        ) from exc

    onsets_by_pitch: dict[str, list[PredictedOnset]] = {}
    for pitch, items in data["onsets"].items():
        onsets_by_pitch[pitch] = [
            PredictedOnset(
                pitch=pitch,
                time=float(item["time"]),
                velocity=int(item["velocity"]),
                modifiers=list(item.get("modifiers", [])),
            )
            for item in items
        ]
    ts = data["timeSignature"]
    return ExtractedJot(
        bpm=float(data["bpm"]),
        time_signature=(int(ts["count"]), int(ts["unit"])),
        onsets_by_pitch=onsets_by_pitch,
    )
