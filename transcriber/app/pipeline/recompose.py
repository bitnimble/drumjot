"""Recompose per-instrument monophonic lines into one Jot.

Thin wrapper around the canonical TypeScript recomposition
(`src/recompose.ts`, invoked via `tools/recompose_jot.ts`). Keeping the
merge logic in TS — next to the parser — means DSL manipulation has a
single source of truth, exactly like `jot_extract.py` delegates parsing
to the TS parser rather than maintaining a second one in Python.

This module only owns the *domain* facts the bridge can't know:

  - `FEET_PITCHES` — which pitches go to the second `||` voice. Just
    the kick: the upstream MDX23C separator produces no separate
    hi-hat-pedal stem, so the struck hi-hat `h` is a hands instrument.
  - `PITCH_DISPLAY_NAMES` — pitch letter → name for `instrumentMapping`
    (mirrors `separate.STEM_NAME_TO_PITCH`).

It shapes those + the beat-tracker `BeatStructure` into the bridge's
JSON contract and returns the merged DSL string.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

from app.pipeline.beats import BeatStructure

log = logging.getLogger(__name__)

TOOL_PATH = Path(
    os.environ.get("RECOMPOSE_JOT_TOOL", "/app/tools/recompose_jot.ts")
)

# Foot instruments → the second `||` voice (stems-down in notation).
FEET_PITCHES: frozenset[str] = frozenset({"k"})

# Pitch letter → display name for the recomposed `instrumentMapping`.
#
# `H` is a synthetic open-hi-hat routing key introduced by
# `pipeline/hihat_split.py` so the per-instrument transcribe pass can see
# closed (`h`) and open (`H`) hits as separate monophonic lines. Drumjot's
# DSL only has ONE notational hi-hat pitch (open vs closed are `:o` / `:c`
# modifiers on `h`), so `H` is a temporary lane: in the current first
# cut it lands in the final Jot as its own "Open Hi-Hat" voice; the
# notation-correct follow-up folds it into the `h` voice with `:o`
# per-note.
PITCH_DISPLAY_NAMES: dict[str, str] = {
    "k": "Kick",
    "s": "Snare",
    "h": "HiHat",
    "H": "Open Hi-Hat",
    "d": "Ride",
    "c": "Crash",
    "t": "Tom",
}


class RecomposeError(Exception):
    """Raised when the recompose bridge fails outright (bad invocation,
    crash, malformed JSON). Per-fragment parse failures are NOT errors —
    those pitches are dropped and reported in the result's `dropped`."""

    def __init__(self, message: str, stderr: str = "") -> None:
        super().__init__(message)
        self.stderr = stderr


def recompose(
    lines_by_pitch: dict[str, str],
    structure: BeatStructure,
    feet_pitches: frozenset[str] = FEET_PITCHES,
    timeout: float = 30.0,
) -> str:
    """Merge per-instrument monophonic DSL fragments into one Jot.

    Fragments that fail to parse are dropped by the bridge (logged
    here); a hard bridge failure raises `RecomposeError`.
    """
    if not TOOL_PATH.exists():
        raise RecomposeError(
            f"recompose tool missing at {TOOL_PATH}; the Docker build "
            "didn't include /app/tools/ correctly."
        )

    payload = {
        "lines": lines_by_pitch,
        "structure": {
            "initialTempo": structure.initial_tempo,
            "initialTimeSig": list(structure.initial_time_signature),
            "hasTempoChanges": structure.has_tempo_changes,
            "hasTimeSigChanges": structure.has_time_sig_changes,
            "bars": [
                {
                    "index": bar.index,
                    "timeSig": list(bar.time_signature),
                    "tempoBpm": bar.tempo_bpm,
                }
                for bar in structure.bars
            ],
        },
        "feetPitches": sorted(feet_pitches),
        "instrumentNames": PITCH_DISPLAY_NAMES,
    }

    try:
        result = subprocess.run(
            ["bun", "run", str(TOOL_PATH)],
            input=json.dumps(payload).encode("utf-8"),
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RecomposeError("Recompose bridge timed out") from exc
    except FileNotFoundError as exc:
        raise RecomposeError(f"`bun` is not installed in PATH: {exc}") from exc

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RecomposeError(
            stderr or "recompose bridge exited non-zero", stderr
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RecomposeError(
            f"Recompose bridge returned non-JSON: {result.stdout[:200]!r}"
        ) from exc

    dropped = data.get("dropped") or []
    if dropped:
        log.warning(
            "recompose: dropped %d unparseable instrument fragment(s): %s",
            len(dropped), dropped,
        )
    return str(data["dsl"])
