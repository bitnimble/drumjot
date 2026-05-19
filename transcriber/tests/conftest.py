"""Shared test setup.

`jot_extract` resolves the bun bridge tool path from `JOT_TO_ONSETS_TOOL`
at import time, defaulting to the in-container `/app/...` path. Point it
at the repo copy so tests that exercise the real TS parser (recompose,
round-trip) work outside Docker. Set before any `app.pipeline` import.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = _REPO_ROOT / "transcriber" / "tools"
os.environ.setdefault(
    "JOT_TO_ONSETS_TOOL", str(_TOOLS / "jot_to_onsets.ts")
)
os.environ.setdefault(
    "RECOMPOSE_JOT_TOOL", str(_TOOLS / "recompose_jot.ts")
)
