"""Canonical pipeline stage names in one torch-free place, so the runner
(`STAGE_ORDER`) and the comms progress reporter (`LIVE_STAGES`) can't silently
drift. comms deliberately avoids importing the heavy runner, so this module must
stay dependency-light (stdlib only)."""
from __future__ import annotations

from enum import StrEnum


class Stage(StrEnum):
    """Named pipeline stages, ordered by data dependency."""

    STEMS_ALL = "stems_all"
    STEMS_PER = "stems_per"
    BEATS = "beats"
    ONSETS = "onsets"
    FILTER = "filter"
    QUANTISE = "quantise"
    TRANSCRIBE = "transcribe"


STAGE_ORDER: list[Stage] = [
    Stage.STEMS_ALL,
    Stage.STEMS_PER,
    Stage.BEATS,
    Stage.ONSETS,
    Stage.FILTER,
    Stage.QUANTISE,
    Stage.TRANSCRIBE,
]


def stage_index(s: Stage) -> int:
    return STAGE_ORDER.index(s)
