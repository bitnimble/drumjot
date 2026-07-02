"""Unit + drift-guard tests for `app.scoring.lanes`.

The lane fold and the two pitch tables are Python ports of TypeScript
sources (`frontend/src/midi/gm.ts`, `frontend/src/schema/rlrr/drums.ts`). The
`drift-guard` tests parse those `.ts` files and assert the Python ports still
match them, so a future TS change that isn't mirrored here fails CI loudly.
"""
from __future__ import annotations

import re
from pathlib import Path

from app.scoring.lanes import (
    GM_NOTE_TO_PITCH,
    LANES,
    PARADIDDLE_CLASS_TO_PITCH,
    class_from_instance_name,
    lane_for_gm_note,
    lane_for_paradiddle_class,
    lane_for_pitch,
)

_REPO = Path(__file__).resolve().parents[2]


def test_lanes_are_the_five_adtof_classes() -> None:
    assert LANES == ("k", "s", "t", "h", "cy")


def test_pitch_fold_maps_kit_pitches_to_lanes() -> None:
    assert lane_for_pitch("k") == "k"
    assert lane_for_pitch("s") == "s"
    assert lane_for_pitch("t") == "t"
    assert lane_for_pitch("f") == "t"  # floor tom folds into toms
    assert lane_for_pitch("h") == "h"
    assert lane_for_pitch("c") == "cy"  # crash
    assert lane_for_pitch("d") == "cy"  # ride merges with crash


def test_pitch_fold_drops_pitches_with_no_adtof_lane() -> None:
    for pitch in ("p", "b", "i", "n", "y", "e", "q", "z", "?"):
        assert lane_for_pitch(pitch) is None


def test_gm_note_to_lane() -> None:
    assert lane_for_gm_note(36) == "k"
    assert lane_for_gm_note(38) == "s"
    assert lane_for_gm_note(41) == "t"  # low floor tom
    assert lane_for_gm_note(48) == "t"
    assert lane_for_gm_note(42) == "h"
    assert lane_for_gm_note(46) == "h"
    assert lane_for_gm_note(49) == "cy"  # crash
    assert lane_for_gm_note(51) == "cy"  # ride
    assert lane_for_gm_note(39) is None  # hand clap
    assert lane_for_gm_note(54) is None  # tambourine
    assert lane_for_gm_note(60) is None  # outside the GM percussion table


def test_paradiddle_class_to_lane() -> None:
    assert lane_for_paradiddle_class("BP_Kick_C") == "k"
    assert lane_for_paradiddle_class("BP_Snare_C") == "s"
    assert lane_for_paradiddle_class("BP_HiHat_C") == "h"
    assert lane_for_paradiddle_class("BP_FloorTom_C") == "t"
    assert lane_for_paradiddle_class("BP_Tom1_C") == "t"
    assert lane_for_paradiddle_class("BP_Crash15_C") == "cy"
    assert lane_for_paradiddle_class("BP_Ride17_C") == "cy"
    assert lane_for_paradiddle_class("BP_Cowbell_C") is None  # aux perc
    assert lane_for_paradiddle_class("BP_Timpani1_C") is None
    assert lane_for_paradiddle_class("BP_NotARealClass_C") is None


def test_class_from_instance_name() -> None:
    assert class_from_instance_name("BP_Snare_C_1") == "BP_Snare_C"
    assert class_from_instance_name("BP_HiHat_C_12") == "BP_HiHat_C"
    assert class_from_instance_name("BP_Kick_C") is None  # no trailing _<idx>
    assert class_from_instance_name("garbage") is None


# ---- Drift guards: parse the TS sources and compare to the Python ports ----


def _parse_gm_ts() -> dict[int, str]:
    """Pull `<note>: { lane: '<x>'` rows out of gm.ts's GM_PERCUSSION."""
    text = (_REPO / "frontend" / "src" / "midi" / "gm.ts").read_text(encoding="utf-8")
    pairs = re.findall(r"^\s*(\d+):\s*\{\s*lane:\s*'([a-z])'", text, re.MULTILINE)
    return {int(note): pitch for note, pitch in pairs}


def _parse_drums_ts() -> dict[str, str]:
    """Pull `<class>: { lane: '<x>'` rows out of drums.ts's CLASS_TO_DRUM."""
    text = (_REPO / "frontend" / "src" / "schema" / "rlrr" / "drums.ts").read_text(encoding="utf-8")
    pairs = re.findall(r"^\s*(BP_\w+):\s*\{\s*lane:\s*'([a-z])'", text, re.MULTILINE)
    return {cls: pitch for cls, pitch in pairs}


def test_gm_table_matches_gm_ts() -> None:
    ts = _parse_gm_ts()
    assert ts, "failed to parse any GM_PERCUSSION rows from gm.ts"
    assert ts == GM_NOTE_TO_PITCH


def test_paradiddle_table_matches_drums_ts() -> None:
    ts = _parse_drums_ts()
    assert ts, "failed to parse any CLASS_TO_DRUM rows from drums.ts"
    assert ts == PARADIDDLE_CLASS_TO_PITCH
