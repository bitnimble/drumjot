"""Drum-lane vocabulary and the two source pitch tables.

The scorer works in the five lanes ADTOF's drum-onset detector exposes
(`app/pipeline/adtof_onsets.py::_LANE_FOR_PITCH`): kick, snare, toms,
hi-hat, and a merged cymbal lane (ADTOF has no separate ride/crash class).

Both pitch tables here are Python ports of TypeScript sources, the canonical
and maintained mappings:

  * `GM_NOTE_TO_PITCH` <- `src/midi/gm.ts::GM_PERCUSSION`
  * `PARADIDDLE_CLASS_TO_PITCH` <- `src/rlrr/drums.ts::CLASS_TO_DRUM`

They are NOT the stale 3-class `benchmarks/core/classes.py` fold. The ports
are guarded by drift tests (`tests/test_scoring_lanes.py`) that parse the
`.ts` files, so a future TS change that isn't mirrored here fails CI.
"""
from __future__ import annotations

import re

# The five scoring lanes, matching ADTOF's output classes.
LANES: tuple[str, ...] = ("k", "s", "t", "h", "cy")

# MIDI note number -> Drumjot DSL pitch. Port of src/midi/gm.ts GM_PERCUSSION.
GM_NOTE_TO_PITCH: dict[int, str] = {
    35: "k",  # Acoustic Bass Drum
    36: "k",  # Kick
    37: "s",  # Side Stick
    38: "s",  # Snare
    39: "p",  # Hand Clap
    40: "s",  # Electric Snare
    41: "f",  # Low Floor Tom
    42: "h",  # Closed Hi-Hat
    43: "f",  # High Floor Tom
    44: "h",  # Pedal Hi-Hat
    45: "t",  # Low Tom
    46: "h",  # Open Hi-Hat
    47: "t",  # Low-Mid Tom
    48: "t",  # Hi-Mid Tom
    49: "c",  # Crash Cymbal 1
    50: "t",  # High Tom
    51: "d",  # Ride Cymbal 1
    52: "c",  # Chinese Cymbal
    53: "d",  # Ride Bell
    54: "b",  # Tambourine
    55: "c",  # Splash Cymbal
    56: "b",  # Cowbell
    57: "c",  # Crash Cymbal 2
    59: "d",  # Ride Cymbal 2
}

# Paradiddle drum class -> Drumjot DSL pitch. Port of
# src/rlrr/drums.ts CLASS_TO_DRUM (pitch column only).
PARADIDDLE_CLASS_TO_PITCH: dict[str, str] = {
    "BP_HiHat_C": "h",
    "BP_Snare_C": "s",
    "BP_Kick_C": "k",
    "BP_Crash13_C": "c",
    "BP_Crash15_C": "c",
    "BP_Crash17_C": "c",
    "BP_China15_C": "c",
    "BP_FloorTom_C": "f",
    "BP_Ride17_C": "d",
    "BP_Ride20_C": "d",
    "BP_Tom1_C": "t",
    "BP_Tom2_C": "t",
    "BP_Timpani1_C": "i",
    "BP_Timpani2_C": "i",
    "BP_Timpani3_C": "i",
    "BP_Triangle_C": "n",
    "BP_BongoH_C": "n",
    "BP_BongoL_C": "n",
    "BP_Xylophone_C": "y",
    "BP_Marimba_C": "y",
    "BP_Glockenspiel_C": "e",
    "BP_Gong_C": "q",
    "BP_Tambourine1_C": "b",
    "BP_Tambourine2_C": "b",
    "BP_Cowbell_C": "b",
}

# DSL pitch -> scoring lane. Pitches absent here (clap `p`, aux perc `b`,
# timpani `i`, triangle/bongo `n`, mallets `y`/`e`, gong `q`, and any
# fallback letter) have no ADTOF lane and are dropped.
_PITCH_TO_LANE: dict[str, str] = {
    "k": "k",
    "s": "s",
    "t": "t",
    "f": "t",  # floor tom folds into the toms lane
    "h": "h",
    "c": "cy",  # crash
    "d": "cy",  # ride merges with crash (ADTOF has no separate class)
}

_INSTANCE_NAME_RE = re.compile(r"^(BP_.+_C)_\d+$")


def lane_for_pitch(pitch: str) -> str | None:
    """Fold a Drumjot DSL pitch into a scoring lane, or None if it has no
    ADTOF lane."""
    return _PITCH_TO_LANE.get(pitch)


def lane_for_gm_note(note: int) -> str | None:
    """Map a General-MIDI percussion note number to a scoring lane, or None
    when the note is outside the GM table or folds to a laneless pitch."""
    pitch = GM_NOTE_TO_PITCH.get(note)
    return lane_for_pitch(pitch) if pitch is not None else None


def lane_for_paradiddle_class(cls: str) -> str | None:
    """Map a Paradiddle drum class (`BP_<Class>_C`) to a scoring lane, or
    None when the class is unknown or folds to a laneless pitch."""
    pitch = PARADIDDLE_CLASS_TO_PITCH.get(cls)
    return lane_for_pitch(pitch) if pitch is not None else None


def class_from_instance_name(name: str) -> str | None:
    """Extract the drum class from an instrument instance name
    (`BP_Snare_C_1` -> `BP_Snare_C`). None when the name lacks the trailing
    `_<idx>`. Port of `src/rlrr/drums.ts::instanceNameToClass`."""
    m = _INSTANCE_NAME_RE.match(name)
    return m.group(1) if m else None
