"""Drum lane vocabulary for the onset-detection training set.

Expanded 11-lane set (was 5): kick, snare, side-stick, toms (merged),
the three hi-hat articulations (closed / pedal / open), ride, crash, and two
fold-up lanes for the sparse tail, misc cymbals (splash + china + ride-bell)
and misc percussion (cowbell + hand-clap + tambourine). General-MIDI
percussion notes fold into these lanes; anything outside the kit maps to None
and is dropped by callers.

Side stick (`ss`) is trained as its own lane and emitted to MIDI on its own
GM-37 track; the frontend folds it onto the snare track as an articulation at
Jot-load time (integration detail, not handled here).
"""
from __future__ import annotations

LANES: tuple[str, ...] = (
    "k", "s", "ss", "t", "hc", "hp", "ho", "rd", "cr", "mc", "mp",
)

LANE_NAMES: dict[str, str] = {
    "k": "kick",
    "s": "snare",
    "ss": "side stick",
    "t": "toms",
    "hc": "closed hi-hat",
    "hp": "pedal hi-hat",
    "ho": "open hi-hat",
    "rd": "ride",
    "cr": "crash",
    "mc": "misc cymbals (splash/china/ride-bell)",
    "mp": "misc percussion (cowbell/clap/tambourine)",
}

# General-MIDI percussion note -> lane.
_GM_NOTE_TO_LANE: dict[int, str] = {
    35: "k", 36: "k",
    37: "ss",                                  # side stick
    38: "s", 40: "s",
    39: "mp",                                  # hand clap
    41: "t", 43: "t", 45: "t", 47: "t", 48: "t", 50: "t",
    42: "hc",                                  # closed hi-hat
    44: "hp",                                  # pedal hi-hat
    46: "ho",                                  # open hi-hat
    49: "cr", 57: "cr",                        # crash 1 / 2
    51: "rd", 59: "rd",                        # ride 1 / 2
    52: "mc", 53: "mc", 55: "mc",              # china / ride bell / splash
    54: "mp", 56: "mp",                        # tambourine / cowbell
}


def lane_for_gm_note(note: int) -> str | None:
    """Return the drum lane for a General-MIDI percussion `note`, or None if
    the note is outside the kit vocabulary."""
    return _GM_NOTE_TO_LANE.get(note)
