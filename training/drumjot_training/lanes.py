"""Drum lane vocabulary for the onset-detection training set.

8-lane set: kick, snare, side-stick, toms (merged), the two hi-hat
articulations (closed / open), ride, and crash. General-MIDI percussion notes
fold into these lanes; anything outside the kit maps to None and is dropped by
callers. Pedal hi-hat (GM 44) folds into closed hi-hat (`hc`): there's no
separate pedal-hat lane.

`mc` (misc cymbals: splash / china / ride-bell) was REMOVED (2026-06): the
per-stem separators don't isolate these rare add-on cymbals and they're low
musical priority. Ride-bell is physically part of the ride cymbal, so it now
folds into `rd`; splash + china map to None (dropped, like misc percussion).

`mp` (misc percussion: cowbell / clap / tambourine) was REMOVED (2026-06): it
has no per-instrument stem, scored ~noise on val, and was the top
cross-instrument leak destination on four of five stems, a garbage-attractor
lane that taught the model to fire on anything percussive. Its source classes
now map to None.

Side stick (`ss`) is trained as its own lane and emitted to MIDI on its own
GM-37 track; the frontend folds it onto the snare track as an articulation at
Jot-load time (integration detail, not handled here).
"""
from __future__ import annotations

LANES: tuple[str, ...] = (
    "k", "s", "ss", "t", "hc", "ho", "rd", "cr",
)

LANE_NAMES: dict[str, str] = {
    "k": "kick",
    "s": "snare",
    "ss": "side stick",
    "t": "toms",
    "hc": "closed hi-hat",
    "ho": "open hi-hat",
    "rd": "ride",
    "cr": "crash",
}

# General-MIDI percussion note -> lane. Clap (39), tambourine (54), cowbell
# (56) are deliberately unmapped (the removed `mp` lane); china (52) and
# splash (55) are unmapped (the removed `mc` lane). Ride bell (53) folds into
# `rd` since it's the same physical cymbal as the ride.
_GM_NOTE_TO_LANE: dict[int, str] = {
    35: "k", 36: "k",
    37: "ss",                                  # side stick
    38: "s", 40: "s",
    41: "t", 43: "t", 45: "t", 47: "t", 48: "t", 50: "t",
    42: "hc",                                  # closed hi-hat
    44: "hc",                                  # pedal hi-hat (folds into closed)
    46: "ho",                                  # open hi-hat
    49: "cr", 57: "cr",                        # crash 1 / 2
    51: "rd", 59: "rd", 53: "rd",              # ride 1 / 2 / bell
}


def lane_for_gm_note(note: int) -> str | None:
    """Return the drum lane for a General-MIDI percussion `note`, or None if
    the note is outside the kit vocabulary."""
    return _GM_NOTE_TO_LANE.get(note)


# Acoustically-confusable siblings, per lane: the lanes whose hits this lane's
# head is empirically prone to firing on. SEEDED FROM MEASURED ParaDB leakage
# (eval_paradb cross-instrument tables, 2026-06), not instrument taxonomy: e.g.
# the hi-hat stem triggering the ride head was the single largest leak, so `rd`
# lists the hats. Used by the loss to up-weight frames where a sibling is active
# (hard negatives when this lane is silent there; harder-positive reward when it
# genuinely co-occurs). Tunable as new leakage data comes in.
CONFUSABLE: dict[str, tuple[str, ...]] = {
    "k": ("t",),                                 # tom stem -> kick leak
    "s": ("ss",),
    "ss": ("s", "k"),
    "t": ("k",),
    "hc": ("ho", "cr", "rd"),
    "ho": ("hc", "cr", "rd"),
    "rd": ("hc", "ho", "cr"),                     # hat->ride: the #1 measured leak
    "cr": ("rd", "ho", "hc"),
}


def sibling_matrix(lanes: tuple[str, ...] = LANES) -> list[list[bool]]:
    """(n_lanes, n_lanes) boolean matrix: S[i][j] is True when `lanes[j]` is a
    confusable sibling of `lanes[i]`. Row-major over `lanes` order; pure Python
    so it stays importable without numpy/torch."""
    idx = {ln: i for i, ln in enumerate(lanes)}
    out = [[False] * len(lanes) for _ in lanes]
    for ln, sibs in CONFUSABLE.items():
        if ln not in idx:
            continue
        for s in sibs:
            if s in idx:
                out[idx[ln]][idx[s]] = True
    return out
