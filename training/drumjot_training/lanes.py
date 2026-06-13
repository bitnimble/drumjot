"""Drum lane vocabulary for the onset-detection training set.

10-lane set: kick, snare, side-stick, toms (merged), the three hi-hat
articulations (closed / pedal / open), ride, crash, and misc cymbals
(splash + china + ride-bell). General-MIDI percussion notes fold into these
lanes; anything outside the kit maps to None and is dropped by callers.

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
    "k", "s", "ss", "t", "hc", "hp", "ho", "rd", "cr", "mc",
)

# Catch-all "negative" lane for real percussive onsets that belong to NO output
# lane: the removed `mp` (cowbell/clap/tambourine) PLUS all non-kit aux percussion
# (congas, bongos, timbales, agogo, claves, woodblocks, guiro, cuica, triangle,
# ...). It gets no head and is never predicted, but the readers still emit its
# onset times (bucketed here) so the loss can treat those frames as HARD NEGATIVES
# for every output lane -- "a hit happened, and it's none of your drums" -- instead
# of silently discarding the false-trigger signal when a class is dropped from the
# model. See train.build_negative_targets / negative_sibling_matrix.
NEGATIVE_LANES: tuple[str, ...] = ("x",)
# Rows the loss-weighting machinery sees: the output lanes plus the ghost lane(s).
WEIGHT_LANES: tuple[str, ...] = LANES + NEGATIVE_LANES

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
}

# General-MIDI percussion note -> lane. Clap (39), tambourine (54) and cowbell
# (56) are deliberately unmapped (the removed `mp` lane).
_GM_NOTE_TO_LANE: dict[int, str] = {
    35: "k", 36: "k",
    37: "ss",                                  # side stick
    38: "s", 40: "s",
    41: "t", 43: "t", 45: "t", 47: "t", 48: "t", 50: "t",
    42: "hc",                                  # closed hi-hat
    44: "hp",                                  # pedal hi-hat
    46: "ho",                                  # open hi-hat
    49: "cr", 57: "cr",                        # crash 1 / 2
    51: "rd", 59: "rd",                        # ride 1 / 2
    52: "mc", 53: "mc", 55: "mc",              # china / ride bell / splash
}


def lane_for_gm_note(note: int) -> str | None:
    """Return the drum lane for a General-MIDI percussion `note`, or None if
    the note is outside the kit vocabulary."""
    return _GM_NOTE_TO_LANE.get(note)


# Non-kit GM percussion -> the catch-all negative lane (NEGATIVE_LANES): hand
# clap (39), tambourine (54), cowbell (56), vibraslap (58), and the latin/aux
# percussion block bongos..open-triangle (60-81). These are genuine onsets the
# kit map drops; they become hard negatives, never outputs.
_GM_NOTE_TO_NEG: dict[int, str] = {n: "x" for n in (39, 54, 56, 58, *range(60, 82))}


def negative_lane_for_gm_note(note: int) -> str | None:
    """Catch-all negative lane for a non-kit GM percussion `note`, else None.
    Only fires for notes that are NOT an output lane (so it never shadows the kit)."""
    if note in _GM_NOTE_TO_LANE:
        return None
    return _GM_NOTE_TO_NEG.get(note)


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
    "hc": ("ho", "hp", "cr", "rd", "mc"),
    "hp": ("hc", "ho", "s", "k"),                # hp fired heavily on snare/kick stems
    "ho": ("hc", "hp", "cr", "rd", "mc"),
    "rd": ("hc", "ho", "hp", "cr", "mc"),        # hat->ride: the #1 measured leak
    "cr": ("rd", "mc", "ho", "hc"),
    "mc": ("cr", "rd", "hc", "ho"),
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


# Which output lanes each NEGATIVE (ghost) lane is a hard negative for. The
# catch-all "x" is a hard negative for EVERY output lane ("this percussive frame
# is none of your drums"). A map (not just "all") so a future split of the ghost
# lane into e.g. low vs metallic aux-perc can target specific lanes.
NEGATIVE_SIBLINGS: dict[str, tuple[str, ...]] = {ln: NEGATIVE_LANES for ln in LANES}


def negative_sibling_matrix(
    lanes: tuple[str, ...] = LANES, negatives: tuple[str, ...] = NEGATIVE_LANES
) -> list[list[bool]]:
    """(len(lanes), len(negatives)) boolean matrix: out[i][j] is True when the
    ghost lane `negatives[j]` is a hard negative for output lane `lanes[i]`.
    Pairs with `sibling_matrix`: the loss extends each lane's sibling activity
    with these ghost columns (the dropped-percussion negatives)."""
    nidx = {ln: j for j, ln in enumerate(negatives)}
    out = [[False] * len(negatives) for _ in lanes]
    for i, ln in enumerate(lanes):
        for g in NEGATIVE_SIBLINGS.get(ln, ()):
            if g in nidx:
                out[i][nidx[g]] = True
    return out
