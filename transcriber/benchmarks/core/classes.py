"""Cross-dataset drum class taxonomy used by the benchmark harness.

The benchmark reports the standard 3-class ADT metric (KD/SD/HH) so
results are directly comparable to the numbers in the N2N paper and
the rest of the field. Each loader translates its dataset's native
labels into this taxonomy; the Jot→events bridge does the same for our
predicted output.
"""
from __future__ import annotations

from enum import Enum


class DrumClass(str, Enum):
    """The three drum classes scored by the benchmark.

    Inheriting from `str` makes these JSON-serialisable and lets us use
    them as dict keys without a custom encoder.
    """

    KD = "KD"
    SD = "SD"
    HH = "HH"


ALL_CLASSES: tuple[DrumClass, ...] = (DrumClass.KD, DrumClass.SD, DrumClass.HH)


# --- Drumjot pitch slug → DrumClass ---
#
# Mirrors `app/pipeline/separate.py:STEM_NAME_TO_PITCH`. Pitches we
# don't score in the 3-class metric (`d`=ride, `c`=crash, `t`=tom) map
# to None and are dropped before scoring.
JOT_PITCH_TO_CLASS: dict[str, DrumClass | None] = {
    "k": DrumClass.KD,
    "s": DrumClass.SD,
    "h": DrumClass.HH,
    "d": None,
    "c": None,
    "t": None,
}


# --- General MIDI percussion → DrumClass (used by the E-GMD loader) ---
#
# Standard 3-class folding used by the field:
#   - 35/36 → KD
#   - 37 (side stick) / 38 / 40 → SD
#   - 42 (closed) / 44 (pedal) / 46 (open) → HH
# Everything else is ignored.
GM_PITCH_TO_CLASS: dict[int, DrumClass] = {
    35: DrumClass.KD,
    36: DrumClass.KD,
    37: DrumClass.SD,
    38: DrumClass.SD,
    40: DrumClass.SD,
    42: DrumClass.HH,
    44: DrumClass.HH,
    46: DrumClass.HH,
}


# --- MDB Drums annotation label → DrumClass ---
#
# MDB Drums also uses `TT`, `CY`, `CB`, ... — those are dropped for the
# 3-class metric. `SDD` (snare drag) and `SDB` (snare buzz) are folded
# into SD; `OH` (open hi-hat) is folded into HH.
MDB_LABEL_TO_CLASS: dict[str, DrumClass] = {
    "KD": DrumClass.KD,
    "SD": DrumClass.SD,
    "SDD": DrumClass.SD,
    "SDB": DrumClass.SD,
    "HH": DrumClass.HH,
    "OH": DrumClass.HH,
}


# --- IDMT-SMT-Drums XML <instrument> → DrumClass ---
#
# IDMT only annotates the three classes natively.
IDMT_LABEL_TO_CLASS: dict[str, DrumClass] = {
    "KD": DrumClass.KD,
    "SD": DrumClass.SD,
    "HH": DrumClass.HH,
}
