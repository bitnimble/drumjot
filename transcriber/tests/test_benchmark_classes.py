"""Tests for the cross-dataset drum class taxonomy mappings."""
from __future__ import annotations

from benchmarks.core.classes import (
    GM_PITCH_TO_CLASS,
    IDMT_LABEL_TO_CLASS,
    JOT_PITCH_TO_CLASS,
    MDB_LABEL_TO_CLASS,
    DrumClass,
)


def test_jot_pitch_three_class_mapping() -> None:
    assert JOT_PITCH_TO_CLASS["k"] == DrumClass.KD
    assert JOT_PITCH_TO_CLASS["s"] == DrumClass.SD
    assert JOT_PITCH_TO_CLASS["h"] == DrumClass.HH


def test_jot_extra_pitches_dropped_from_3_class() -> None:
    # 3-class metric ignores ride / crash / tom.
    assert JOT_PITCH_TO_CLASS["d"] is None
    assert JOT_PITCH_TO_CLASS["c"] is None
    assert JOT_PITCH_TO_CLASS["t"] is None


def test_gm_canonical_pitches() -> None:
    assert GM_PITCH_TO_CLASS[36] == DrumClass.KD
    assert GM_PITCH_TO_CLASS[38] == DrumClass.SD
    assert GM_PITCH_TO_CLASS[42] == DrumClass.HH
    # Open hi-hat folds into HH per ADT field convention.
    assert GM_PITCH_TO_CLASS[46] == DrumClass.HH


def test_gm_non_3class_pitches_absent() -> None:
    # Tom (41/43/45/47/48/50) and cymbals (49/51/57/59) are not mapped.
    for pitch in (41, 43, 45, 47, 48, 49, 50, 51, 57, 59):
        assert pitch not in GM_PITCH_TO_CLASS


def test_mdb_label_mapping() -> None:
    assert MDB_LABEL_TO_CLASS["KD"] == DrumClass.KD
    assert MDB_LABEL_TO_CLASS["SD"] == DrumClass.SD
    assert MDB_LABEL_TO_CLASS["SDD"] == DrumClass.SD  # snare drag folded in
    assert MDB_LABEL_TO_CLASS["OH"] == DrumClass.HH  # open HH folded in
    assert "TT" not in MDB_LABEL_TO_CLASS
    assert "CY" not in MDB_LABEL_TO_CLASS


def test_idmt_label_mapping() -> None:
    assert IDMT_LABEL_TO_CLASS == {
        "KD": DrumClass.KD,
        "SD": DrumClass.SD,
        "HH": DrumClass.HH,
    }
