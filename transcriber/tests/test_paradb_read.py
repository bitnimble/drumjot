"""Unit tests for `app.scoring.paradb_read`: a ParaDB `.zip` pack ->
chosen chart + extracted audio bytes. Port of paradb.ts::loadParadbZip on
stdlib zipfile."""
from __future__ import annotations

import io
import json
import zipfile

import pytest

from app.scoring.paradb_read import load_paradb_bytes

_INSTRUMENTS = [{"name": "BP_Kick_C_1", "class": "BP_Kick_C"}]
_EVENTS = [{"name": "BP_Kick_C_1", "vel": 100, "loc": 0, "time": 0.5}]


def _rlrr(complexity: int, song: list[str], drums: list[str]) -> bytes:
    doc = {
        "version": 0.7,
        "recordingMetadata": {"complexity": complexity},
        "audioFileData": {"songTracks": song, "drumTracks": drums},
        "instruments": _INSTRUMENTS,
        "events": _EVENTS,
        "bpmEvents": [],
    }
    return json.dumps(doc).encode("utf-8")


def _make_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_picks_highest_complexity_and_extracts_audio() -> None:
    pack = _make_zip(
        {
            "Song - Easy.rlrr": _rlrr(1, ["song.ogg"], ["drums.ogg"]),
            "Song - Expert.rlrr": _rlrr(4, ["song.ogg"], ["drums.ogg"]),
            "song.ogg": b"SONGDATA",
            "drums.ogg": b"DRUMDATA",
        }
    )
    result = load_paradb_bytes(pack)
    assert "Expert" in result.rlrr_name
    assert result.chart.onsets_by_lane["k"] == [0.5]
    assert [(a.name, a.data) for a in result.song_audio] == [("song.ogg", b"SONGDATA")]
    assert [(a.name, a.data) for a in result.drum_audio] == [("drums.ogg", b"DRUMDATA")]


def test_difficulty_name_breaks_complexity_tie() -> None:
    pack = _make_zip(
        {
            "chart_medium.rlrr": _rlrr(2, ["s.ogg"], []),
            "chart_expert.rlrr": _rlrr(2, ["s.ogg"], []),  # same complexity
            "s.ogg": b"X",
        }
    )
    assert "expert" in load_paradb_bytes(pack).rlrr_name.lower()


def test_missing_drum_track_yields_song_only() -> None:
    pack = _make_zip(
        {
            "chart.rlrr": _rlrr(3, ["song.ogg"], []),
            "song.ogg": b"SONG",
        }
    )
    result = load_paradb_bytes(pack)
    assert [a.name for a in result.song_audio] == ["song.ogg"]
    assert result.drum_audio == []


def test_audio_resolved_by_basename_case_insensitive() -> None:
    # Author references a bare/cased name; the entry sits in a folder.
    pack = _make_zip(
        {
            "chart.rlrr": _rlrr(3, ["Song.OGG"], ["Audio/Drums.ogg"]),
            "audio/song.ogg": b"S",
            "audio/drums.ogg": b"D",
        }
    )
    result = load_paradb_bytes(pack)
    assert [a.name for a in result.song_audio] == ["song.ogg"]
    assert [a.name for a in result.drum_audio] == ["drums.ogg"]


def test_no_rlrr_raises() -> None:
    pack = _make_zip({"readme.txt": b"hello", "song.ogg": b"S"})
    with pytest.raises(ValueError, match="rlrr"):
        load_paradb_bytes(pack)
