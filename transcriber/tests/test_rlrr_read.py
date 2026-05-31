"""Unit tests for `app.scoring.rlrr_read`: `.rlrr` chart -> per-lane onset
seconds. Port of the chart side of src/rlrr (drums.ts / schema.ts)."""
from __future__ import annotations

import json

from app.scoring.rlrr_read import chart_from_rlrr, decode_rlrr_text

_INSTRUMENTS = [
    {"name": "BP_Kick_C_1", "class": "BP_Kick_C"},
    {"name": "BP_Snare_C_1", "class": "BP_Snare_C"},
    {"name": "BP_HiHat_C_1", "class": "BP_HiHat_C"},
    {"name": "BP_Crash15_C_1", "class": "BP_Crash15_C"},
    {"name": "BP_Cowbell_C_1", "class": "BP_Cowbell_C"},
]


def _rlrr(events: list[dict], **extra: object) -> dict:
    return {
        "version": 0.7,
        "recordingMetadata": {},
        "instruments": _INSTRUMENTS,
        "events": events,
        "bpmEvents": [],
        **extra,
    }


def test_maps_events_to_lanes() -> None:
    data = _rlrr(
        [
            {"name": "BP_Kick_C_1", "vel": 100, "loc": 0, "time": 0.0},
            {"name": "BP_Snare_C_1", "vel": 100, "loc": 0, "time": 0.5},
            {"name": "BP_HiHat_C_1", "vel": 80, "loc": 0, "time": 0.25},
            {"name": "BP_Crash15_C_1", "vel": 110, "loc": 0, "time": 1.0},
        ]
    )
    chart = chart_from_rlrr(data)
    assert chart.onsets_by_lane["k"] == [0.0]
    assert chart.onsets_by_lane["s"] == [0.5]
    assert chart.onsets_by_lane["h"] == [0.25]
    assert chart.onsets_by_lane["cy"] == [1.0]  # crash folds into cymbals
    assert chart.unmapped_events == 0


def test_event_time_accepts_string_and_number() -> None:
    data = _rlrr(
        [
            {"name": "BP_Kick_C_1", "vel": 100, "loc": 0, "time": "1.2345"},
            {"name": "BP_Kick_C_1", "vel": 100, "loc": 0, "time": 2.0},
        ]
    )
    chart = chart_from_rlrr(data)
    assert chart.onsets_by_lane["k"] == [1.2345, 2.0]


def test_unmapped_class_counted_not_fatal() -> None:
    data = _rlrr(
        [
            {"name": "BP_Kick_C_1", "vel": 100, "loc": 0, "time": 0.0},
            {"name": "BP_Cowbell_C_1", "vel": 100, "loc": 0, "time": 0.3},  # no lane
        ]
    )
    chart = chart_from_rlrr(data)
    assert chart.onsets_by_lane.get("k") == [0.0]
    assert chart.unmapped_events == 1


def test_onsets_sorted_ascending() -> None:
    data = _rlrr(
        [
            {"name": "BP_Kick_C_1", "vel": 100, "loc": 0, "time": 2.0},
            {"name": "BP_Kick_C_1", "vel": 100, "loc": 0, "time": 0.5},
            {"name": "BP_Kick_C_1", "vel": 100, "loc": 0, "time": 1.0},
        ]
    )
    assert chart_from_rlrr(data).onsets_by_lane["k"] == [0.5, 1.0, 2.0]


def test_resolve_class_falls_back_to_instance_name_regex() -> None:
    # Event references an instance not in `instruments`; the BP_<Class>_C_N
    # regex still recovers the class.
    data = _rlrr([{"name": "BP_Snare_C_9", "vel": 100, "loc": 0, "time": 0.0}])
    data["instruments"] = []
    assert chart_from_rlrr(data).onsets_by_lane["s"] == [0.0]


def test_audio_track_refs_extracted() -> None:
    data = _rlrr(
        [],
        audioFileData={"songTracks": ["song.ogg"], "drumTracks": ["drums.ogg"]},
    )
    chart = chart_from_rlrr(data)
    assert chart.song_tracks == ["song.ogg"]
    assert chart.drum_tracks == ["drums.ogg"]


def test_decode_rlrr_text_handles_encodings() -> None:
    doc = {"hello": "world"}
    text = json.dumps(doc)
    # UTF-8, UTF-8 BOM, UTF-16 (BOM), bare UTF-16LE all round-trip.
    assert json.loads(decode_rlrr_text(text.encode("utf-8"))) == doc
    assert json.loads(decode_rlrr_text(b"\xef\xbb\xbf" + text.encode("utf-8"))) == doc
    assert json.loads(decode_rlrr_text(text.encode("utf-16"))) == doc
    assert json.loads(decode_rlrr_text(text.encode("utf-16-le"))) == doc
