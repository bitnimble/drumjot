"""Read a Paradiddle `.rlrr` chart into per-lane onset seconds.

A `.rlrr` is a JSON document; `events` are drum hits at absolute recording
seconds (`event.time`, number or 4-decimal string; `loc` is a hit-zone
index, not a timestamp). Each event names an instrument *instance*, which
the `instruments[]` array maps to a drum class; the class folds to a
scoring lane. We read raw `time` and never the quantised musical grid that
`rlrr_to_jot` builds. Port of the chart side of src/rlrr (schema.ts,
drums.ts, the encoding sniff in paradb.ts).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.scoring.lanes import class_from_instance_name, lane_for_paradiddle_class


@dataclass
class RlrrChart:
    onsets_by_lane: dict[str, list[float]] = field(default_factory=dict)
    unmapped_events: int = 0
    song_tracks: list[str] = field(default_factory=list)
    drum_tracks: list[str] = field(default_factory=list)


def decode_rlrr_text(raw: bytes) -> str:
    """Decode `.rlrr` bytes to text. Paradiddle (a Unity/Windows app) writes
    UTF-8, UTF-8-with-BOM, or UTF-16 (LE/BE, sometimes BOM-less). Port of
    `paradb.ts::decodeRlrrText`. Unlike JS `TextDecoder`, Python's
    `bytes.decode` keeps a leading BOM character, so we strip it explicitly."""
    if raw[:3] == b"\xef\xbb\xbf":
        text = raw.decode("utf-8")
    elif raw[:2] == b"\xff\xfe":
        text = raw.decode("utf-16-le")
    elif raw[:2] == b"\xfe\xff":
        text = raw.decode("utf-16-be")
    # BOM-less UTF-16: an ASCII first char leaves a NUL in the other byte.
    elif len(raw) >= 2 and raw[0] != 0 and raw[1] == 0:
        text = raw.decode("utf-16-le")
    elif len(raw) >= 2 and raw[0] == 0 and raw[1] != 0:
        text = raw.decode("utf-16-be")
    else:
        text = raw.decode("utf-8")
    return text[1:] if text and ord(text[0]) == 0xFEFF else text


def load_rlrr(path: Path) -> dict[str, Any]:
    """Read + decode + parse a `.rlrr` file from disk."""
    return parse_rlrr_bytes(path.read_bytes())


def parse_rlrr_bytes(raw: bytes) -> dict[str, Any]:
    return json.loads(decode_rlrr_text(raw))


def _event_time_seconds(ev: dict[str, Any]) -> float:
    """Read `event.time` whether it arrives as a number or a string."""
    t = ev.get("time", 0.0)
    return float(t)


def chart_from_rlrr(data: dict[str, Any]) -> RlrrChart:
    """Fold a parsed `.rlrr` document into per-lane ascending onset seconds.
    Events whose class has no scoring lane (aux percussion, mallets, ...) are
    counted in `unmapped_events`, never fatal."""
    name_to_class: dict[str, str] = {
        inst["name"]: inst["class"]
        for inst in data.get("instruments", [])
        if isinstance(inst, dict) and "name" in inst and "class" in inst
    }

    onsets_by_lane: dict[str, list[float]] = {}
    unmapped = 0
    for ev in data.get("events", []):
        name = ev.get("name", "")
        cls = name_to_class.get(name) or class_from_instance_name(name)
        lane = lane_for_paradiddle_class(cls) if cls else None
        if lane is None:
            unmapped += 1
            continue
        onsets_by_lane.setdefault(lane, []).append(_event_time_seconds(ev))

    for times in onsets_by_lane.values():
        times.sort()

    audio = data.get("audioFileData") or {}
    return RlrrChart(
        onsets_by_lane=onsets_by_lane,
        unmapped_events=unmapped,
        song_tracks=list(audio.get("songTracks") or []),
        drum_tracks=list(audio.get("drumTracks") or []),
    )
