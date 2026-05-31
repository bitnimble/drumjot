"""Read a MIDI file into per-lane onset seconds.

Iterating `mido.MidiFile` yields messages merged across all tracks in
absolute order with `msg.time` already in seconds and tempo applied, so a
conductor track's `set_tempo` correctly times a separate drum track. We
must NOT iterate `mid.tracks` (per-track, raw ticks, default 120 BPM) -
that is the multi-track mis-timing bug from research §4.

Channel policy: prefer channel 9 (GM drums); if a file has no channel-9
note-ons, fall back to all channels, since external charts are less
disciplined than our own output. MIDI is a secondary test input (the
primary corpus is ParaDB).
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import mido

from app.scoring.lanes import lane_for_gm_note

_DRUM_CHANNEL = 9  # GM percussion channel (MIDI channel 10, 0-indexed)


@dataclass
class MidiChart:
    onsets_by_lane: dict[str, list[float]] = field(default_factory=dict)
    unmapped_notes: int = 0
    used_all_channels: bool = False


def onsets_from_midi(path: Path) -> MidiChart:
    return _onsets(mido.MidiFile(filename=str(path)))


def onsets_from_midi_bytes(data: bytes) -> MidiChart:
    return _onsets(mido.MidiFile(file=io.BytesIO(data)))


def _onsets(mid: mido.MidiFile) -> MidiChart:
    # First pass: every note-on (velocity > 0) at its absolute time in
    # seconds, with its channel, in merged tempo-aware order.
    events: list[tuple[float, int, int]] = []  # (time_sec, channel, note)
    t = 0.0
    for msg in mid:
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            events.append((t, msg.channel, msg.note))

    has_ch9 = any(channel == _DRUM_CHANNEL for _, channel, _ in events)
    kept = [e for e in events if e[1] == _DRUM_CHANNEL] if has_ch9 else events

    onsets_by_lane: dict[str, list[float]] = {}
    unmapped = 0
    for time_sec, _channel, note in kept:
        lane = lane_for_gm_note(note)
        if lane is None:
            unmapped += 1
            continue
        onsets_by_lane.setdefault(lane, []).append(time_sec)

    if not onsets_by_lane:
        raise ValueError("no drum channel found (no drum-mapped note-ons in the MIDI)")

    for times in onsets_by_lane.values():
        times.sort()

    return MidiChart(
        onsets_by_lane=onsets_by_lane,
        unmapped_notes=unmapped,
        used_all_channels=not has_ch9,
    )
