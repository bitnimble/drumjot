"""MIDI -> `OnsetEvent` list, for both ground-truth files and the
`filter` pathway's predicted `prediction.mid`.

The 3-class folding (`GM_PITCH_TO_CLASS`) is identical to the E-GMD
ground-truth path, so a predicted MIDI is scored *symmetrically* with
the reference — no Jot in the loop. `app/pipeline/onsets_midi.py`
renders kick=36, snare=38, hi-hat=42 on channel 9, which fold to
KD/SD/HH here; ride/crash/tom (49/51/50) map to None and drop out,
matching the 3-class metric the rest of the benchmark reports.
"""
from __future__ import annotations

import io
from pathlib import Path

import mido

from .classes import GM_PITCH_TO_CLASS
from .events import OnsetEvent


def _events_from_mido(mid: mido.MidiFile) -> list[OnsetEvent]:
    """Extract drum onsets from a parsed GM MIDI file, mapped to the
    3-class taxonomy.

    Iterates note-on events (velocity > 0) on channel 9 (0-indexed
    drums) across all tracks, accumulating ticks-since-start with the
    file's current tempo to recover absolute seconds.
    """
    events: list[OnsetEvent] = []
    for track in mid.tracks:
        tempo = 500_000  # default 120 BPM if no set_tempo seen
        elapsed = 0.0
        for msg in track:
            if msg.time:
                elapsed += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
            if msg.type == "set_tempo":
                tempo = msg.tempo
                continue
            if msg.type == "note_on" and msg.velocity > 0 and msg.channel == 9:
                drum_class = GM_PITCH_TO_CLASS.get(msg.note)
                if drum_class is None:
                    continue
                events.append(OnsetEvent(time=elapsed, drum_class=drum_class))
    events.sort(key=lambda e: (e.time, e.drum_class.value))
    return events


def midi_file_to_events(midi_path: Path) -> list[OnsetEvent]:
    """Parse a MIDI file on disk into 3-class onset events."""
    return _events_from_mido(mido.MidiFile(str(midi_path)))


def midi_bytes_to_events(data: bytes) -> list[OnsetEvent]:
    """Parse in-memory MIDI bytes into 3-class onset events."""
    return _events_from_mido(mido.MidiFile(file=io.BytesIO(data)))
