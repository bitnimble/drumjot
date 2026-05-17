"""E-GMD (Expanded Groove MIDI Dataset) loader.

Reads `e-gmd-v1.0.0.csv` and yields the test split (configurable via
`--split`). Ground truth comes from the paired MIDI file's drum-channel
note-on events, mapped to KD/SD/HH via `GM_PITCH_TO_CLASS`.
"""
from __future__ import annotations

import csv
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import mido

from ..core.classes import GM_PITCH_TO_CLASS
from ..core.events import OnsetEvent
from .base import LoadedTrack

log = logging.getLogger(__name__)

CSV_NAME = "e-gmd-v1.0.0.csv"


@dataclass
class EgmdLoader:
    name: str = "e-gmd"
    split: str = "test"

    def iter_tracks(self, root: Path) -> Iterator[LoadedTrack]:
        csv_path = root / CSV_NAME
        if not csv_path.exists():
            raise FileNotFoundError(
                f"E-GMD CSV missing at {csv_path}. "
                f"Paste the extracted contents of e-gmd-v1.0.0.zip into {root}."
            )

        with csv_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("split") != self.split:
                    continue
                audio_rel = row.get("audio_filename")
                midi_rel = row.get("midi_filename")
                if not audio_rel or not midi_rel:
                    continue

                audio_path = root / audio_rel
                midi_path = root / midi_rel
                if not audio_path.exists():
                    log.warning("E-GMD: audio missing, skipping: %s", audio_path)
                    continue
                if not midi_path.exists():
                    log.warning("E-GMD: midi missing, skipping: %s", midi_path)
                    continue

                try:
                    reference = _midi_to_events(midi_path)
                except Exception as exc:
                    log.warning("E-GMD: failed to parse %s: %s", midi_path, exc)
                    continue

                yield LoadedTrack(
                    track_id=audio_rel,
                    audio_path=audio_path,
                    reference=reference,
                )


def _midi_to_events(midi_path: Path) -> list[OnsetEvent]:
    """Extract drum onsets from a GM MIDI file, mapped to the 3-class taxonomy.

    Iterates note-on events (velocity > 0) on channel 9 (0-indexed
    drums) across all tracks, accumulating ticks-since-start with the
    file's current tempo to recover absolute seconds.
    """
    mid = mido.MidiFile(str(midi_path))
    events: list[OnsetEvent] = []
    for track in mid.tracks:
        tempo = 500_000  # default 120 BPM if no set_tempo seen
        ticks = 0
        elapsed = 0.0
        for msg in track:
            ticks += msg.time
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


LOADER = EgmdLoader()
