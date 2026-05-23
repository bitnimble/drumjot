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

from ..core.midi_events import midi_file_to_events
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
                    reference = midi_file_to_events(midi_path)
                except Exception as exc:
                    log.warning("E-GMD: failed to parse %s: %s", midi_path, exc)
                    continue

                yield LoadedTrack(
                    track_id=audio_rel,
                    audio_path=audio_path,
                    reference=reference,
                )


LOADER = EgmdLoader()
