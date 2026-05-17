"""MDB Drums loader.

Pairs each `annotations/class/<track>_class.txt` with the matching
audio file. Prefers `audio/<track>_MIX.wav` (full mix — matches N2N's
protocol) and falls back to `audio/<track>_Drum.wav` with a warning if
only the drum-stem-only mix is available.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ..core.classes import MDB_LABEL_TO_CLASS
from ..core.events import OnsetEvent
from .base import LoadedTrack

log = logging.getLogger(__name__)


@dataclass
class MdbDrumsLoader:
    name: str = "mdb-drums"

    def iter_tracks(self, root: Path) -> Iterator[LoadedTrack]:
        ann_dir = root / "annotations" / "class"
        audio_dir = root / "audio"
        if not ann_dir.is_dir():
            raise FileNotFoundError(
                f"MDB Drums annotations missing at {ann_dir}. "
                f"Clone CarlSouthall/MDBDrums into {root} (see {root}/README.md)."
            )
        if not audio_dir.is_dir():
            raise FileNotFoundError(
                f"MDB Drums audio missing at {audio_dir}. "
                "MedleyDB audio must be placed there (see this folder's README)."
            )

        for ann_path in sorted(ann_dir.glob("*_class.txt")):
            track_id = ann_path.stem.removesuffix("_class")

            audio_path = audio_dir / f"{track_id}_MIX.wav"
            if not audio_path.exists():
                fallback = audio_dir / f"{track_id}_Drum.wav"
                if fallback.exists():
                    log.warning(
                        "MDB Drums: full mix missing for %s, using drum-only fallback",
                        track_id,
                    )
                    audio_path = fallback
                else:
                    log.warning("MDB Drums: no audio for %s, skipping", track_id)
                    continue

            try:
                reference = _parse_annotation(ann_path)
            except Exception as exc:
                log.warning("MDB Drums: failed to parse %s: %s", ann_path, exc)
                continue

            yield LoadedTrack(
                track_id=track_id,
                audio_path=audio_path,
                reference=reference,
            )


def _parse_annotation(path: Path) -> list[OnsetEvent]:
    """Read tab-separated `<onsetSec>\\t<label>` lines, filter to 3 classes."""
    events: list[OnsetEvent] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                onset = float(parts[0])
            except ValueError:
                continue
            label = parts[1].upper()
            drum_class = MDB_LABEL_TO_CLASS.get(label)
            if drum_class is None:
                continue
            events.append(OnsetEvent(time=onset, drum_class=drum_class))
    events.sort(key=lambda e: (e.time, e.drum_class.value))
    return events


LOADER = MdbDrumsLoader()
