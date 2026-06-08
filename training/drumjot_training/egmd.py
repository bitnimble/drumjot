"""E-GMD dataset indexing.

E-GMD ships a metadata CSV (`e-gmd-v1.0.0.csv`) with one row per clip:
drummer/session/id/style/bpm/.../midi_filename/audio_filename/duration/split.
`read_index` turns that into `EgmdClip` records with absolute paths; the
helpers carve out a split or a small duration-capped subset for the Phase-0
smoke test (design spec §2).

Known E-GMD caveats (data owner's experience): some mislabels, duplicates,
and offset timing. Those are handled in the cleaning stage (spec §3 / the
`dedup` module + the scoring quality pass), not here; this is pure indexing.
"""
from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EgmdClip:
    audio_path: Path
    midi_path: Path
    split: str
    duration: float
    bpm: float | None


def _parse_bpm(raw: str) -> float | None:
    raw = (raw or "").strip()
    return float(raw) if raw else None


def read_index(csv_path: str | Path, root: str | Path) -> list[EgmdClip]:
    """Read the E-GMD metadata CSV into `EgmdClip`s with paths under `root`."""
    root = Path(root)
    clips: list[EgmdClip] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            clips.append(
                EgmdClip(
                    audio_path=root / row["audio_filename"],
                    midi_path=root / row["midi_filename"],
                    split=row["split"].strip(),
                    duration=float(row["duration"]),
                    bpm=_parse_bpm(row.get("bpm", "")),
                )
            )
    return clips


def for_split(clips: Iterable[EgmdClip], split: str) -> list[EgmdClip]:
    """Clips belonging to `split` ("train" / "validation" / "test")."""
    return [c for c in clips if c.split == split]


def take_duration(clips: Sequence[EgmdClip], max_seconds: float) -> list[EgmdClip]:
    """A prefix of `clips` whose cumulative duration stays <= `max_seconds`.

    Greedy in list order; a clip that would push the total over the cap is
    skipped (not truncated), and iteration continues so a later shorter clip
    can still fit. Used to grab the ~30 min smoke-test subset.
    """
    out: list[EgmdClip] = []
    total = 0.0
    for c in clips:
        if total + c.duration <= max_seconds:
            out.append(c)
            total += c.duration
    return out
