"""STAR Drums dataset loader.

STAR (Zenodo 15690078, BSD-3) annotates drums as plain text:
`time<TAB>CLASS<TAB>velocity`, one onset per line, over an 18-class
vocabulary. We fold to our expanded 11-lane set: kick, snare, side-stick,
toms, the three hat articulations, ride, crash, and the two fold-up lanes
(misc cymbals = splash/china/ride-bell; misc percussion = cowbell/clap/
tambourine). Out-of-kit classes map to None and are dropped.

STAR's labels are accurate by construction (the audio is re-synthesized from
these annotations), so there's no timing-offset / mislabel cleanup needed,
which makes it a clean Phase-0 set. Note (vs STAR's own 5-class reduction):
we keep ride/crash split and the three hats separate, and `RC` is mapped
defensively to ride (it appears in some kit CSVs; STAR's official 5-class
dict omits it).
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from drumjot_training.lanes import LANES

_STAR_TO_LANE: dict[str, str] = {
    "BD": "k",
    "SD": "s",
    "SS": "ss",
    "HT": "t", "MT": "t", "LT": "t",
    "CHH": "hc",
    "PHH": "hp",
    "OHH": "ho",
    "RD": "rd", "RC": "rd",  # RC: defensive ride alias (some kit CSVs use it)
    "CRC": "cr",
    "SPC": "mc", "CHC": "mc", "RB": "mc",  # splash / china / ride bell
    "CB": "mp", "CL": "mp", "CLP": "mp", "TB": "mp",  # cowbell / clap / tambourine
}


def lane_for_star_class(cls: str) -> str | None:
    """Fold a STAR class abbreviation to our lane, or None if out-of-kit."""
    return _STAR_TO_LANE.get(cls)


def onsets_by_lane(annotation_path: str | Path) -> dict[str, list[float]]:
    """Parse a STAR `.txt` annotation into per-lane onset times (seconds).

    Lines are `time<TAB>class<TAB>velocity`; velocity is ignored here.
    Always returns all lanes (empty lists for absent ones).
    """
    out: dict[str, list[float]] = {lane: [] for lane in LANES}
    with open(annotation_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            lane = lane_for_star_class(parts[1].strip())
            if lane is not None:
                out[lane].append(float(parts[0]))
    return out


_SPLITS = ("training", "validation", "test")


@dataclass(frozen=True)
class StarClip:
    audio_path: Path
    annotation_path: Path
    split: str


def _split_from(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    for s in _SPLITS:
        if s in parts:
            return s
    return "unknown"


def index(root: str | Path) -> list[StarClip]:
    """Pair every `annotation/<name>.txt` with `audio/mix/<name>.flac`.

    STAR lays each song out as `<...>/annotation/<name>.txt` alongside
    `<...>/audio/mix/<name>.flac` (the full mix). Split is inferred from the
    path (training / validation / test). Annotations with no matching mix
    audio are skipped.
    """
    root = Path(root)
    clips: list[StarClip] = []
    for ann in sorted(root.rglob("annotation/*.txt")):
        mix = ann.parent.parent / "audio" / "mix" / f"{ann.stem}.flac"
        if mix.exists():
            clips.append(StarClip(audio_path=mix, annotation_path=ann, split=_split_from(ann)))
    return clips


def for_split(clips: Iterable[StarClip], split: str) -> list[StarClip]:
    """Clips belonging to `split`."""
    return [c for c in clips if c.split == split]
