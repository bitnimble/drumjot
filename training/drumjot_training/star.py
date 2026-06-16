"""STAR Drums dataset loader.

STAR (Zenodo 15690078, BSD-3) annotates drums as plain text:
`time<TAB>CLASS<TAB>velocity`, one onset per line, over an 18-class
vocabulary. We fold to our 10-lane set: kick, snare, side-stick, toms, the
three hat articulations, ride, crash, and misc cymbals (= splash/china/
ride-bell). Non-kit percussion (cowbell/clap/tambourine, ...) folds into the
catch-all `x` negative lane (a hard negative for every output lane, never
predicted); anything else maps to None and is dropped.

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
    "RB": "rd",              # ride bell folds into ride (same physical cymbal)
    "CRC": "cr",
    # SPC (splash) / CHC (china) dropped: removed `mc` lane, no per-stem target
}

def lane_for_star_class(cls: str) -> str | None:
    """Fold a STAR class abbreviation to our lane, or None if out-of-kit."""
    return _STAR_TO_LANE.get(cls)


def onsets_by_lane(annotation_path: str | Path) -> dict[str, list[float]]:
    """Parse a STAR `.txt` annotation into per-lane onset times (seconds).

    Lines are `time<TAB>class<TAB>velocity`; velocity is ignored here. Always
    returns all output lanes (empty lists for absent ones); out-of-kit classes
    (cowbell/clap/tambourine, ...) map to None and are dropped.
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


# --- per-instrument (separation-aware) mode --------------------------------
# Each MDX23C drum-piece stem is trained as its own example with ONLY the lanes
# that belong to it labelled; the other lanes are empty so the model learns to
# stay silent on an isolated stem (i.e. ignore cross-instrument bleed). Matches
# the per-instrument eval (eval_paradb.STEM_TO_LANES) and the inference path.
# `mp` has no stem, so it is not trained in this mode.
PERSTEM_TO_LANES: dict[str, tuple[str, ...]] = {
    "k": ("k",),
    "s": ("s", "ss"),
    "h": ("hc", "hp", "ho"),
    "c": ("rd", "cr"),
    "t": ("t",),
}


@dataclass(frozen=True)
class StarPerstemClip:
    audio_path: Path  # audio/perstem/<pitch>/<name>.flac
    annotation_path: Path
    pitch: str  # k / s / h / c / t
    split: str


def perstem_index(root: str | Path) -> list[StarPerstemClip]:
    """Index per-instrument stems: one entry per (song, drum-piece pitch).

    Pairs each `audio/perstem/<pitch>/<name>.flac` (written by
    `scripts/separate_star_dataset.py`) with its `annotation/<name>.txt`. Stems
    that weren't produced are skipped."""
    root = Path(root)
    clips: list[StarPerstemClip] = []
    for ann in sorted(root.rglob("annotation/*.txt")):
        per_dir = ann.parent.parent / "audio" / "perstem"
        for pitch in PERSTEM_TO_LANES:
            audio = per_dir / pitch / f"{ann.stem}.flac"
            if audio.exists():
                clips.append(StarPerstemClip(audio, ann, pitch, _split_from(ann)))
    return clips


def restricted_onsets(annotation_path: str | Path, pitch: str) -> dict[str, list[float]]:
    """STAR onsets keeping ONLY the lanes that belong to `pitch`'s stem; all other
    lanes are empty (so the isolated-stem example teaches bleed suppression).
    Always returns all 10 lanes."""
    full = onsets_by_lane(annotation_path)
    keep = set(PERSTEM_TO_LANES.get(pitch, ()))
    return {lane: (full[lane] if lane in keep else []) for lane in LANES}
