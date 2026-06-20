"""MDB-Drums loader (annotations + audio paths) for the SOTA-comparable eval.

MDB-Drums (Southall et al., ISMIR 2017): ~23 `MusicDelta_*` tracks from MedleyDB
with hand onset annotations. Layout (the repo clones to `.../MDBDrums/MDB Drums/`,
note the space):

  MDB Drums/
    audio/full_mix/<track>_MIX.wav      full song mix (the realistic ADT condition)
    audio/drum_only/<track>_Drum.wav    isolated drum track
    annotations/subclass/<track>_subclass.txt   "<time>\\t<LABEL>" per onset
    annotations/class/<track>_class.txt          coarse 6-class (we use subclass)

The subclass labels (verified against the clone) fold to our 9 lanes; tambourine
(TMB) is dropped (not in our taxonomy). Annotation lines are whitespace-separated
`<time> <LABEL>` with stray spaces around both, so we `.split()` and `.strip()`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# MDB subclass label -> our 9-lane vocab. TMB (tambourine) -> None (dropped).
MDB_TO_LANE: dict[str, str | None] = {
    "KD": "k",
    "SD": "s", "SDB": "s", "SDG": "s", "SDF": "s", "SDD": "s", "SDNS": "s",
    "SST": "ss",
    "HIT": "t", "MHT": "t", "HFT": "t", "LFT": "t",
    "CHH": "hc", "PHH": "hp", "OHH": "ho",
    "RDC": "rd", "RDB": "rd",
    "CRC": "cr", "CHC": "cr", "SPC": "cr",
    "TMB": None,  # tambourine -- not a kit lane, dropped
}


@dataclass
class MdbClip:
    track: str
    full_mix: Path | None
    drum_only: Path | None
    subclass_ann: Path


def _root(root: str | Path) -> Path:
    """Resolve to the 'MDB Drums' dir whether given the repo root or it directly."""
    p = Path(root)
    inner = p / "MDB Drums"
    return inner if inner.is_dir() else p


def index(root: str | Path) -> list[MdbClip]:
    """One MdbClip per track (keyed by the subclass annotation, the canonical list)."""
    base = _root(root)
    ann_dir = base / "annotations" / "subclass"
    fm_dir = base / "audio" / "full_mix"
    do_dir = base / "audio" / "drum_only"
    clips: list[MdbClip] = []
    for ann in sorted(ann_dir.glob("*_subclass.txt")):
        track = ann.stem[: -len("_subclass")]
        fm = next((p for p in (fm_dir / f"{track}_MIX.wav", fm_dir / f"{track}.wav") if p.exists()), None)
        do = next((p for p in (do_dir / f"{track}_Drum.wav", do_dir / f"{track}_Drums.wav") if p.exists()), None)
        clips.append(MdbClip(track, fm, do, ann))
    return clips


def onsets_by_lane(subclass_ann: str | Path) -> dict[str, list[float]]:
    """Parse a subclass annotation -> {lane: sorted onset times}. Unknown/dropped
    labels (e.g. TMB) are skipped; malformed lines ignored."""
    out: dict[str, list[float]] = {}
    for line in Path(subclass_ann).read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            t = float(parts[0])
        except ValueError:
            continue
        lane = MDB_TO_LANE.get(parts[1].strip())
        if lane is None:
            continue
        out.setdefault(lane, []).append(t)
    return {lane: sorted(v) for lane, v in out.items()}
