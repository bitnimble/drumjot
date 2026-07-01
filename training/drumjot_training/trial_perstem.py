"""Self-contained per-stem loader for the 6-way-drumsep A/B trial.

Reads a paradb-shaped tree: ``perstem/<pitch>/<cid>.flac`` + ``onsets/<cid>.json``
(an all-lane onset dict, baked snapped/offset-corrected at staging time). Two
layouts, auto-detected from which cymbal dirs exist:

  * **5-way** (baseline): pitches ``k/s/h/c/t``; ``c`` = merged ride+crash cymbal stem.
  * **6-way** (treatment): pitches ``k/s/h/rd/cr/t``; the aufr33-jarredou 6-way
    MDX23C splits the cymbal stem into isolated ``rd`` (ride) + ``cr`` (crash).

Kept OUT of paradb.py so the production 5-way hard-routing invariant (one lane ->
one stem, guarded by ``test_perstem_to_lanes_covers_all_lanes_no_overlap``) is
untouched: ``c`` and ``rd``/``cr`` are alternative layouts of the SAME tree, never
both present, so no lane is double-routed within one layout. Onsets are baked into
each json at staging time (identical across arms), so there is NO aligned-store
lookup here -- the two arms differ ONLY in the cymbal/hat stem audio.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .lanes import LANES

LANES_5WAY: dict[str, tuple[str, ...]] = {
    "k": ("k",), "s": ("s", "ss"), "h": ("hc", "ho"), "c": ("rd", "cr"), "t": ("t",),
}
LANES_6WAY: dict[str, tuple[str, ...]] = {
    "k": ("k",), "s": ("s", "ss"), "h": ("hc", "ho"), "rd": ("rd",), "cr": ("cr",), "t": ("t",),
}


@dataclass(frozen=True)
class TrialPerstemClip:
    audio_path: Path
    onsets_path: Path
    pitch: str
    map_id: str
    lanes: tuple[str, ...]  # lanes this stem legitimately carries


def lane_map(root: str | Path) -> dict[str, tuple[str, ...]]:
    """6-way if the tree has an isolated ride dir, else 5-way."""
    return LANES_6WAY if (Path(root) / "perstem" / "rd").is_dir() else LANES_5WAY


def perstem_index(root: str | Path) -> list[TrialPerstemClip]:
    root = Path(root)
    lm = lane_map(root)
    onsets_dir, perstem_dir = root / "onsets", root / "perstem"
    clips: list[TrialPerstemClip] = []
    for oj in sorted(onsets_dir.glob("*.json")):
        cid = oj.stem
        for pitch, lanes in lm.items():
            audio = perstem_dir / pitch / f"{cid}.flac"
            if audio.exists():
                clips.append(TrialPerstemClip(audio, oj, pitch, cid, lanes))
    return clips


def onsets_by_lane(onsets_path: str | Path) -> dict[str, list[float]]:
    raw = json.loads(Path(onsets_path).read_text())
    return {ln: sorted(float(t) for t in raw.get(ln, [])) for ln in LANES}


def restricted_onsets(onsets_path: str | Path, lanes: tuple[str, ...]) -> dict[str, list[float]]:
    full = onsets_by_lane(onsets_path)
    keep = set(lanes)
    return {ln: (full[ln] if ln in keep else []) for ln in LANES}


def _hash_frac(cid: str, salt: str) -> float:
    h = hashlib.sha1(f"{salt}:{cid}".encode()).hexdigest()
    return (int(h[:8], 16) % 1_000_000) / 1_000_000.0


def perstem_for_split(clips, split: str, *, val_frac: float = 0.1, salt: str = "trial6-val"):
    """Split by cid hash (all of a clip's stems in one split). val_frac 0.1 (>
    paradb's 0.05) so the small trial keeps enough validation cymbals."""
    want_val = split in ("validation", "val")
    return [c for c in clips if (_hash_frac(c.map_id, salt) < val_frac) == want_val]
