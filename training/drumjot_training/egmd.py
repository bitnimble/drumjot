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

from drumjot_training.lanes import LANES


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


# --- per-instrument (separation-aware) mode --------------------------------
# MDX23C drum-piece stems written by scripts/separate_egmd_dataset.py; each is
# trained with ONLY its own lanes labelled so the model learns to ignore
# cross-instrument bleed. Identical routing to STAR/ENST and the per-instrument
# eval (eval_paradb.STEM_TO_LANES): side stick rides the snare stem, the three
# hats share the hi-hat stem, ride/crash share the cymbal stem.
PERSTEM_TO_LANES: dict[str, tuple[str, ...]] = {
    "k": ("k",),
    "s": ("s", "ss"),
    "h": ("hc", "hp", "ho"),
    "c": ("rd", "cr"),
    "t": ("t",),
}


@dataclass(frozen=True)
class EgmdPerstemClip:
    audio_path: Path  # audio/perstem/<pitch>/<uid>.flac
    midi_path: Path
    pitch: str  # k / s / h / c / t
    split: str
    duration: float


def perstem_index(
    root: str | Path, csv_name: str = "e-gmd-v1.0.0.csv"
) -> list[EgmdPerstemClip]:
    """Index per-instrument stems written by `separate_egmd_dataset.py`.

    Reads the sep tree's CSV (for split + MIDI label per clip) and, for each
    clip, enumerates `audio/perstem/<pitch>/<uid>.flac` (uid = the drum-stem's
    filename stem). Stems that weren't produced are skipped."""
    root = Path(root)
    clips: list[EgmdPerstemClip] = []
    for clip in read_index(root / csv_name, root):
        uid = clip.audio_path.stem
        for pitch in PERSTEM_TO_LANES:
            audio = root / "audio" / "perstem" / pitch / f"{uid}.flac"
            if audio.exists():
                clips.append(
                    EgmdPerstemClip(audio, clip.midi_path, pitch, clip.split, clip.duration)
                )
    return clips


def restricted_onsets(midi_path: str | Path, pitch: str) -> dict[str, list[float]]:
    """E-GMD MIDI onsets keeping ONLY the lanes that belong to `pitch`'s stem;
    all other lanes are empty (so the isolated-stem example teaches bleed
    suppression). Always returns all lanes."""
    from drumjot_training import midi_labels  # lazy: keeps egmd indexing mido-free

    full = midi_labels.onsets_from_path(midi_path)
    keep = set(PERSTEM_TO_LANES.get(pitch, ()))
    return {lane: (full[lane] if lane in keep else []) for lane in LANES}
