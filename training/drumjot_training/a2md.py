"""A2MD loader: real (YouTube) songs + aligned full-song MIDI, separated into
per-instrument stems by `scripts/separate_a2md.py`.

Unlike the drums-only datasets (E-GMD / ENST / STAR), A2MD's aligned MIDI is the
WHOLE song (bass, melody, ...), so drum onsets are taken ONLY from the GM
percussion channel (channel 9, 0-indexed = MIDI channel 10) -- otherwise a
melodic note that happens to land on a GM drum note number (e.g. a bass D2 = 38)
would be miscounted as a snare. Labels are aligned arrangements (approximate, not
hand-transcribed), so prefer the low-`dist` buckets and treat onset-F1 on A2MD as
training signal, not a precise eval.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mido

from drumjot_training.lanes import LANES, lane_for_gm_note

DRUM_CHANNEL = 9  # GM percussion (MIDI channel 10, 1-indexed)

# stem pitch -> the lanes it carries (matches the other datasets + eval_paradb)
PERSTEM_TO_LANES: dict[str, tuple[str, ...]] = {
    "k": ("k",),
    "s": ("s", "ss"),
    "h": ("hc", "ho"),
    "c": ("rd", "cr"),
    "t": ("t",),
}


@dataclass(frozen=True)
class A2mdPerstemClip:
    audio_path: Path  # audio/perstem/<pitch>/<id>.flac
    midi_path: Path   # annotation/<id>.mid (full-song aligned MIDI)
    pitch: str        # k / s / h / c / t


def drum_onsets_by_lane(midi_path: str | Path) -> dict[str, list[float]]:
    """Per-lane ascending onset times (s) from the GM percussion channel only.

    Tempo-aware `mido.MidiFile` iteration (msg.time is delta seconds). `note_on`
    with velocity > 0 on `DRUM_CHANNEL` is an onset; everything else is ignored.
    Always returns all lanes (empty when absent)."""
    out: dict[str, list[float]] = {lane: [] for lane in LANES}
    t = 0.0
    for msg in mido.MidiFile(str(midi_path)):
        t += msg.time
        if msg.type != "note_on" or msg.velocity <= 0:
            continue
        if getattr(msg, "channel", None) != DRUM_CHANNEL:
            continue
        lane = lane_for_gm_note(msg.note)
        if lane is not None:
            out[lane].append(t)
    return {lane: sorted(v) for lane, v in out.items()}


def restricted_onsets(midi_path: str | Path, pitch: str) -> dict[str, list[float]]:
    """Drum onsets keeping ONLY the lanes that belong to `pitch`'s stem; all other
    lanes empty (isolated-stem bleed suppression). Always returns all lanes."""
    full = drum_onsets_by_lane(midi_path)
    keep = set(PERSTEM_TO_LANES.get(pitch, ()))
    return {lane: (full[lane] if lane in keep else []) for lane in LANES}


def perstem_index(root: str | Path) -> list[A2mdPerstemClip]:
    """One entry per (track, drum-piece pitch). Pairs each
    `audio/perstem/<pitch>/<id>.flac` (from separate_a2md.py) with its
    `annotation/<id>.mid`. Stems that weren't produced are skipped."""
    root = Path(root)
    clips: list[A2mdPerstemClip] = []
    for ann in sorted((root / "annotation").glob("*.mid")):
        per_dir = root / "audio" / "perstem"
        for pitch in PERSTEM_TO_LANES:
            audio = per_dir / pitch / f"{ann.stem}.flac"
            if audio.exists():
                clips.append(A2mdPerstemClip(audio, ann, pitch))
    return clips
