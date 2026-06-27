"""ADTOF dataset loader (Zehren et al., ISMIR 2021 / Signals 2023).

ADTOF is the current ADT SOTA *training* set: ~114-359 h of REAL (non-synthetic)
music, labelled by an automatic-cleansing pipeline over crowdsourced rhythm-game
charts -- the SAME chart-derived approach as our ParaDB pipeline (`rlrr.py`), so
its labels are the shape we already handle. It's the prime target for the
data-bound cymbal lanes (RESULTS: crash lifts on more REAL data).

  github.com/MZehren/ADTOF -- code + chart-cleansing pipeline (NO audio: the
  songs are copyrighted game charts, redistributed only as mel-spectrograms on
  Zenodo on request). This loader consumes the **built** dataset that
  `bin/automaticGrooming.py` writes from user-supplied PhaseShift charts (see
  research/DATASETS.md "Acquiring ADTOF audio").

================================ CYMBAL TAXONOMY ==============================

CRITICAL for crash-vs-ride training signal: **the default ADTOF build LUMPS ride
and crash into ONE cymbal class.** Verified against the repo (commit b3968fb):

  adtof/config.py:
      LABELS_5    = [35, 38, 47, 42, 49]
      LABELS_5TXT = ["BD", "SD", "TT", "HH", "CY+RD"]   <- ride+crash = ONE class

  adtof/converters/phaseShiftConverter.py convertTrack(): only task "5"
  (MIDI_REDUCED_5) or "7" (MIDI_REDUCED_7) are buildable, and `bin/
  automaticGrooming.py` defaults to task="5". In MIDI_REDUCED_5
  (instrumentsMapping.py) every cymbal collapses to 49:
      49 Crash1, 51 Ride1, 52 China, 53 Ride bell, 55 Splash, 57 Crash2,
      59 Ride2   -> 49   ("CY", the merged cymbal class)

So the 5-class default gives NO ride/crash separation. We map that merged class
to **crash** (`cr`), our data-bound lane -- NOT ride -- because (a) in most rock
charts the merged cymbal is crash-dominated, and (b) crash is the lane more REAL
data is known to lift. The cost: ADTOF ride hits become crash labels (a known,
documented bias), and the `rd` lane gets NO positive signal from ADTOF.

There IS a path to real ride/crash: rebuild the dataset with
`automaticGrooming.py -t 7`. task="7" chains MIDI_REDUCED_7, built on
MIDI_REDUCED_6 ("Splitting CY and RD"), so ride (51, 53 bell, 59) -> 51 and
crash (49, 57) -> 49 become SEPARATE classes (+ open hat 46 split out). With a
task=7 build the loader routes 51 -> rd and 49 -> cr automatically -- the pitch
map below is a task-agnostic superset (a task=5 tree just never contains 51/46),
so a task=7 rebuild yields genuine ride-vs-crash signal with no loader change.
Caveat (the ADTOF authors' own note,
phaseShiftConverter.convertPitches docstring): chart ride/crash labels are noisy
("RD can be wrongly annotated and used for a third crash"), so even task=7 ride
is imperfect; but it is the ONLY way ADTOF adds ride signal at all.

==============================================================================

On-disk layout written by `bin/automaticGrooming.py <charts> <out>` (config.py):

    <out>/audio/audio/<artist> - <name>.ogg            # rendered song audio
    <out>/annotations/aligned_drum/<artist> - <name>.txt   # the final GT

The aligned_drum `.txt` is MIREX-style, TAB-separated `<time>\t<reduced_pitch>`
(correctAlignmentConverter.writteBeats writes exactly two columns; no velocity),
one line per onset, `<reduced_pitch>` drawn from LABELS_5 (task=5) or LABELS_7
(task=7). We parse those reduced MIDI pitches straight to lanes -- no chart
parsing here (the cleansing pipeline already did it).
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from drumjot_training.lanes import LANES

# ADTOF *reduced* MIDI pitch -> our lane. Keyed on the post-reduction pitch the
# cleansing pipeline writes into aligned_drum/*.txt, NOT raw chart pitches.
#
# Shared rows (both task=5 and task=7):
#   35 BD -> k ; 38 SD -> s ; 47 TT -> t ; 42 HH(closed) -> hc
# task=7 also emits:
#   46 OH -> ho ; 51 RD -> rd ; 49 CY -> cr   (ride/crash SEPARATE)
# task=5 emits the merged cymbal as 49 only (-> cr; see CYMBAL TAXONOMY above).
#
# Note: ADTOF's reductions fold side stick (37) into snare (38) and pedal/open
# hat into 42 at task=5, so we get no `ss`/`ho` lane from a task=5 build (both
# become their merged class). That matches our taxonomy folds elsewhere (pedal
# hat -> hc) except side stick, which ADTOF simply doesn't isolate.
ADTOF_PITCH_TO_LANE: dict[int, str] = {
    35: "k",   # BD  (Acoustic + Electric Bass Drum reduced to 35)
    38: "s",   # SD  (snare; side stick 37 + claps 39 also fold to 38 upstream)
    47: "t",   # TT  (all toms reduced to 47)
    42: "hc",  # HH  closed (closed + pedal; + open at task=5)
    46: "ho",  # OH  open hi-hat (task=7 only; absent in a task=5 build)
    51: "rd",  # RD  ride (task=7 only; in task=5 ride is folded into 49)
    49: "cr",  # CY  cymbal: crash always; ALSO merged ride+crash at task=5
}


def onsets_by_lane(annotation_path: str | Path) -> dict[str, list[float]]:
    """Parse an ADTOF `aligned_drum/*.txt` into per-lane onset times (seconds).

    Lines are TAB-separated `<time>\\t<reduced_midi_pitch>` (MIREX style; the
    cleansing pipeline writes no velocity column). Pitches outside
    `ADTOF_PITCH_TO_LANE` and malformed lines are dropped. Always returns all
    output lanes (empty lists for absent ones); each list sorted ascending.
    Signature mirrors enst/paradb `onsets_by_lane` so the pooled spec builder
    uses it directly as the parsed-onset `reader`."""
    out: dict[str, list[float]] = {lane: [] for lane in LANES}
    with open(annotation_path) as f:
        for line in f:
            parts = line.replace("\r\n", "").replace("\n", "").split("\t")
            if len(parts) < 2:
                continue
            try:
                t = float(parts[0])
                pitch = int(float(parts[1]))
            except ValueError:
                continue  # header / non-numeric
            lane = ADTOF_PITCH_TO_LANE.get(pitch)
            if lane is None:
                continue
            out[lane].append(t)
    for ts in out.values():
        ts.sort()
    return out


# --- mix-level clips --------------------------------------------------------
@dataclass(frozen=True)
class AdtofClip:
    audio_path: Path        # audio/audio/<track>.ogg  (or .flac after separation)
    annotation_path: Path   # annotations/aligned_drum/<track>.txt
    track: str


def _audio_for(root: Path, track: str) -> Path | None:
    """Locate `<root>/audio/audio/<track>.<ext>` (flac first so a sep tree
    pairs, then the original ogg/wav), or None if no audio exists."""
    audio_dir = root / "audio" / "audio"
    for ext in ("flac", "ogg", "wav"):
        p = audio_dir / f"{track}.{ext}"
        if p.exists():
            return p
    return None


def index(root: str | Path) -> list[AdtofClip]:
    """Pair every `annotations/aligned_drum/<track>.txt` with its
    `audio/audio/<track>.{flac,ogg,wav}`. Annotations with no matching audio
    are skipped. `root` = the `automaticGrooming.py` output folder."""
    root = Path(root)
    ann_dir = root / "annotations" / "aligned_drum"
    clips: list[AdtofClip] = []
    for ann in sorted(ann_dir.glob("*.txt")):
        audio = _audio_for(root, ann.stem)
        if audio is not None:
            clips.append(AdtofClip(audio_path=audio, annotation_path=ann, track=ann.stem))
    return clips


def _hash_frac(track: str, salt: str) -> float:
    """Stable [0,1) hash of a track name (machine-independent; no RNG state).
    Mirrors paradb._hash_frac so the held-out split is frozen + reproducible."""
    h = hashlib.sha1(f"{salt}:{track}".encode()).hexdigest()
    return (int(h[:8], 16) % 1_000_000) / 1_000_000.0


def for_split(
    clips: Iterable[AdtofClip], split: str, *, val_frac: float = 0.05, salt: str = "adtof-val"
) -> list[AdtofClip]:
    """Track-name-hash split (ADTOF has official splits, but our pooled trainer
    just needs a held-IN val for threshold tuning / early stop). `split` is
    'train' or 'validation'. Deterministic across runs/machines (sha1)."""
    want_val = split in ("validation", "val")
    return [c for c in clips if (_hash_frac(c.track, salt) < val_frac) == want_val]


# --- per-instrument (separation-aware) mode --------------------------------
# Each MDX23C drum-piece stem (written by scripts/separate_adtof_dataset.py) is
# trained with ONLY its own lanes labelled so the model learns to suppress
# cross-instrument bleed. Identical routing to STAR/ENST/E-GMD/ParaDB: side
# stick rides the snare stem, the three hats share the hi-hat stem, ride/crash
# share the cymbal stem. Reused verbatim from egmd.PERSTEM_TO_LANES.
PERSTEM_TO_LANES: dict[str, tuple[str, ...]] = {
    "k": ("k",),
    "s": ("s", "ss"),
    "h": ("hc", "ho"),
    "c": ("rd", "cr"),
    "t": ("t",),
}


@dataclass(frozen=True)
class AdtofPerstemClip:
    audio_path: Path        # perstem/<pitch>/<track>.flac
    annotation_path: Path   # annotations/aligned_drum/<track>.txt
    pitch: str              # k / s / h / c / t
    track: str


def perstem_index(root: str | Path) -> list[AdtofPerstemClip]:
    """Index per-instrument stems: one entry per (track, drum-piece pitch).

    Pairs each `perstem/<pitch>/<track>.flac` (written by
    `scripts/separate_adtof_dataset.py`) with its
    `annotations/aligned_drum/<track>.txt`. Stems that weren't produced are
    skipped; a track with no annotation is skipped."""
    root = Path(root)
    ann_dir = root / "annotations" / "aligned_drum"
    perstem_dir = root / "perstem"
    clips: list[AdtofPerstemClip] = []
    for ann in sorted(ann_dir.glob("*.txt")):
        track = ann.stem
        for pitch in PERSTEM_TO_LANES:
            audio = perstem_dir / pitch / f"{track}.flac"
            if audio.exists():
                clips.append(AdtofPerstemClip(audio, ann, pitch, track))
    return clips


def restricted_onsets(annotation_path: str | Path, pitch: str) -> dict[str, list[float]]:
    """ADTOF onsets keeping ONLY the lanes that belong to `pitch`'s stem; all
    other lanes empty (so the isolated-stem example teaches bleed suppression).
    Always returns all lanes."""
    full = onsets_by_lane(annotation_path)
    keep = set(PERSTEM_TO_LANES.get(pitch, ()))
    return {ln: (full[ln] if ln in keep else []) for ln in LANES}


def perstem_for_split(
    clips: Iterable[AdtofPerstemClip], split: str, *, val_frac: float = 0.05,
    salt: str = "adtof-val",
) -> list[AdtofPerstemClip]:
    """Track-hash split for per-instrument clips (all of a track's stems land in
    the same split -> no cross-stem leakage). See `for_split`."""
    want_val = split in ("validation", "val")
    return [c for c in clips if (_hash_frac(c.track, salt) < val_frac) == want_val]
