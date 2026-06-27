"""Separate ADTOF full-song tracks into per-instrument stems (the `adtof-sep`
tree), mirroring `enst-sep` / `mdb-sep` / `paradb-sep` so the pooled trainer
consumes them identically.

Per track: full song mix -> BS-Roformer drum stem -> MDX23C 5-class
per-instrument -> `perstem/<pitch>/<track>.flac` for pitch in k/s/h/c/t. This is
our DEPLOYMENT separation, so the training audio carries the same artifacts the
model sees in production. Idempotent (skips tracks whose 5 perstems all exist).

ADTOF audio is the FULL song mix (the cleansing pipeline renders the game
chart's `song.ogg` into `audio/audio/<track>.ogg`), so we always go through
BS-Roformer first -- there is no isolated drum-only track to shortcut from.

The output tree reuses ADTOF's annotation folder so `adtof.perstem_index` pairs
the stems with the unchanged GT:

    <out>/perstem/<pitch>/<track>.flac          # pitch in k/s/h/c/t
    <out>/annotations/aligned_drum/<track>.txt  # COPIED from the source tree

  MODELS_DIR=/codebox-workspace/drumjot/models-cache \\
  scripts/sandbox-run env PYTHONPATH=transcriber:training:dsp \\
    python3 training/scripts/separate_adtof_dataset.py \\
      /codebox-workspace/datasets/adtof_built /codebox-workspace/datasets/adtof-sep

`--limit N` processes only the first N tracks (smoke test).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import soundfile as sf

_REPO = Path(__file__).resolve().parents[2]  # training/scripts/ -> repo root (portable)
for _p in ("transcriber", "training", "dsp"):
    sys.path.insert(0, str(_REPO / _p))

from drumjot_training import adtof  # noqa: E402

_PERSTEM_PITCHES = ("k", "s", "h", "c", "t")


def _write(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    y, sr = sf.read(str(src))
    sf.write(str(dst), y, sr, format="FLAC")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adtof_root", type=Path, help="ADTOF built dataset root (automaticGrooming.py output)")
    ap.add_argument("out_dir", type=Path, help="output dir for the adtof-sep per-stem tree")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N tracks (0=all)")
    args = ap.parse_args()

    clips = adtof.index(args.adtof_root)
    if args.limit:
        clips = clips[: args.limit]
    log = lambda s: print(s, flush=True)  # noqa: E731

    from app.pipeline.separate import Separator
    sep = Separator()
    log(f"=== separate ADTOF: {len(clips)} tracks -> {args.out_dir} ===")
    t0 = time.perf_counter()
    done = processed = 0  # done = ready (incl pre-existing skips); processed = newly separated
    for i, clip in enumerate(clips, 1):
        track = clip.track
        targets = {p: args.out_dir / "perstem" / p / f"{track}.flac" for p in _PERSTEM_PITCHES}
        # mirror the GT so adtof.perstem_index pairs against the sep tree directly
        ann_dst = args.out_dir / "annotations" / "aligned_drum" / f"{track}.txt"
        if all(t.exists() for t in targets.values()) and ann_dst.exists():
            log(f"  [{i}/{len(clips)}] {track}: already done, skip")
            done += 1
            continue
        src = clip.audio_path  # adtof.index() only yields clips with existing audio
        if not src.exists():
            log(f"  [{i}/{len(clips)}] {track}: no audio, skip")
            continue
        log(f"  [{i}/{len(clips)}] {track}: separating...")
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            drum = sep.run_stems_all(src, tdp / "s1", build_no_drums=False).drum_stem
            per = sep.run_stems_per(drum, tdp / "s2").per_instrument  # {pitch: path}
            for p in _PERSTEM_PITCHES:
                if per.get(p):
                    _write(Path(per[p]), targets[p])
        ann_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(clip.annotation_path, ann_dst)
        done += 1
        processed += 1
        el = time.perf_counter() - t0
        log(f"  [{i}/{len(clips)}] {track}: done ({el / processed:.0f}s/track avg)")
    log(f"\nseparated {processed} new ({done}/{len(clips)} ready) -> {args.out_dir}")


if __name__ == "__main__":
    main()
