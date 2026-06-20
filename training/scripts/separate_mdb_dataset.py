"""Separate MDB-Drums full_mix tracks into per-instrument stems (the `mdb-sep`
tree), mirroring `enst-sep` so `sota_eval.py` consumes them identically.

Per track: full_mix -> BS-Roformer drum stem -> MDX23C 5-class per-instrument ->
`perstem/<pitch>/<track>.flac` for pitch in k/s/h/c/t. This is our DEPLOYMENT
separation, so the eval audio carries the same artifacts the model sees in
production. Idempotent (skips tracks whose 5 perstems all exist).

  MODELS_DIR=/codebox-workspace/drumjot/models-cache \
  scripts/sandbox-run env PYTHONPATH=transcriber:training:dsp \
    python3 training/scripts/separate_mdb_dataset.py \
      /codebox-workspace/datasets/MDBDrums /codebox-workspace/datasets/mdb-sep

`--source drum_only` skips BS-Roformer (MDB's isolated drum track straight into
MDX23C) for a cleaner / faster variant; default `full_mix` is the realistic one.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import soundfile as sf

_REPO = Path(__file__).resolve().parents[2]  # training/scripts/ -> repo root (portable)
for _p in ("transcriber", "training", "dsp"):
    sys.path.insert(0, str(_REPO / _p))

from drumjot_training import mdb  # noqa: E402

_PERSTEM_PITCHES = ("k", "s", "h", "c", "t")


def _write(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    y, sr = sf.read(str(src))
    sf.write(str(dst), y, sr, format="FLAC")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mdb_root", type=Path, help="MDBDrums clone root (with 'MDB Drums/')")
    ap.add_argument("out_dir", type=Path, help="output dir for the mdb-sep per-stem tree")
    ap.add_argument("--source", default="full_mix", choices=("full_mix", "drum_only"))
    ap.add_argument("--limit", type=int, default=0, help="process only the first N tracks (0=all)")
    args = ap.parse_args()

    clips = mdb.index(args.mdb_root)
    if args.limit:
        clips = clips[: args.limit]
    log = lambda s: print(s, flush=True)  # noqa: E731

    from app.pipeline.separate import Separator
    sep = Separator()
    log(f"=== separate MDB ({args.source}): {len(clips)} tracks -> {args.out_dir} ===")
    t0 = time.perf_counter()
    done = 0
    for i, clip in enumerate(clips, 1):
        track = clip.track
        targets = {p: args.out_dir / "perstem" / p / f"{track}.flac" for p in _PERSTEM_PITCHES}
        if all(t.exists() for t in targets.values()):
            log(f"  [{i}/{len(clips)}] {track}: already done, skip")
            done += 1
            continue
        src = clip.full_mix if args.source == "full_mix" else clip.drum_only
        if src is None or not src.exists():
            log(f"  [{i}/{len(clips)}] {track}: no {args.source} audio, skip")
            continue
        log(f"  [{i}/{len(clips)}] {track}: separating...")
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            if args.source == "full_mix":
                drum = sep.run_stems_all(src, tdp / "s1", build_no_drums=False).drum_stem
            else:
                drum = src
            per = sep.run_stems_per(drum, tdp / "s2").per_instrument  # {pitch: path}
            for p in _PERSTEM_PITCHES:
                if per.get(p):
                    _write(Path(per[p]), targets[p])
        done += 1
        el = time.perf_counter() - t0
        log(f"  [{i}/{len(clips)}] {track}: done ({el/done:.0f}s/track avg)")
    log(f"\nseparated {done}/{len(clips)} tracks -> {args.out_dir}")


if __name__ == "__main__":
    main()
