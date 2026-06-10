"""Separation-AWARE training data: run our drum separator over STAR mixes so the
training audio carries the SAME artifacts (bleed, residual, smearing) as the
real-separator outputs we feed at inference -- while keeping STAR's EXACT labels.

This distribution-matches train to test (the model learns to ignore separation
artifacts instead of seeing pristine synthetic stems), the most direct attack on
the synthetic->real gap. STAR gives clean labels for free: its `mix` =
non_drum + re_synth_drum, so separating the mix yields a drum stem with realistic
bleed/residual while the annotations still describe the same hits.

Input is a STAR subset dir with mix audio + annotations (e.g. star_balanced).
Output mirrors the layout, writing TWO stages per clip:
  - `audio/mix/<name>.flac`        = BS-Roformer drum stem (so `star.index` pairs
                                      it -> `--dataset star` full-drum training)
  - `audio/perstem/<pitch>/<name>.flac` for pitch in k/s/h/c/t = MDX23C 5-class
                                      per-instrument stems (for a future
                                      per-instrument-stem training loop)
plus the annotation copied unchanged. Selection mirrors a training run:
for_split(training)[:n_train] + (validation+test)[:n_val]. Idempotent per stage
(skips clips whose drum stem + all 5 per-instrument stems already exist).

Run in the CUDA sandbox (needs the transcriber `app` separator + GPU +
`MODELS_DIR`). PYTHONPATH must include the dsp + training packages.

Usage: separate_star_dataset.py <ref_dir> <out_dir> [n_train] [n_val]
"""
import os
import sys
import tempfile
from pathlib import Path

import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))  # training/
sys.path.insert(0, os.path.join(_HERE, "..", "..", "transcriber"))  # transcriber app

from drumjot_training import star  # noqa: E402

# MDX23C 5-class drum-piece pitches (kick/snare/hi-hat/cymbals/toms), keyed as in
# eval_paradb.STEM_TO_LANES. Written under audio/perstem/<pitch>/ for a future
# per-instrument-stem training loop (train each lane on its isolated stem so the
# model learns to ignore cross-instrument bleed).
_PERSTEM_PITCHES = ("k", "s", "h", "c", "t")


def main():
    ref = Path(sys.argv[1])
    out = Path(sys.argv[2])
    n_train = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    n_val = int(sys.argv[4]) if len(sys.argv) > 4 else 48

    clips = star.index(ref)
    tr = star.for_split(clips, "training")[:n_train]
    held = star.for_split(clips, "validation") + star.for_split(clips, "test")
    va = held[:n_val]
    sel = tr + va
    print(f"selection: {len(tr)} train + {len(va)} val/test = {len(sel)}", flush=True)

    from app.pipeline.separate import Separator

    sep = Separator()
    sep.load()

    def _write_flac(src, dst: Path) -> None:
        y, sr = sf.read(str(src))
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst), y, sr, format="FLAC")

    done = skipped = 0
    for i, clip in enumerate(sel):
        name = clip.audio_path.name  # <stem>.flac
        out_drum = out / clip.audio_path.relative_to(ref)  # audio/mix/<name>.flac (full drum stem)
        audio_root = out_drum.parent.parent  # .../audio
        per_targets = {p: audio_root / "perstem" / p / name for p in _PERSTEM_PITCHES}
        out_ann = out / clip.annotation_path.relative_to(ref)

        drum_done = out_drum.exists()
        per_done = all(pt.exists() for pt in per_targets.values())
        if drum_done and per_done:
            skipped += 1
            continue
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            if not drum_done:  # stage 1: BS-Roformer mix -> drum stem
                _write_flac(sep.run_stems_all(clip.audio_path, tdp).drum_stem, out_drum)
            if not per_done:  # stage 2: MDX23C drum stem -> 5 per-instrument stems
                per = sep.run_stems_per(out_drum, tdp).per_instrument  # {pitch: path}
                for p, src in per.items():
                    if p in per_targets:
                        _write_flac(src, per_targets[p])
        out_ann.parent.mkdir(parents=True, exist_ok=True)
        out_ann.write_text(clip.annotation_path.read_text())
        done += 1
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(sel)} (done {done}, skipped {skipped})", flush=True)
    print(f"DONE. processed {done}, skipped {skipped} -> {out}", flush=True)


if __name__ == "__main__":
    main()
