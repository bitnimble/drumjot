"""Separation-AWARE ENST training data: build a realistic song mix from each
ENST take (isolated drums + its musical backing), run our drum separator over
it, and keep ENST's EXACT hand annotations -- so the training audio carries the
SAME artifacts (bleed, residual, smearing) as the real-separator outputs we feed
at inference, on genuine acoustic-drum recordings.

This is the ENST analogue of `separate_star_dataset.py` (which does it for STAR's
synthetic stems). ENST is real acoustic drums with hand-aligned labels, so it
attacks the synthetic->real gap from the other side: real source audio + real
separation artifacts + accurate labels.

Mix recipe (per take): `wet_mix` (isolated kit: close mics + overheads) +
`accompaniment` (the minus-one musical backing; digital silence on the ~80% of
takes with no backing). A straight stereo sum preserves ENST's natural
drum/backing balance (the drummer played to that very backing); a clip guard
scales the sum down only if it would clip. So the ~64 `minus-one` takes become
full-band song mixes and the rest are drums-only -- both still pick up the
separator's artifacts.

Output mirrors the ENST + STAR-sep layout, preserving the `drummer_N/` dirs so
`enst.for_split` still holds out drummer_3:
  - `drummer_N/audio/sep_drum/<take>.flac`        = BS-Roformer drum stem
                                      (`enst.index(..., mix="sep_drum")` pairs it
                                      -> `--dataset enst --enst-mix sep_drum`)
  - `drummer_N/audio/perstem/<pitch>/<take>.flac` for pitch in k/s/h/c/t =
                                      MDX23C 5-class per-instrument stems
                                      (-> `--dataset enst_perstem`)
plus the annotation copied unchanged. Idempotent per take (skips takes whose
drum stem + all 5 per-instrument stems already exist).

Run in the CUDA sandbox (needs the transcriber `app` separator + GPU +
`MODELS_DIR`). PYTHONPATH must include the dsp + training packages.

Usage: separate_enst_dataset.py <enst_root> <out_dir> [--limit N]
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))  # training/
sys.path.insert(0, os.path.join(_HERE, "..", "..", "transcriber"))  # transcriber app

from drumjot_training import enst  # noqa: E402

# MDX23C 5-class drum-piece pitches (kick/snare/hi-hat/cymbals/toms), keyed as in
# enst.PERSTEM_TO_LANES / eval_paradb.STEM_TO_LANES.
_PERSTEM_PITCHES = tuple(enst.PERSTEM_TO_LANES)  # ("k", "s", "h", "c", "t")

# BS-Roformer's chunking errors out on very short takes ("size of tensor a (0)..."
# -> no stem produced) -- it hits the ~80 short isolated-"hits" practice takes
# (~12 s) but not the longer phrase/minus-one takes. Pad short mixes with trailing
# silence to this length before separation, then trim the stems back; onset labels
# are unaffected by trailing silence and longer takes pass through unchanged.
MIN_SEP_SECONDS = 30.0


def combine_mix(wet_path: Path, acc_path: Path | None) -> tuple[np.ndarray, int]:
    """Sum `wet_mix` (drums) and `accompaniment` (backing) into one mix.

    Returns (samples, sr). Channels/length are aligned to the wet take; the
    backing is summed in where present (it's digital silence on most takes) and a
    clip guard scales the whole mix down only if the sum would exceed full scale,
    so ENST's natural drum/backing balance is preserved. If `acc_path` is missing,
    the wet drums pass through unchanged."""
    wet, sr = sf.read(str(wet_path), always_2d=True)
    mix = wet.astype(np.float64)
    if acc_path is not None and acc_path.exists():
        acc, acc_sr = sf.read(str(acc_path), always_2d=True)
        if acc_sr != sr:
            raise ValueError(f"sr mismatch: wet {sr} vs acc {acc_sr} ({wet_path.name})")
        acc = acc.astype(np.float64)
        n = min(mix.shape[0], acc.shape[0])
        c = min(mix.shape[1], acc.shape[1])
        mix = mix[:n].copy()
        mix[:, :c] += acc[:n, :c]
    peak = float(np.abs(mix).max()) if mix.size else 0.0
    if peak > 1.0:
        mix /= peak
    return mix.astype(np.float32), sr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("enst_root", type=Path, help="extracted ENST root (with drummer_*/)")
    ap.add_argument("out_dir", type=Path, help="output dir for the separation-aware tree")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N takes (0 = all)")
    args = ap.parse_args()

    ref, out = args.enst_root, args.out_dir
    clips = enst.index(ref, mix="wet_mix")  # all takes, every drummer
    if args.limit:
        clips = clips[: args.limit]
    print(f"selection: {len(clips)} takes from {ref}", flush=True)

    from app.pipeline.separate import Separator

    sep = Separator()
    sep.load()

    def _write_flac(y, sr, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst), y, sr, format="FLAC")

    def _copy_flac_trim(src, dst: Path, keep_seconds: float) -> None:
        """Copy a separated stem to `dst` as FLAC, trimmed to `keep_seconds` (drops
        the trailing silence padded on for the separator). Trim is by duration so
        it's correct whatever the stem's own sample rate."""
        y, sr = sf.read(str(src), always_2d=True)
        _write_flac(y[: int(round(keep_seconds * sr))], sr, dst)

    done = skipped = 0
    for i, clip in enumerate(clips):
        stem = clip.annotation_path.stem
        rel_drummer = clip.annotation_path.parent.parent.relative_to(ref)  # drummer_N
        audio_root = out / rel_drummer / "audio"
        out_drum = audio_root / "sep_drum" / f"{stem}.flac"
        per_targets = {p: audio_root / "perstem" / p / f"{stem}.flac" for p in _PERSTEM_PITCHES}
        out_ann = out / clip.annotation_path.relative_to(ref)

        # all-or-nothing per take: short takes are padded before stage 1, so stage 2
        # must read the padded drum stem (not a trimmed on-disk one) -- regenerate
        # both stages together rather than resuming a half-done take.
        if out_drum.exists() and all(pt.exists() for pt in per_targets.values()):
            skipped += 1
            continue

        acc_path = clip.audio_path.parent.parent / "accompaniment" / f"{stem}.wav"
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            mix_y, mix_sr = combine_mix(clip.audio_path, acc_path)
            keep_seconds = mix_y.shape[0] / mix_sr
            need = int(MIN_SEP_SECONDS * mix_sr)
            if mix_y.shape[0] < need:  # pad short takes so BS-Roformer doesn't choke
                pad = np.zeros((need - mix_y.shape[0], mix_y.shape[1]), dtype=mix_y.dtype)
                mix_y = np.concatenate([mix_y, pad], axis=0)
            mix_dir = tdp / "_mixin"  # keep the combined mix out of the separator work_dir
            mix_dir.mkdir()
            # NEUTRAL input filename: the separator picks its drum stem by
            # `"drum" in filename`, and ENST take names contain "drum"
            # ("snare-drum", "bass-drum"), which would make EVERY stem
            # (bass/other/...) match and the wrong one win. "mix" is safe.
            mix_path = mix_dir / "mix.wav"
            sf.write(str(mix_path), mix_y, mix_sr)
            # stage 1: BS-Roformer combined mix -> drum stem (kept padded for stage 2).
            # each stage gets its OWN work dir so stage 2 doesn't pick up stage 1's
            # sibling stems (bass/other/...) left in a shared dir.
            drum_tmp = sep.run_stems_all(mix_path, tdp / "stage1").drum_stem
            # stage 2: MDX23C drum stem -> 5 per-instrument stems
            per = sep.run_stems_per(drum_tmp, tdp / "stage2").per_instrument  # {pitch: path}
            _copy_flac_trim(drum_tmp, out_drum, keep_seconds)
            for p, src in per.items():
                if p in per_targets:
                    _copy_flac_trim(src, per_targets[p], keep_seconds)
        out_ann.parent.mkdir(parents=True, exist_ok=True)
        out_ann.write_text(clip.annotation_path.read_text())
        done += 1
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(clips)} (done {done}, skipped {skipped})", flush=True)
    print(f"DONE. processed {done}, skipped {skipped} -> {out}", flush=True)


if __name__ == "__main__":
    main()
