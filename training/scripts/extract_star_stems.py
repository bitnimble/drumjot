"""Mirror a STAR subset dir but with DRUM-STEM audio instead of the full mix.

For a controlled mix-vs-stem training comparison: the mix is `non_drum +
re_synthesized_drum`, so the re-synth drum stem has onsets at identical times
to the mix (labels stay accurate) and isolates the effect of removing the
non-drum instruments.

Reads the re-synth drum flac for every clip in `ref_dir` (a star_subset/
star_balanced dir) straight from the parts and writes it under
`audio/mix/<annotation_stem>.flac` in `out_dir` (the annotation stem name, so
`star.index` pairs it). The selection mirrors what the training run sees:
for_split(training)[:n_train] + (validation+test)[:n_val].

Stem-file naming: `<id>_mix_<kit>` (annotation/mix) -> `<id>_re_synth_drum_<kit>`
in audio/re_synthesized_drum/.

Usage: extract_star_stems.py <parts_dir> <ref_dir> <out_dir> [n_train] [n_val]
"""
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from extract_star_subset import MultiPartFile  # noqa: E402

from drumjot_training import star  # noqa: E402


def main():
    parts_dir, ref_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    n_train = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    n_val = int(sys.argv[5]) if len(sys.argv) > 5 else 22
    ref = Path(ref_dir)
    out = Path(out_dir)

    idx = star.index(ref)
    train_sel = star.for_split(idx, "training")[:n_train]
    held = star.for_split(idx, "validation") + star.for_split(idx, "test")
    val_sel = held[:n_val]
    sel = train_sel + val_sel
    print(f"selection: {len(train_sel)} train + {len(val_sel)} val/test = {len(sel)}", flush=True)

    parts = sorted(os.path.join(parts_dir, f) for f in os.listdir(parts_dir) if ".zip.part-" in f)
    z = zipfile.ZipFile(MultiPartFile(parts))
    names = set(z.namelist())

    extracted = skipped = 0
    for clip in sel:
        ann_member = str(clip.annotation_path.relative_to(ref))
        stem = clip.annotation_path.stem  # <id>_mix_<kit>
        prefix = ann_member[: ann_member.rindex("/annotation/")]
        synth_name = stem.replace("_mix_", "_re_synth_drum_") + ".flac"
        synth_member = f"{prefix}/audio/re_synthesized_drum/{synth_name}"
        if synth_member not in names:
            print(f"  SKIP no stem for {stem} ({synth_member})", flush=True)
            skipped += 1
            continue
        z.extract(ann_member, out)  # annotation at its normal path
        dst = out / prefix / "audio" / "mix" / f"{stem}.flac"  # stem audio, named as mix
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(z.read(synth_member))
        extracted += 1
        if extracted % 100 == 0:
            print(f"  {extracted}/{len(sel)}", flush=True)
    print(f"DONE. extracted {extracted}, skipped {skipped} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
