"""Build the param-predictor corpus from the model's per-stem TRAINING datasets.

Unlike build_param_dataset.py (real .rlrr maps -> separation -> stems), the
training datasets (STAR / ENST / E-GMD) are already separated into per-instrument
stems with per-lane labels. This reuses each loader's `perstem_index` +
`restricted_onsets` (the exact pairing training uses) to enumerate
`(stem_audio, pitch, GT onsets)`, then -- for the original stem plus N
onset-preserving augmented variants -- runs the FROZEN model and emits
`{features -> oracle params}` rows (drumjot_training.parampred.dataset).

For the hat+cymbal checkpoint, point it at the `h` and `c` stems (default
`--pitches h,c` / `--lanes hc,hp,ho,rd,cr`). These are TRAINING datasets; keep
ParaDB as the held-out test (eval_paradb.py --oracle-report --param-predictor).

Must run with a GPU where drumjot_training imports + the datasets are visible.
  PYTHONPATH=dsp:training python3 training/scripts/build_param_dataset_perstem.py \
      --checkpoint <ckpt dir> --out <table.npz> \
      --star-root /codebox-workspace/datasets/star_balanced_sep \
      --enst-root /codebox-workspace/datasets/enst-sep \
      --egmd-root /codebox-workspace/datasets/egmd_sep \
      [--variants 4] [--max-clips-per-dataset 200] [--max-seconds 60]
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # training/

from drumjot_training import egmd, enst, inference, runtime, star  # noqa: E402
from drumjot_training.parampred import augment, dataset  # noqa: E402

SR = 44100  # load/augment stems at full band (matches build_param_dataset)

# dataset name -> (perstem_index, restricted_onsets(ann_or_midi, pitch))
_LOADERS = {
    "star": (star.perstem_index, star.restricted_onsets, "annotation_path"),
    "enst": (enst.perstem_index, enst.restricted_onsets, "annotation_path"),
    "egmd": (egmd.perstem_index, egmd.restricted_onsets, "midi_path"),
}


def _iter_clips(name, root, pitches, max_clips, rng):
    """Yield (audio_path, pitch, onsets_by_lane) for `pitches`, capped + shuffled."""
    index_fn, onsets_fn, ann_attr = _LOADERS[name]
    clips = [c for c in index_fn(root) if c.pitch in pitches]
    rng.shuffle(clips)
    if max_clips:
        clips = clips[:max_clips]
    for c in clips:
        yield c.audio_path, c.pitch, onsets_fn(getattr(c, ann_attr), c.pitch)


def _stem_rows(stem_path, pitch, gt, restrict, model, meta, encoder, args, rng, librosa, sf):
    """Rows for one stem: identity + `args.variants` augmented variants."""
    rows = []
    wave0, sr = librosa.load(str(stem_path), sr=SR, mono=True)
    song_id = Path(stem_path).stem  # shared by a song's h/c stems -> grouped in the split
    for v in range(args.variants + 1):
        tmp = None
        if v == 0:
            audio_path, wave, aug = str(stem_path), wave0, "identity"
        else:
            wave, aug = augment.random_chain(wave0, sr, rng, use_codec=args.codec)
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            sf.write(tmp, wave, sr)
            audio_path = tmp
        try:
            probs, fps = inference.stitched_probs(
                audio_path, model, meta, encoder, args.max_seconds, args.window_seconds
            )
            rows.extend(dataset.build_rows_for_song(
                probs, fps, meta["lanes"], meta["thresholds"], gt, wave, sr,
                song_id=song_id, aug=aug,
                default_threshold=meta["peak_threshold"], tolerance=meta["onset_tolerance_s"],
                restrict_lanes=restrict,
            ))
        finally:
            if tmp:
                Path(tmp).unlink(missing_ok=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Build the param-predictor corpus from per-stem datasets")
    ap.add_argument("--checkpoint", required=True, help="frozen onset checkpoint dir")
    ap.add_argument("--out", required=True, help="output npz Table")
    ap.add_argument("--star-root", default=None)
    ap.add_argument("--enst-root", default=None)
    ap.add_argument("--egmd-root", default=None)
    ap.add_argument("--pitches", default="h,c", help="drum-stem pitches to ingest (default h,c)")
    ap.add_argument("--lanes", default="hc,hp,ho,rd,cr", help="lanes to keep (intersected per stem)")
    ap.add_argument("--variants", type=int, default=4, help="augmented variants per stem (plus the original)")
    ap.add_argument("--max-clips-per-dataset", type=int, default=200, help="cap stems per dataset (0 = all)")
    ap.add_argument("--no-codec", dest="codec", action="store_false", help="skip lossy-codec augmentation")
    ap.add_argument("--max-seconds", type=float, default=60.0, help="cap each stem to N s before encoding")
    ap.add_argument("--window-seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    import librosa
    import soundfile as sf
    import torch

    runtime.tee_stdio(args.log)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pitches = {p.strip() for p in args.pitches.split(",")}
    req_lanes = {ln.strip() for ln in args.lanes.split(",")}
    roots = {"star": args.star_root, "enst": args.enst_root, "egmd": args.egmd_root}
    roots = {k: v for k, v in roots.items() if v}
    if not roots:
        ap.error("provide at least one of --star-root / --enst-root / --egmd-root")

    model, meta = inference.load_model(args.checkpoint, device)
    from drumjot_training import embeddings
    encoder = embeddings.MertEncoder(name=meta["encoder"], layer=meta["encoder_layer"])
    rng = np.random.default_rng(args.seed)
    rows: list[dataset.ParamSample] = []

    for name, root in roots.items():
        clips = list(_iter_clips(name, root, pitches, args.max_clips_per_dataset, rng))
        print(f"\n=== {name}: {len(clips)} stems x {args.variants + 1} variants ===", flush=True)
        for i, (stem_path, pitch, gt) in enumerate(clips):
            restrict = set(star.PERSTEM_TO_LANES.get(pitch, ())) & req_lanes
            if not restrict:
                continue
            rows.extend(_stem_rows(stem_path, pitch, gt, restrict, model, meta, encoder, args, rng, librosa, sf))
            if (i + 1) % 25 == 0:
                print(f"  {name}: {i + 1}/{len(clips)} stems -> {len(rows)} rows", flush=True)

    if not rows:
        print("no rows produced; nothing written", flush=True)
        return
    table = dataset.Table.from_rows(rows)
    table.save(args.out)
    print(f"\nwrote {len(table)} rows ({len(table.lanes())} lanes, "
          f"{len(set(table.song.tolist()))} songs) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
