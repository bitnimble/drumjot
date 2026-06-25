"""Build the param-predictor training corpus from a folder of labeled maps.

For each map: reconstruct the song, separate it into per-instrument drum stems
(reusing eval_paradb's verified plumbing), then for the original stem plus N
onset-preserving augmented variants, run the FROZEN model to get activation
curves and emit {features -> oracle params} rows (drumjot_training.parampred.
dataset). The stacked rows are saved as an npz `Table` for
train_param_predictor.py.

Augmentation is applied to the SEPARATED stems (not the pre-separation mix): it
directly varies what the model sees -- the activation curve the params bite on --
without paying to re-separate every variant. The onsets don't move, so the
chart's per-lane GT (and the once-computed global offset) stay valid for every
variant; the oracle just re-sweeps on the new curve.

Sources: `--maps-dir` (a folder of training zips, e.g. A2MD), or `--paradb-sep` (a
paradb-sep tree = the ParaDB KEPT-TRAINING split, with the held-out eval ids
already excluded by separate_paradb_dataset.py). Never point `--maps-dir` at the
raw ParaDB eval set -- the held-out eval maps must stay out of training. Pool
multiple corpora at fit time: `train_param_predictor.py --dataset a2md.npz paradb.npz`.

Must run where the transcriber app (separation) and drumjot_training (model) are
importable, with a GPU (e.g. the sandbox). Mirrors eval_paradb's env:
  MODELS_DIR=<models-cache> PYTHONPATH=dsp:training \
  python3 training/scripts/build_param_dataset.py --maps-dir <folder> \
      --checkpoint <dir> --out <table.npz> [--variants 6] [--stems-cache <dir>]
"""
import argparse
import gc
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # eval_paradb (sibling script)
sys.path.insert(0, os.path.join(_HERE, ".."))  # training/
sys.path.insert(0, os.path.join(_HERE, "..", "..", "transcriber"))

import eval_paradb as ep  # noqa: E402

from drumjot_training import (  # noqa: E402
    embeddings,
    forced_align,
    inference,
    postfilter,
    rlrr,
    runtime,
)
from drumjot_training.parampred import augment, dataset  # noqa: E402


def _separate_maps(zips, stems_cache, args):
    """Phase A: per map -> (zip, gt, drum_stem, {pitch: stem_path}). Mirrors
    eval_paradb so the corpus model input matches the eval input."""
    from app.pipeline.separate import Separator

    sep = Separator()
    sep.load()
    maps = []
    for zp in zips:
        print(f"\n=== {zp.name} (separate) ===", flush=True)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with zipfile.ZipFile(zp) as z:
                z.extractall(root)
            chart = ep._pick_rlrr(root)
            if chart is None:
                print("  no .rlrr; skipping", flush=True)
                continue
            gt = {
                ln: [t for t in ts if args.max_seconds is None or t < args.max_seconds]
                for ln, ts in rlrr.onsets_by_lane(chart).items()
            }
            drum_cached = stems_cache / f"{zp.stem}.drum.flac"
            if not drum_cached.exists():
                mix_wav = root / "_mix.wav"
                ok, case = ep.build_mix(
                    root, rlrr.song_tracks(chart), rlrr.drum_tracks(chart),
                    ep.SEP_SR, mix_wav, args.max_seconds, args.drum_corr_threshold,
                )
                if not ok:
                    print("  no resolvable audio; skipping", flush=True)
                    continue
                print(f"  mix: {case}", flush=True)
                drum_cached.write_bytes(Path(sep.run_stems_all(mix_wav, root).drum_stem).read_bytes())
            piece_cached = {p: stems_cache / f"{zp.stem}.{p}.flac" for p in ep.STEM_TO_LANES}
            if all(pp.exists() for pp in piece_cached.values()):
                pieces = dict(piece_cached)
            else:
                per = sep.run_stems_per(drum_cached, root).per_instrument
                pieces = {}
                for p, path in per.items():
                    if p in ep.STEM_TO_LANES:
                        piece_cached[p].write_bytes(Path(path).read_bytes())
                        pieces[p] = piece_cached[p]
            maps.append((zp, gt, drum_cached, pieces))
    del sep
    gc.collect()
    return maps


def _offset_corrected_gt(gt, drum_stem, args):
    """Apply eval_paradb's once-per-song global GT->audio offset (timing is
    augmentation-invariant, so it holds for every variant)."""
    env, env_fps = forced_align.onset_envelope(drum_stem, max_seconds=args.max_seconds)
    floor = postfilter.support_floor_from_env(env, args.support_percentile)
    off, _s0 = ep._global_offset(gt, env, env_fps, floor, args.align_window, args.offset_window)
    if not (args.offset_correct and abs(off) > args.offset_correct_min):
        return gt
    return {ln: [t + off for t in ts] for ln, ts in gt.items()}


def _paradb_sep_maps(root, args):
    """Maps from a `paradb-sep` tree (separate_paradb_dataset.py output): the
    kept-training per-instrument stems + already-offset-corrected onsets. No
    separation or offset-correction here -- it's all precomputed, so this reuses
    that GPU work instead of re-separating. Returns [(map_id, gt, pieces)]."""
    from drumjot_training import paradb

    root = Path(root)
    out = []
    for oj in sorted((root / "onsets").glob("*.json")):
        mid = oj.stem
        pieces = {p: root / "perstem" / p / f"{mid}.flac" for p in ep.STEM_TO_LANES}
        pieces = {p: a for p, a in pieces.items() if a.exists()}
        if not pieces:
            continue
        gt = paradb.onsets_by_lane(oj)
        if args.max_seconds is not None:
            gt = {ln: [t for t in ts if t < args.max_seconds] for ln, ts in gt.items()}
        out.append((mid, gt, pieces))
    return out


def main():
    ap = argparse.ArgumentParser(description="Build the param-predictor training corpus")
    ap.add_argument("--maps-dir", default=None, help="folder of .zip labeled maps (NOT the ParaDB eval set)")
    ap.add_argument("--paradb-sep", default=None,
                    help="alternative source: a paradb-sep tree (kept-training stems + corrected onsets) "
                    "from separate_paradb_dataset.py -- reuses its separation instead of re-separating. "
                    "Pool the resulting table with the real corpus via train_param_predictor --dataset a.npz b.npz")
    ap.add_argument("--checkpoint", required=True, help="frozen onset checkpoint (model.pt + meta.json)")
    ap.add_argument("--out", required=True, help="output npz Table")
    ap.add_argument("--variants", type=int, default=6, help="augmented variants per song (plus the original)")
    ap.add_argument("--no-codec", dest="codec", action="store_false", help="skip lossy-codec augmentation")
    ap.add_argument("--lanes", default=None,
                    help="comma-separated lanes to build rows for (e.g. hc,ho,rd,cr); stems carrying none "
                    "of them are skipped (the model never runs on them). Default: all checkpoint lanes.")
    ap.add_argument("--stems-cache", default=None, help="dir to cache separated stems across runs")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--window-seconds", type=float, default=30.0)
    ap.add_argument("--support-percentile", type=float, default=60.0)
    ap.add_argument("--align-window", type=float, default=0.03)
    ap.add_argument("--offset-window", type=float, default=0.05)
    ap.add_argument("--offset-correct-min", type=float, default=0.025)
    ap.add_argument("--no-offset-correct", dest="offset_correct", action="store_false")
    ap.add_argument("--drum-corr-threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    import librosa
    import soundfile as sf
    import torch

    runtime.tee_stdio(args.log)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.stems_cache:
        stems_cache = Path(args.stems_cache)
        stems_cache.mkdir(parents=True, exist_ok=True)
        stems_tmp = None
    else:
        stems_tmp = tempfile.TemporaryDirectory()
        stems_cache = Path(stems_tmp.name)

    # source -> unified [(song_id, offset-corrected gt, {pitch: stem_path})]
    if args.paradb_sep:
        songs = _paradb_sep_maps(args.paradb_sep, args)
        print(f"{len(songs)} paradb-sep maps (kept-training, pre-separated); "
              f"{args.variants} variants/song; checkpoint={args.checkpoint}", flush=True)
    else:
        if not args.maps_dir:
            ap.error("one of --maps-dir or --paradb-sep is required")
        zips = sorted(Path(args.maps_dir).glob("*.zip"))
        print(f"{len(zips)} maps; {args.variants} variants/song; checkpoint={args.checkpoint}", flush=True)
        maps = _separate_maps(zips, stems_cache, args)
        songs = [(zp.stem, _offset_corrected_gt(gt, drum, args), pieces) for zp, gt, drum, pieces in maps]
    if device == "cuda":
        torch.cuda.empty_cache()

    model, meta = inference.load_model(args.checkpoint, device)
    encoder = embeddings.MertEncoder(name=meta["encoder"], layer=meta["encoder_layer"])
    rng = np.random.default_rng(args.seed)
    rows: list[dataset.ParamSample] = []

    for song_id, gt_scored, pieces in songs:
        print(f"\n=== {song_id} (rows) ===", flush=True)
        req = {ln.strip() for ln in args.lanes.split(",")} if args.lanes else None
        for v in range(args.variants + 1):
            for pitch, stem_path in pieces.items():
                restrict = set(ep.STEM_TO_LANES.get(pitch, ()))
                if req is not None:
                    restrict &= req
                if not restrict:
                    continue
                wave, sr = librosa.load(str(stem_path), sr=ep.SEP_SR, mono=True)
                tmp_wav = None
                if v == 0:
                    audio_path, aug_label = str(stem_path), "identity"
                else:
                    wave, aug_label = augment.random_chain(wave, sr, rng, use_codec=args.codec)
                    fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
                    os.close(fd)
                    sf.write(tmp_wav, wave, sr)
                    audio_path = tmp_wav
                try:
                    probs, fps = inference.stitched_probs(
                        audio_path, model, meta, encoder, args.max_seconds, args.window_seconds
                    )
                    rows.extend(dataset.build_rows_for_song(
                        probs, fps, meta["lanes"], meta["thresholds"], gt_scored, wave, sr,
                        song_id=song_id, aug=aug_label,
                        default_threshold=meta["peak_threshold"], tolerance=meta["onset_tolerance_s"],
                        restrict_lanes=restrict,
                    ))
                finally:
                    if tmp_wav:
                        Path(tmp_wav).unlink(missing_ok=True)
            print(f"  variant {v}: {len(rows)} rows total", flush=True)

    if not rows:
        print("no rows produced; nothing written", flush=True)
        return
    table = dataset.Table.from_rows(rows)
    table.save(args.out)
    print(f"\nwrote {len(table)} rows ({len(table.lanes())} lanes, "
          f"{len(set(table.song.tolist()))} songs) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
