"""Build the param-predictor corpus from the per-stem TRAINING datasets, reusing
the MERT feature cache the model was trained against.

train.py windows each stem (`_window_plan.json`, nudged ~30 s cuts) and caches
per-window MERT+high-band features under `mert_cache`. So the IDENTITY
pass is free: load cached features -> run the tiny GRU heads -> activation curve,
NO MERT. Augmented variants -- different audio -- are encoded on the GPU and
persisted to a probs cache (probs_cache) so re-runs / more variants / resumed
builds never re-encode.

GPU-feeding pipeline (so the GPU never waits on the CPU codec): a pool of CPU
worker threads runs augmentation AHEAD into a bounded queue; the main thread does
only encode + heads. Augmented work is bounded to `--aug-windows` windows/stem x
`--variants` so an overnight run fits; identity uses those same windows (free).

Datasets NOT in train.py's MERT cache (A2MD / MDB -- real songs through our
separation) take a fresh-encode path instead: windows are planned on the fly and
the identity window is encoded on the GPU (its probs are still cached, so a
resumed build never re-encodes). This is the real-domain corpus that the
synthetic-only predictor lacked.

Per-window granularity: each window (+ its window-relative GT) is one oracle unit.
Hat+cymbal: `--pitches h,c`. These are TRAINING datasets; ParaDB stays the test.

  PYTHONPATH=dsp:training python3 training/scripts/build_param_dataset_perstem.py \
      --checkpoint <ckpt> --out <table.npz> \
      --star-root .../star_balanced_sep --enst-root .../enst-sep --egmd-root .../egmd_sep \
      --a2md-root .../a2md_sep \
      --feature-cache .../mert_cache [--variants 4] [--aug-windows 1] [--workers 3]
"""
import argparse
import hashlib
import json
import os
import queue
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # training/

from drumjot_training import (  # noqa: E402
    a2md,
    clean,
    egmd,
    embeddings,
    enst,
    inference,
    postfilter,
    runtime,
    star,
)
from drumjot_training.parampred import dataset, probs_cache  # noqa: E402

SR = embeddings.HB_SR  # 44100; encode resamples to MERT sr, high-band wants 44.1k
WINDOW, SEARCH = 30.0, 3.0  # must match train.py (win = max_seconds or 30.0; window_search 3.0)

_LOADERS = {
    # name -> (perstem_index, restricted_onsets, annotation-attr, split-group attr | None)
    "star": (star.perstem_index, star.restricted_onsets, "annotation_path", "split"),
    "enst": (enst.perstem_index, enst.restricted_onsets, "annotation_path", "drummer"),
    "egmd": (egmd.perstem_index, egmd.restricted_onsets, "midi_path", "split"),
    # A2MD: real songs separated to per-instrument stems; full-song MIDI (drum
    # onsets are channel-9-only, see a2md.py). NOT in the model's MERT cache, so
    # _prepare_stem encodes it fresh. No split metadata (all of it is corpus).
    "a2md": (a2md.perstem_index, a2md.restricted_onsets, "midi_path", None),
}


def _iter_clips(name, root, pitches, max_clips, rng, splits=None):
    index_fn, onsets_fn, ann_attr, group_attr = _LOADERS[name]
    clips = [c for c in index_fn(root) if c.pitch in pitches]
    if splits:  # keep only clips whose split / drummer is requested (held-out vs train)
        if group_attr is None:
            raise SystemExit(f"--splits not supported for dataset '{name}' (no split metadata)")
        clips = [c for c in clips if getattr(c, group_attr) in splits]
    rng.shuffle(clips)
    if max_clips:
        clips = clips[:max_clips]
    for c in clips:
        yield str(c.audio_path), c.pitch, onsets_fn(getattr(c, ann_attr), c.pitch)


def _stem_seed(stem_path):
    return int(hashlib.sha1(str(stem_path).encode()).hexdigest()[:8], 16)


def _support_gate(wgt, wave_w, args, librosa):
    """Per-(window, lane) label-quality gate: drop a lane's GT when too few of its
    onsets land on a real transient of THIS stem (mislabeled / mis-aligned -- e.g.
    an A2MD ride that doesn't match the recording). Dataset-agnostic; a no-op on
    clean labels (synthetic STAR scores ~1.0). Uses the ORIGINAL window audio so
    the decision is identical for every augmented variant."""
    if args.min_support <= 0.0:
        return wgt
    env = librosa.onset.onset_strength(y=wave_w, sr=SR, hop_length=64).astype(np.float64)
    floor = postfilter.support_floor_from_env(env, args.support_percentile)
    filtered, _ = clean.filter_lanes_by_support(
        wgt, env, SR / 64.0, support_floor=floor, min_support=args.min_support,
        window_s=args.support_window, snap=True)  # snap kept onsets onto their stem's transient
    return filtered


# ---- stage 1 (CPU worker threads): prepare encode-ready sub-items per stem ----
def _prepare_stem(task, plan, meta, args, req_lanes):
    """Pure-CPU: load the stem wave, pick windows, and for each (window, variant)
    build a sub-item ready for the GPU. No model / GPU here.

    Stems in train.py's window-plan + MERT cache take the FREE identity path
    (cached per-window features -> heads). Stems NOT in the cache (A2MD / MDB
    real-domain data) plan windows fresh here and encode the identity window on
    the GPU too (its probs are still cached, so resumes don't re-encode)."""
    import librosa

    from drumjot_training import train

    stem_path, pitch, gt = task
    restrict = set(star.PERSTEM_TO_LANES.get(pitch, ())) & req_lanes
    if not restrict:
        return []
    wins = plan.get(probs_cache.window_plan_key(stem_path, WINDOW, SEARCH))
    cached = wins is not None
    if not cached:  # A2MD/MDB: not in the model's MERT cache -> plan + encode fresh
        wins = train.plan_windows(stem_path, WINDOW, SEARCH, max_windows=0)
    if not wins:
        return []
    variant_tok = embeddings.feat_variant(meta.get("high_band", True))
    wave44, _ = librosa.load(stem_path, sr=SR, mono=True)
    song_id = Path(stem_path).stem
    # windows that actually carry our lanes, then pick up to --aug-windows of them
    usable = [(s, ln) for s, ln in wins if any(probs_cache.window_onsets(gt, s, ln).get(x) for x in restrict)]
    rng = np.random.default_rng(_stem_seed(stem_path))
    rng.shuffle(usable)
    items = []
    for start, length in usable[: max(1, args.aug_windows)]:
        wgt = probs_cache.window_onsets(gt, start, length)
        wave_w = wave44[int(start * SR): int((start + length) * SR)]
        wgt = _support_gate(wgt, wave_w, args, librosa)  # drop mis-aligned (lane) labels
        if not any(wgt.get(ln) for ln in restrict):
            continue  # the gate removed every lane this window carried
        if cached:  # free: cached per-window MERT features -> heads
            feat = probs_cache.load_window_features(
                args.feature_cache, stem_path, start, length,
                encoder=meta["encoder"], layer=meta["encoder_layer"], variant=variant_tok)
            if feat is not None:
                items.append(dict(kind="feat", payload=feat, wave=wave_w, gt=wgt,
                                  song=song_id, aug="identity", restrict=restrict))
        else:  # not cached: encode the identity window on the GPU (probs-cached like a variant)
            key = probs_cache.probs_key(
                f"{stem_path}@{start:.3f}", "identity", encoder=meta["encoder"], layer=meta["encoder_layer"],
                in_dim=meta["in_dim"], max_seconds=length, window_seconds=None)
            hit = probs_cache.load_probs(args.probs_cache, key)
            if hit is not None:
                items.append(dict(kind="probs", payload=hit[0], wave=wave_w, gt=wgt,
                                  song=song_id, aug="identity", restrict=restrict))
            else:
                items.append(dict(kind="encode", payload=wave_w, key=key, wave=wave_w, gt=wgt,
                                  song=song_id, aug="identity", restrict=restrict))
        for v in range(1, args.variants + 1):
            aug_wave, recipe = probs_cache.variant_audio(f"{stem_path}@{start:.3f}", v, wave_w, SR, codec=args.codec)
            key = probs_cache.probs_key(
                f"{stem_path}@{start:.3f}", recipe, encoder=meta["encoder"], layer=meta["encoder_layer"],
                in_dim=meta["in_dim"], max_seconds=length, window_seconds=None)
            hit = probs_cache.load_probs(args.probs_cache, key)
            if hit is not None:
                items.append(dict(kind="probs", payload=hit[0], wave=aug_wave, gt=wgt,
                                  song=song_id, aug=recipe, restrict=restrict))
            else:
                items.append(dict(kind="encode", payload=aug_wave, key=key, wave=aug_wave, gt=wgt,
                                  song=song_id, aug=recipe, restrict=restrict))
    return items


def _prefetch(tasks, prepare, n_workers, buffer):
    """Run `prepare` over `tasks` in `n_workers` threads, yielding results as they
    finish. A bounded out-queue gives backpressure so workers stay only `buffer`
    items ahead of the consumer (GPU) -- enough that the GPU never waits, without
    buffering the whole dataset."""
    tasks = list(tasks)
    task_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue(maxsize=buffer)
    for t in tasks:
        task_q.put(t)
    for _ in range(n_workers):
        task_q.put(None)

    def worker():
        while True:
            t = task_q.get()
            if t is None:
                return
            try:
                out_q.put(("ok", prepare(t)))
            except Exception as e:  # noqa: BLE001 -- surface, don't kill the run
                out_q.put(("err", (t, repr(e))))

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_workers)]
    for th in threads:
        th.start()
    for _ in range(len(tasks)):
        yield out_q.get()
    for th in threads:
        th.join()


# ---- stage 2 (main thread / GPU): encode + heads, then oracle inline ----
def _heads_probs(model, feat, device):
    import torch

    x = torch.as_tensor(np.asarray(feat, dtype=np.float32), device=device).unsqueeze(0)
    with torch.no_grad(), runtime.autocast():
        return torch.sigmoid(model(x))[0].float().cpu().numpy()


def _encode_window(wave44, meta, encoder):
    feat = encoder.encode(wave44, SR)
    if meta.get("high_band", meta["in_dim"] > embeddings.MERT_DIM):
        feat = np.concatenate([feat, embeddings.highband_from_wave(wave44, feat.shape[0])], axis=1)
    return feat


def _consume(item, model, meta, encoder, device, args):
    """GPU + oracle for one prepared sub-item -> rows."""
    if item["kind"] == "probs":
        probs = item["payload"]
    elif item["kind"] == "feat":
        probs = _heads_probs(model, item["payload"], device)
    else:  # encode (augmented, GPU)
        probs = _heads_probs(model, _encode_window(item["payload"], meta, encoder), device)
        probs_cache.save_probs(args.probs_cache, item["key"], probs, meta["encoder_fps"])
    return dataset.build_rows_for_song(
        probs, meta["encoder_fps"], meta["lanes"], meta["thresholds"], item["gt"], item["wave"], SR,
        song_id=item["song"], aug=item["aug"],
        default_threshold=meta["peak_threshold"], tolerance=meta["onset_tolerance_s"],
        restrict_lanes=item["restrict"])


def _validate(clips, plan, model, meta, encoder, device, args):
    """A cached-feature identity curve must match a fresh re-encode of the same
    window audio (proves the cache reuse + slicing are correct)."""
    import librosa

    variant_tok = embeddings.feat_variant(meta.get("high_band", True))
    n = 0
    for stem_path, _pitch, _gt in clips:
        wins = plan.get(probs_cache.window_plan_key(stem_path, WINDOW, SEARCH))
        if not wins:
            continue
        wave44, _ = librosa.load(stem_path, sr=SR, mono=True)
        start, length = wins[0]
        feat_c = probs_cache.load_window_features(args.feature_cache, stem_path, start, length,
                    encoder=meta["encoder"], layer=meta["encoder_layer"], variant=variant_tok)
        if feat_c is None:
            continue
        wave_w = wave44[int(start * SR): int((start + length) * SR)]
        feat_f = _encode_window(wave_w, meta, encoder)
        m = min(feat_c.shape[0], feat_f.shape[0])
        fd = float(np.max(np.abs(feat_c[:m].astype(np.float32) - feat_f[:m])))
        pc, pf = _heads_probs(model, feat_c, device), _heads_probs(model, feat_f, device)
        pm = min(pc.shape[1], pf.shape[1])
        print(f"  {Path(stem_path).name[:38]:38s} feat|d|={fd:.4f} probs|d|={np.max(np.abs(pc[:, :pm]-pf[:, :pm])):.4f} "
              f"frames c={feat_c.shape[0]} f={feat_f.shape[0]}", flush=True)
        n += 1
        if n >= args.validate:
            break


def main():
    ap = argparse.ArgumentParser(description="Build the param corpus from per-stem datasets (cache-aware, pipelined)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--star-root", default=None)
    ap.add_argument("--enst-root", default=None)
    ap.add_argument("--egmd-root", default=None)
    ap.add_argument("--a2md-root", default=None, help="A2MD separated root (real-domain; encoded fresh)")
    ap.add_argument("--feature-cache", default="/codebox-workspace/mert_cache")
    ap.add_argument("--probs-cache", default="/codebox-workspace/datasets/_cache_param_probs")
    ap.add_argument("--pitches", default="h,c")
    ap.add_argument("--lanes", default="hc,ho,rd,cr")
    ap.add_argument("--splits", default=None,
                    help="comma list of split/drummer names to keep (e.g. test,validation,drummer_3 "
                    "for held-out; training,train,drummer_1,drummer_2 for trained-on). Default: all.")
    ap.add_argument("--min-support", type=float, default=0.95,
                    help="per-(window,lane) label-quality gate: drop a lane whose onsets' support "
                    "(fraction landing on a real transient of its stem) is below this. 0 = off. "
                    "Applies to ALL datasets (no-op on clean labels; rescues noisy real ones like A2MD/ParaDB).")
    ap.add_argument("--support-percentile", type=float, default=60.0, help="adaptive support floor percentile")
    ap.add_argument("--support-window", type=float, default=0.04,
                    help="+/- window (s) for the support check + onset snap (kept onsets are snapped onto "
                    "their stem's transient within this window)")
    ap.add_argument("--variants", type=int, default=4, help="augmented variants per window (plus free identity)")
    ap.add_argument("--aug-windows", type=int, default=1, help="windows per stem to augment (identity uses the same)")
    ap.add_argument("--max-clips-per-dataset", type=int, default=0, help="cap stems per dataset (0 = all)")
    ap.add_argument("--workers", type=int, default=3, help="CPU augmentation worker threads (feed the GPU)")
    ap.add_argument("--prefetch", type=int, default=4, help="stems prepared ahead of the GPU")
    ap.add_argument("--no-codec", dest="codec", action="store_false")
    ap.add_argument("--save-every", type=int, default=150, help="checkpoint the table every N stems")
    ap.add_argument("--validate", type=int, default=0, help="run N cache-vs-fresh checks then exit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    import torch

    runtime.tee_stdio(args.log)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.probs_cache).mkdir(parents=True, exist_ok=True)
    pitches = {p.strip() for p in args.pitches.split(",")}
    req_lanes = {ln.strip() for ln in args.lanes.split(",")}
    roots = {k: v for k, v in {"star": args.star_root, "enst": args.enst_root,
                               "egmd": args.egmd_root, "a2md": args.a2md_root}.items() if v}
    if not roots:
        ap.error("provide at least one of --star-root / --enst-root / --egmd-root / --a2md-root")

    plan_path = Path(args.feature_cache) / "_window_plan.json"  # absent for a pure fresh-encode (A2MD-only) build
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    model, meta = inference.load_model(args.checkpoint, device)
    model = model.to(device)  # load_model leaves heads on CPU; run them on the GPU too
    encoder = embeddings.MertEncoder(name=meta["encoder"], layer=meta["encoder_layer"])
    rng = np.random.default_rng(args.seed)

    splits = {s.strip() for s in args.splits.split(",")} if args.splits else None
    all_clips = []
    for name, root in roots.items():
        cs = list(_iter_clips(name, root, pitches, args.max_clips_per_dataset, rng, splits))
        print(f"{name}: {len(cs)} stems", flush=True)
        all_clips += cs

    if args.validate:
        print(f"\n== validate: cached features vs fresh re-encode ({args.validate}) ==", flush=True)
        _validate(all_clips, plan, model, meta, encoder, device, args)
        return

    rows: list = []
    done = errs = 0
    for status, payload in _prefetch(
        all_clips, lambda t: _prepare_stem(t, plan, meta, args, req_lanes), args.workers, args.prefetch
    ):
        if status == "err":
            errs += 1
            print(f"  WARN prepare failed: {payload[0][0]} {payload[1]}", flush=True)
            continue
        for item in payload:  # GPU + oracle on the main thread (augmentation already done)
            rows += _consume(item, model, meta, encoder, device, args)
        done += 1
        if done % args.save_every == 0:
            dataset.Table.from_rows(rows).save(args.out)
            print(f"  {done}/{len(all_clips)} stems -> {len(rows)} rows (checkpointed; {errs} errs)", flush=True)

    if not rows:
        print("no rows produced; nothing written", flush=True)
        return
    table = dataset.Table.from_rows(rows)
    table.save(args.out)
    print(f"\nwrote {len(table)} rows ({len(table.lanes())} lanes, {len(set(table.song.tolist()))} songs; "
          f"{errs} prepare errors) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
