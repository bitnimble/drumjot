"""Head-capacity x data-scale sweep on the cymbal+hat stems.

Tests "is the per-lane RNN head too small to use the data" -- sweep head_hidden
(e.g. 128 vs 512) over the cymbal+hat per-stem pool, everything else fixed to the
validated baseline, BATCH HELD CONSTANT across arms, per-lane keep_best. If
crash/ride/hc climb with hidden size (at a converged epoch budget + enough data),
the cymbal ceiling is capacity/under-fit; if flat, it isn't.

History (RESULTS.md "Phase 1"): the first pass (cap-100, 12 epochs, single-window
train) came up flat but UNDER-TRAINED both arms (the per-epoch curves showed
ho/cr still climbing). The convergence run found ~30 epochs needed and that the
crash "spike" was small-val noise. So a fair capacity test needs: more data (now
unblocked -- full windowing + the full-windowed cache), >=~30 epochs, and an
enlarged val. That's what this script runs.

Reuses the production path: pooled per-stem indexing (restricted = full filtered to
the stem's lanes; full rides along as the sibling-weighting source) -> full
windowing -> materialize -> CachedClips -> train_loop. Features come from the
full-windowed _cache_mert_pooled (layer 10, hb16), so it's all cache hits (no
re-encode) as long as --layer / high-band / --window-search match how the cache
was built.

NFS READ PIPELINE: pass --num-workers>0 so the DataLoader streams .npy from the
(NFS) cache in worker processes that prefetch the next batches WHILE the GPU
trains the current one (pin_memory + persistent_workers are already on in
train_loop). Without it (num_workers=0) reads are serial and the GPU starves on
NFS latency. The only writes during training are occasional checkpoints
(infrequent, not per-step), so reads are the thing to pipeline.

Run with DRUMJOT_STAR/ENST/EGMD pointing at the sep trees, e.g.:
  python training/scripts/head_capacity_sweep.py \
      --hidden 128,512 --pool-cap 0 --epochs 40 --num-workers 8 \
      --cache /codebox-workspace/datasets/_cache_mert_pooled
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))  # training/
sys.path.insert(0, os.path.join(_HERE, "..", "..", "dsp"))  # dsp/

from drumjot_training import egmd, embeddings, enst, midi_labels, paths, runtime, star  # noqa: E402
from drumjot_training.config import Config  # noqa: E402
from drumjot_training.lanes import LANES  # noqa: E402
from drumjot_training.model import MultiLaneHeads  # noqa: E402
from drumjot_training.targets import pos_weights_from_targets  # noqa: E402
from drumjot_training.train import (  # noqa: E402
    _cap_by_clip,
    _window_specs,
    evaluate_clip,
    materialize,
    train_loop,
    tune_thresholds,
)

CYM_LANES = {"rd", "cr", "mc"}
HAT_LANES = {"hc", "hp", "ho"}
LANES_CH: tuple[str, ...] = ("hc", "hp", "ho", "rd", "cr", "mc")  # report order


def _pitch_to_stem(p2l) -> dict:
    """pitch -> 'c'/'h' for the cymbal/hat stems of one source's PERSTEM map."""
    out = {}
    for pitch, lanes in p2l.items():
        ls = set(lanes)
        if ls & CYM_LANES:
            out[pitch] = "c"
        elif ls & HAT_LANES:
            out[pitch] = "h"
    return out


def _source(name: str):
    """(train_clips, val_clips, ann_of, reader, p2l) for one sep-tree source,
    each clip list already filtered to the cymbal+hat stems."""
    root = paths.dataset_path(name)
    if name == "star":
        allper = star.perstem_index(root)
        tr = [c for c in allper if c.split == "training"]
        va = [c for c in allper if c.split in ("validation", "test")]
        ann_of, reader, p2l = (lambda c: c.annotation_path), star.onsets_by_lane, star.PERSTEM_TO_LANES
    elif name == "enst":
        allper = enst.perstem_index(root)
        tr = enst.perstem_for_split(allper, "train")
        va = enst.perstem_for_split(allper, "validation")
        ann_of, reader, p2l = (lambda c: c.annotation_path), enst.onsets_by_lane, enst.PERSTEM_TO_LANES
    elif name == "egmd":
        allper = egmd.perstem_index(root)
        tr = [c for c in allper if c.split == "train"]
        va = [c for c in allper if c.split == "validation"]
        ann_of, reader, p2l = (lambda c: c.midi_path), midi_labels.onsets_from_path, egmd.PERSTEM_TO_LANES
    else:
        raise SystemExit(f"unknown source {name!r}")
    stem = _pitch_to_stem(p2l)
    tr = [c for c in tr if stem.get(c.pitch) in ("c", "h")]
    va = [c for c in va if stem.get(c.pitch) in ("c", "h")]
    return tr, va, ann_of, reader, p2l


def build_specs(sources, cap, cache):
    """Pooled cymbal+hat per-stem specs (audio, restricted_onsets, full_onsets),
    capped per source by distinct source song. Restricted = full filtered to the
    stem's own lanes; full rides along for sibling weighting. Parsed onsets are
    memoized to _onsets.json beside the feature cache (same as _pooled_specs)."""
    ocp = Path(cache) / "_onsets.json"
    try:
        ocache = json.loads(ocp.read_text()) if ocp.exists() else {}
    except Exception:  # noqa: BLE001
        ocache = {}
    dirty = False

    def _full(path, reader):
        nonlocal dirty
        v = ocache.get(str(path))
        if v is None or any(ln not in v for ln in LANES):
            r = reader(path)
            v = {ln: list(r.get(ln, [])) for ln in LANES}
            ocache[str(path)] = v
            dirty = True
        return v

    tr_specs, va_specs = [], []
    for name in sources:
        tr, va, ann_of, reader, p2l = _source(name)

        def _spec(c, ann_of=ann_of, reader=reader, p2l=p2l):
            full = _full(ann_of(c), reader)
            keep = set(p2l.get(c.pitch, ()))
            restricted = {ln: (full[ln] if ln in keep else []) for ln in LANES}
            return (c.audio_path, restricted, full)

        tr_c = _cap_by_clip(tr, ann_of, cap)
        tr_specs += [_spec(c) for c in tr_c]
        va_specs += [_spec(c) for c in va]
        print(f"  {name:5} train={len(tr_c):5d}  val={len(va):5d}", flush=True)
    if dirty:
        tmp = ocp.with_name(ocp.name + ".tmp")
        tmp.write_text(json.dumps(ocache))
        os.replace(tmp, ocp)
    return tr_specs, va_specs


def make_cfg(hidden, layers):
    return Config(
        encoder=embeddings.MERT_NAME, encoder_fps=embeddings.MERT_FPS, encoder_layer=10,
        high_band=True, lanes=LANES_CH, head_hidden=hidden, head_layers=layers,
    )


def eval_per_lane(model, val_clips, cfg, thresholds):
    """mean per-lane F1 over val clips that actually carry that lane's onsets."""
    agg: dict[str, list] = {ln: [] for ln in cfg.lanes}
    for c in val_clips:
        f1 = evaluate_clip(model, c, cfg, thresholds)
        for ln in cfg.lanes:
            if c.onsets_by_lane.get(ln):
                agg[ln].append(f1[ln])
    return {ln: (sum(v) / len(v) if v else float("nan")) for ln, v in agg.items()}


def main():
    ap = argparse.ArgumentParser(description="Head-capacity x data-scale sweep (cymbals+hats)")
    ap.add_argument("--pool-sources", default="star,enst,egmd")
    ap.add_argument("--pool-cap", type=int, default=0, help="max source-songs per dataset (0 = all)")
    ap.add_argument("--hidden", default="128,512", help="comma list of head hidden sizes to sweep")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=80, help="absolute epoch cap (early-stop ends sooner)")
    ap.add_argument("--early-stop", default=True, action=argparse.BooleanOptionalAction,
                    help="stop each arm once every lane's val-F1 has converged (flat trend + low "
                    "jitter over --es-window epochs); --epochs is the cap. --no-early-stop = fixed epochs")
    ap.add_argument("--es-window", type=int, default=8, help="epochs in the convergence window")
    ap.add_argument("--es-slope", type=float, default=0.002, help="max |val-F1 slope| (per epoch) to be 'flat'")
    ap.add_argument("--es-jitter", type=float, default=0.015, help="max residual std around the trend")
    ap.add_argument("--es-min-epochs", type=int, default=20, help="never stop before this many epochs")
    ap.add_argument("--batch", type=int, default=8, help="held constant across arms (fair capacity A/B)")
    ap.add_argument("--num-workers", type=int, default=8,
                    help="DataLoader prefetch workers: stream .npy from the (NFS) cache in parallel "
                    "WHILE the GPU trains, so it doesn't starve on NFS latency. 0 = serial reads.")
    ap.add_argument("--train-max-windows", type=int, default=0,
                    help="windows per TRAIN song (0 = all = full windowing; uses all the data). Must "
                    "match how the cache was built to stay cache-hit (the cache is full-windowed).")
    ap.add_argument("--val-max-windows", type=int, default=4,
                    help="windows per VAL song (>1 enlarges/diversifies val -> damps lucky-epoch F1; "
                    "0 = all). The first N windows are a subset of the full-windowed cache, so still hits.")
    ap.add_argument("--cache", default="/codebox-workspace/datasets/_cache_mert_pooled")
    ap.add_argument("--out-json", default="head_capacity_results.json")
    ap.add_argument("--probe-timing", action="store_true",
                    help="materialize, then train 1 epoch at the LARGEST hidden and exit "
                    "(measure s/epoch before committing to the full sweep)")
    args = ap.parse_args()

    import torch

    runtime.configure_backends()
    sources = [s.strip() for s in args.pool_sources.split(",") if s.strip()]
    hiddens = [int(h) for h in args.hidden.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    cache = Path(args.cache)
    log = lambda s: print(s, flush=True)  # noqa: E731

    log(f"=== head-capacity sweep: cymbals+hats, cap={args.pool_cap}, hidden={hiddens} "
        f"layers={args.layers} seeds={seeds} epochs={args.epochs} batch={args.batch} "
        f"num_workers={args.num_workers} train_win={args.train_max_windows} val_win={args.val_max_windows} ===")
    log("indexing pooled cym+hat specs:")
    tr_specs, va_specs = build_specs(sources, args.pool_cap, cache)
    log(f"total: {len(tr_specs)} train / {len(va_specs)} val cym+hat stem clips")

    # one materialize pass (features are head-size independent); reuse across arms.
    cfg0 = make_cfg(hiddens[0], args.layers)
    encoder = embeddings.make_encoder(cfg0.encoder, cfg0.encoder_layer)
    tr_w = _window_specs(tr_specs, 30.0, 3.0, args.train_max_windows, plan_cache_dir=cache)
    va_w = _window_specs(va_specs, 30.0, 3.0, args.val_max_windows, plan_cache_dir=cache)
    t0 = time.perf_counter()
    train_clips = materialize(tr_w, encoder, cfg0, cache, 30.0, "train", log)
    val_clips = materialize(va_w, encoder, cfg0, cache, 30.0, "val", log)
    log(f"materialize done in {time.perf_counter() - t0:.0f}s "
        f"({len(train_clips)} train / {len(val_clips)} val windows)")
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pos_w = pos_weights_from_targets(train_clips.iter_targets(), cap=50.0)
    log(f"pos_weights: {dict((ln, round(float(w), 1)) for ln, w in zip(cfg0.lanes, pos_w, strict=True))}")
    in_dim = embeddings.feat_dim(True)

    if args.probe_timing:
        H = max(hiddens)
        cfg = make_cfg(H, args.layers)
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        model = MultiLaneHeads(in_dim=in_dim, hidden=H, num_layers=args.layers, lane_names=cfg.lanes)
        if torch.cuda.is_available():
            model = model.cuda()
        ep0 = time.perf_counter()
        val_probe = [val_clips[i] for i in range(min(20, len(val_clips)))]
        train_loop(model, train_clips, cfg, epochs=1, pos_weight=pos_w, batch_size=args.batch,
                   num_workers=args.num_workers, val_clips=val_probe, keep_best=True, log=log)
        dt = time.perf_counter() - ep0
        n_arms = len(hiddens) * len(seeds)
        log(f"\n[timing] hidden={H} batch={args.batch} workers={args.num_workers}: ~{dt:.0f}s/epoch")
        log(f"[timing] full sweep ~= {n_arms} arms x {args.epochs} epochs (smaller hidden faster) "
            f"-> rough upper bound {n_arms * args.epochs * dt / 3600:.1f}h")
        return

    results = {}
    for H in hiddens:
        for seed in seeds:
            cfg = make_cfg(H, args.layers)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model = MultiLaneHeads(in_dim=in_dim, hidden=H, num_layers=args.layers, lane_names=cfg.lanes)
            if torch.cuda.is_available():
                model = model.cuda()
            t1 = time.perf_counter()
            hist = train_loop(model, train_clips, cfg, epochs=args.epochs, pos_weight=pos_w,
                              batch_size=args.batch, num_workers=args.num_workers,
                              val_clips=val_clips, keep_best=True, log=log,
                              early_stop=args.early_stop, es_window=args.es_window,
                              es_slope=args.es_slope, es_jitter=args.es_jitter,
                              es_min_epochs=args.es_min_epochs)
            thr = tune_thresholds(model, val_clips, cfg)
            f1 = eval_per_lane(model, val_clips, cfg, thr)
            # dense per-epoch curves (UNTUNED 0.5-thr val F1, every epoch) -> tell if a
            # lane is still climbing at the final epoch (under-trained) vs plateaued.
            best_ep = hist.get("best_epoch_by_lane")
            curve = {"val_macro": hist.get("val_f1", [])}
            for ln in cfg.lanes:
                curve[f"vf1_{ln}"] = hist.get(f"vf1_{ln}", [])
            results[f"h{H}_s{seed}"] = {
                "hidden": H, "seed": seed, "f1": f1,
                "thr": {ln: thr.get(ln) for ln in cfg.lanes},
                "best_epoch_by_lane": {ln: (int(best_ep[i]) if best_ep else None)
                                       for i, ln in enumerate(cfg.lanes)},
                "curve": curve,
            }
            log(f"\n>>> hidden={H} seed={seed} ({time.perf_counter() - t1:.0f}s): "
                + " ".join(f"{ln}={f1[ln]:.3f}" for ln in LANES_CH))
            log(f"--- h{H} s{seed} per-epoch UNTUNED val F1 curve (climb check) ---")
            log("ep   " + "".join(f"{ln:>7s}" for ln in LANES_CH) + "   macro")
            for e in range(len(curve["val_macro"])):
                cells = "".join(f"{curve[f'vf1_{ln}'][e]:7.3f}" if e < len(curve[f"vf1_{ln}"])
                                else f"{'-':>7s}" for ln in LANES_CH)
                log(f"{e:<3d} {cells}  {curve['val_macro'][e]:6.3f}")
            # atomic per-arm checkpoint (a SIGKILL mid-write must not truncate results)
            _tmp = Path(args.out_json + ".tmp")
            _tmp.write_text(json.dumps(results, indent=2))
            os.replace(_tmp, args.out_json)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    log("\n==== summary: per-lane F1, mean+/-std over seeds ====")
    log("  hidden " + "".join(f"{ln:>14s}" for ln in LANES_CH))
    for H in hiddens:
        row = f"  h{H:<5d}"
        for ln in LANES_CH:
            vals = [results[f"h{H}_s{s}"]["f1"][ln] for s in seeds
                    if not np.isnan(results[f"h{H}_s{s}"]["f1"][ln])]
            row += f"  {np.mean(vals):.3f}+/-{np.std(vals):.3f}" if vals else f"{'nan':>14s}"
        log(row)
    cym_sum = {H: sum(np.nanmean([results[f"h{H}_s{s}"]["f1"][ln] for s in seeds]) for ln in CYM_LANES)
               for H in hiddens}
    log(f"\ncymbal-sum (rd+cr+mc) by hidden: {dict((H, round(v, 3)) for H, v in cym_sum.items())}")
    log(f"\nresults -> {args.out_json}")


if __name__ == "__main__":
    main()
