"""Overnight cymbal loss A/B: focal & crash-oversample vs the baseline.

The miss-typing (RESULTS.md 2026-06-18) showed the cymbal recall gap is the head
UNDER-FIRING on true onsets (`dead` misses: ride 43.7%, crash 74.6%), worst on
the rare class -- and that plain frequency reweighting is already saturated
(crash pos_weight is pinned at the cap-50 ceiling and still dead). So the lever is
a different training signal. This A/B trains two arms against that baseline and
scores them with the SAME decomposition + miss-typing, so the comparison is
apples-to-apples and the key number is crash's `dead`-rate:

  - baseline         : load the existing checkpoint, score only (the control).
  - focal            : CenterNet penalty-reduced focal (`--loss focal`) -- ignores
                       pos_weight, concentrates gradient on HARD frames (= the dead
                       crashes). Never A/B'd before (RESULTS marked it "pending").
  - crash_oversample : duplicate the crash stems N x in the train pool so crash
                       gets more distinct gradient steps (data-level balance).

Built for an unattended overnight run on the idle 3080:
  - arms run SEQUENTIALLY and are FAIL-INDEPENDENT (a crash in one still runs the
    rest and still writes the comparison table);
  - each training arm is RESUMABLE (per-epoch resume checkpoint) -- re-run the
    same command to continue after a kill;
  - self-logs next to --out-json; --epochs 60 caps each arm ~3.5 h (the baseline
    converged at ep50; cymbal bests land by ~ep25), so both fit well inside 9 h.

  DRUMJOT_STAR=/codebox-workspace/datasets/star_balanced_sep \
  DRUMJOT_ENST=/codebox-workspace/datasets/enst-sep \
  DRUMJOT_EGMD=/codebox-workspace/datasets/egmd_sep \
  python training/scripts/cymbal_loss_ab.py \
      --cache /codebox-workspace/datasets/_cache_mert_pooled \
      --out-json /codebox-workspace/cymbal_loss_ab.json

Validate the oversample logic without cache/GPU:  python ... --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # head_capacity_sweep, cymbal_recall_confusion, cymbal_miss_typing
sys.path.insert(0, os.path.join(_HERE, ".."))  # training/
sys.path.insert(0, os.path.join(_HERE, "..", "..", "dsp"))  # dsp/

CYM = ("rd", "cr")
ALL_ARMS = ("baseline", "focal", "crash_oversample")


def _use_cuda(args) -> bool:
    import torch

    return args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())


def crash_oversample_specs(tr_specs, factor: int):
    """Duplicate crash-stem specs `factor`x (1 = no-op). A crash stem has onsets
    in the 'cr' lane and none in 'rd' (per-stem maps are single-lane). Returns
    (new_specs, n_crash_specs)."""
    crash = [s for s in tr_specs if s[1].get("cr") and not s[1].get("rd")]
    if factor <= 1:
        return list(tr_specs), len(crash)
    return list(tr_specs) + crash * (factor - 1), len(crash)


def _miss_typing(model, val_clips, cfg, thresholds):
    """One forward over val -> per-lane miss typing (the `dead`-rate is the key
    A/B metric: did the head start firing on the rare class?)."""
    import numpy as np
    from cymbal_miss_typing import type_misses

    from drumjot_training import metrics
    from drumjot_training.train import _clip_probs

    lane_i = {ln: i for i, ln in enumerate(cfg.lanes)}
    acts = {ln: [] for ln in CYM}
    refs = {ln: [] for ln in CYM}
    for clip in val_clips:
        probs = _clip_probs(model, clip)
        for ln in CYM:
            acts[ln].append(np.ascontiguousarray(probs[lane_i[ln]]))
            refs[ln].append(np.asarray(clip.onsets_by_lane.get(ln, []), dtype=np.float64))
    fps, tol_s = cfg.encoder_fps, cfg.onset_tolerance_s
    typing = {}
    for ln in CYM:
        base = dict(metrics.LANE_PEAK_PARAMS[ln])
        thr = thresholds.get(ln, cfg.peak_threshold)
        cats, n_ref, n_hit = type_misses(acts[ln], refs[ln], fps, thr, base, tol_s)
        n_miss = n_ref - n_hit
        typing[ln] = {"n_ref": n_ref, "n_hit": n_hit, "n_miss": n_miss,
                      "dead": cats["dead"], "dead_rate": (cats["dead"] / n_miss if n_miss else 0.0),
                      "cats": cats}
    return typing


def score_arm(model, val_clips, cfg, thresholds, log):
    """Decomposition (hit/confused/missed/fp + R/P/F1) + miss typing for one arm."""
    from cymbal_recall_confusion import _report, decompose

    agg = decompose(model, val_clips, cfg, thresholds, log)
    decomp = _report(agg, log)
    typing = _miss_typing(model, val_clips, cfg, thresholds)
    return {"decomp": decomp, "typing": typing,
            "thresholds": {ln: thresholds.get(ln, cfg.peak_threshold) for ln in CYM}}


def _comparison_table(results, log):
    log("\n==== cymbal loss A/B comparison (val) ====")
    log(f"  {'arm':16s} | {'rd R':>5s} {'rd P':>5s} {'rd F1':>5s} {'rd dead':>7s} | "
        f"{'cr R':>5s} {'cr P':>5s} {'cr F1':>5s} {'cr dead':>7s}")
    for arm in ALL_ARMS:
        r = results.get(arm)
        if not r:
            log(f"  {arm:16s} | (not run / failed)")
            continue
        d, t = r["decomp"], r["typing"]
        cells = []
        for ln in CYM:
            o = d.get(ln, {})
            cells.append(f"{o.get('recall', 0):5.3f} {o.get('precision', 0):5.3f} "
                         f"{o.get('f1', 0):5.3f} {t[ln]['dead_rate']:7.1%}")
        log(f"  {arm:16s} | {cells[0]} | {cells[1]}")
    base = results.get("baseline")
    if base:
        for arm in ("focal", "crash_oversample"):
            r = results.get(arm)
            if not r:
                continue
            d_cr = r["decomp"].get("cr", {}).get("recall", 0) - base["decomp"].get("cr", {}).get("recall", 0)
            d_dead = r["typing"]["cr"]["dead_rate"] - base["typing"]["cr"]["dead_rate"]
            log(f"  -> {arm}: crash recall {d_cr:+.3f}, crash dead-rate {d_dead:+.1%} vs baseline")


def _selftest():
    specs = [
        ("ride1.wav", {"rd": [1.0], "cr": []}, {}),
        ("crash1.wav", {"rd": [], "cr": [2.0]}, {}),
        ("crash2.wav", {"rd": [], "cr": [3.0]}, {}),
        ("hat1.wav", {"rd": [], "cr": [], "hc": [4.0]}, {}),
    ]
    out, n = crash_oversample_specs(specs, 1)
    assert n == 2 and len(out) == 4, (n, len(out))
    out, n = crash_oversample_specs(specs, 3)
    assert n == 2 and len(out) == 4 + 2 * 2, (n, len(out))  # 2 crash specs x (3-1) extra
    crash_paths = [s[0] for s in out if s[0].startswith("crash")]
    assert crash_paths.count("crash1.wav") == 3 and crash_paths.count("crash2.wav") == 3, crash_paths
    print("SELFTEST OK (crash_oversample_specs duplicates only crash stems)", flush=True)


def _train_arm(arm, args, cache, sources, cfg, in_dim, log):
    """Train one arm -> (model, thresholds). Reuses the warm cache (no encoder)."""
    import torch
    from cymbal_miss_typing import _CacheKeyEncoder
    from head_capacity_sweep import build_specs

    from drumjot_training.model import MultiLaneHeads
    from drumjot_training.targets import pos_weights_from_targets
    from drumjot_training.train import _window_specs, materialize, train_loop, tune_thresholds

    tr_specs, va_specs = build_specs(sources, args.pool_cap, cache, aligned_path=args.aligned_onsets)
    if arm == "crash_oversample":
        tr_specs, n_crash = crash_oversample_specs(tr_specs, args.crash_oversample)
        log(f"  crash-oversample x{args.crash_oversample}: {n_crash} crash stems -> "
            f"{len(tr_specs)} train specs")
    enc = _CacheKeyEncoder(cfg.encoder, cfg.encoder_layer)
    tr_w = _window_specs(tr_specs, 30.0, 3.0, 0, plan_cache_dir=cache)
    va_w = _window_specs(va_specs, 30.0, 3.0, 4, plan_cache_dir=cache)
    train_clips = materialize(tr_w, enc, cfg, cache, 30.0, "train", log)  # type: ignore[arg-type]
    val_clips = materialize(va_w, enc, cfg, cache, 30.0, "val", log)  # type: ignore[arg-type]

    uc = _use_cuda(args)
    model = MultiLaneHeads(in_dim=in_dim, hidden=args.hidden, num_layers=args.layers,
                           lane_names=cfg.lanes)
    if uc:
        model = model.cuda()
    if args.loss_for(arm) == "focal":
        pos_w = 1.0  # focal ignores pos_weight (targets hard frames directly)
    else:
        pos_w = pos_weights_from_targets(train_clips.iter_targets(), cap=args.pos_weight_cap)
        log(f"  pos_weights: {dict((ln, round(float(w), 1)) for ln, w in zip(cfg.lanes, pos_w, strict=True))}")
    torch.manual_seed(args.seed)
    if uc:
        torch.cuda.manual_seed_all(args.seed)
    resume = str(Path(args.ckpt_dir) / f"loss_ab_{arm}.resume.pt")
    train_loop(model, train_clips, cfg, epochs=args.epochs, pos_weight=pos_w,
               batch_size=args.batch, num_workers=args.num_workers, val_clips=val_clips,
               keep_best=True, log=log, early_stop=True, es_min_epochs=20,
               warmup_steps=args.warmup, loss_fn=args.loss_for(arm), resume_path=resume)
    thresholds = tune_thresholds(model, val_clips, cfg)
    log(f"  tuned thresholds: {dict((ln, round(thresholds.get(ln, cfg.peak_threshold), 2)) for ln in cfg.lanes)}")
    save = str(Path(args.ckpt_dir) / f"loss_ab_{arm}.pt")
    torch.save({"state_dict": model.state_dict(), "thresholds": thresholds,
                "lanes": list(cfg.lanes), "hidden": args.hidden, "num_layers": args.layers}, save)
    log(f"  saved -> {save}")
    return model, thresholds, val_clips


def main():
    ap = argparse.ArgumentParser(description="Overnight cymbal loss A/B (focal, crash-oversample)")
    ap.add_argument("--arms", default="baseline,focal,crash_oversample",
                    help="comma list from baseline,focal,crash_oversample (run in this order)")
    ap.add_argument("--pool-sources", default="star,enst,egmd")
    ap.add_argument("--pool-cap", type=int, default=3000, help="keep =3000 to stay cache-hit")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60, help="cap; ~3.5 h/arm on a 3080 (early-stops ~ep50)")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--pos-weight-cap", type=float, default=50.0)
    ap.add_argument("--crash-oversample", type=int, default=2, help="duplicate crash stems Nx")
    ap.add_argument("--cache", default="/codebox-workspace/datasets/_cache_mert_pooled")
    ap.add_argument("--aligned-onsets", default=None,
                    help="opt-in _onsets_aligned.json -> train+score on audio-snapped/filtered targets")
    ap.add_argument("--ckpt-dir", default="/codebox-workspace/checkpoints")
    ap.add_argument("--baseline-ckpt", default="/codebox-workspace/checkpoints/h128_cymhat_s1.pt")
    ap.add_argument("--out-json", default="/codebox-workspace/cymbal_loss_ab.json")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                    help="auto/cuda for the real run; cpu is scoring/smoke only (training on cpu is too slow)")
    ap.add_argument("--selftest", action="store_true", help="validate oversample logic; no cache/GPU")
    args = ap.parse_args()
    args.loss_for = lambda arm: "focal" if arm == "focal" else "bce"

    if args.selftest:
        _selftest()
        return

    import torch
    from head_capacity_sweep import make_cfg

    from drumjot_training import embeddings, runtime
    from drumjot_training.model import MultiLaneHeads

    runtime.tee_stdio(Path(args.out_json).with_suffix(".log"))
    runtime.configure_backends()
    log = lambda s: print(s, flush=True)  # noqa: E731
    sources = [s.strip() for s in args.pool_sources.split(",") if s.strip()]
    cache = Path(args.cache)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    cfg = make_cfg(args.hidden, args.layers, 3e-4)
    in_dim = embeddings.feat_dim(cfg.high_band)
    log(f"=== cymbal loss A/B: arms={arms} epochs={args.epochs} seed={args.seed} ===")

    uc = _use_cuda(args)
    log(f"device: {'cuda' if uc else 'cpu'}")
    results = {}
    for arm in arms:
        t0 = time.perf_counter()
        log(f"\n########## arm: {arm} ##########")
        try:
            if arm == "baseline":
                from cymbal_miss_typing import _CacheKeyEncoder
                from head_capacity_sweep import build_specs

                from drumjot_training.train import _window_specs, materialize
                ck = torch.load(args.baseline_ckpt, map_location="cuda" if uc else "cpu")
                _, va_specs = build_specs(sources, args.pool_cap, cache, aligned_path=args.aligned_onsets)
                enc = _CacheKeyEncoder(cfg.encoder, cfg.encoder_layer)
                va_w = _window_specs(va_specs, 30.0, 3.0, 4, plan_cache_dir=cache)
                val_clips = materialize(va_w, enc, cfg, cache, 30.0, "val", log)  # type: ignore[arg-type]
                model = MultiLaneHeads(in_dim=in_dim, hidden=int(ck["hidden"]),
                                       num_layers=int(ck.get("num_layers", args.layers)),
                                       lane_names=cfg.lanes)
                model.load_state_dict(ck["state_dict"])
                if uc:
                    model = model.cuda()
                thresholds = ck["thresholds"]
                log(f"  loaded baseline <- {args.baseline_ckpt}")
            else:
                model, thresholds, val_clips = _train_arm(arm, args, cache, sources, cfg, in_dim, log)
            results[arm] = score_arm(model, val_clips, cfg, thresholds, log)
            results[arm]["minutes"] = round((time.perf_counter() - t0) / 60, 1)
            del model
            if uc:
                torch.cuda.empty_cache()
            # write incrementally so a later-arm crash never loses earlier results
            Path(args.out_json).write_text(json.dumps({"config": {k: v for k, v in vars(args).items()
                                                                  if not callable(v)}, "arms": results}, indent=2))
        except Exception:  # noqa: BLE001  -- fail-independent: keep going to the next arm
            log(f"  !! arm {arm} FAILED:\n{traceback.format_exc()}")

    _comparison_table(results, log)
    Path(args.out_json).write_text(json.dumps({"config": {k: v for k, v in vars(args).items()
                                                          if not callable(v)}, "arms": results}, indent=2))
    log(f"\nresults -> {args.out_json}")


if __name__ == "__main__":
    main()
