"""Ride/crash recall-vs-confusion decomposition (cymbal F1 error analysis).

The feature probe (RESULTS.md 2026-06-18) showed ride/crash are ~0.84 linearly
separable at the onset frame, so the ~0.59/0.64 end-to-end F1 is NOT pure feature
confusion. This script splits the end-to-end ride/crash error into its parts to
see which dominates -- the thing the probe can't tell us because it's handed the
true onset:

  - HIT      : a predicted onset in the SAME lane within +/-tolerance
  - CONFUSED : missed by its own lane but caught by the OTHER cymbal lane
               (ride<->crash cross-firing)
  - MISSED   : detected by neither cymbal lane (a pure RECALL failure)
  - FALSE-POS: predicted onsets in a lane with no matching truth (precision loss)

If MISSED dominates -> the lever is detection/recall (proposer, activity head,
labels), and the ~16% feature-confusion floor is moot until recall improves.
If CONFUSED dominates -> the joint ride/crash decision / separation is the lever.

Trains the matched h128 cym+hat arm (same config as the width A/B: lr 3e-4,
warmup 500, batch 8, seed 1, early-stop) so the decomposition reflects the model
whose F1 we're explaining, tunes per-lane thresholds, then categorizes every val
ride/crash onset with mir_eval's matcher. Reuses the warm cap-3000 cache (all
cache hits). Saves the trained model so a re-run can `--load` it.

Run on a box with the warm cache (3080):
  DRUMJOT_STAR=/codebox-workspace/datasets/star_balanced_sep \
  DRUMJOT_ENST=/codebox-workspace/datasets/enst-sep \
  DRUMJOT_EGMD=/codebox-workspace/datasets/egmd_sep \
  python training/scripts/cymbal_recall_confusion.py \
      --cache /codebox-workspace/datasets/_cache_mert_pooled \
      --out-json /codebox-workspace/cymbal_recall_confusion.json

Validate the categorizer without cache/GPU:  python ... --selftest
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
sys.path.insert(0, _HERE)  # head_capacity_sweep
sys.path.insert(0, os.path.join(_HERE, ".."))  # training/
sys.path.insert(0, os.path.join(_HERE, "..", "..", "dsp"))  # dsp/

CYM = ("rd", "cr")
OTHER = {"rd": "cr", "cr": "rd"}


def _categorize(ref_main, est_main, est_other, tol):
    """(hits, confused, missed, false_pos) for one lane on one clip.

    hits/confused/missed partition the TRUE onsets (ref_main); false_pos counts
    predicted onsets in this lane that match no truth. Uses mir_eval's optimal
    bipartite match so a ref/est pairs at most once."""
    from mir_eval.util import match_events

    ref_main = np.sort(np.asarray(ref_main, dtype=np.float64))
    est_main = np.sort(np.asarray(est_main, dtype=np.float64))
    est_other = np.sort(np.asarray(est_other, dtype=np.float64))
    m = match_events(ref_main, est_main, tol) if len(ref_main) and len(est_main) else []
    matched_ref = {i for i, _ in m}
    matched_est = {j for _, j in m}
    hits = len(matched_ref)
    unmatched = np.asarray(
        [ref_main[i] for i in range(len(ref_main)) if i not in matched_ref], dtype=np.float64
    )
    mo = match_events(unmatched, est_other, tol) if len(unmatched) and len(est_other) else []
    confused = len(mo)
    missed = len(ref_main) - hits - confused
    false_pos = len(est_main) - len(matched_est)
    return hits, confused, missed, false_pos


def decompose(model, val_clips, cfg, thresholds, log):
    """Aggregate ride/crash (hits, confused, missed, fp) over all val clips."""
    from drumjot_training import metrics
    from drumjot_training.train import _clip_probs

    lane_i = {ln: i for i, ln in enumerate(cfg.lanes)}
    agg = {ln: [0, 0, 0, 0] for ln in CYM}  # hits, confused, missed, fp
    t0 = time.perf_counter()
    for k, clip in enumerate(val_clips, 1):
        probs = _clip_probs(model, clip)
        est = {}
        for ln in CYM:
            thr = thresholds.get(ln, cfg.peak_threshold)
            est[ln] = metrics.pick_onsets_lane(probs[lane_i[ln]], cfg.encoder_fps, ln, thr)
        for ln in CYM:
            ref = clip.onsets_by_lane.get(ln, [])
            if not len(ref):
                continue
            h, c, m, fp = _categorize(ref, est[ln], est[OTHER[ln]], cfg.onset_tolerance_s)
            agg[ln][0] += h
            agg[ln][1] += c
            agg[ln][2] += m
            agg[ln][3] += fp
        if log and k % 200 == 0:
            log(f"  decompose: {k} clips ({time.perf_counter() - t0:.0f}s)")
    return agg


def _report(agg, log):
    out = {}
    log("\n==== ride/crash error decomposition (val) ====")
    log(f"  {'lane':5s} {'N':>6s} {'hit':>7s} {'confuse':>8s} {'miss':>7s} "
        f"{'recall':>7s} {'prec':>6s} {'F1':>6s}")
    for ln in CYM:
        h, c, m, fp = agg[ln]
        n = h + c + m
        if n == 0:
            continue
        recall = h / n
        prec = h / (h + fp) if (h + fp) else 0.0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0
        log(f"  {ln:5s} {n:6d} {h / n:7.3f} {c / n:8.3f} {m / n:7.3f} "
            f"{recall:7.3f} {prec:6.3f} {f1:6.3f}")
        out[ln] = {"n": n, "hit": h, "confused": c, "missed": m, "false_pos": fp,
                   "hit_rate": h / n, "confused_rate": c / n, "missed_rate": m / n,
                   "recall": recall, "precision": prec, "f1": f1}
    # which dominates the error?
    for ln in CYM:
        if ln in out:
            o = out[ln]
            err = o["confused"] + o["missed"]
            if err:
                lead = "MISSED (recall)" if o["missed"] >= o["confused"] else "CONFUSED (cross-lane)"
                log(f"  -> {ln}: of {err} non-hits, {o['missed']} missed / {o['confused']} confused "
                    f"-> {lead} dominates")
    return out


def _selftest():
    tol = 0.05
    # ref [1,2,3,4]: 1,2 hit in-lane; 3 caught by other lane (confused); 4 missed.
    # est_main also has 9.0 (false-pos).
    h, c, m, fp = _categorize([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 9.0], [3.0], tol)
    assert (h, c, m, fp) == (2, 1, 1, 1), (h, c, m, fp)
    # all hit, no other-lane est
    h, c, m, fp = _categorize([1.0, 2.0], [1.0, 2.0], [], tol)
    assert (h, c, m, fp) == (2, 0, 0, 0), (h, c, m, fp)
    # all missed (no est anywhere)
    h, c, m, fp = _categorize([1.0, 2.0], [], [], tol)
    assert (h, c, m, fp) == (0, 0, 2, 0), (h, c, m, fp)
    # outside tolerance = miss + false-pos (est at 1.2 doesn't match ref at 1.0)
    h, c, m, fp = _categorize([1.0], [1.2], [], tol)
    assert (h, c, m, fp) == (0, 0, 1, 1), (h, c, m, fp)
    print("SELFTEST OK (_categorize hit/confused/missed/fp correct)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Ride/crash recall-vs-confusion decomposition")
    ap.add_argument("--pool-sources", default="star,enst,egmd")
    ap.add_argument("--pool-cap", type=int, default=3000, help="keep =3000 to stay cache-hit")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=80, help="early-stop usually ends ~ep60")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--train-max-windows", type=int, default=0)
    ap.add_argument("--val-max-windows", type=int, default=4)
    ap.add_argument("--cache", default="/codebox-workspace/datasets/_cache_mert_pooled")
    ap.add_argument("--load", default=None, help="load a saved model .pt instead of training")
    ap.add_argument("--save", default=None, help="save the trained model+thresholds here (.pt)")
    ap.add_argument("--out-json", default="/codebox-workspace/cymbal_recall_confusion.json")
    ap.add_argument("--selftest", action="store_true", help="validate the categorizer; no cache/GPU")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    import torch
    from head_capacity_sweep import build_specs, make_cfg

    from drumjot_training import embeddings, runtime
    from drumjot_training.targets import pos_weights_from_targets
    from drumjot_training.train import _window_specs, materialize, train_loop, tune_thresholds

    runtime.tee_stdio(Path(args.out_json).with_suffix(".log"))  # self-log next to --out-json
    runtime.configure_backends()
    log = lambda s: print(s, flush=True)  # noqa: E731
    sources = [s.strip() for s in args.pool_sources.split(",") if s.strip()]
    cache = Path(args.cache)

    log(f"=== ride/crash decomposition: cap={args.pool_cap} h{args.hidden} seed={args.seed} ===")
    tr_specs, va_specs = build_specs(sources, args.pool_cap, cache)
    cfg = make_cfg(args.hidden, args.layers, args.lr)
    from drumjot_training.model import MultiLaneHeads
    in_dim = embeddings.feat_dim(cfg.high_band)
    encoder = embeddings.make_encoder(cfg.encoder, cfg.encoder_layer)
    tr_w = _window_specs(tr_specs, 30.0, 3.0, args.train_max_windows, plan_cache_dir=cache)
    va_w = _window_specs(va_specs, 30.0, 3.0, args.val_max_windows, plan_cache_dir=cache)
    train_clips = materialize(tr_w, encoder, cfg, cache, 30.0, "train", log)
    val_clips = materialize(va_w, encoder, cfg, cache, 30.0, "val", log)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = MultiLaneHeads(in_dim=in_dim, hidden=args.hidden, num_layers=args.layers, lane_names=cfg.lanes)
    if torch.cuda.is_available():
        model = model.cuda()

    if args.load:
        ck = torch.load(args.load, map_location="cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(ck["state_dict"])
        thresholds = ck["thresholds"]
        log(f"loaded model + thresholds <- {args.load}")
    else:
        pos_w = pos_weights_from_targets(train_clips.iter_targets(), cap=50.0)
        log(f"pos_weights: {dict((ln, round(float(w), 1)) for ln, w in zip(cfg.lanes, pos_w, strict=True))}")
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        train_loop(model, train_clips, cfg, epochs=args.epochs, pos_weight=pos_w,
                   batch_size=args.batch, num_workers=args.num_workers, val_clips=val_clips,
                   keep_best=True, log=log, early_stop=True, es_min_epochs=20, warmup_steps=args.warmup)
        thresholds = tune_thresholds(model, val_clips, cfg)
        log(f"tuned thresholds: {dict((ln, round(thresholds.get(ln, cfg.peak_threshold), 2)) for ln in cfg.lanes)}")
        if args.save:
            torch.save({"state_dict": model.state_dict(), "thresholds": thresholds,
                        "lanes": list(cfg.lanes), "hidden": args.hidden}, args.save)
            log(f"saved model -> {args.save}")

    agg = decompose(model, val_clips, cfg, thresholds, log)
    out = _report(agg, log)
    Path(args.out_json).write_text(json.dumps({"config": vars(args), "decomp": out}, indent=2))
    log(f"\nresults -> {args.out_json}")


if __name__ == "__main__":
    main()
