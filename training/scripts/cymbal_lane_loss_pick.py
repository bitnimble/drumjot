"""Per-lane loss picker: focal vs bce F1-optimal, per cym+hat lane (no retrain).

The loss A/B trained whole-model focal vs bce. Since the heads are independent, the
best deployable model uses, per lane, whichever loss trained the better head. This
re-scores the saved focal & bce keep_best checkpoints (no retrain), sweeping each
lane's height threshold at its PRODUCTION prominence (the per-lane peak-pick params)
to its F1 optimum, and reports the winner per lane -> the `--focal-lanes` map.

Per-lane F1 uses the per-stem isolation convention (score a lane only on clips that
carry its ground truth), matching `cymbal_recall_confusion.decompose`.

  DRUMJOT_STAR=/codebox-workspace/datasets/star_balanced_sep \
  DRUMJOT_ENST=/codebox-workspace/datasets/enst-sep \
  DRUMJOT_EGMD=/codebox-workspace/datasets/egmd_sep \
  OMP_NUM_THREADS=8 python training/scripts/cymbal_lane_loss_pick.py \
      --out-json /codebox-workspace/cymbal_lane_loss_pick.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "dsp"))

LANES = ("hc", "ho", "rd", "cr")
CKPTS = {
    "bce": "/codebox-workspace/checkpoints/loss_ab_bce.pt",
    "focal": "/codebox-workspace/checkpoints/loss_ab_focal.pt",
    "mixed": "/codebox-workspace/checkpoints/loss_ab_mixed.pt",  # focal hc,rd + bce rest
}


def _best_threshold(acts, refs, fps, base, grid, tol):
    """Sweep height over `grid` at the lane's production prominence; F1-optimal point.
    Per-stem isolation: skip clips with no ground truth for this lane."""
    from mir_eval.util import match_events

    from drumjot_training import metrics

    best = {"thr": None, "f1": -1.0, "recall": 0.0, "precision": 0.0}
    for thr in grid:
        H = FP = N = 0
        for a, r in zip(acts, refs, strict=True):
            if not len(r):
                continue
            est = metrics.pick_onsets(a, fps, thr, base["min_distance_s"], prominence=base["prominence"],
                                      decay_reset_frac=base["decay_reset_frac"],
                                      decay_reset_floor=base["decay_reset_floor"])
            N += len(r)
            if not len(est):
                continue
            h = len(match_events(r, np.sort(est), tol))
            H += h
            FP += len(est) - h
        recall = H / N if N else 0.0
        prec = H / (H + FP) if (H + FP) else 0.0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0
        if f1 > best["f1"]:
            best = {"thr": round(float(thr), 2), "f1": f1, "recall": recall, "precision": prec}
    return best


def main():
    ap = argparse.ArgumentParser(description="Per-lane loss picker: focal vs bce")
    ap.add_argument("--pool-sources", default="star,enst,egmd")
    ap.add_argument("--pool-cap", type=int, default=1000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--cache", default="/codebox-workspace/datasets/_cache_mert_pooled")
    ap.add_argument("--aligned-onsets", default="/codebox-workspace/datasets/_onsets_aligned.json")
    ap.add_argument("--margin", type=float, default=0.01, help="min F1 gain for focal to win a lane")
    ap.add_argument("--out-json", default="/codebox-workspace/cymbal_lane_loss_pick.json")
    args = ap.parse_args()

    import torch
    from cymbal_miss_typing import _CacheKeyEncoder
    from head_capacity_sweep import build_specs, make_cfg

    from drumjot_training import embeddings, metrics, runtime
    from drumjot_training.model import MultiLaneHeads
    from drumjot_training.train import _clip_probs, _window_specs, materialize

    runtime.tee_stdio(Path(args.out_json).with_suffix(".log"))
    runtime.configure_backends()
    log = lambda s: print(s, flush=True)  # noqa: E731
    sources = [s.strip() for s in args.pool_sources.split(",") if s.strip()]
    cache = Path(args.cache)
    cfg = make_cfg(args.hidden, args.layers, 3e-4)
    in_dim = embeddings.feat_dim(cfg.high_band)
    uc = torch.cuda.is_available()
    fps, tol = cfg.encoder_fps, cfg.onset_tolerance_s
    grid = [round(0.05 * i, 2) for i in range(1, 19)]
    lane_i = {ln: i for i, ln in enumerate(cfg.lanes)}

    log(f"=== lane loss pick: focal vs bce, lanes={LANES} cap={args.pool_cap} ===")
    _, va_specs = build_specs(sources, args.pool_cap, cache, aligned_path=args.aligned_onsets)
    enc = _CacheKeyEncoder(cfg.encoder, cfg.encoder_layer)
    va_w = _window_specs(va_specs, 30.0, 3.0, 4, plan_cache_dir=cache)
    val_clips = materialize(va_w, enc, cfg, cache, 30.0, "val", log)  # type: ignore[arg-type]

    best = {}  # arm -> lane -> {thr,f1,...}
    for arm, ckpt_path in CKPTS.items():
        ck = torch.load(ckpt_path, map_location="cuda" if uc else "cpu")
        model = MultiLaneHeads(in_dim=in_dim, hidden=int(ck["hidden"]),
                               num_layers=int(ck.get("num_layers", args.layers)), lane_names=cfg.lanes)
        model.load_state_dict(ck["state_dict"])
        if uc:
            model = model.cuda()
        acts = {ln: [] for ln in LANES}
        refs = {ln: [] for ln in LANES}
        for clip in val_clips:
            probs = _clip_probs(model, clip)
            for ln in LANES:
                acts[ln].append(np.ascontiguousarray(probs[lane_i[ln]]))
                refs[ln].append(np.sort(np.asarray(clip.onsets_by_lane.get(ln, []), dtype=np.float64)))
        best[arm] = {ln: _best_threshold(acts[ln], refs[ln], fps, dict(metrics.LANE_PEAK_PARAMS[ln]), grid, tol)
                     for ln in LANES}
        del model
        if uc:
            torch.cuda.empty_cache()

    extra = [a for a in best if a not in ("bce", "focal")]  # e.g. the deployable 'mixed'
    hdr = f"\n  {'lane':4s} | {'bce F1':>7s}(thr) | {'focal F1':>8s}(thr) | {'Δ(f-b)':>7s} | winner"
    for a in extra:
        hdr += f" | {a + ' F1':>9s}(thr)"
    log(hdr)
    focal_lanes = []
    out = {}
    for ln in LANES:
        b, f = best["bce"][ln], best["focal"][ln]
        d = f["f1"] - b["f1"]
        win = "focal" if d >= args.margin else "bce"
        if win == "focal":
            focal_lanes.append(ln)
        line = (f"  {ln:4s} | {b['f1']:7.3f}({b['thr']:.2f}) | {f['f1']:8.3f}({f['thr']:.2f}) | "
                f"{d:+7.3f} | {win:6s}")
        for a in extra:
            e = best[a][ln]
            line += f" | {e['f1']:9.3f}({e['thr']:.2f})"
        log(line)
        out[ln] = {"bce": b, "focal": f, "delta": d, "winner": win,
                   **{a: best[a][ln] for a in extra}}
    log(f"\n  => --focal-lanes {','.join(focal_lanes) if focal_lanes else '(none)'}  "
        f"(margin {args.margin}); all other lanes -> bce")
    Path(args.out_json).write_text(json.dumps(
        {"config": vars(args), "lanes": out, "focal_lanes": focal_lanes}, indent=2))
    log(f"\nresults -> {args.out_json}")


if __name__ == "__main__":
    main()
