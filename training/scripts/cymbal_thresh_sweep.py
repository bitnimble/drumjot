"""Per-lane height re-tune at prominence 0.1 vs 0.2 (no retrain).

The picker retest (RESULTS 2026-06-19) loosened height AND prominence at once and
overshot recall (net F1 down). This does it properly: for each saved loss-A/B
checkpoint, cache the cymbal activations once (one model pass), then sweep the
height threshold over a grid at BOTH the production cym prominence (0.20) and the
lowered 0.10 -- same grid, only prominence differs -- and report each lane's
F1-OPTIMAL operating point. Answers: is there a (prominence, threshold) where
focal's crash recall converts to an F1 win, vs just trading off?

Per-lane F1 depends only on that lane's own picks (precision = hits/est, recall =
hits/refs; the rd<->cr `confused` split is diagnostic only), so each lane's
threshold is swept independently. mir_eval +/-50 ms matcher, same as the decompose.

  DRUMJOT_STAR=/codebox-workspace/datasets/star_balanced_sep \
  DRUMJOT_ENST=/codebox-workspace/datasets/enst-sep \
  DRUMJOT_EGMD=/codebox-workspace/datasets/egmd_sep \
  OMP_NUM_THREADS=8 python training/scripts/cymbal_thresh_sweep.py \
      --aligned-onsets /codebox-workspace/datasets/_onsets_aligned.json \
      --out-json /codebox-workspace/cymbal_thresh_sweep.json
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

CYM = ("rd", "cr")
ARMS = {
    "baseline": "/codebox-workspace/checkpoints/h128_cymhat_s1.pt",
    "bce": "/codebox-workspace/checkpoints/loss_ab_bce.pt",
    "focal": "/codebox-workspace/checkpoints/loss_ab_focal.pt",
    "crash_oversample": "/codebox-workspace/checkpoints/loss_ab_crash_oversample.pt",
}


def _best_threshold(acts, refs, fps, base, prom, grid, tol):
    """Sweep height over `grid` at fixed `prom`; return the F1-optimal point."""
    from mir_eval.util import match_events

    from drumjot_training import metrics

    best = {"thr": None, "f1": -1.0, "recall": 0.0, "precision": 0.0}
    for thr in grid:
        H = FP = N = 0
        for a, r in zip(acts, refs, strict=True):
            # per-stem isolation convention (matches cymbal_recall_confusion.decompose):
            # score a lane ONLY on clips that carry its ground truth; cross-stem
            # firing is leakage, not an F1 false-positive here.
            if not len(r):
                continue
            est = metrics.pick_onsets(a, fps, thr, base["min_distance_s"], prominence=prom,
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
    ap = argparse.ArgumentParser(description="Per-lane height re-tune at prominence 0.1 vs 0.2")
    ap.add_argument("--pool-sources", default="star,enst,egmd")
    ap.add_argument("--pool-cap", type=int, default=1000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--cache", default="/codebox-workspace/datasets/_cache_mert_pooled")
    ap.add_argument("--aligned-onsets", default="/codebox-workspace/datasets/_onsets_aligned.json")
    ap.add_argument("--proms", default="0.2,0.1", help="cym prominences to compare")
    ap.add_argument("--out-json", default="/codebox-workspace/cymbal_thresh_sweep.json")
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
    grid = [round(0.05 * i, 2) for i in range(1, 19)]  # 0.05 .. 0.90
    proms = [float(p) for p in args.proms.split(",")]
    lane_i = {ln: i for i, ln in enumerate(cfg.lanes)}

    log(f"=== thresh sweep: proms={proms} grid=[{grid[0]}..{grid[-1]}] cap={args.pool_cap} ===")
    _, va_specs = build_specs(sources, args.pool_cap, cache, aligned_path=args.aligned_onsets)
    enc = _CacheKeyEncoder(cfg.encoder, cfg.encoder_layer)
    va_w = _window_specs(va_specs, 30.0, 3.0, 4, plan_cache_dir=cache)
    val_clips = materialize(va_w, enc, cfg, cache, 30.0, "val", log)  # type: ignore[arg-type]

    log(f"\n  {'arm':16s} {'lane':4s} | {'prom':>4s} {'thr*':>4s} {'R':>5s} {'P':>5s} {'F1':>5s}  (best per prom)")
    results = {}
    for arm, ckpt_path in ARMS.items():
        if not Path(ckpt_path).exists():
            log(f"  {arm}: missing {ckpt_path}")
            continue
        ck = torch.load(ckpt_path, map_location="cuda" if uc else "cpu")
        model = MultiLaneHeads(in_dim=in_dim, hidden=int(ck["hidden"]),
                               num_layers=int(ck.get("num_layers", args.layers)), lane_names=cfg.lanes)
        model.load_state_dict(ck["state_dict"])
        if uc:
            model = model.cuda()
        # cache activations + sorted refs per cym lane (one model pass)
        acts = {ln: [] for ln in CYM}
        refs = {ln: [] for ln in CYM}
        for clip in val_clips:
            probs = _clip_probs(model, clip)
            for ln in CYM:
                acts[ln].append(np.ascontiguousarray(probs[lane_i[ln]]))
                refs[ln].append(np.sort(np.asarray(clip.onsets_by_lane.get(ln, []), dtype=np.float64)))
        results[arm] = {}
        for ln in CYM:
            base = dict(metrics.LANE_PEAK_PARAMS[ln])
            results[arm][ln] = {}
            for prom in proms:
                b = _best_threshold(acts[ln], refs[ln], fps, base, prom, grid, tol)
                results[arm][ln][f"prom{prom}"] = b
                log(f"  {arm:16s} {ln:4s} | {prom:4.2f} {b['thr']:4.2f} {b['recall']:5.3f} "
                    f"{b['precision']:5.3f} {b['f1']:5.3f}")
        del model
        if uc:
            torch.cuda.empty_cache()

    # delta table: F1 at prom 0.1 best vs prom 0.2 best
    if len(proms) == 2:
        hi, lo = proms[0], proms[1]
        log(f"\n  ==== F1 delta: prom{lo} best - prom{hi} best ====")
        for arm in results:
            for ln in CYM:
                d = results[arm][ln][f"prom{lo}"]["f1"] - results[arm][ln][f"prom{hi}"]["f1"]
                log(f"  {arm:16s} {ln:4s}  {results[arm][ln][f'prom{hi}']['f1']:.3f} -> "
                    f"{results[arm][ln][f'prom{lo}']['f1']:.3f}  ({d:+.3f})")
    Path(args.out_json).write_text(json.dumps({"config": vars(args), "results": results}, indent=2))
    log(f"\nresults -> {args.out_json}")


if __name__ == "__main__":
    main()
