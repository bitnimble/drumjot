"""Re-score the saved loss-A/B checkpoints with a more permissive cymbal picker.

The loss A/B (RESULTS 2026-06-19) found focal turns crash into a high-precision /
low-recall lane under the production picker (cym min-dist 70 ms, prominence 0.20,
decay-reset, + tuned height). Question: how much of focal's lost crash recall is
the PICKER being too strict vs the model itself? This loads each arm's keep_best
checkpoint (NO retrain) and re-runs the same ride/crash decompose with the height
threshold scaled and the cymbal prominence lowered, alongside the original params
as a sanity reproduction of the A/B table.

  DRUMJOT_STAR=/codebox-workspace/datasets/star_balanced_sep \
  DRUMJOT_ENST=/codebox-workspace/datasets/enst-sep \
  DRUMJOT_EGMD=/codebox-workspace/datasets/egmd_sep \
  OMP_NUM_THREADS=8 python training/scripts/cymbal_picker_retest.py \
      --aligned-onsets /codebox-workspace/datasets/_onsets_aligned.json \
      --thr-scale 0.5 --prominence 0.1 \
      --out-json /codebox-workspace/cymbal_picker_retest.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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


def _rpf(agg, ln):
    h, c, m, fp = agg[ln]
    n = h + c + m
    if not n:
        return 0.0, 0.0, 0.0
    r = h / n
    p = h / (h + fp) if (h + fp) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return r, p, f


def main():
    ap = argparse.ArgumentParser(description="Re-score loss-A/B checkpoints with a permissive cymbal picker")
    ap.add_argument("--pool-sources", default="star,enst,egmd")
    ap.add_argument("--pool-cap", type=int, default=1000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--cache", default="/codebox-workspace/datasets/_cache_mert_pooled")
    ap.add_argument("--aligned-onsets", default="/codebox-workspace/datasets/_onsets_aligned.json")
    ap.add_argument("--thr-scale", type=float, default=0.5, help="multiply each tuned height threshold")
    ap.add_argument("--prominence", type=float, default=0.1, help="cymbal (rd/cr) prominence override")
    ap.add_argument("--out-json", default="/codebox-workspace/cymbal_picker_retest.json")
    args = ap.parse_args()

    import torch
    from cymbal_miss_typing import _CacheKeyEncoder
    from cymbal_recall_confusion import decompose
    from head_capacity_sweep import build_specs, make_cfg

    from drumjot_training import embeddings, metrics, runtime
    from drumjot_training.model import MultiLaneHeads
    from drumjot_training.train import _window_specs, materialize

    runtime.tee_stdio(Path(args.out_json).with_suffix(".log"))
    runtime.configure_backends()
    log = lambda s: print(s, flush=True)  # noqa: E731
    sources = [s.strip() for s in args.pool_sources.split(",") if s.strip()]
    cache = Path(args.cache)
    cfg = make_cfg(args.hidden, args.layers, 3e-4)
    in_dim = embeddings.feat_dim(cfg.high_band)
    uc = torch.cuda.is_available()

    log(f"=== picker retest: thr-scale={args.thr_scale} prominence(cym)={args.prominence} "
        f"cap={args.pool_cap} aligned={args.aligned_onsets} ===")
    _, va_specs = build_specs(sources, args.pool_cap, cache, aligned_path=args.aligned_onsets)
    enc = _CacheKeyEncoder(cfg.encoder, cfg.encoder_layer)
    va_w = _window_specs(va_specs, 30.0, 3.0, 4, plan_cache_dir=cache)
    val_clips = materialize(va_w, enc, cfg, cache, 30.0, "val", log)  # type: ignore[arg-type]

    # snapshot the production cym prominence so we can restore between passes
    orig_prom = {ln: metrics.LANE_PEAK_PARAMS[ln]["prominence"] for ln in CYM}

    def set_prom(val):
        for ln in CYM:
            metrics.LANE_PEAK_PARAMS[ln]["prominence"] = val

    results = {}
    log(f"\n  {'arm':16s} {'variant':12s} | {'rd R':>5s} {'rd P':>5s} {'rdF1':>5s} | "
        f"{'cr R':>5s} {'cr P':>5s} {'crF1':>5s}")
    for arm, ckpt_path in ARMS.items():
        if not Path(ckpt_path).exists():
            log(f"  {arm:16s} (missing ckpt {ckpt_path})")
            continue
        ck = torch.load(ckpt_path, map_location="cuda" if uc else "cpu")
        model = MultiLaneHeads(in_dim=in_dim, hidden=int(ck["hidden"]),
                               num_layers=int(ck.get("num_layers", args.layers)), lane_names=cfg.lanes)
        model.load_state_dict(ck["state_dict"])
        if uc:
            model = model.cuda()
        thr = ck["thresholds"]
        scaled = {ln: thr.get(ln, cfg.peak_threshold) * args.thr_scale for ln in cfg.lanes}

        results[arm] = {}
        for variant, thresholds, prom in (
            ("orig", thr, None),
            (f"thr*{args.thr_scale} p{args.prominence}", scaled, args.prominence),
        ):
            set_prom(prom if prom is not None else orig_prom["rd"])  # rd/cr share the cym default
            if prom is None:  # restore the genuine per-lane production values
                for ln in CYM:
                    metrics.LANE_PEAK_PARAMS[ln]["prominence"] = orig_prom[ln]
            agg = decompose(model, val_clips, cfg, thresholds, log=None)
            rr, rp, rf = _rpf(agg, "rd")
            cr_, cp, cf = _rpf(agg, "cr")
            log(f"  {arm:16s} {variant:12s} | {rr:5.3f} {rp:5.3f} {rf:5.3f} | "
                f"{cr_:5.3f} {cp:5.3f} {cf:5.3f}")
            results[arm][variant] = {"rd": {"R": rr, "P": rp, "F1": rf},
                                     "cr": {"R": cr_, "P": cp, "F1": cf}}
        del model
        if uc:
            torch.cuda.empty_cache()

    set_prom(orig_prom["rd"])
    for ln in CYM:
        metrics.LANE_PEAK_PARAMS[ln]["prominence"] = orig_prom[ln]
    Path(args.out_json).write_text(json.dumps({"config": vars(args), "results": results}, indent=2))
    log(f"\nresults -> {args.out_json}")


if __name__ == "__main__":
    main()
