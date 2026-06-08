"""Score a trained checkpoint on a dataset's val set, raw model vs +deterministic
envelope post-processing, without retraining. Same comparison `train.py` prints
at the end of a run, runnable standalone on any checkpoint.

Usage (DRUMJOT_STAR / DRUMJOT_EGMD must point at the dataset):
  python3 eval_filtered.py --dataset star --checkpoint <dir> [--val-clips N]
      [--max-seconds 30] [--align-window 0.03] [--support-percentile 60]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from drumjot_training import checkpoint, embeddings, train  # noqa: E402
from drumjot_training.config import Config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="raw vs +filter onset-F1 for a checkpoint")
    ap.add_argument("--dataset", choices=("egmd", "star"), default="star")
    ap.add_argument("--checkpoint", required=True, help="checkpoint dir (model.pt + meta.json)")
    ap.add_argument("--train-clips", type=int, default=0)  # unused; val only
    ap.add_argument("--val-clips", type=int, default=120)
    ap.add_argument("--train-min", type=float, default=0.0)
    ap.add_argument("--val-min", type=float, default=30.0)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--align-window", type=float, default=0.03)
    ap.add_argument("--support-percentile", type=float, default=60.0)
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = checkpoint.load(args.checkpoint, device)
    cfg = Config(encoder_layer=meta["encoder_layer"])
    thresholds = meta["thresholds"]

    specs = train._star_specs(args) if args.dataset == "star" else train._egmd_specs(args)
    _, val_specs, cache = specs
    encoder = embeddings.MertEncoder(name=cfg.encoder, layer=cfg.encoder_layer)
    print(f"val: {len(val_specs)} clips (cache {cache})", flush=True)
    val_clips = train.materialize(val_specs, encoder, cfg, cache, args.max_seconds, "val")

    train._report(model, val_clips, cfg, thresholds)
    train._report_compare(
        model, val_clips, cfg, thresholds, max_seconds=args.max_seconds,
        align_window_s=args.align_window, support_percentile=args.support_percentile,
    )


if __name__ == "__main__":
    main()
