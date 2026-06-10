"""MERT layer-sweep probe: which hidden layer carries each lane's information?

Trains the small per-lane heads on SEVERAL different single MERT layers (raw
MERT features only, no high-band block) over a small balanced subset, and
prints per-lane F1 vs layer. Purpose (see RESULTS.md / the architecture
review): layer 10 is N2N's untested pick; if hats/cymbals peak at a different
layer this justifies per-lane layers or a learned layer mix, and if NO layer is
good for them it implicates the missing >12 kHz band instead (the high-band
pathway), not layer choice.

Each clip is encoded ONCE (one forward yields all requested layers), cached per
layer under the probe cache (variant ""), then each layer gets a short
from-scratch training run + threshold tune + per-lane val F1.

Usage (CUDA box; PYTHONPATH must include dsp + training):
  python3 layer_sweep.py <star_layout_root> [--layers 1,4,7,10,13,16,19,22]
      [--clips 200] [--val-clips 24] [--epochs 15] [--max-seconds 30]
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from drumjot_training import embeddings, star  # noqa: E402
from drumjot_training.config import Config  # noqa: E402
from drumjot_training.model import MultiLaneHeads  # noqa: E402
from drumjot_training.targets import pos_weights_from_targets  # noqa: E402
from drumjot_training.train import (  # noqa: E402
    Clip,
    build_targets,
    evaluate_clip,
    train_loop,
    tune_thresholds,
)


def main():
    ap = argparse.ArgumentParser(description="MERT layer sweep: per-lane F1 vs hidden layer")
    ap.add_argument("root", help="dataset root (star layout: annotation/ + audio/mix/)")
    ap.add_argument("--layers", default="1,4,7,10,13,16,19,22")
    ap.add_argument("--clips", type=int, default=200)
    ap.add_argument("--val-clips", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--cache-dir", default=None, help="probe feature cache (default <root>/_cache_sweep)")
    args = ap.parse_args()

    import torch

    layers = [int(x) for x in args.layers.split(",")]
    root = Path(args.root)
    cache = Path(args.cache_dir) if args.cache_dir else root / "_cache_sweep"
    cache.mkdir(parents=True, exist_ok=True)

    clips_all = star.index(root)
    tr = star.for_split(clips_all, "training")[: args.clips]
    held = star.for_split(clips_all, "validation") + star.for_split(clips_all, "test")
    va = held[: args.val_clips]
    sel = [(c, "train") for c in tr] + [(c, "val") for c in va]
    print(f"sweep: layers={layers}  {len(tr)} train + {len(va)} val clips", flush=True)

    cfg0 = Config()  # encoder name/fps; per-layer cfgs built below

    def _cache_path(audio, layer):
        key = embeddings.cache_key(audio, cfg0.encoder, layer, args.max_seconds, variant="")
        return cache / f"{key}.npy"

    # ---- encode pass: one forward per clip yields every requested layer ----
    need = [
        (c, sp) for c, sp in sel
        if not all(_cache_path(c.audio_path, li).exists() for li in layers)
    ]
    if need:
        enc = embeddings.MertEncoder(name=cfg0.encoder, layer=layers[0])
        for i, (c, _sp) in enumerate(need):
            y = embeddings.load_audio(c.audio_path, sr=enc.sr)
            y = y[: int(args.max_seconds * enc.sr)]
            feats = enc.encode_layers(y, enc.sr, layers)
            for li, feat in feats.items():
                np.save(_cache_path(c.audio_path, li), feat.astype("float16"))
            if (i + 1) % 25 == 0:
                print(f"  encoded {i + 1}/{len(need)}", flush=True)
        del enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("encode pass done (cached)", flush=True)

    # ---- per-layer: short train + tune + per-lane F1 ----
    results: dict[int, dict[str, float]] = {}
    for li in layers:
        cfg = Config(encoder_layer=li)

        def _clip(c, li=li, cfg=cfg):
            feat = np.load(_cache_path(c.audio_path, li))
            onsets = {
                ln: [t for t in ts if t < args.max_seconds]
                for ln, ts in star.onsets_by_lane(c.annotation_path).items()
            }
            return Clip(
                features=feat,
                targets=build_targets(onsets, feat.shape[0], cfg),
                onsets_by_lane=onsets,
            )

        tr_clips = [_clip(c) for c in tr]
        va_clips = [_clip(c) for c in va]
        pos_w = pos_weights_from_targets(c.targets for c in tr_clips)
        model = MultiLaneHeads(in_dim=embeddings.MERT_DIM, hidden=cfg.head_hidden,
                               num_layers=cfg.head_layers)
        if torch.cuda.is_available():
            model = model.cuda()
        train_loop(
            model, tr_clips, cfg, epochs=args.epochs, pos_weight=pos_w,
            batch_size=args.batch_size, log=lambda s: None,
        )
        thresholds = tune_thresholds(model, va_clips, cfg)
        per_lane: dict[str, list[float]] = defaultdict(list)
        for c in va_clips:
            f1 = evaluate_clip(model, c, cfg, thresholds)
            for ln, ts in c.onsets_by_lane.items():
                if ts:
                    per_lane[ln].append(f1[ln])
        results[li] = {ln: sum(v) / len(v) for ln, v in per_lane.items()}
        print(f"layer {li:2d} done: " + " ".join(
            f"{ln}={results[li].get(ln, float('nan')):.2f}" for ln in cfg.lanes
        ), flush=True)
        del model, tr_clips, va_clips
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- report: lanes x layers matrix + best layer per lane ----
    print("\n==== per-lane F1 by MERT layer (val) ====", flush=True)
    print("  lane " + "".join(f"  L{li:<4d}" for li in layers), flush=True)
    for ln in cfg0.lanes:
        row = [results[li].get(ln) for li in layers]
        if all(v is None for v in row):
            continue
        cells = "".join(f"  {v:.3f}" if v is not None else "      -" for v in row)
        best_li = max((li for li in layers if results[li].get(ln) is not None),
                      key=lambda li: results[li][ln])
        print(f"  {ln:4s}{cells}   best=L{best_li}", flush=True)


if __name__ == "__main__":
    main()
