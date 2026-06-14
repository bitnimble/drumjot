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
    ap.add_argument(
        "--high-band-modes", default="none",
        help="comma list of feature variants to compare, each trained at every --layer x "
        "--seed. A variant is 'none' (raw MERT) or a '+'-composed set of blocks appended to "
        "MERT: 'energy' (6-20 kHz log-mel, embeddings.highband_features) or 'flux' (its "
        "positive temporal difference). E.g. 'none,energy,energy+flux'. All reuse the "
        "raw-MERT encode cache (no re-encode); blocks are layer-independent, computed "
        "once per clip.",
    )
    ap.add_argument(
        "--seeds", default="0",
        help="comma list of torch init seeds; each (mode,layer) is trained once per seed and "
        "reported as mean+/-std, so a high-band delta is only believed if it clears the seed "
        "noise band. The DataLoader shuffle is separately fixed (train_loop seeds it).",
    )
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

    modes = [m.strip() for m in args.high_band_modes.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"modes={modes}  seeds={seeds}", flush=True)

    # Feature blocks are layer- and seed-independent (derived from raw audio),
    # so each is computed ONCE per clip and reused across every mode/layer/seed.
    # A mode is "none" or a "+"-joined composition of blocks, e.g. "energy+flux":
    # energy = 6-20 kHz log-mel; flux = its positive time-diff.
    hb_energy: dict = {}

    def _hb_energy(c, n_frames):
        e = hb_energy.get(c.audio_path)
        if e is None or e.shape[0] != n_frames:
            e = embeddings.highband_features(c.audio_path, n_frames, args.max_seconds)
            hb_energy[c.audio_path] = e
        return e

    def _blocks(c, n_frames, mode):
        if mode == "none":
            return None
        parts = []
        for p in mode.split("+"):
            if p == "energy":
                parts.append(_hb_energy(c, n_frames))
            elif p == "flux":  # positive temporal difference per band (onset-sharpened)
                e = _hb_energy(c, n_frames)
                parts.append(np.diff(e, axis=0, prepend=e[:1]).clip(min=0))
            else:
                raise SystemExit(f"unknown feature block {p!r} in mode {mode!r}")
        return np.concatenate(parts, axis=1)

    def _clip(c, li, cfg, mode):
        feat = np.load(_cache_path(c.audio_path, li))
        hb = _blocks(c, feat.shape[0], mode)
        if hb is not None:
            feat = np.concatenate([feat, hb], axis=1).astype(feat.dtype, copy=False)
        onsets = {
            ln: [t for t in ts if t < args.max_seconds]
            for ln, ts in star.onsets_by_lane(c.annotation_path).items()
        }
        return Clip(
            features=feat,
            targets=build_targets(onsets, feat.shape[0], cfg),
            onsets_by_lane=onsets,
        )

    # ---- (mode, layer, seed): short train + tune + per-lane F1 ----
    # results[(mode, li)][lane] = [f1 per seed]
    results: dict[tuple[str, int], dict[str, list[float]]] = {}
    for li in layers:
        cfg = Config(encoder_layer=li)
        for mode in modes:
            tr_clips = [_clip(c, li, cfg, mode) for c in tr]
            va_clips = [_clip(c, li, cfg, mode) for c in va]
            pos_w = pos_weights_from_targets(c.targets for c in tr_clips)
            per_lane: dict[str, list[float]] = defaultdict(list)
            for seed in seeds:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                model = MultiLaneHeads(in_dim=tr_clips[0].features.shape[1],
                                       hidden=cfg.head_hidden, num_layers=cfg.head_layers)
                if torch.cuda.is_available():
                    model = model.cuda()
                train_loop(
                    model, tr_clips, cfg, epochs=args.epochs, pos_weight=pos_w,
                    batch_size=args.batch_size, log=lambda s: None,
                )
                thresholds = tune_thresholds(model, va_clips, cfg)
                lane_f1: dict[str, list[float]] = defaultdict(list)
                for c in va_clips:
                    f1 = evaluate_clip(model, c, cfg, thresholds)
                    for ln in cfg.lanes:  # output lanes only (onsets may carry the `x` ghost lane)
                        ts = c.onsets_by_lane.get(ln)
                        if ts:
                            lane_f1[ln].append(f1[ln])
                for ln, v in lane_f1.items():
                    per_lane[ln].append(sum(v) / len(v))  # this seed's mean-over-clips
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            results[(mode, li)] = dict(per_lane)
            summ = " ".join(
                f"{ln}={np.mean(per_lane[ln]):.2f}" for ln in cfg.lanes if per_lane.get(ln)
            )
            print(f"L{li:<2d} {mode:6s} done ({len(seeds)} seeds): {summ}", flush=True)
            del tr_clips, va_clips

    # ---- report: per lane, per layer, mode comparison as mean+/-std over seeds ----
    def _ms(vals):
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), 0.0)

    print(f"\n==== per-lane F1: mean+/-std over {len(seeds)} seeds ====", flush=True)
    head = "  lane  layer  " + "".join(f"{m:>16s}" for m in modes)
    print(head, flush=True)
    for ln in cfg0.lanes:
        rows = [(li, [results[(m, li)].get(ln, []) for m in modes]) for li in layers]
        if all(not any(cols) for _li, cols in rows):
            continue
        for li, cols in rows:
            if not any(cols):
                continue
            cells = ""
            for v in cols:
                mu, sd = _ms(v)
                cells += f"   {mu:.3f}+/-{sd:.3f}"
            print(f"  {ln:4s}  L{li:<4d}{cells}", flush=True)


if __name__ == "__main__":
    main()
