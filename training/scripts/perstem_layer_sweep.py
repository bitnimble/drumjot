"""Per-stem MERT layer sweep on the pooled, separation-aware data -- find which
hidden layer carries each lane's onset best, in the DEPLOYMENT domain.

Unlike scripts/layer_sweep.py (STAR full-mix, bare heads, raw MERT), this:
  - trains on POOLED per-stem examples from star+enst+egmd sep trees (each
    instrument's isolated stem -> its own lanes), the real inference domain;
  - runs the FULL pipeline (high-band block + aux ring-activity + sibling
    weighting), so per-lane F1 is trustworthy (the bare probe under-detected
    ride badly -- see RESULTS);
  - encodes EVERY swept layer in ONE MERT forward per clip (encode_layers), so an
    N-layer sweep costs one encode pass, not N, and the cache it writes is the
    SAME one train.py reads (so a later `train.py --layer L` is a cache hit).

Output: a lane x layer F1 matrix (mean+/-std over --seeds) + the best layer per
lane, to lock per-lane encoder layers (or inform a layer-concat head).

Models load OFFLINE (run training/scripts/fetch_models.py once first).
DRUMJOT_STAR/ENST/EGMD must point at the sep trees. Usage:
  perstem_layer_sweep.py --pool-sources star,enst,egmd --pool-cap 150 \
    --layers 1,4,7,10,13,16,19,22 --seeds 0,1,2 [--pool-cache /nvme/cache]
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))  # training/

from drumjot_training import embeddings  # noqa: E402
from drumjot_training.config import Config  # noqa: E402
from drumjot_training.model import MultiLaneHeads  # noqa: E402
from drumjot_training.targets import pos_weights_from_targets  # noqa: E402
from drumjot_training.train import (  # noqa: E402
    _pooled_specs,
    _window_specs,
    evaluate_clip,
    materialize,
    train_loop,
    tune_thresholds,
)


def _encode_all_layers(specs, enc, layers, cache: Path, cfg: Config, log) -> None:
    """One MERT forward per (audio, window) -> cache features for EVERY layer, in
    the exact layout train.py's embed_clip reads ([MERT_layer | high-band] under
    cache_key(..., layer, length, 'hb16', start)). So materialize() below is a
    pure cache hit at every layer -- the sweep pays a single encode pass."""
    variant = embeddings.feat_variant(cfg.high_band)  # "hb16"

    def _key(audio, layer, length, start):
        k = embeddings.cache_key(audio, enc.name, layer, length, variant, start)
        return cache / f"{k}.npy"

    windows, seen = [], set()
    for audio, _o, _w, start, length in specs:
        sig = (str(audio), start, length)
        if sig in seen:
            continue
        seen.add(sig)
        if not all(_key(audio, li, length, start).exists() for li in layers):
            windows.append((audio, start, length))
    cache.mkdir(parents=True, exist_ok=True)
    log(f"encode pass: {len(windows)} windows x {len(layers)} layers (one forward each)")
    for i, (audio, start, length) in enumerate(windows):
        y = embeddings.load_audio(audio, sr=enc.sr)
        a = int(start * enc.sr)
        b = a + int(length * enc.sr)
        feats = enc.encode_layers(y[a:b], enc.sr, layers)  # {layer: (T, MERT_DIM)}
        nT = next(iter(feats.values())).shape[0]
        hb = embeddings.highband_features(audio, nT, length, start, enc.fps) if cfg.high_band else None
        for li, mert in feats.items():
            feat = np.concatenate([mert, hb], axis=1) if hb is not None else mert
            np.save(_key(audio, li, length, start), feat.astype(cfg.cache_dtype, copy=False))
        if (i + 1) % 25 == 0:
            log(f"  encoded {i + 1}/{len(windows)}")
    log("encode pass done (all layers cached)")


def main():
    ap = argparse.ArgumentParser(description="per-stem pooled MERT layer sweep (full pipeline)")
    ap.add_argument("--pool-sources", default="star,enst,egmd")
    ap.add_argument("--pool-cap", type=int, default=150, help="clips per source (0 = all)")
    ap.add_argument("--pool-balance", action="store_true", help="oversample smaller sources")
    ap.add_argument("--pool-cache", default=None, help="feature-cache dir (point at LOCAL NVMe)")
    ap.add_argument("--layers", default="1,4,7,10,13,16,19,22")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--enst-mix", default="sep_drum")
    ap.add_argument("--log", default=None,
                    help="tee stdout+stderr to this file (self-log; no manual redirect needed)")
    args = ap.parse_args()

    import torch

    from drumjot_training import runtime

    runtime.tee_stdio(args.log)
    runtime.configure_backends()
    layers = [int(x) for x in args.layers.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    enc_name = embeddings.MERT_NAME
    enc_fps = embeddings.MERT_FPS
    cfg0 = Config(encoder=enc_name, encoder_fps=enc_fps, high_band=True)

    train_specs, val_specs, cache = _pooled_specs(args)
    # single window per clip (the sweep ranks layers; windowing is orthogonal)
    tr = _window_specs(train_specs, args.max_seconds, 3.0, 1)  # single window per clip
    va = _window_specs(val_specs, args.max_seconds, 3.0, 1)
    log = lambda s: print(s, flush=True)  # noqa: E731
    enc = embeddings.make_encoder(cfg0.encoder, layers[0])
    # clamp to the encoder's real depth (MERT exposes 25 hidden states, 0..24) so an
    # out-of-range --layers entry is dropped instead of crashing with an IndexError.
    n_hs = enc.n_hidden_states()
    valid = [li for li in layers if 0 <= li < n_hs]
    if valid != layers:
        log(f"encoder has {n_hs} hidden states (valid 0..{n_hs - 1}); dropping "
            f"{[li for li in layers if li not in valid]}, using {valid}")
        layers = valid
    if not layers:
        raise SystemExit(f"no requested layer is valid for this encoder ({n_hs} hidden states)")
    print(f"per-stem layer sweep [MERT, {enc_fps:.0f}fps]: layers={layers} seeds={seeds}  "
          f"{len(tr)} train / {len(va)} val windows  epochs={args.epochs}", flush=True)
    _encode_all_layers(tr + va, enc, layers, cache, cfg0, log)

    # results[layer][lane] = [F1 per seed]
    results: dict[int, dict[str, list[float]]] = {li: defaultdict(list) for li in layers}
    for li in layers:
        cfg = Config(encoder=enc_name, encoder_fps=enc_fps, encoder_layer=li, high_band=True)
        enc.layer = li  # so materialize's embed_clip hits the layer-li cache
        train_clips = materialize(tr, enc, cfg, cache, args.max_seconds, f"L{li} train", log)
        val_clips = materialize(va, enc, cfg, cache, args.max_seconds, f"L{li} val", log)
        pos_w = pos_weights_from_targets(train_clips.iter_targets())
        for seed in seeds:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model = MultiLaneHeads(in_dim=embeddings.feat_dim(cfg.high_band),
                                   hidden=cfg.head_hidden, num_layers=cfg.head_layers)
            if torch.cuda.is_available():
                model = model.cuda()
            train_loop(model, train_clips, cfg, epochs=args.epochs, pos_weight=pos_w,
                       batch_size=args.batch_size, log=lambda s: None)
            thr = tune_thresholds(model, val_clips, cfg)
            per_lane: dict[str, list[float]] = defaultdict(list)
            for c in val_clips:
                f1 = evaluate_clip(model, c, cfg, thr)
                for ln, ts in c.onsets_by_lane.items():
                    if ts:
                        per_lane[ln].append(f1[ln])
            for ln, v in per_lane.items():
                results[li][ln].append(sum(v) / len(v))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        summ = " ".join(f"{ln}={np.mean(results[li][ln]):.2f}" for ln in cfg.lanes if results[li][ln])
        print(f"L{li:2d} done ({len(seeds)} seeds): {summ}", flush=True)

    # ---- report: lane x layer F1 (mean+/-std), best layer per lane ----
    print(f"\n==== per-stem per-lane F1 by MERT layer "
          f"(mean+/-std over {len(seeds)} seeds) ====", flush=True)
    print("  lane " + "".join(f"   L{li:<8d}" for li in layers), flush=True)
    for ln in cfg0.lanes:
        cells, best, bestv = "", None, -1.0
        any_data = False
        for li in layers:
            v = results[li].get(ln, [])
            if v:
                any_data = True
                m, s = float(np.mean(v)), float(np.std(v))
                cells += f"  {m:.2f}+/-{s:.2f}"
                if m > bestv:
                    bestv, best = m, li
            else:
                cells += "        -    "
        if any_data:
            print(f"  {ln:4s}{cells}   best=L{best}", flush=True)


if __name__ == "__main__":
    main()
