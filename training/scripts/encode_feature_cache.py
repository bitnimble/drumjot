"""Pre-encode the pooled MERT feature cache with FULL windowing.

Training now slices every clip into as many ~max-seconds windows as fit (the
former `--max-windows` flag is gone), so the first full-windowed run would
otherwise re-encode every clip's later windows on whatever box runs training.
This script does that one-time encode pass up front on a fast GPU, populating the
SAME cache (same keys) the trainer reads -- so a subsequent `train.py` run is all
cache hits, no encoding.

It reuses the production code paths verbatim -- `train._pooled_specs` (indexing +
restricted/full onsets + onset JSON cache), `train._window_specs(..., 0)` (full
windowing, with the low-energy cut nudge and the short-tail merge), and
`train.materialize` (per-window embed + the .rings.json aux-activity side-cache).
So what it caches is exactly what the trainer asks for; nothing can drift.

Resumable: `embed_clip` skips any window already cached, so re-running after an
interruption only encodes what's missing. Start small with `--pool-cap 50` to
sanity-check throughput before the full pass.

=== Running on a CUDA 13.0 (cu130) box, e.g. an RTX 3080 ===
This script is pure torch/transformers; cu130 is purely an *environment* concern:

  1. A torch built for CUDA 13.0. With uv (the project's tool):
       uv pip install --python <venv> torch --torch-backend=cu130
     or with pip:
       pip install torch --index-url https://download.pytorch.org/whl/cu130
     The 3080 is Ampere (sm_86), supported by the cu130 wheels. Keep the rest of
     the training deps (transformers, librosa, soundfile, ...) from
     transcriber/pyproject.toml. (Per project policy, install deps yourself --
     ordering is fragile.)
  2. MERT weights cached locally: HF is forced offline, so run
       python3 training/scripts/fetch_models.py
     on this box first (else you get a clear "run fetch_models" error).
  3. Dataset roots: export DRUMJOT_STAR / DRUMJOT_ENST / DRUMJOT_EGMD to the
     sep-tree paths ON THIS BOX (star_balanced_sep / enst-sep / egmd_sep), and
     point --pool-cache at where the trainer will read the cache from.

The script prints torch/CUDA/device at startup so you can confirm it's actually
on the 3080 with a cu130 runtime before it spends hours encoding.

Usage:
  export DRUMJOT_STAR=/data/star_balanced_sep DRUMJOT_ENST=/data/enst-sep \
         DRUMJOT_EGMD=/data/egmd_sep
  python3 training/scripts/encode_feature_cache.py \
      --pool-cache /data/_cache_mert_pooled            # full set, all stems
  python3 training/scripts/encode_feature_cache.py --pool-cap 50 ...  # quick trial
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))  # training/
sys.path.insert(0, os.path.join(_HERE, "..", "..", "dsp"))  # dsp/ (high-band block)

from drumjot_training import embeddings, runtime, train  # noqa: E402
from drumjot_training.config import Config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Pre-encode the full-windowed pooled MERT cache")
    ap.add_argument("--pool-sources", default="star,enst,egmd",
                    help="comma list of sep-tree sources (DRUMJOT_<SRC> must point at each)")
    ap.add_argument("--pool-cap", type=int, default=0,
                    help="max source-clips per dataset (0 = ALL); each expands to ~5 stems x N windows")
    ap.add_argument("--pool-cache", default=None,
                    help="feature-cache dir the trainer will read (default: <common parent of "
                    "sources>/_cache_mert_pooled, matching train.py)")
    ap.add_argument("--layer", type=int, default=10, help="MERT hidden layer (must match training)")
    ap.add_argument("--high-band", default=True, action=argparse.BooleanOptionalAction,
                    help="append the 6-20 kHz high-band block (default on; must match training)")
    ap.add_argument("--max-seconds", type=float, default=30.0, help="window length (must match training)")
    ap.add_argument("--window-search", type=float, default=3.0,
                    help="low-energy cut nudge radius (must match training)")
    ap.add_argument("--cache-dtype", choices=("float16", "float32"), default="float16")
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch

    print(f"torch {torch.__version__}  built-for-cuda {torch.version.cuda}  "
          f"cuda_available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability(0)
        print(f"device: {torch.cuda.get_device_name(0)}  compute capability {cc[0]}.{cc[1]}", flush=True)
    else:
        print("WARNING: no CUDA visible -- encoding on CPU is impractically slow. Check the "
              "cu130 torch install + drivers before continuing.", flush=True)

    runtime.configure_backends()
    cfg = Config(
        encoder=embeddings.MERT_NAME, encoder_fps=embeddings.MERT_FPS, encoder_layer=args.layer,
        cache_dtype=args.cache_dtype, high_band=args.high_band,
    )
    print(f"encoder={cfg.encoder} layer={cfg.encoder_layer} high_band={cfg.high_band} "
          f"variant={embeddings.feat_variant(cfg.high_band)} feat_dim={embeddings.feat_dim(cfg.high_band)} "
          f"cache_dtype={cfg.cache_dtype}", flush=True)

    # Reuse the production pooled indexing -> identical specs + cache keys as train.main().
    # pool_balance is off: it only oversamples (repeats) specs, which is pointless for
    # encoding (a repeat hits the same cache key).
    ns = argparse.Namespace(
        pool_sources=args.pool_sources, pool_cap=args.pool_cap,
        pool_balance=False, pool_cache=args.pool_cache,
    )
    train_specs, val_specs, cache = train._pooled_specs(ns)

    encoder = embeddings.make_encoder(cfg.encoder, cfg.encoder_layer)
    nhs = encoder.n_hidden_states()
    if not 0 <= cfg.encoder_layer < nhs:
        raise SystemExit(f"--layer {cfg.encoder_layer} out of range: valid 0..{nhs - 1}")

    # FULL windowing (max_windows=0) for BOTH splits -- exactly what train.main() now does.
    win = args.max_seconds or 30.0
    tr_w = train._window_specs(train_specs, win, args.window_search, 0)
    va_w = train._window_specs(val_specs, win, args.window_search, 0)
    print(f"\nfull windowing: {len(train_specs)} train clips -> {len(tr_w)} windows | "
          f"{len(val_specs)} val clips -> {len(va_w)} windows", flush=True)
    print(f"cache dir: {cache}  (resumable: already-cached windows are skipped)\n", flush=True)

    log = lambda s: print(s, flush=True)  # noqa: E731
    t0 = time.perf_counter()
    train.materialize(tr_w, encoder, cfg, cache, args.max_seconds, "train", log)
    train.materialize(va_w, encoder, cfg, cache, args.max_seconds, "val", log)
    n_npy = len(list(Path(cache).glob("*.npy")))
    print(f"\ndone in {(time.perf_counter() - t0) / 60:.1f} min  ({n_npy} .npy now in {cache})", flush=True)


if __name__ == "__main__":
    main()
