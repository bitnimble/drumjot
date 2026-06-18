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

=== Environment (any CUDA box, e.g. an RTX 3080) ===
This script is pure torch/transformers. The project pins the **cu128** torch stack
(transcriber/pyproject.toml), which is forward-compatible with newer drivers (a
cu128 build runs fine on a CUDA 13.0 driver), so you do NOT need cu130 wheels.

  1. torch + torchvision + torchaudio ALL from the cu128 index, together (their
     ABIs must match; a torchvision from generic PyPI vs a cu128 torch yields
     "operator torchvision::nms does not exist"). `uv sync` should produce this;
     if torchvision came from PyPI, force it:
       uv pip install --reinstall --no-deps torchvision==<ver>+cu128 \
         --index-url https://download.pytorch.org/whl/cu128
     (Per project policy, install deps yourself -- ordering is fragile.)
  2. MERT weights cached locally: HF is forced offline, so run
       python3 training/scripts/fetch_models.py
     on this box first (else you get a clear "run fetch_models" error).
  3. Dataset roots: export DRUMJOT_STAR / DRUMJOT_ENST / DRUMJOT_EGMD to the
     sep-tree paths ON THIS BOX (star_balanced_sep / enst-sep / egmd_sep), and
     point --pool-cache at where the trainer will read the cache from.

The script prints torch/CUDA/device at startup so you can confirm the GPU + a
working torch before it spends hours encoding.

CROSS-BOX RESUME CAVEAT: the cache key includes the clip's ABSOLUTE audio path, so
resuming a partial cache on a DIFFERENT box only skips already-done windows if the
datasets resolve at the SAME absolute path (same NFS mount point) AND the config
(layer / high-band / max-seconds / window-search) is identical. Mount the share at
the same path on both boxes (or symlink) or the second box re-encodes from scratch.

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


def encode_pipelined(specs, encoder, cfg, cache, workers, writers, log):
    """Encode all window-specs into `cache`, optimised for an NFS source.

    Three things the serial `materialize` doesn't do, which matter when the audio
    lives on NFS (else the GPU starves on I/O, like the stem-split job):
    - **Load each clip ONCE** (at both sample rates) and slice all its windows from
      memory, instead of re-decoding the whole file per window x per sample rate
      (`embed_clip` otherwise reloads the entire file 2x for EVERY window).
    - **Parallel loader threads** overlap NFS reads with the GPU forward and raise
      aggregate NFS throughput (librosa releases the GIL during decode).
    - **Background .npy writes** so saves don't stall the forward loop.

    Byte-identical to `materialize`: it reuses `embed_clip` / `_rings_for_clip` with
    the preloaded audio passed in, so only WHERE the bytes come from changes.
    Resumable: clips whose every feature (+ needed rings) already exist are skipped.
    """
    import queue
    import threading
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    cache = Path(cache)
    variant = embeddings.feat_variant(cfg.high_band)

    def feat_path(audio, start, length):
        k = embeddings.cache_key(audio, cfg.encoder, cfg.encoder_layer, length, variant, start)
        return cache / f"{k}.npy"

    def rings_path(audio, start, length):
        k = embeddings.cache_key(audio, cfg.encoder, cfg.encoder_layer, length, start=start)
        return cache / f"{k}.rings.json"

    by_clip = defaultdict(list)  # audio -> [(onsets, start, length)]
    for audio, onsets, _weight, start, length in specs:
        by_clip[audio].append((onsets, start, length))

    def needs_work(audio, wins):
        for onsets, start, length in wins:
            if not feat_path(audio, start, length).exists():
                return True
            capped = train._cap_onsets(onsets, length)
            if (any(capped.get(ln) for ln in train.SUSTAINED_LANES)
                    and not rings_path(audio, start, length).exists()):
                return True
        return False

    todo = [(a, w) for a, w in by_clip.items() if needs_work(a, w)]
    n_win = sum(len(w) for _, w in todo)
    log(f"{len(by_clip)} clips total, {len(todo)} need work ({n_win} windows); "
        f"{workers} loaders / {writers} writers")
    if not todo:
        return

    load_q: queue.Queue = queue.Queue()
    ready_q: queue.Queue = queue.Queue(maxsize=workers * 2)  # backpressure -> bounded RAM
    for item in todo:
        load_q.put(item)
    for _ in range(workers):
        load_q.put(None)  # one stop sentinel per loader

    def loader():
        import librosa
        while True:
            item = load_q.get()
            if item is None:
                return
            audio, wins = item
            try:
                y24 = embeddings.load_audio(audio, sr=encoder.sr)
                y44 = (librosa.load(str(audio), sr=embeddings.HB_SR, mono=True)[0]
                       if cfg.high_band else None)
                ready_q.put((audio, wins, y24, y44))
            except Exception as e:  # noqa: BLE001
                log(f"  load fail {Path(audio).name}: {e!r}")
                ready_q.put((audio, wins, None, None))  # keep the consumer count balanced

    loaders = [threading.Thread(target=loader, daemon=True) for _ in range(workers)]
    for t in loaders:
        t.start()

    def _save(path, feat):
        try:
            # tmp MUST end in .npy, else np.save appends .npy (-> hash.tmp.npy.npy)
            # and the os.replace below renames a nonexistent file.
            tmp = path.with_name(path.stem + ".tmp.npy")
            np.save(tmp, feat)
            os.replace(tmp, path)
        except Exception as e:  # noqa: BLE001
            log(f"  write fail {path.name}: {e!r}")

    writer = ThreadPoolExecutor(max_workers=writers)
    t0 = time.perf_counter()
    n_done = n_enc = 0
    for _ in range(len(todo)):  # GPU consumer (main thread; only it touches the model)
        audio, wins, y24, y44 = ready_q.get()
        if y24 is None:
            n_done += 1
            continue
        for onsets, start, length in wins:
            fp = feat_path(audio, start, length)
            if not fp.exists():
                try:
                    feat = embeddings.embed_clip(
                        audio, encoder, cache_dir=None, max_seconds=length,
                        cache_dtype=cfg.cache_dtype, high_band=cfg.high_band,
                        start_seconds=start, y_full=y24, y44_full=y44,
                    )
                    writer.submit(_save, fp, feat)
                    n_enc += 1
                except Exception as e:  # noqa: BLE001
                    log(f"  skip {Path(audio).name}@{start:.0f}s: {e!r}")
                    continue
            try:  # rings reuse the preloaded y24 (no reload); idempotent json side-cache
                train._rings_for_clip(audio, onsets, cfg, cache, length, start, y_full=y24)
            except Exception as e:  # noqa: BLE001
                log(f"  rings fail {Path(audio).name}@{start:.0f}s: {e!r}")
        n_done += 1
        if n_done % 50 == 0:
            rate = n_enc / max(1e-9, time.perf_counter() - t0)
            log(f"  {n_done}/{len(todo)} clips, {n_enc} windows ({rate:.1f} win/s)")
    writer.shutdown(wait=True)
    for t in loaders:
        t.join()
    log(f"pipelined encode: {n_enc} new windows from {len(todo)} clips")


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
    ap.add_argument("--workers", type=int, default=6,
                    help="parallel loader threads (overlap NFS audio reads with the GPU; raise if "
                    "the GPU is still I/O-starved, lower if NFS/RAM is the limit)")
    ap.add_argument("--writers", type=int, default=4, help="background .npy writer threads")
    ap.add_argument("--no-pipeline", action="store_true",
                    help="use the simple serial materialize instead of the load-once/prefetch "
                    "pipeline (debug / parity check)")
    ap.add_argument("--log", default=None,
                    help="tee stdout+stderr to this file (self-log; no manual redirect needed)")
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch

    runtime.tee_stdio(args.log)
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
    tr_w = train._window_specs(train_specs, win, args.window_search, 0, plan_cache_dir=cache)
    va_w = train._window_specs(val_specs, win, args.window_search, 0, plan_cache_dir=cache)
    print(f"\nfull windowing: {len(train_specs)} train clips -> {len(tr_w)} windows | "
          f"{len(val_specs)} val clips -> {len(va_w)} windows", flush=True)
    print(f"cache dir: {cache}  (resumable: already-cached windows are skipped)\n", flush=True)

    log = lambda s: print(s, flush=True)  # noqa: E731
    t0 = time.perf_counter()
    if args.no_pipeline:
        train.materialize(tr_w, encoder, cfg, cache, args.max_seconds, "train", log)
        train.materialize(va_w, encoder, cfg, cache, args.max_seconds, "val", log)
    else:
        # train+val keys are split-independent, so encode them in one pooled pass
        encode_pipelined(tr_w + va_w, encoder, cfg, cache, args.workers, args.writers, log)
    n_npy = len(list(Path(cache).glob("*.npy")))
    print(f"\ndone in {(time.perf_counter() - t0) / 60:.1f} min  ({n_npy} .npy now in {cache})", flush=True)


if __name__ == "__main__":
    main()
