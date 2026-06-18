"""Cymbal feature-separability probe (diagnostic for the cymbal F1 ceiling).

The width A/B (h128 vs h512) and the data-scale sweep both plateaued, ruling out
head CAPACITY and data VOLUME as the cymbal bottleneck. This probe isolates the
remaining input-side suspect: how much ride/crash (and hat) discriminative
information is LINEARLY present in the FROZEN features AT the true onset frames --
decoupled from the GRU, recall, peak-picking and threshold tuning.

Method: reuse the exact pooled cym+hat per-stem clips + on-disk MERT|HB cache the
head-capacity sweep uses (apples-to-apples; all cache hits). At each GROUND-TRUTH
onset frame take the frozen feature vector and fit a LINEAR probe (multinomial
logistic regression -- convex, so it measures linear separability, not modelling
power) to predict the lane. Evaluate on the held-out val songs.

Three feature slices localise where the information lives (or doesn't):
  - MERT    : the 1024-d frozen encoder layer alone (24 kHz -> 12 kHz Nyquist)
  - MERT+HB : + the 16-d 6-20 kHz high-band block (the full model input)
  - HB-only : the 16-d high-band block alone
and two frame variants: the single onset frame vs a short post-onset mean-pool
(cymbal identity lives in the tail, so pooling may separate ride/crash where a
single frame can't).

HEADLINE: ride-vs-crash binary balanced accuracy. If even the best slice/variant
tops out ~0.6, the ride/crash information is NOT in the features -> the ceiling is
an INPUT/representation problem (richer HF front-end, multi-layer fusion, better
separation), not the head. If it's high (~0.85+), the features are fine and the
ceiling is downstream (recall / localisation / labels / joint decision).

Run on a box with the warm cache (3080):
  DRUMJOT_STAR=/codebox-workspace/datasets/star_balanced_sep \
  DRUMJOT_ENST=/codebox-workspace/datasets/enst-sep \
  DRUMJOT_EGMD=/codebox-workspace/datasets/egmd_sep \
  python training/scripts/cymbal_feature_probe.py \
      --cache /codebox-workspace/datasets/_cache_mert_pooled \
      --out-json /codebox-workspace/cymbal_feature_probe.json

Validate the code path without the cache/GPU:  python ... --selftest
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

from drumjot_training import embeddings  # noqa: E402

MERT_DIM = embeddings.MERT_DIM  # 1024
HB_BANDS = embeddings.HB_BANDS  # 16

# Feature column slices over the cached MERT|HB vector (1040-d).
SLICES: dict[str, tuple[int, int]] = {
    "MERT": (0, MERT_DIM),
    "MERT+HB": (0, MERT_DIM + HB_BANDS),
    "HB-only": (MERT_DIM, MERT_DIM + HB_BANDS),
}


def extract_lane_samples(clips, lanes, fps, pool_frames):
    """One sample per (lane, onset): (single-frame feat, pooled feat, lane idx).

    Pooled = mean over [onset, onset+pool_frames) (captures attack + early tail).
    Frames outside the feature range are skipped."""
    lane_idx = {ln: i for i, ln in enumerate(lanes)}
    xs, xp, y = [], [], []
    for clip in clips:
        feats = np.asarray(clip.features, dtype=np.float32)  # (T, D)
        n = feats.shape[0]
        for ln in lanes:
            for t in clip.onsets_by_lane.get(ln, []):
                f = int(round(float(t) * fps))
                if f < 0 or f >= n:
                    continue
                xs.append(feats[f])
                xp.append(feats[f : min(n, f + max(1, pool_frames))].mean(axis=0))
                y.append(lane_idx[ln])
    if not xs:
        d = MERT_DIM + HB_BANDS
        return np.zeros((0, d), np.float32), np.zeros((0, d), np.float32), np.zeros((0,), np.int64)
    return np.asarray(xs, np.float32), np.asarray(xp, np.float32), np.asarray(y, np.int64)


def extract_binary_rd_cr(clips, fps, pool_frames):
    """Ride(0)-vs-crash(1) samples, EXCLUDING frames where both fire (the
    ambiguous simultaneous hits) so the binary target is unambiguous."""
    xs, xp, y = [], [], []
    for clip in clips:
        feats = np.asarray(clip.features, dtype=np.float32)
        n = feats.shape[0]
        rd = {int(round(float(t) * fps)) for t in clip.onsets_by_lane.get("rd", [])}
        cr = {int(round(float(t) * fps)) for t in clip.onsets_by_lane.get("cr", [])}
        both = rd & cr
        for label, frames in ((0, rd - both), (1, cr - both)):
            for f in frames:
                if 0 <= f < n:
                    xs.append(feats[f])
                    xp.append(feats[f : min(n, f + max(1, pool_frames))].mean(axis=0))
                    y.append(label)
    if not xs:
        d = MERT_DIM + HB_BANDS
        return np.zeros((0, d), np.float32), np.zeros((0, d), np.float32), np.zeros((0,), np.int64)
    return np.asarray(xs, np.float32), np.asarray(xp, np.float32), np.asarray(y, np.int64)


def _metrics(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[t, p] += 1
    acc = float((y_true == y_pred).mean()) if len(y_true) else float("nan")
    per, recalls = [], []
    for c in range(n_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per.append({"prec": prec, "rec": rec, "f1": f1, "n": int(cm[c].sum())})
        if cm[c].sum():
            recalls.append(rec)
    return {
        "acc": acc,
        "bal_acc": float(np.mean(recalls)) if recalls else float("nan"),
        "macro_f1": float(np.mean([p["f1"] for p in per])),
        "per_class": per,
        "confusion": cm.tolist(),
    }


def linear_probe(x_tr, y_tr, x_va, y_va, n_classes, *, steps, lr, wd, device, seed=0):
    """Convex multinomial logistic probe. Standardises on train stats, trains
    class-weighted (inverse-freq) full-batch to convergence, evaluates on val."""
    import torch

    torch.manual_seed(seed)
    mu = x_tr.mean(0, keepdims=True)
    sd = x_tr.std(0, keepdims=True) + 1e-6
    xt = torch.tensor((x_tr - mu) / sd, device=device)
    xv = torch.tensor((x_va - mu) / sd, device=device)
    yt = torch.tensor(y_tr, device=device)
    counts = torch.bincount(yt, minlength=n_classes).float().clamp(min=1.0)
    weight = (counts.sum() / (counts * n_classes)).to(device)
    lin = torch.nn.Linear(xt.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    lossf = torch.nn.CrossEntropyLoss(weight=weight)
    for _ in range(steps):
        opt.zero_grad()
        lossf(lin(xt), yt).backward()
        opt.step()
    with torch.no_grad():
        pred = lin(xv).argmax(1).cpu().numpy()
    return _metrics(y_va, pred, n_classes)


def _sliced(x, name):
    a, b = SLICES[name]
    return x[:, a:b]


def run_probes(samp_tr, samp_va, lanes, *, steps, lr, wd, device, log):
    """samp_* = dict variant -> (X, y). Returns nested results[task][variant][slice]."""
    results: dict = {}
    n_lane = len(lanes)
    # 5-way lane probe
    results["lane5"] = {}
    for variant in ("single", "pooled"):
        xtr, ytr = samp_tr["lane"][variant]
        xva, yva = samp_va["lane"][variant]
        results["lane5"][variant] = {}
        for sl in SLICES:
            m = linear_probe(_sliced(xtr, sl), ytr, _sliced(xva, sl), yva, n_lane,
                             steps=steps, lr=lr, wd=wd, device=device)
            results["lane5"][variant][sl] = m
            log(f"  lane5 {variant:6s} {sl:8s}  acc {m['acc']:.3f}  bal {m['bal_acc']:.3f}  "
                f"macroF1 {m['macro_f1']:.3f}")
    # ride-vs-crash binary probe
    results["rd_cr"] = {}
    for variant in ("single", "pooled"):
        xtr, ytr = samp_tr["rdcr"][variant]
        xva, yva = samp_va["rdcr"][variant]
        results["rd_cr"][variant] = {}
        for sl in SLICES:
            m = linear_probe(_sliced(xtr, sl), ytr, _sliced(xva, sl), yva, 2,
                             steps=steps, lr=lr, wd=wd, device=device)
            results["rd_cr"][variant][sl] = m
            log(f"  rd_cr {variant:6s} {sl:8s}  bal {m['bal_acc']:.3f}  "
                f"F1 {m['macro_f1']:.3f}  (n_val {len(yva)})")
    return results


def _build_samples(clips, lanes, fps, pool_frames):
    xs, xp, y = extract_lane_samples(clips, lanes, fps, pool_frames)
    bxs, bxp, by = extract_binary_rd_cr(clips, fps, pool_frames)
    return {
        "lane": {"single": (xs, y), "pooled": (xp, y)},
        "rdcr": {"single": (bxs, by), "pooled": (bxp, by)},
    }


def _print_summary(results, lanes, log):
    log("\n==== SUMMARY ====")
    log("ride-vs-crash binary balanced accuracy (HEADLINE):")
    log(f"  {'variant':8s}" + "".join(f"{s:>10s}" for s in SLICES))
    best = 0.0
    for variant in ("single", "pooled"):
        row = f"  {variant:8s}"
        for sl in SLICES:
            v = results["rd_cr"][variant][sl]["bal_acc"]
            best = max(best, v)
            row += f"{v:10.3f}"
        log(row)
    log(f"  -> best ride/crash linear separability: {best:.3f}")
    log("     (~0.6 => info NOT in features = INPUT ceiling;  ~0.85+ => features fine, ceiling is downstream)")
    log("\n5-way lane balanced accuracy:")
    log(f"  {'variant':8s}" + "".join(f"{s:>10s}" for s in SLICES))
    for variant in ("single", "pooled"):
        row = f"  {variant:8s}"
        for sl in SLICES:
            row += f"{results['lane5'][variant][sl]['bal_acc']:10.3f}"
        log(row)
    # confusion of the best 5-way (pooled, MERT+HB) for the ride/crash story
    cm = np.array(results["lane5"]["pooled"]["MERT+HB"]["confusion"])
    log("\n5-way confusion (pooled, MERT+HB)  rows=true cols=pred:")
    log("        " + "".join(f"{ln:>7s}" for ln in lanes))
    for i, ln in enumerate(lanes):
        log(f"  {ln:5s} " + "".join(f"{int(cm[i, j]):7d}" for j in range(len(lanes))))


def _selftest():
    """Validate extraction + probe + slicing on synthetic clips, no cache/GPU.
    Plants ride/crash signal ONLY in HB columns, so HB-only & MERT+HB should
    separate (~1.0) while MERT-only stays near chance."""
    import types

    rng = np.random.default_rng(0)
    fps, d = embeddings.MERT_FPS, MERT_DIM + HB_BANDS
    lanes = list(LANES_CH)

    def make_clips(n_clips):
        clips = []
        for _ in range(n_clips):
            tt = 12.0
            nfr = int(tt * fps)
            feats = rng.standard_normal((nfr, d)).astype(np.float32)
            onsets = {ln: [] for ln in lanes}
            for _ in range(40):
                f = int(rng.integers(0, nfr - 10))
                ln = lanes[int(rng.integers(0, len(lanes)))]
                onsets[ln].append(f / fps)
                # plant separable signal in HB cols by lane (esp rd vs cr)
                sig = {"rd": 3.0, "cr": -3.0, "hc": 1.5, "ho": -1.5, "hp": 0.0}[ln]
                feats[f : f + 8, MERT_DIM:] += sig
            clips.append(types.SimpleNamespace(features=feats, onsets_by_lane=onsets))
        return clips

    log = lambda s: print(s, flush=True)  # noqa: E731
    tr = _build_samples(make_clips(40), lanes, fps, 8)
    va = _build_samples(make_clips(12), lanes, fps, 8)
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = run_probes(tr, va, lanes, steps=800, lr=0.05, wd=1e-4, device=dev, log=log)
    _print_summary(res, lanes, log)
    hb = res["rd_cr"]["pooled"]["HB-only"]["bal_acc"]
    mert = res["rd_cr"]["pooled"]["MERT"]["bal_acc"]
    assert hb > 0.9, f"selftest: HB-only should recover planted rd/cr signal, got {hb:.3f}"
    assert mert < 0.75, f"selftest: MERT-only should be ~chance on HB-planted signal, got {mert:.3f}"
    print(f"\nSELFTEST OK (HB-only rd/cr {hb:.3f} >> MERT-only {mert:.3f})", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Cymbal feature-separability probe")
    ap.add_argument("--pool-sources", default="star,enst,egmd")
    ap.add_argument("--pool-cap", type=int, default=3000,
                    help="train WINDOWS per source; keep =3000 to stay cache-hit with the sweep")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--train-max-windows", type=int, default=0)
    ap.add_argument("--val-max-windows", type=int, default=4,
                    help="keep =4 (the sweep's val cap) so val features are cache hits")
    ap.add_argument("--pool-frames", type=int, default=8, help="post-onset mean-pool length (~107 ms)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--cache", default="/codebox-workspace/datasets/_cache_mert_pooled")
    ap.add_argument("--out-json", default="/codebox-workspace/cymbal_feature_probe.json")
    ap.add_argument("--selftest", action="store_true", help="synthetic end-to-end check (no cache/GPU)")
    args = ap.parse_args()

    global LANES_CH, build_specs, make_cfg, _window_specs, materialize
    from head_capacity_sweep import LANES_CH, build_specs, make_cfg  # noqa: F401

    from drumjot_training.train import _window_specs, materialize  # noqa: F401

    if args.selftest:
        _selftest()
        return

    import torch

    from drumjot_training import runtime
    runtime.configure_backends()
    log = lambda s: print(s, flush=True)  # noqa: E731
    sources = [s.strip() for s in args.pool_sources.split(",") if s.strip()]
    cache = Path(args.cache)
    lanes = list(LANES_CH)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log(f"=== cymbal feature probe: cap={args.pool_cap} layer={args.layer} pool_frames={args.pool_frames} "
        f"device={device} ===")
    tr_specs, va_specs = build_specs(sources, args.pool_cap, cache)
    log(f"total: {len(tr_specs)} train / {len(va_specs)} val cym+hat stem clips")

    import dataclasses
    cfg = make_cfg(128, 2)  # only encoder/layer/lanes/high_band matter here
    cfg = dataclasses.replace(cfg, encoder_layer=args.layer)  # Config is frozen
    encoder = embeddings.make_encoder(cfg.encoder, cfg.encoder_layer)
    tr_w = _window_specs(tr_specs, 30.0, 3.0, args.train_max_windows, plan_cache_dir=cache)
    va_w = _window_specs(va_specs, 30.0, 3.0, args.val_max_windows, plan_cache_dir=cache)
    t0 = time.perf_counter()
    train_clips = materialize(tr_w, encoder, cfg, cache, 30.0, "train", log)
    val_clips = materialize(va_w, encoder, cfg, cache, 30.0, "val", log)
    log(f"materialize done in {time.perf_counter() - t0:.0f}s")
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fps = cfg.encoder_fps
    log("extracting onset-frame features...")
    samp_tr = _build_samples(train_clips, lanes, fps, args.pool_frames)
    samp_va = _build_samples(val_clips, lanes, fps, args.pool_frames)
    n_tr = len(samp_tr["lane"]["single"][1])
    n_va = len(samp_va["lane"]["single"][1])
    n_rdcr_tr = len(samp_tr["rdcr"]["single"][1])
    n_rdcr_va = len(samp_va["rdcr"]["single"][1])
    counts = np.bincount(samp_tr["lane"]["single"][1], minlength=len(lanes))
    log(f"samples: {n_tr} train / {n_va} val onsets across 5 lanes "
        f"({dict(zip(lanes, counts.tolist(), strict=True))}); "
        f"ride-vs-crash {n_rdcr_tr} train / {n_rdcr_va} val")

    results = run_probes(samp_tr, samp_va, lanes, steps=args.steps, lr=args.lr, wd=args.wd,
                         device=device, log=log)
    _print_summary(results, lanes, log)

    out = {
        "config": vars(args),
        "lanes": lanes,
        "n_samples": {"train": int(n_tr), "val": int(n_va),
                      "rdcr_train": int(n_rdcr_tr), "rdcr_val": int(n_rdcr_va),
                      "per_lane_train": counts.tolist()},
        "results": results,
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    log(f"\nresults -> {args.out_json}")


if __name__ == "__main__":
    main()
