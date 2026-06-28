"""Type the ride/crash MISSES + sweep the cymbal peak-picker (recall error analysis).

The recall-vs-confusion decomposition (RESULTS.md 2026-06-18) showed the dominant
cymbal error is MISSED detection, not ride<->crash confusion (ride 15.1% missed /
0.7% confused; crash 27.6% missed / 17.3% confused). This script answers the
follow-up: *why* are they missed, and how much is recoverable with the picker
alone vs needs a retrain.

It reuses the already-trained checkpoint (no new training) -- one forward pass of
the small head over the WARM pooled-MERT cache gives per-frame activations, then
everything else is pure numpy:

  Part A -- MISS TYPING. Every true ride/crash onset the lane's own picker fails
  to detect is bucketed by *why*, inspecting this lane's activation around it:
    - dead         : activation ~0 in the window (no bump). The model doesn't see
                     it -> needs RETRAIN (class-weighting / sigma / domain / features).
    - subthreshold : a bump exists but peaks below the tuned height -> lower threshold.
    - merge        : clears height & is a peak, but suppressed by the 70 ms
                     min-distance (a stronger neighbour is <70 ms away) -> dense
                     ride collapsed; loosen min-distance.
    - decay        : killed by the decay-reset filter (rides on a ringing tail)
                     -> loosen decay_reset_frac.
    - prominence   : killed by the prominence floor -> lower prominence.
  dead => retrain is the only lever; merge/decay/prominence/subthreshold => the
  picker grid-sweep (Part B) recovers it with no GPU.

  Part B -- PICKER SWEEP. For ride and crash, grid over (min_distance, decay_reset,
  prominence, threshold-scale), re-pick from the cached activations, and report
  recall/precision/F1 vs the current config -- the recall/precision trade-off and
  the best achievable without retraining.

Run on a box with the warm cache + the saved checkpoint (the 3080 is idle after
the decomposition; this is light enough to also run cpu-side without touching a
training GPU -- pass --device cpu):

  DRUMJOT_STAR=/codebox-workspace/datasets/star_balanced_sep \
  DRUMJOT_ENST=/codebox-workspace/datasets/enst-sep \
  DRUMJOT_EGMD=/codebox-workspace/datasets/egmd_sep \
  python training/scripts/cymbal_miss_typing.py \
      --load /codebox-workspace/checkpoints/h128_cymhat_s1.pt \
      --cache /codebox-workspace/mert_cache \
      --out-json /codebox-workspace/cymbal_miss_typing.json

Validate the typing + sweep logic without cache/GPU:  python ... --selftest
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

CYM = ("rd", "cr")
MISS_CATS = ("dead", "subthreshold", "merge", "decay", "prominence", "height_other")


class _CacheKeyEncoder:
    """Encoder stand-in exposing only .name/.layer for materialize's cache-key
    lookup. The warm cache means embed_clip never runs, so the real MERT model
    (and its heavy transformers/torchvision imports) is never constructed -- the
    diagnostic runs on a box whose torchvision is broken or whose GPU is busy."""

    def __init__(self, name: str, layer: int):
        self.name = name
        self.layer = layer


def _picked_frames(act, fps, thr, params):
    """Kept onset frame indices for one activation under `params` (the picker)."""
    from drumjot_training.metrics import pick_onsets

    on = pick_onsets(act, fps, thr, **params)
    if len(on) == 0:
        return np.empty(0, dtype=int)
    return np.round(np.asarray(on) * fps).astype(int)


def _near(frames, f, tolf):
    return frames.size > 0 and bool(np.any(np.abs(frames - f) <= tolf))


def classify_miss(act, f, fps, thr, params, tolf):
    """Bucket one missed onset (frame `f`) by *why* the picker drops it.

    Priority: dead < subthreshold < merge < decay < prominence < height_other.
    `merge`/`decay`/`prominence` are attributed by relaxing exactly ONE picker
    constraint and checking whether a kept peak then lands within `tolf` of `f`.
    """
    lo, hi = max(0, f - tolf), min(act.size, f + tolf + 1)
    win = act[lo:hi]
    if win.size == 0:
        return "dead"
    vmax = float(win.max())
    floor = max(0.10, 0.5 * thr)
    if vmax < floor:
        return "dead"
    if vmax < thr:
        return "subthreshold"
    # vmax >= thr: a height-clearing bump exists but the full picker drops it.
    # Find which single relaxation recovers it.
    if _near(_picked_frames(act, fps, thr, {**params, "min_distance_s": 1.0 / fps}), f, tolf):
        return "merge"
    if params.get("decay_reset_frac", 0.0) > 0.0 and _near(
        _picked_frames(act, fps, thr, {**params, "decay_reset_frac": 0.0}), f, tolf
    ):
        return "decay"
    if params.get("prominence") and _near(
        _picked_frames(act, fps, thr, {**params, "prominence": 0.0}), f, tolf
    ):
        return "prominence"
    return "height_other"  # clears height but no single relax recovers (combination/edge)


def type_misses(acts, refs, fps, thr, params, tol_s):
    """Counts of (own-lane misses by category) + n_ref + n_hit for one lane.

    `acts`/`refs` are per-clip lists (activation array, ref onset times). A ref is
    an own-lane miss when the picker keeps no onset within tolerance of it."""
    from mir_eval.util import match_events

    tolf = int(round(tol_s * fps))
    cats = dict.fromkeys(MISS_CATS, 0)
    n_ref = n_hit = 0
    for act, ref in zip(acts, refs, strict=True):
        ref = np.sort(np.asarray(ref, dtype=np.float64))
        n_ref += len(ref)
        if len(ref) == 0:
            continue
        est = _picked_frames(act, fps, thr, params).astype(np.float64) / fps
        m = match_events(ref, np.sort(est), tol_s) if len(est) else []
        matched = {i for i, _ in m}
        n_hit += len(matched)
        for i in range(len(ref)):
            if i in matched:
                continue
            cats[classify_miss(act, int(round(ref[i] * fps)), fps, thr, params, tolf)] += 1
    return cats, n_ref, n_hit


def _rpf(acts, refs, fps, thr, params, tol_s):
    """(recall, precision, f1, hits, n_ref, n_est) for one lane under `params`."""
    from mir_eval.util import match_events

    hits = n_ref = n_est = 0
    for act, ref in zip(acts, refs, strict=True):
        ref = np.sort(np.asarray(ref, dtype=np.float64))
        est = np.sort(_picked_frames(act, fps, thr, params).astype(np.float64) / fps)
        n_ref += len(ref)
        n_est += len(est)
        if len(ref) and len(est):
            hits += len(match_events(ref, est, tol_s))
    recall = hits / n_ref if n_ref else 0.0
    prec = hits / n_est if n_est else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0
    return recall, prec, f1, hits, n_ref, n_est


def sweep_picker(acts, refs, fps, base_thr, base_params, tol_s):
    """Grid-sweep the picker for one lane; return a list of result rows.

    Axes: min-distance, decay-reset fraction, prominence, threshold scale. Each
    row has the params + recall/precision/f1. The first row is the current config."""
    grids = {
        "min_distance_s": [base_params["min_distance_s"], 0.050, 0.040, 0.030],
        "decay_reset_frac": [base_params["decay_reset_frac"], 0.3, 0.0],
        "prominence": [base_params["prominence"], 0.10],
        "thr_scale": [1.0, 0.85],
    }
    # de-dup while preserving order so the current value isn't tried twice
    md = list(dict.fromkeys(grids["min_distance_s"]))
    dr = list(dict.fromkeys(grids["decay_reset_frac"]))
    pr = list(dict.fromkeys(grids["prominence"]))
    ts = list(dict.fromkeys(grids["thr_scale"]))
    rows = []
    for s in ts:
        for d in md:
            for k in dr:
                for p in pr:
                    params = {**base_params, "min_distance_s": d, "decay_reset_frac": k,
                              "prominence": p}
                    thr = base_thr * s
                    r, pre, f1, *_ = _rpf(acts, refs, fps, thr, params, tol_s)
                    rows.append({"min_distance_s": d, "decay_reset_frac": k, "prominence": p,
                                 "thr_scale": s, "recall": r, "precision": pre, "f1": f1})
    return rows


def _report(typing, sweeps, base, log):
    out = {"typing": {}, "sweep": {}, "base": base}
    log("\n==== ride/crash MISS TYPING (own-lane misses, current picker) ====")
    for ln in CYM:
        cats, n_ref, n_hit = typing[ln]
        n_miss = n_ref - n_hit
        log(f"  {ln}: recall {n_hit}/{n_ref} = {(n_hit / n_ref if n_ref else 0.0):.3f}  "
            f"({n_miss} misses)")
        if n_miss:
            for c in MISS_CATS:
                if cats[c]:
                    log(f"      {c:13s} {cats[c]:5d}  ({cats[c] / n_miss:5.1%})")
        recoverable = sum(cats[c] for c in ("subthreshold", "merge", "decay", "prominence"))
        log(f"      -> picker-recoverable {recoverable}/{n_miss} "
            f"({(recoverable / n_miss if n_miss else 0):.1%}); dead/retrain {cats['dead']} "
            f"({(cats['dead'] / n_miss if n_miss else 0):.1%})")
        out["typing"][ln] = {"n_ref": n_ref, "n_hit": n_hit, "n_miss": n_miss,
                             "recall": n_hit / n_ref if n_ref else 0.0, "cats": cats,
                             "picker_recoverable": recoverable}

    log("\n==== picker SWEEP (recall / precision / f1 per lane) ====")
    for ln in CYM:
        rows = sweeps[ln]
        cur = rows[0]
        best_f1 = max(rows, key=lambda r: r["f1"])
        # max recall while keeping precision >= current
        no_prec_loss = [r for r in rows if r["precision"] >= cur["precision"] - 1e-9]
        best_recall = max(no_prec_loss or rows, key=lambda r: r["recall"])

        def fmt(lane, tag, r):
            log(f"  {lane} {tag:18s} md={r['min_distance_s']:.3f} dr={r['decay_reset_frac']:.1f} "
                f"pr={r['prominence']:.2f} thr*{r['thr_scale']:.2f}  "
                f"R={r['recall']:.3f} P={r['precision']:.3f} F1={r['f1']:.3f}")
        fmt(ln, "current", cur)
        fmt(ln, "best-F1", best_f1)
        fmt(ln, "max-R @ P>=cur", best_recall)
        out["sweep"][ln] = {"current": cur, "best_f1": best_f1, "max_recall_no_prec_loss": best_recall,
                            "all": rows}
    return out


def _selftest():
    fps = 75.0  # 0.070 s min-dist -> 5 frames; 0.020 s -> ~2 frames
    tolf = 4
    base = {"min_distance_s": 0.070, "prominence": 0.20, "decay_reset_frac": 0.6,
            "decay_reset_floor": 0.05}
    thr = 0.5
    n = 400

    def ck(act, f, want):
        got = classify_miss(act, f, fps, thr, base, tolf)
        assert got == want, f"frame {f}: got {got!r}, want {want!r}"

    # dead: flat ~0 around the onset -> model sees nothing
    ck(np.zeros(n), 100, "dead")

    # subthreshold: a clean bump that peaks below thr
    a = np.zeros(n)
    a[99], a[100], a[101] = 0.2, 0.30, 0.2
    ck(a, 100, "subthreshold")

    # merge: two isolated peaks 3 frames apart (<5), 2nd has full prominence
    # (deep dip between) so ONLY the 70 ms min-distance suppresses it.
    a = np.zeros(n)
    a[100], a[101], a[102], a[103] = 0.9, 0.0, 0.0, 0.8
    ck(a, 103, "merge")

    # decay: 2nd hit rides a sustain that never decays below the reset level
    # (0.6*0.9=0.54); the dip stays at 0.6 -> ONLY decay-reset drops it.
    a = np.zeros(n)
    a[100] = 0.9
    a[101:119] = 0.6
    a[119] = 0.85
    a[120:140] = 0.6
    ck(a, 119, "decay")

    # prominence: dip drops below the decay reset (0.50<0.54, so decay PASSES)
    # but the peak's prominence (0.65-0.50=0.15) is under 0.20 -> ONLY prominence.
    a = np.zeros(n)
    a[100] = 0.9
    a[101:109] = 0.6
    a[109] = 0.50
    a[110] = 0.65
    a[111] = 0.40
    ck(a, 110, "prominence")

    # sweep monotonicity: loosening min-distance recovers the merged 2nd hit
    merged = np.zeros(n)
    merged[100], merged[101], merged[102], merged[103] = 0.9, 0.0, 0.0, 0.8
    acts = [merged]
    refs = [np.array([100 / fps, 103 / fps])]
    r_tight, *_ = _rpf(acts, refs, fps, thr, base, 0.05)
    r_loose, *_ = _rpf(acts, refs, fps, thr, {**base, "min_distance_s": 0.020}, 0.05)
    assert r_loose > r_tight, (r_tight, r_loose)
    print("SELFTEST OK (classify_miss buckets + sweep recall behave)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Type ride/crash misses + sweep the cymbal picker")
    ap.add_argument("--load", default="/codebox-workspace/checkpoints/h128_cymhat_s1.pt",
                    help="trained checkpoint (state_dict + thresholds + hidden)")
    ap.add_argument("--pool-sources", default="star,enst,egmd")
    ap.add_argument("--pool-cap", type=int, default=3000, help="keep =3000 to stay cache-hit")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--val-max-windows", type=int, default=4)
    ap.add_argument("--cache", default="/codebox-workspace/mert_cache")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                    help="cpu keeps a busy training GPU untouched (the head is tiny)")
    ap.add_argument("--out-json", default="/codebox-workspace/cymbal_miss_typing.json")
    ap.add_argument("--selftest", action="store_true", help="validate the logic; no cache/GPU")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    import torch
    from head_capacity_sweep import build_specs, make_cfg

    from drumjot_training import embeddings, metrics, runtime
    from drumjot_training.model import MultiLaneHeads
    from drumjot_training.train import _clip_probs, _window_specs, materialize

    runtime.tee_stdio(Path(args.out_json).with_suffix(".log"))  # self-log next to --out-json
    runtime.configure_backends()
    log = lambda s: print(s, flush=True)  # noqa: E731
    sources = [s.strip() for s in args.pool_sources.split(",") if s.strip()]
    cache = Path(args.cache)

    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())

    ck = torch.load(args.load, map_location="cpu")
    hidden = int(ck["hidden"])
    thresholds = ck["thresholds"]
    log(f"=== miss typing + picker sweep: {args.load} (h{hidden}) ===")

    _, va_specs = build_specs(sources, args.pool_cap, cache)
    cfg = make_cfg(hidden, args.layers, 3e-4)
    in_dim = embeddings.feat_dim(cfg.high_band)
    # With a warm cap-3000 cache every window is a hit, so materialize only needs
    # the encoder's .name/.layer for the cache key -- it never loads MERT. Use a
    # stand-in to skip the heavy (and here torchvision-broken / GPU-bound)
    # transformers import entirely. A genuine cache miss would just skip that
    # window (logged), which is correct for a diagnostic that requires the cache.
    encoder = _CacheKeyEncoder(cfg.encoder, cfg.encoder_layer)
    va_w = _window_specs(va_specs, 30.0, 3.0, args.val_max_windows, plan_cache_dir=cache)
    val_clips = materialize(va_w, encoder, cfg, cache, 30.0, "val", log)  # type: ignore[arg-type]  # duck-typed stand-in (warm cache => .name/.layer only)
    del encoder
    if use_cuda:
        torch.cuda.empty_cache()
    model = MultiLaneHeads(in_dim=in_dim, hidden=hidden, num_layers=args.layers, lane_names=cfg.lanes)
    model.load_state_dict(ck["state_dict"])
    if use_cuda:
        model = model.cuda()
    log(f"device: {'cuda' if use_cuda else 'cpu'}; "
        f"thresholds: {dict((ln, round(thresholds.get(ln, cfg.peak_threshold), 2)) for ln in CYM)}")

    # one forward pass -> cache per-clip cymbal activations + ref onsets (then numpy only)
    lane_i = {ln: i for i, ln in enumerate(cfg.lanes)}
    acts = {ln: [] for ln in CYM}
    refs = {ln: [] for ln in CYM}
    t0 = time.perf_counter()
    for k, clip in enumerate(val_clips, 1):
        probs = _clip_probs(model, clip)
        for ln in CYM:
            acts[ln].append(np.ascontiguousarray(probs[lane_i[ln]]))
            refs[ln].append(np.asarray(clip.onsets_by_lane.get(ln, []), dtype=np.float64))
        if k % 200 == 0:
            log(f"  forward: {k}/{len(val_clips)} clips ({time.perf_counter() - t0:.0f}s)")
    del model
    if use_cuda:
        torch.cuda.empty_cache()

    fps, tol_s = cfg.encoder_fps, cfg.onset_tolerance_s
    base = {ln: dict(metrics.LANE_PEAK_PARAMS[ln]) for ln in CYM}
    typing = {ln: type_misses(acts[ln], refs[ln], fps,
                              thresholds.get(ln, cfg.peak_threshold), base[ln], tol_s) for ln in CYM}
    sweeps = {ln: sweep_picker(acts[ln], refs[ln], fps,
                               thresholds.get(ln, cfg.peak_threshold), base[ln], tol_s) for ln in CYM}

    base_meta = {ln: {"threshold": thresholds.get(ln, cfg.peak_threshold), **base[ln]} for ln in CYM}
    out = _report({ln: typing[ln] for ln in CYM}, sweeps, base_meta, log)
    Path(args.out_json).write_text(json.dumps({"config": vars(args), **out}, indent=2))
    log(f"\nresults -> {args.out_json}")


if __name__ == "__main__":
    main()
