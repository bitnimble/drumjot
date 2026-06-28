"""Crash label-quality audit using the CANONICAL support gate (forced_align/clean).

The miss-typing (RESULTS.md 2026-06-18) found the head leaves 74.6% of crash
onsets `dead`. The onset cleaner that should catch bad targets -- `forced_align`'s
per-note envelope snap + `support_floor` gate, scored by `clean.support_score` --
EXISTS but is only wired into the ParaDB eval (`eval_paradb.py`), NOT the training
data path: `build_specs` -> the dataset readers feed RAW onsets straight into the
pool. So the model trains on uncleaned targets.

This runs that exact cleaner over every crash onset to answer: how many of the
model's crash targets would the existing support gate DISCARD as unsupported (no
real transient in the stem within +/-window)? It reuses the canonical code
(`forced_align.onset_envelope` + `postfilter.support_floor_from_env` +
`forced_align.align_lane`) with the ParaDB defaults (percentile 60, window 0.03 s),
so the number is "what your cleaner would do," not a hand-rolled proxy.

Caveat: the percentile floor was tuned for the full DRUM stem (constant activity);
on an isolated, mostly-silent crash stem the 60th-pct floor sits near the noise
floor and the gate runs LENIENT, so this is a *lower bound* on the unsupported
rate. The per-source floors are reported so that's visible.

Per-source split (egmd has known MIDI<->audio drift; star labels are clean by
design so its misses are separation drops; enst is real+manual) attributes cause.

Validate the support logic without audio/GPU:  python ... --selftest

  DRUMJOT_STAR=/codebox-workspace/datasets/star_balanced_sep \
  DRUMJOT_ENST=/codebox-workspace/datasets/enst-sep \
  DRUMJOT_EGMD=/codebox-workspace/datasets/egmd_sep \
  OMP_NUM_THREADS=4 python training/scripts/cymbal_label_audit.py \
      --splits train,val --out-json /codebox-workspace/cymbal_label_audit_canonical.json
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

SOURCES_IN_PATH = ("star", "enst", "egmd")


def _source_of(path) -> str:
    p = str(path).lower()
    for s in SOURCES_IN_PATH:
        if s in p:
            return s
    return "other"


def audit(specs, log, *, percentile, window_s, max_clips=0):
    """Per-source canonical support over every crash onset (crash stems only).

    For each crash stem: build the onset-strength envelope (`forced_align`), derive
    the per-clip support floor (`postfilter`, the `percentile`-th pct), then
    `forced_align.align_lane` flags each crash onset supported/unsupported within
    +/-`window_s`. Unsupported == what `clean`'s cleaning stage would discard."""
    from drumjot_training import forced_align, postfilter

    crash = [(a, o["cr"]) for (a, o, _full) in specs if o.get("cr")]
    if max_clips:
        crash = crash[:max_clips]
    log(f"crash stems to audit: {len(crash)}")
    agg = {}  # source -> counters
    t0 = time.perf_counter()
    for k, (audio_path, onsets) in enumerate(crash, 1):
        src = _source_of(audio_path)
        a = agg.setdefault(src, {"n": 0, "unsup": 0, "floors": []})
        try:
            env, fps = forced_align.onset_envelope(audio_path)
        except Exception as e:  # noqa: BLE001
            log(f"  skip {Path(audio_path).name}: {e!r}")
            continue
        floor = postfilter.support_floor_from_env(env, percentile)
        a["floors"].append(float(floor))
        for _t, ok in forced_align.align_lane(onsets, env, fps, window_s, floor):
            a["n"] += 1
            if not ok:
                a["unsup"] += 1
        if k % 100 == 0:
            log(f"  audited {k}/{len(crash)} stems ({time.perf_counter() - t0:.0f}s)")
    return agg


def _report(agg, log):
    out = {}
    log("\n==== crash label audit -- CANONICAL support gate (forced_align/clean) ====")
    log(f"  {'source':7s} {'onsets':>7s} {'unsupp':>7s} {'%unsupp':>8s} {'medFloor':>9s}")
    tot = {"n": 0, "unsup": 0}
    for src in sorted(agg):
        a = agg[src]
        n = a["n"]
        if n == 0:
            continue
        frac = a["unsup"] / n
        medf = float(np.median(a["floors"])) if a["floors"] else 0.0
        log(f"  {src:7s} {n:7d} {a['unsup']:7d} {frac:8.1%} {medf:9.4f}")
        out[src] = {"onsets": n, "unsupported": a["unsup"], "pct_unsupported": frac,
                    "median_floor": medf}
        tot["n"] += n
        tot["unsup"] += a["unsup"]
    if tot["n"]:
        log(f"  {'ALL':7s} {tot['n']:7d} {tot['unsup']:7d} {tot['unsup'] / tot['n']:8.1%}")
        out["ALL"] = {"onsets": tot["n"], "unsupported": tot["unsup"],
                      "pct_unsupported": tot["unsup"] / tot["n"]}
    log("\n  %unsupp = crash targets the existing cleaner would DISCARD (no transient")
    log("  in +/-window). It's a LOWER bound: the percentile floor was tuned for the")
    log("  full drum stem and runs lenient on a sparse crash stem (see medFloor).")
    return out


def _selftest():
    from drumjot_training import forced_align, postfilter

    fps = 44100 / 64  # onset_envelope's default frame rate
    n = 2000
    env = np.full(n, 0.1)  # flat noise floor
    env[1000] = 5.0  # one clear transient
    # explicit floor isolates the support logic from the percentile leniency
    res = forced_align.align_lane([1000 / fps, 500 / fps], env, fps, 0.03, support_floor=1.0)
    supported = [ok for _t, ok in res]
    assert supported == [True, False], supported  # peak supported, flat-region unsupported
    # percentile floor on a flat env is ~the floor value
    f = postfilter.support_floor_from_env(env, 60.0)
    assert abs(f - 0.1) < 1e-6, f
    print("SELFTEST OK (align_lane support + percentile floor)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Crash label audit via the canonical support gate")
    ap.add_argument("--pool-sources", default="star,enst,egmd")
    ap.add_argument("--pool-cap", type=int, default=3000, help="match the training pool")
    ap.add_argument("--splits", default="train,val", help="train, val, or train,val")
    ap.add_argument("--cache", default="/codebox-workspace/mert_cache",
                    help="only for the onset memo (_onsets.json); no features read")
    ap.add_argument("--support-percentile", type=float, default=60.0, help="eval_paradb default")
    ap.add_argument("--window-s", type=float, default=0.03, help="eval_paradb align window (s)")
    ap.add_argument("--max-clips", type=int, default=0, help="cap crash stems (0 = all); for a dry run")
    ap.add_argument("--out-json", default="/codebox-workspace/cymbal_label_audit_canonical.json")
    ap.add_argument("--selftest", action="store_true", help="validate the gate; no audio/GPU")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    from head_capacity_sweep import build_specs

    from drumjot_training import runtime
    runtime.tee_stdio(Path(args.out_json).with_suffix(".log"))
    log = lambda s: print(s, flush=True)  # noqa: E731
    sources = [s.strip() for s in args.pool_sources.split(",") if s.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    cache = Path(args.cache)
    log(f"=== crash label audit (CANONICAL): sources={sources} splits={splits} "
        f"pct={args.support_percentile} window={args.window_s}s ===")

    tr_specs, va_specs = build_specs(sources, args.pool_cap, cache)
    specs = []
    if "train" in splits:
        specs += tr_specs
    if "val" in splits:
        specs += va_specs
    agg = audit(specs, log, percentile=args.support_percentile, window_s=args.window_s,
                max_clips=args.max_clips)
    out = _report(agg, log)
    Path(args.out_json).write_text(json.dumps({"config": vars(args), "audit": out}, indent=2))
    log(f"\nresults -> {args.out_json}")


if __name__ == "__main__":
    main()
