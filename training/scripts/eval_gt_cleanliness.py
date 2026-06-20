"""Eval-GT cleanliness probe: flag ONLY completely-dead labels (onset on silence).

If a benchmark's ground-truth has onsets where the dataset's OWN reference audio is
silent, the model correctly stays quiet there but gets scored as a miss (free FN)
-> our SOTA-comparable F is unfairly depressed. This finds those.

STRICT by design (per request): flags an onset ONLY if the peak amplitude within
+/-window (= the mir_eval tolerance) is essentially ZERO relative to the track --
i.e. true silence in the reference recording. It does NOT flag soft hits or
wrong-lane notes: a phantom cymbal during a kick still has mix energy at that time
(loud), so it isn't flagged; only labels sitting in genuine silence are.

Reference = the dataset's own recording (ENST `wet_mix`), NOT our separation -- so a
flag is a DATASET annotation error, not a separation drop.

  scripts/sandbox-run env PYTHONPATH=training:dsp python3 \
      training/scripts/eval_gt_cleanliness.py \
      --enst-public-root /codebox-workspace/datasets/ENST-drums-public \
      --out-json /codebox-workspace/eval_gt_cleanliness_enst.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "dsp"))

FOLD5 = {"KD": ("k",), "SD": ("s", "ss"), "HH": ("hc", "hp", "ho"), "TT": ("t",), "CY": ("rd", "cr")}
LANE_TO_CLASS = {ln: c for c, lns in FOLD5.items() for ln in lns}


def _enst_clips(root, split, val_drummer, exclude, mix):
    from drumjot_training import enst
    clips = enst.index(root, mix=mix)
    if split != "all":
        clips = enst.for_split(clips, split, val_drummer=val_drummer)
    return [c for c in clips if not (exclude and exclude in c.audio_path.stem)]


def main():
    ap = argparse.ArgumentParser(description="Eval-GT cleanliness probe (dead-onset detector)")
    ap.add_argument("--dataset", default="enst", choices=("enst", "mdb"))
    ap.add_argument("--enst-public-root", default="/codebox-workspace/datasets/ENST-drums-public",
                    help="ORIGINAL ENST (with audio/wet_mix) -- the dataset's own reference recording")
    ap.add_argument("--mdb-root", default="/codebox-workspace/datasets/MDBDrums",
                    help="MDBDrums clone (reference = its full_mix)")
    ap.add_argument("--split", default="test", help="ENST: test=held-out drummer; 'all'=whole dataset")
    ap.add_argument("--val-drummer", default="drummer_3")
    ap.add_argument("--exclude-takes", default="hits", help="skip ENST isolated-technique demos")
    ap.add_argument("--mix", default="wet_mix", help="ENST reference audio (wet_mix = mixed recording)")
    ap.add_argument("--window-s", type=float, default=0.05, help="+/- search (= mir_eval tolerance)")
    ap.add_argument("--rel-floor", type=float, default=0.01,
                    help="dead if peak in window < this * track peak (0.01 = ~-40 dB, 'silent')")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--out-json", default="/codebox-workspace/eval_gt_cleanliness.json")
    args = ap.parse_args()

    import librosa

    from drumjot_training import runtime
    runtime.tee_stdio(Path(args.out_json).with_suffix(".log"))
    log = lambda s: print(s, flush=True)  # noqa: E731

    # unified clip source -> (reference_audio_path, onsets_by_lane, name)
    def _iter():
        if args.dataset == "enst":
            from drumjot_training import enst
            for c in _enst_clips(args.enst_public_root, args.split, args.val_drummer,
                                 args.exclude_takes, args.mix):
                name = c.audio_path.stem if c.audio_path else c.annotation_path.stem
                yield c.audio_path, enst.onsets_by_lane(c.annotation_path), name
        else:  # mdb -- reference = full_mix
            from drumjot_training import mdb
            for c in mdb.index(args.mdb_root):
                yield c.full_mix, mdb.onsets_by_lane(c.subclass_ann), c.track

    triples = list(_iter())
    ref = args.mix if args.dataset == "enst" else "full_mix"
    ctx = f" split={args.split}" if args.dataset == "enst" else ""
    log(f"=== eval-GT cleanliness: {args.dataset}{ctx} mix={ref} "
        f"win=+/-{args.window_s*1000:.0f}ms floor={args.rel_floor:.3f}*peak  ({len(triples)} tracks) ===")

    per_lane = defaultdict(lambda: {"n": 0, "dead": 0})
    examples = []  # (ratio, name, lane, t)
    w = int(args.window_s * args.sr)
    for ci, (audio, onsets, name) in enumerate(triples, 1):
        if audio is None or not Path(audio).exists():
            log(f"  skip {name}: no reference audio")
            continue
        y, _ = librosa.load(str(audio), sr=args.sr, mono=True)
        tp = float(np.max(np.abs(y))) if y.size else 0.0
        if tp <= 0:
            continue
        for lane, times in onsets.items():
            if lane not in LANE_TO_CLASS:
                continue
            for t in times:
                i = int(t * args.sr)
                seg = y[max(0, i - w): i + w + 1]
                peak = float(np.max(np.abs(seg))) if seg.size else 0.0
                ratio = peak / tp
                per_lane[lane]["n"] += 1
                if ratio < args.rel_floor:
                    per_lane[lane]["dead"] += 1
                    examples.append((ratio, name, lane, round(float(t), 3)))
        if ci % 20 == 0:
            log(f"  scanned {ci}/{len(triples)} tracks")

    # report per-lane + folded 5-class + overall
    log(f"\n==== DEAD onsets (peak < {args.rel_floor:.3f} x track peak within +/-{args.window_s*1000:.0f}ms) ====")
    log(f"  {'lane':5s} | {'onsets':>7s} {'dead':>6s} {'dead%':>7s}")
    cls_tot = defaultdict(lambda: {"n": 0, "dead": 0})
    out_lanes = {}
    for lane in sorted(per_lane):
        d = per_lane[lane]
        frac = d["dead"] / d["n"] if d["n"] else 0.0
        log(f"  {lane:5s} | {d['n']:7d} {d['dead']:6d} {frac:7.2%}")
        out_lanes[lane] = {"onsets": d["n"], "dead": d["dead"], "dead_frac": frac}
        c = LANE_TO_CLASS[lane]
        cls_tot[c]["n"] += d["n"]
        cls_tot[c]["dead"] += d["dead"]
    log("  " + "-" * 32)
    tot = {"n": 0, "dead": 0}
    for c in ("KD", "SD", "HH", "TT", "CY"):
        d = cls_tot[c]
        if not d["n"]:
            continue
        log(f"  {c:5s} | {d['n']:7d} {d['dead']:6d} {(d['dead']/d['n']):7.2%}")
        tot["n"] += d["n"]
        tot["dead"] += d["dead"]
    log(f"  {'ALL':5s} | {tot['n']:7d} {tot['dead']:6d} {(tot['dead']/tot['n'] if tot['n'] else 0):7.2%}")

    examples.sort()
    log("\n  most-dead labels (lowest peak/track ratio) -- eyeball/ear-check these:")
    for ratio, take, lane, t in examples[:25]:
        log(f"    {ratio:6.4f}  {lane:3s} @ {t:7.3f}s  {take}")
    log(f"\n  NB reference = the dataset's own {ref}, so a dead label = a DATASET annotation error")
    log("  (not a separation drop). Strict: only true-silence labels; soft/wrong-lane NOT flagged.")
    out = {"config": vars(args), "per_lane": out_lanes,
           "by_class": {c: dict(v) for c, v in cls_tot.items()},
           "all": tot, "examples": [{"ratio": r, "take": tk, "lane": ln, "t": t}
                                    for r, tk, ln, t in examples[:200]]}
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    log(f"\nresults -> {args.out_json}")


if __name__ == "__main__":
    main()
