"""Unattended overnight batch runner for cym+hat loss/seed/scale sweeps.

Runs a value-ordered queue of `cymbal_loss_ab.py` configs SEQUENTIALLY, one per
GPU box, each in its OWN checkpoint dir (no collision, independently resumable).
Fail-independent: a crashed/timed-out run is logged and the queue continues. Two
boxes share /codebox-workspace (cache + aligned onsets), so namespace with --tag.

Each --run is `arm:cap:seed:epochs:hidden` (arm in baseline,bce,focal,
crash_oversample,mixed). Example (1660 sandbox -- seed error bars at cap-1000):

  DRUMJOT_STAR=/codebox-workspace/datasets/star_balanced_sep \
  DRUMJOT_ENST=/codebox-workspace/datasets/enst-sep \
  DRUMJOT_EGMD=/codebox-workspace/datasets/egmd_sep OMP_NUM_THREADS=8 \
  python training/scripts/overnight_runner.py --tag ovn1660 \
    --run mixed:1000:2:30:128 --run mixed:1000:3:30:128 \
    --run bce:1000:2:30:128 --run focal:1000:2:30:128

3080 (best recipe at h256 + scale): same, --tag ovn3080 and h256/cap-3000/cap-0
runs. Re-running the same command RESUMES (per-config resume checkpoints).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
LANES_ORDER = ("hc", "hp", "ho", "rd", "cr", "k", "s", "ss", "t")


def _timeout_for(cap: int, hidden: int) -> int:
    base = {1000: 7200, 3000: 21600, 0: 50400}.get(cap, 21600)  # 2h / 6h / 14h
    return base * (2 if hidden >= 256 else 1)


def main():
    ap = argparse.ArgumentParser(description="Overnight cym+hat batch runner")
    ap.add_argument("--tag", required=True, help="box namespace (e.g. ovn1660, ovn3080)")
    ap.add_argument("--run", action="append", default=[], dest="runs",
                    help="arm:cap:seed:epochs:hidden (repeatable, run in order)")
    ap.add_argument("--ckpt-root", default="/codebox-workspace/checkpoints")
    ap.add_argument("--out-root", default="/codebox-workspace/ovn")
    ap.add_argument("--cache", default="/codebox-workspace/datasets/_cache_mert_pooled")
    ap.add_argument("--aligned-onsets", default="/codebox-workspace/datasets/_onsets_aligned.json")
    ap.add_argument("--focal-lanes", default="hc,rd")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    out_root = Path(args.out_root) / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    master = out_root / "_runner.log"
    t_start = time.perf_counter()

    def log(s):
        line = f"[{(time.perf_counter() - t_start) / 60:7.1f}m] {s}"
        print(line, flush=True)
        with master.open("a") as f:
            f.write(line + "\n")

    configs = []
    for spec in args.runs:
        arm, cap, seed, epochs, hidden = spec.split(":")
        configs.append({"arm": arm, "cap": int(cap), "seed": int(seed),
                        "epochs": int(epochs), "hidden": int(hidden), "spec": spec})

    log(f"=== overnight runner tag={args.tag}: {len(configs)} configs ===")
    for c in configs:
        log(f"  queued: {c['spec']}")

    done = []
    for i, c in enumerate(configs, 1):
        name = f"{c['arm']}_c{c['cap']}_h{c['hidden']}_s{c['seed']}"
        ckpt_dir = Path(args.ckpt_root) / args.tag / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_root / f"{name}.json"
        if out_json.exists():
            log(f"[{i}/{len(configs)}] SKIP {name} (out-json exists)")
            done.append((c, out_json))
            continue
        to = _timeout_for(c["cap"], c["hidden"])
        log(f"[{i}/{len(configs)}] START {name} (epochs={c['epochs']} timeout={to / 3600:.1f}h)")
        cmd = [sys.executable, os.path.join(_HERE, "cymbal_loss_ab.py"),
               "--arms", c["arm"], "--pool-cap", str(c["cap"]), "--hidden", str(c["hidden"]),
               "--layers", "2", "--epochs", str(c["epochs"]), "--seed", str(c["seed"]),
               "--focal-lanes", args.focal_lanes, "--batch", str(args.batch),
               "--num-workers", str(args.num_workers), "--cache", args.cache,
               "--aligned-onsets", args.aligned_onsets, "--ckpt-dir", str(ckpt_dir),
               "--out-json", str(out_json)]
        t0 = time.perf_counter()
        try:
            subprocess.run(cmd, timeout=to, check=False)
            mins = (time.perf_counter() - t0) / 60
            if out_json.exists():
                log(f"[{i}/{len(configs)}] DONE  {name} ({mins:.0f}m)")
                done.append((c, out_json))
            else:
                log(f"[{i}/{len(configs)}] NO-OUTPUT {name} ({mins:.0f}m) -- check {ckpt_dir}")
        except subprocess.TimeoutExpired:
            log(f"[{i}/{len(configs)}] TIMEOUT {name} after {to / 3600:.1f}h -- skipping")
        except Exception as e:  # noqa: BLE001  -- fail-independent
            log(f"[{i}/{len(configs)}] FAILED {name}: {e!r}")

    # aggregate per-lane tuned F1 across all completed runs
    log("\n==== per-lane tuned F1 (all completed runs) ====")
    rows = {}
    for c, oj in done:
        try:
            data = json.loads(Path(oj).read_text())
            lf = data["arms"][c["arm"]].get("lane_f1", {})
        except Exception:  # noqa: BLE001
            continue
        key = f"{c['arm']}_c{c['cap']}_h{c['hidden']}"
        rows.setdefault(key, []).append(lf)
    present = [ln for ln in LANES_ORDER if any(ln in lf for lfs in rows.values() for lf in lfs)]
    log(f"  {'config':22s} n | " + " ".join(f"{ln:>5s}" for ln in present))
    for key, lfs in sorted(rows.items()):
        cells = []
        for ln in present:
            vals = [lf[ln] for lf in lfs if ln in lf]
            cells.append(f"{sum(vals) / len(vals):5.3f}" if vals else f"{'--':>5s}")
        log(f"  {key:22s} {len(lfs)} | " + " ".join(cells))
    log(f"\nDONE: {len(done)}/{len(configs)} configs. results under {out_root}")


if __name__ == "__main__":
    main()
