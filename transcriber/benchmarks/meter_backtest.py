"""Meter-detection regression backtest for the beat tracker.

Unlike `run_benchmark.py` (onset F1 against the running transcriber service),
this is a standalone, offline check of *beats-per-bar* accuracy. It runs Beat
This! directly on E-GMD audio and compares the modal detected bar length to
E-GMD's ground-truth `time_signature`, with the meter-recovery pass
(`beats._recover_bar_length_if_incoherent`) OFF vs ON.

Its job is to guard the recovery pass: confirm it doesn't regress the common
meters (4/4, 3/4) while it rescues the odd meters (5/4, 7/4, 7/8) that Beat
This!'s DBN-free downbeat head cannot group. Beat This! runs once per song; OLD
vs NEW then diverge only at the downbeat-grouping stage (`_raw_to_structure`),
since alignment/tempo passes don't change the bar grouping.

Run (from the repo root, transcriber venv):

    PYTHONPATH=transcriber transcriber/.venv/bin/python3 -m benchmarks.meter_backtest \
        --root /codebox-workspace/datasets/e-gmd-v1.0.0 --workers 4
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

CSV_NAME = "e-gmd-v1.0.0.csv"
DEFAULT_ROOT = os.environ.get("DRUMJOT_EGMD_RAW", "/codebox-workspace/datasets/e-gmd-v1.0.0")

# How many songs to sample per E-GMD meter (deterministic: sorted, first N).
DEFAULT_SAMPLE = {"4-4": 60, "3-4": 60, "5-4": 43, "5-8": 43, "6-8": 40}


def expected_beats_per_bar(time_signature: str) -> set[int]:
    """Acceptable modal beats-per-bar for an E-GMD `time_signature`.

    A compound eighth meter (6/8, 9/8, 12/8) is also correct read as its
    dotted-quarter count (6/8 as 2), which is how Beat This! tracks it.
    """
    num, den = (int(x) for x in time_signature.split("-"))
    if den == 8 and num % 3 == 0:
        return {num, num // 3}
    return {num}


def pick_songs(root: Path, sample: dict[str, int], min_duration: float
               ) -> list[tuple[str, str]]:
    by_meter: dict[str, list[str]] = defaultdict(list)
    with (root / CSV_NAME).open(newline="") as fh:
        for row in csv.DictReader(fh):
            ts = row["time_signature"]
            if ts in sample and float(row["duration"]) >= min_duration:
                by_meter[ts].append(row["audio_filename"])
    songs: list[tuple[str, str]] = []
    for ts, n in sample.items():
        for rel in sorted(by_meter[ts])[:n]:
            songs.append((ts, str(root / rel)))
    return songs


def _init_worker() -> None:
    import torch
    torch.set_num_threads(3)


def _score_song(item: tuple[str, str]) -> tuple[str, int | None, int | None, bool]:
    ts, path = item
    import numpy as np

    from app.pipeline.beats import (
        _beat_this_model,
        _beats_downbeats_to_raw,
        _raw_to_structure,
        _recover_bar_length_if_incoherent,
    )

    try:
        beats, downbeats = _beat_this_model()(path)
        old = _raw_to_structure(_beats_downbeats_to_raw(beats, downbeats))
        recovered = _recover_bar_length_if_incoherent(beats, downbeats, Path(path))
        new = _raw_to_structure(_beats_downbeats_to_raw(beats, recovered))
        fired = len(recovered) != len(downbeats) or not np.array_equal(
            np.asarray(sorted(float(x) for x in recovered)),
            np.asarray(sorted(float(x) for x in downbeats)),
        )
    except Exception:
        return (ts, None, None, False)
    return (ts, old.initial_time_signature[0], new.initial_time_signature[0], fired)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="E-GMD dataset root (holds the CSV).")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-duration", type=float, default=25.0,
                    help="Skip clips shorter than this many seconds.")
    args = ap.parse_args()

    songs = pick_songs(Path(args.root), DEFAULT_SAMPLE, args.min_duration)
    print(f"{len(songs)} songs from {args.root}", flush=True)

    results: list[tuple[str, int | None, int | None, bool]] = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        results.extend(ex.map(_score_song, songs))

    agg: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "old_ok": 0, "new_ok": 0, "fired": 0,
                 "old": Counter(), "new": Counter()}
    )
    for ts, old_num, new_num, fired in results:
        if old_num is None:
            continue
        exp = expected_beats_per_bar(ts)
        a = agg[ts]
        a["n"] += 1
        a["old_ok"] += old_num in exp
        a["new_ok"] += new_num in exp
        a["fired"] += fired
        a["old"][old_num] += 1
        a["new"][new_num] += 1

    print(f"\n{'meter':>6} {'n':>4} {'OLD':>6} {'NEW':>6} {'fired':>6}   distributions")
    for ts in DEFAULT_SAMPLE:
        if ts not in agg:
            continue
        a = agg[ts]
        n = a["n"]
        print(f"{ts:>6} {n:>4} {a['old_ok']/n:>6.2f} {a['new_ok']/n:>6.2f} "
              f"{a['fired']:>6}   old={dict(a['old'])} new={dict(a['new'])}")


if __name__ == "__main__":
    main()
