"""Extract a CLASS-BALANCED STAR subset from the parts, biased toward the
rare lanes (side-stick, ride, crash, misc-cymbal, pedal-hat, toms, misc-perc)
that a head-of-distribution sort misses.

Scans every pairable annotation (reads the tiny .txt straight from the zip, no
extraction), counts per-lane onsets, then greedily selects clips by marginal
coverage gain `sum_lane count/(1+covered)` so under-covered (rare) lanes pull
their clips in first. Train is drawn from the training split, val from
validation+test (song-disjoint, so rare-lane F1 is actually measurable). Only
the selected clips' mix flac + annotation are extracted, preserving the
data/<split>/.../{annotation,audio/mix} layout for star.index.

Usage: extract_star_balanced.py <parts_dir> <out_dir> [n_train] [n_val]
"""
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from extract_star_subset import MultiPartFile  # noqa: E402

from drumjot_training.lanes import LANES  # noqa: E402
from drumjot_training.star import lane_for_star_class  # noqa: E402


def _split_of(name):
    low = name.lower().split("/")
    for s in ("training", "validation", "test"):
        if s in low:
            return s
    return "?"


def _mix_for(ann):
    i = ann.rindex("/annotation/")
    stem = ann[ann.rindex("/") + 1 : -4]
    return f"{ann[:i]}/audio/mix/{stem}.flac"


def _counts(text):
    c = {ln: 0 for ln in LANES}
    for line in text.splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        lane = lane_for_star_class(parts[1].strip())
        if lane is not None:
            c[lane] += 1
    return c


def greedy(pool, n):
    """Greedily pick n clips maximizing marginal rare-lane coverage."""
    covered = dict.fromkeys(LANES, 0)
    chosen, remaining = [], list(pool)
    while remaining and len(chosen) < n:
        best_i, best = -1, -1.0
        for i, (_name, c) in enumerate(remaining):
            score = sum(c[ln] / (1.0 + covered[ln]) for ln in LANES)
            if score > best:
                best, best_i = score, i
        name, c = remaining.pop(best_i)
        chosen.append((name, c))
        for ln in LANES:
            covered[ln] += c[ln]
    return chosen, covered


def _report(tag, chosen):
    print(f"\n{tag}: {len(chosen)} clips", flush=True)
    for ln in LANES:
        onsets = sum(c[ln] for _, c in chosen)
        nclips = sum(1 for _, c in chosen if c[ln] > 0)
        print(f"  {ln:3s} onsets={onsets:7d}  clips={nclips:4d}", flush=True)


def main():
    parts_dir, out_dir = sys.argv[1], sys.argv[2]
    n_train = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    n_val = int(sys.argv[4]) if len(sys.argv) > 4 else 120

    parts = sorted(
        os.path.join(parts_dir, f) for f in os.listdir(parts_dir) if ".zip.part-" in f
    )
    print("parts:", [os.path.basename(p) for p in parts], flush=True)
    z = zipfile.ZipFile(MultiPartFile(parts))
    names = z.namelist()
    mix_set = {n for n in names if "/audio/mix/" in n and n.endswith(".flac")}
    anns = [n for n in names if "/annotation/" in n and n.endswith(".txt")]
    print(f"annotations: {len(anns)}  mix flac: {len(mix_set)}", flush=True)

    train_pool, val_pool = [], []
    for i, ann in enumerate(anns):
        if _mix_for(ann) not in mix_set:
            continue
        c = _counts(z.read(ann).decode("utf-8", "replace"))
        sp = _split_of(ann)
        if sp == "training":
            train_pool.append((ann, c))
        elif sp in ("validation", "test"):
            val_pool.append((ann, c))
        if (i + 1) % 1000 == 0:
            print(f"  scanned {i + 1}/{len(anns)}", flush=True)
    print(f"pairable: {len(train_pool)} train-split, {len(val_pool)} val/test-split", flush=True)

    val_sel, _ = greedy(val_pool, n_val)
    train_sel, _ = greedy(train_pool, n_train)
    _report("TRAIN (selected)", train_sel)
    _report("VAL (selected)", val_sel)

    members = set()
    for ann, _ in train_sel + val_sel:
        members.add(ann)
        members.add(_mix_for(ann))
    members.update(n for n in names if n.endswith(".csv"))
    print(f"\nextracting {len(members)} members -> {out_dir}", flush=True)
    for i, m in enumerate(sorted(members)):
        z.extract(m, out_dir)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(members)}", flush=True)
    print("DONE. dataset root:", out_dir, flush=True)


if __name__ == "__main__":
    main()
