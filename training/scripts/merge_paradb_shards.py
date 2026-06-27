"""Combine the --dump pickles from sharded eval_paradb runs into one report.

Each shard wrote (agg, leak, gap_records, flagged) for its maps[I::N] subset; this
concatenates the per-song F1 lists, sums the leakage counters, pools the gap
records, and prints the SAME report a single eval_paradb run prints.

  python3 merge_paradb_shards.py shard_*.pkl [--predictor] [--no-oracle]
"""
import argparse
import os
import pickle
import sys
from collections import Counter, defaultdict

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)  # for `import eval_paradb` (print_reports)
sys.path.insert(0, os.path.join(_here, ".."))  # training/
sys.path.insert(0, os.path.join(_here, "..", "..", "transcriber"))

from eval_paradb import print_reports  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Merge sharded eval_paradb --dump pickles")
    ap.add_argument("dumps", nargs="+", help="the shard .pkl files")
    ap.add_argument("--predictor", action="store_true",
                    help="the shards used --param-predictor (enables the predicted/hybrid columns)")
    ap.add_argument("--no-oracle", action="store_true",
                    help="the shards ran without --oracle-report (skip the oracle/hybrid block)")
    ap.add_argument("--expect", type=int, default=0,
                    help="expected shard-dump count; abort if fewer (a crashed worker writes no "
                    "dump, so a silent partial merge would under-report songs)")
    args = ap.parse_args()
    if args.expect and len(args.dumps) != args.expect:
        raise SystemExit(f"expected {args.expect} shard dumps, got {len(args.dumps)} -- a worker "
                         "likely crashed (no dump); aborting rather than merge a partial result")

    agg: dict = defaultdict(lambda: defaultdict(list))
    leak: dict = defaultdict(lambda: {"matched": 0, "leaked": 0, "to": Counter()})
    gap_records: list = []
    flagged: list = []
    for path in args.dumps:
        with open(path, "rb") as fh:
            a, lk, gr, fl = pickle.loads(fh.read())
        for label, metrics in a.items():
            for m, vals in metrics.items():
                agg[label][m].extend(vals)
        for pitch, d in lk.items():
            leak[pitch]["matched"] += d["matched"]
            leak[pitch]["leaked"] += d["leaked"]
            leak[pitch]["to"].update(d["to"])
        gap_records.extend(gr)
        flagged.extend(fl)

    print(f"merged {len(args.dumps)} shards; gap_records={len(gap_records)}", flush=True)
    print_reports(agg, leak, gap_records, flagged,
                  oracle_report=not args.no_oracle, predictor=args.predictor)


if __name__ == "__main__":
    main()
