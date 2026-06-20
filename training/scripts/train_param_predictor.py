"""Fit the per-song param predictor from a dataset table.

Consumes the npz `Table` built by build_param_dataset.py ({features -> oracle
params} rows) and fits one `ParamRegressor` (per-lane, per-param HistGBR) over
it, then writes a joblib artifact for `eval_paradb.py --param-predictor` and the
transcriber-side inference.

Holdout is grouped BY SONG so a song's augmented variants never straddle the
train/val split (that would leak). The reported metric is per-lane held-out
param MAE; the real test is the oracle-gap report on ParaDB (held-out real audio).

Usage:
  PYTHONPATH=dsp:training python3 training/scripts/train_param_predictor.py \
      --dataset <table.npz> --out <predictor.joblib> [--val-frac 0.2]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # training/

from drumjot_training import metrics  # noqa: E402
from drumjot_training.parampred import dataset, regressor  # noqa: E402


def _song_split(songs: np.ndarray, val_frac: float, seed: int) -> np.ndarray:
    """Boolean train mask, grouped by unique song (variants stay together)."""
    uniq = sorted(set(songs.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(val_frac * len(uniq)))) if len(uniq) > 1 else 0
    val_songs = set(uniq[:n_val])
    return np.array([s not in val_songs for s in songs.tolist()])


def _lane_defaults(lane: str) -> dict[str, float]:
    """Seed fallback for untrained params: the current global per-lane params."""
    return {"threshold": 0.5, **metrics.LANE_PEAK_PARAMS.get(lane, metrics.DEFAULT_PEAK_PARAMS)}


def main():
    ap = argparse.ArgumentParser(description="Fit the per-song param predictor")
    ap.add_argument("--dataset", required=True, help="npz Table from build_param_dataset.py")
    ap.add_argument("--out", required=True, help="output predictor .joblib")
    ap.add_argument("--val-frac", type=float, default=0.2, help="song fraction held out for MAE report")
    ap.add_argument("--min-rows", type=int, default=20, help="skip lanes with fewer train rows")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    table = dataset.Table.load(args.dataset)
    print(f"loaded {len(table)} rows, {len(table.lanes())} lanes, "
          f"{len(set(table.song.tolist()))} songs", flush=True)
    train_mask = _song_split(table.song, args.val_frac, args.seed)

    reg = regressor.ParamRegressor(feature_names=table.feature_names)
    for lane in table.lanes():
        lane_mask = (table.lane == lane) & train_mask
        n = int(lane_mask.sum())
        if n < args.min_rows:
            print(f"  {lane}: {n} train rows < {args.min_rows}, skipped", flush=True)
            continue
        sub = _subset(table, lane_mask)
        X, targets = sub.training_matrices(lane)
        reg.fit_lane(lane, X, targets, _lane_defaults(lane))
        maes = _val_mae(table, lane, ~train_mask, reg)
        mae_str = " ".join(f"{p}:{m:.3f}" for p, m in maes.items()) or "(no val rows)"
        print(f"  {lane}: trained {sorted(targets)} on {n} rows | val MAE {mae_str}", flush=True)

    reg.save(args.out)
    print(f"\nwrote predictor -> {args.out}", flush=True)


def _subset(table: dataset.Table, mask: np.ndarray) -> dataset.Table:
    return dataset.Table(
        lane=table.lane[mask], song=table.song[mask], aug=table.aug[mask],
        X=table.X[mask], Y=table.Y[mask], swept=table.swept[mask],
        oracle_f1=table.oracle_f1[mask], baseline_f1=table.baseline_f1[mask],
        feature_names=table.feature_names, param_names=table.param_names,
    )


def _val_mae(table: dataset.Table, lane: str, val_mask: np.ndarray, reg: regressor.ParamRegressor):
    """Per-param mean-absolute-error of the predictor vs the oracle on held-out rows."""
    mask = (table.lane == lane) & val_mask
    if not mask.any() or not reg.trained_params(lane):
        return {}
    X = table.X[mask]
    Y = table.Y[mask]
    pidx = {p: i for i, p in enumerate(table.param_names)}
    errs: dict[str, float] = {}
    preds = [reg.predict_row(lane, x) for x in X]
    for p in reg.trained_params(lane):
        truth = Y[:, pidx[p]]
        got = np.array([pr[p] for pr in preds])
        errs[p] = float(np.mean(np.abs(got - truth)))
    return errs


if __name__ == "__main__":
    main()
