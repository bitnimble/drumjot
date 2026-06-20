"""Training corpus for the param predictor: {features -> oracle params} rows.

A `ParamSample` is one (song, augmentation, lane) row: the label-free feature
vector, the per-song oracle param vector (the regression target), a `swept` mask
marking which params the oracle actually optimized for that lane (clean lanes
hold decay-reset at the seed, so it isn't a target), and bookkeeping. `Table`
stacks rows into parallel arrays, persists as npz, and hands the regressor the
per-lane training matrices.

`build_rows_for_song` is the pure bridge (probs + GT + audio -> rows), reused by
the GPU corpus builder (scripts/build_param_dataset.py). Pure numpy; the frozen
model runs in that script, not here.
"""
from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drumjot_training import metrics
from drumjot_training.parampred import features, oracle


@dataclass(frozen=True)
class ParamSample:
    """One (song, aug, lane) training row."""

    song: str
    aug: str
    lane: str
    features: np.ndarray   # (n_features,) in features.FEATURE_NAMES order
    params: np.ndarray     # (n_params,) oracle params in oracle.PARAM_NAMES order
    swept: np.ndarray      # (n_params,) bool: oracle actually optimized this param
    oracle_f1: float
    baseline_f1: float


def build_rows_for_song(
    probs: np.ndarray,
    fps: float,
    lanes: Sequence[str],
    thresholds: Mapping[str, float],
    gt: Mapping[str, Sequence[float]],
    waveform: np.ndarray,
    sr: int,
    *,
    song_id: str,
    aug: str,
    default_threshold: float = 0.5,
    tolerance: float = 0.05,
    beat_period_s: float | None = None,
    restrict_lanes: Collection[str] | None = None,
) -> list[ParamSample]:
    """Feature + oracle-label rows for one song's stem(s). One row per lane that
    has ground truth (and is in `restrict_lanes`)."""
    rows: list[ParamSample] = []
    for i, lane in enumerate(lanes):
        if restrict_lanes is not None and lane not in restrict_lanes:
            continue
        ref = gt.get(lane)
        if not ref:
            continue
        seed = {
            "threshold": float(thresholds.get(lane, default_threshold)),
            **metrics.LANE_PEAK_PARAMS.get(lane, metrics.DEFAULT_PEAK_PARAMS),
        }
        grids = oracle.default_grids(seed)
        ores = oracle.oracle_lane_params(probs[i], fps, ref, seed=seed, grids=grids, tolerance=tolerance)
        feat = features.feature_vector(
            probs[i], fps, seed["min_distance_s"], waveform, sr, beat_period_s=beat_period_s
        )
        rows.append(ParamSample(
            song=song_id, aug=aug, lane=lane, features=feat,
            params=np.array([ores.params[p] for p in oracle.PARAM_NAMES], dtype=np.float64),
            swept=np.array([p in grids for p in oracle.PARAM_NAMES], dtype=bool),
            oracle_f1=ores.f1, baseline_f1=ores.baseline_f1,
        ))
    return rows


class Table:
    """Stacked `ParamSample` rows as parallel arrays, with npz persistence."""

    def __init__(
        self, lane, song, aug, X, Y, swept, oracle_f1, baseline_f1,
        feature_names=features.FEATURE_NAMES, param_names=oracle.PARAM_NAMES,
    ):
        self.lane = np.asarray(lane)
        self.song = np.asarray(song)
        self.aug = np.asarray(aug)
        self.X = np.asarray(X, dtype=np.float64)
        self.Y = np.asarray(Y, dtype=np.float64)
        self.swept = np.asarray(swept, dtype=bool)
        self.oracle_f1 = np.asarray(oracle_f1, dtype=np.float64)
        self.baseline_f1 = np.asarray(baseline_f1, dtype=np.float64)
        self.feature_names = tuple(feature_names)
        self.param_names = tuple(param_names)

    @staticmethod
    def from_rows(rows: Sequence[ParamSample]) -> Table:
        if not rows:
            raise ValueError("no rows")
        return Table(
            lane=[r.lane for r in rows],
            song=[r.song for r in rows],
            aug=[r.aug for r in rows],
            X=np.stack([r.features for r in rows]),
            Y=np.stack([r.params for r in rows]),
            swept=np.stack([r.swept for r in rows]),
            oracle_f1=[r.oracle_f1 for r in rows],
            baseline_f1=[r.baseline_f1 for r in rows],
        )

    def __len__(self) -> int:
        return len(self.lane)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            Path(path), lane=self.lane, song=self.song, aug=self.aug, X=self.X, Y=self.Y,
            swept=self.swept, oracle_f1=self.oracle_f1, baseline_f1=self.baseline_f1,
            feature_names=np.asarray(self.feature_names), param_names=np.asarray(self.param_names),
        )

    @staticmethod
    def load(path: str | Path) -> Table:
        d = np.load(Path(path), allow_pickle=False)
        return Table(
            lane=d["lane"], song=d["song"], aug=d["aug"], X=d["X"], Y=d["Y"], swept=d["swept"],
            oracle_f1=d["oracle_f1"], baseline_f1=d["baseline_f1"],
            feature_names=tuple(str(s) for s in d["feature_names"]),
            param_names=tuple(str(s) for s in d["param_names"]),
        )

    def lanes(self) -> list[str]:
        return sorted(set(self.lane.tolist()))

    def training_matrices(self, lane: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """`(X, targets)` for one lane: feature matrix and per-param target
        vectors, restricted to the params swept for *every* row of that lane (a
        param the oracle never optimized for this lane is not a learnable target)."""
        mask = self.lane == lane
        X = self.X[mask]
        swept = self.swept[mask]
        targets: dict[str, np.ndarray] = {}
        for j, param in enumerate(self.param_names):
            if swept.shape[0] and bool(swept[:, j].all()):
                targets[param] = self.Y[mask][:, j]
        return X, targets
