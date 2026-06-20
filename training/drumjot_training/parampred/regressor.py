"""Per-lane feature -> peakpick-param regressor.

One `HistGradientBoostingRegressor` per (lane, param): the params are nearly
independent and the per-lane data is small, so gradient-boosted trees train in
seconds, give interpretable feature importances, and need no new dependency
(scikit-learn already rides in transitively via librosa). A param not trained
for a lane (e.g. decay-reset on a clean lane) falls back to the lane's seed
default. Predictions are clamped to each param's valid range so the picker never
sees an out-of-band value. Artifact persists via joblib (design spec §regressor).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

#: Valid range per param; predictions are clipped to this band.
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "threshold": (0.05, 0.90),
    "prominence": (0.0, 0.40),
    "min_distance_s": (0.005, 0.12),
    "decay_reset_frac": (0.0, 0.85),
    "decay_reset_floor": (0.0, 0.12),
}


def _clip_param(name: str, value: float) -> float:
    lo, hi = PARAM_BOUNDS[name]
    return float(np.clip(value, lo, hi))


class ParamRegressor:
    """Trainable map from a feature row to per-lane peakpick params.

    `fit_lane` trains the supplied params for one lane (others are left to the
    seed default passed in); `predict_row` returns the full param dict for a
    feature vector. Feature order is fixed by `feature_names` and must match the
    rows passed at fit and predict time.
    """

    def __init__(self, feature_names: Sequence[str]):
        self.feature_names: tuple[str, ...] = tuple(feature_names)
        self._models: dict[tuple[str, str], Any] = {}
        self._defaults: dict[str, dict[str, float]] = {}

    def fit_lane(
        self,
        lane: str,
        X: np.ndarray,
        targets: Mapping[str, np.ndarray],
        default_params: Mapping[str, float],
    ) -> None:
        """Train regressors for `targets` (param -> per-sample target) on lane
        `lane`. `default_params` is the seed fallback for untrained params."""
        from sklearn.ensemble import HistGradientBoostingRegressor

        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"X has {X.shape[1]} cols, expected {len(self.feature_names)}")
        self._defaults[lane] = {k: float(v) for k, v in default_params.items()}
        for param, y in targets.items():
            if param not in PARAM_BOUNDS:
                raise KeyError(f"unknown param {param!r}")
            model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_depth=3)
            model.fit(X, np.asarray(y, dtype=np.float64))
            self._models[(lane, param)] = model

    def trained_params(self, lane: str) -> tuple[str, ...]:
        return tuple(p for (ln, p) in self._models if ln == lane)

    def predict_row(self, lane: str, x: np.ndarray) -> dict[str, float]:
        """Full param dict for one feature row: trained params from their models
        (clamped to bounds), untrained params from the lane's seed default."""
        if lane not in self._defaults:
            raise KeyError(f"lane {lane!r} was never fit")
        out = dict(self._defaults[lane])
        row = np.asarray(x, dtype=np.float64).reshape(1, -1)
        for param in PARAM_BOUNDS:
            model = self._models.get((lane, param))
            if model is not None:
                out[param] = _clip_param(param, float(model.predict(row)[0]))
        return out

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump(self, Path(path))

    @staticmethod
    def load(path: str | Path) -> ParamRegressor:
        import joblib

        obj = joblib.load(Path(path))
        if not isinstance(obj, ParamRegressor):
            raise TypeError(f"{path} is not a ParamRegressor")
        return obj
