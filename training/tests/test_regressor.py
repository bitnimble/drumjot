import numpy as np

from drumjot_training.parampred import regressor


def _linear_data(n, slope, intercept, seed):
    rng = np.random.default_rng(seed)
    x0 = rng.random(n)
    X = np.column_stack([x0, rng.random(n), rng.random(n)])
    y = np.clip(intercept + slope * x0 + 0.01 * rng.standard_normal(n), 0.05, 0.9)
    return X, y, x0


def test_predicts_a_learned_threshold_relationship():
    X, y, _ = _linear_data(300, slope=0.7, intercept=0.1, seed=0)
    reg = regressor.ParamRegressor(feature_names=("f0", "f1", "f2"))
    default = {"threshold": 0.5, "prominence": 0.1, "min_distance_s": 0.05,
               "decay_reset_frac": 0.0, "decay_reset_floor": 0.0}
    reg.fit_lane("k", X, {"threshold": y}, default)
    lo = reg.predict_row("k", np.array([0.0, 0.5, 0.5]))["threshold"]
    hi = reg.predict_row("k", np.array([1.0, 0.5, 0.5]))["threshold"]
    assert lo < hi                  # learned the monotone relationship
    assert abs(lo - 0.1) < 0.15     # x0=0 -> ~intercept
    assert abs(hi - 0.8) < 0.15     # x0=1 -> ~intercept+slope


def test_untrained_param_falls_back_to_default():
    X, y, _ = _linear_data(120, 0.5, 0.2, seed=1)
    reg = regressor.ParamRegressor(feature_names=("f0", "f1", "f2"))
    default = {"threshold": 0.5, "prominence": 0.13, "min_distance_s": 0.05,
               "decay_reset_frac": 0.6, "decay_reset_floor": 0.05}
    reg.fit_lane("hc", X, {"threshold": y}, default)        # only threshold trained
    out = reg.predict_row("hc", X[0])
    assert out["prominence"] == 0.13
    assert out["decay_reset_frac"] == 0.6                   # untrained -> seed default


def test_predictions_clamp_to_param_bounds():
    # train threshold targets pinned at the upper bound; predictions must not exceed it
    X = np.random.default_rng(2).random((150, 3))
    y = np.full(150, regressor.PARAM_BOUNDS["threshold"][1])
    reg = regressor.ParamRegressor(feature_names=("f0", "f1", "f2"))
    default = {"threshold": 0.5, "prominence": 0.1, "min_distance_s": 0.05,
               "decay_reset_frac": 0.0, "decay_reset_floor": 0.0}
    reg.fit_lane("k", X, {"threshold": y}, default)
    pred = reg.predict_row("k", X[0])["threshold"]
    lo, hi = regressor.PARAM_BOUNDS["threshold"]
    assert lo <= pred <= hi


def test_save_load_round_trips(tmp_path):
    X, y, _ = _linear_data(150, 0.6, 0.15, seed=3)
    reg = regressor.ParamRegressor(feature_names=("f0", "f1", "f2"))
    default = {"threshold": 0.5, "prominence": 0.1, "min_distance_s": 0.05,
               "decay_reset_frac": 0.0, "decay_reset_floor": 0.0}
    reg.fit_lane("k", X, {"threshold": y}, default)
    path = tmp_path / "pred.joblib"
    reg.save(path)
    loaded = regressor.ParamRegressor.load(path)
    a = reg.predict_row("k", X[5])["threshold"]
    b = loaded.predict_row("k", X[5])["threshold"]
    assert a == b
    assert loaded.feature_names == ("f0", "f1", "f2")
