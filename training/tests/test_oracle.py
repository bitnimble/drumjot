import numpy as np

import drumjot_training.targets as targets
from drumjot_training.parampred import oracle


def _curve(onset_times, n_frames, fps, *, scale=1.0, sigma=1.0):
    arr = targets.onsets_to_target(onset_times, n_frames=n_frames, fps=fps, sigma_frames=sigma)
    return arr * scale


def test_oracle_raises_threshold_to_drop_spurious_bumps():
    fps = 100.0
    n = 400
    real = [0.5, 1.5, 2.5]
    # real onsets at full height; three spurious bumps at 0.35 height that a low
    # seed threshold would wrongly pick.
    curve = np.maximum(_curve(real, n, fps), _curve([1.0, 2.0, 3.0], n, fps, scale=0.35))
    seed = {"threshold": 0.10, "prominence": 0.0, "min_distance_s": 0.02,
            "decay_reset_frac": 0.0, "decay_reset_floor": 0.0}
    res = oracle.oracle_lane_params(
        curve, fps, real, seed=seed,
        grids={"threshold": oracle.THRESHOLD_GRID},
    )
    assert res.baseline_f1 < 1.0            # seed picks the spurious bumps -> false positives
    assert res.f1 == 1.0                    # a higher threshold recovers a perfect score
    assert res.params["threshold"] >= 0.35  # above the 0.35 spurious bumps, below the real peaks
    assert res.params["threshold"] < 1.0
    assert res.f1 >= res.baseline_f1


def test_oracle_never_scores_below_seed():
    fps = 100.0
    n = 300
    rng = np.random.default_rng(0)
    curve = np.clip(_curve([0.3, 0.9, 1.7, 2.4], n, fps) + 0.15 * rng.random(n), 0.0, 1.0)
    seed = {"threshold": 0.30, "prominence": 0.0, "min_distance_s": 0.02,
            "decay_reset_frac": 0.0, "decay_reset_floor": 0.0}
    res = oracle.oracle_lane_params(
        curve, fps, [0.3, 0.9, 1.7, 2.4], seed=seed,
        grids={"threshold": oracle.THRESHOLD_GRID, "prominence": oracle.PROMINENCE_GRID},
    )
    assert res.f1 >= res.baseline_f1 - 1e-9


def test_default_grids_sweep_decay_only_for_sustained_lanes():
    clean_seed = {"threshold": 0.5, "prominence": 0.1, "min_distance_s": 0.02,
                  "decay_reset_frac": 0.0, "decay_reset_floor": 0.0}
    sustained_seed = {**clean_seed, "decay_reset_frac": 0.6, "decay_reset_floor": 0.05}
    clean = oracle.default_grids(clean_seed)
    sustained = oracle.default_grids(sustained_seed)
    assert set(clean) == {"threshold", "prominence", "min_distance_s"}
    assert "decay_reset_frac" in sustained and "decay_reset_floor" in sustained


def test_flat_f1_region_breaks_tie_toward_seed():
    fps = 100.0
    n = 400
    # onsets 0.5 s apart: every min-distance in the grid scores F1 = 1.0, so the
    # oracle must return the value closest to the seed (well-conditioned labels).
    onsets = [0.5, 1.0, 1.5, 2.0]
    curve = _curve(onsets, n, fps)
    seed = {"threshold": 0.5, "prominence": 0.0, "min_distance_s": 0.05,
            "decay_reset_frac": 0.0, "decay_reset_floor": 0.0}
    res = oracle.oracle_lane_params(
        curve, fps, onsets, seed=seed,
        grids={"min_distance_s": oracle.MIN_DISTANCE_GRID},
    )
    assert res.f1 == 1.0
    assert res.params["min_distance_s"] == 0.05
