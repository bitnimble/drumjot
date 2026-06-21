import numpy as np

import drumjot_training.targets as targets
from drumjot_training.parampred import eval_gap


def _curve(onset_times, n, fps, *, scale=1.0):
    return targets.onsets_to_target(onset_times, n_frames=n, fps=fps, sigma_frames=1.0) * scale


def _two_lane_probs(fps, n):
    # lane 0 (k): clean, oracle == current. lane 1 (s): spurious 0.35 bumps a low
    # global threshold picks -> oracle beats current by raising the threshold.
    k = _curve([0.5, 1.5, 2.5], n, fps)
    s = np.maximum(_curve([0.6, 1.6, 2.6], n, fps), _curve([1.1, 2.1, 3.1], n, fps, scale=0.35))
    return np.stack([k, s])


def test_records_show_oracle_at_or_above_current():
    fps = 100.0
    n = 400
    probs = _two_lane_probs(fps, n)
    gt = {"k": [0.5, 1.5, 2.5], "s": [0.6, 1.6, 2.6]}
    recs = eval_gap.lane_gap_records(
        probs, fps, ["k", "s"], thresholds={"k": 0.3, "s": 0.1}, gt=gt,
    )
    by_lane = {r.lane: r for r in recs}
    assert set(by_lane) == {"k", "s"}
    for r in recs:
        assert r.oracle_f1 >= r.current_f1 - 1e-9
        assert r.predicted_f1 == r.current_f1     # no predictor -> predicted == current
    assert by_lane["s"].oracle_f1 > by_lane["s"].current_f1  # the prize is real on snare


def test_records_carry_a_deterministic_point():
    fps = 100.0
    probs = _two_lane_probs(fps, 400)
    gt = {"k": [0.5, 1.5, 2.5], "s": [0.6, 1.6, 2.6]}
    recs = eval_gap.lane_gap_records(probs, fps, ["k", "s"], thresholds={"k": 0.3, "s": 0.1}, gt=gt)
    for r in recs:
        assert r.deterministic_f1 is not None
        assert 0.0 <= r.deterministic_f1 <= 1.0
    # on snare (spurious low bumps a global 0.1 threshold picks), the self-calibrated
    # knee threshold should beat the current global params
    s = next(r for r in recs if r.lane == "s")
    assert s.deterministic_f1 >= s.current_f1


def test_restrict_lanes_filters_records():
    fps = 100.0
    probs = _two_lane_probs(fps, 400)
    gt = {"k": [0.5, 1.5, 2.5], "s": [0.6, 1.6, 2.6]}
    recs = eval_gap.lane_gap_records(
        probs, fps, ["k", "s"], thresholds={"k": 0.3, "s": 0.1}, gt=gt,
        restrict_lanes={"k"},
    )
    assert {r.lane for r in recs} == {"k"}


def test_lane_without_ground_truth_is_skipped():
    fps = 100.0
    probs = _two_lane_probs(fps, 400)
    recs = eval_gap.lane_gap_records(
        probs, fps, ["k", "s"], thresholds={"k": 0.3, "s": 0.1}, gt={"k": [0.5, 1.5, 2.5]},
    )
    assert {r.lane for r in recs} == {"k"}


class _FixedPredictor:
    """Stand-in predictor: returns a high threshold that clears the spurious
    bumps, so the snare's predicted F1 should reach the oracle."""

    def predict_row(self, lane, x):
        return {"threshold": 0.6, "prominence": 0.0, "min_distance_s": 0.02,
                "decay_reset_frac": 0.0, "decay_reset_floor": 0.0}


def test_predictor_path_scores_predicted_params():
    fps = 100.0
    n = 400
    probs = _two_lane_probs(fps, n)
    gt = {"s": [0.6, 1.6, 2.6]}
    wave = np.sin(2 * np.pi * 1000 * np.arange(44100) / 44100).astype(np.float32)
    recs = eval_gap.lane_gap_records(
        probs, fps, ["k", "s"], thresholds={"k": 0.3, "s": 0.1}, gt=gt,
        predictor=_FixedPredictor(), waveform=wave, sr=44100,
    )
    s = next(r for r in recs if r.lane == "s")
    assert s.predicted_f1 == 1.0
    assert s.predicted_f1 > s.current_f1
