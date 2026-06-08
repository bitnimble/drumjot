import drumjot_training.metrics as metrics
import drumjot_training.targets as targets


def test_perfect_match_is_f1_one():
    r = metrics.onset_f1([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], tolerance=0.05)
    assert r["f"] == 1.0
    assert r["p"] == 1.0
    assert r["r"] == 1.0


def test_est_within_tolerance_counts_as_hit():
    r = metrics.onset_f1([1.0], [1.03], tolerance=0.05)
    assert r["f"] == 1.0


def test_est_outside_tolerance_misses():
    r = metrics.onset_f1([1.0], [1.20], tolerance=0.05)
    assert r["r"] == 0.0


def test_unsorted_inputs_are_handled():
    r = metrics.onset_f1([3.0, 1.0, 2.0], [2.0, 1.0, 3.0], tolerance=0.05)
    assert r["f"] == 1.0


def test_pick_onsets_recovers_a_planted_onset():
    arr = targets.onsets_to_target([0.5], n_frames=200, fps=100.0, sigma_frames=1.0)
    times = metrics.pick_onsets(arr, fps=100.0, threshold=0.5, min_distance_s=0.03)
    assert len(times) == 1
    assert abs(times[0] - 0.5) < 0.02
