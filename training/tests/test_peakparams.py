from drumjot_training import metrics
from drumjot_training.lanes import LANES


def test_lane_params_cover_every_lane():
    assert set(metrics.LANE_PEAK_PARAMS) == set(LANES)
    # cymbals get the widest spacing + decay-reset; clean drums the tightest
    assert metrics.LANE_PEAK_PARAMS["rd"]["min_distance_s"] > metrics.LANE_PEAK_PARAMS["k"]["min_distance_s"]
    assert metrics.LANE_PEAK_PARAMS["cr"]["decay_reset_frac"] > 0.0
    assert metrics.LANE_PEAK_PARAMS["k"]["decay_reset_frac"] == 0.0
