import drumjot_training.lanes as lanes


def test_lanes_are_the_expanded_set():
    assert lanes.LANES == ("k", "s", "ss", "t", "hc", "hp", "ho", "rd", "cr", "mc")


def test_kick():
    assert lanes.lane_for_gm_note(35) == "k"
    assert lanes.lane_for_gm_note(36) == "k"


def test_side_stick_is_its_own_lane():
    assert lanes.lane_for_gm_note(37) == "ss"


def test_snare():
    assert lanes.lane_for_gm_note(38) == "s"
    assert lanes.lane_for_gm_note(40) == "s"


def test_toms_merged():
    for n in (41, 43, 45, 47, 48, 50):
        assert lanes.lane_for_gm_note(n) == "t"


def test_hat_subclasses_split():
    assert lanes.lane_for_gm_note(42) == "hc"  # closed
    assert lanes.lane_for_gm_note(44) == "hp"  # pedal
    assert lanes.lane_for_gm_note(46) == "ho"  # open


def test_ride_and_crash_split():
    for n in (51, 59):
        assert lanes.lane_for_gm_note(n) == "rd"  # ride 1/2
    for n in (49, 57):
        assert lanes.lane_for_gm_note(n) == "cr"  # crash 1/2


def test_misc_cymbals_folded():
    for n in (52, 53, 55):  # china, ride bell, splash
        assert lanes.lane_for_gm_note(n) == "mc"


def test_misc_percussion_dropped():
    # mp lane removed: clap / tambourine / cowbell are out-of-kit now
    for n in (39, 54, 56):
        assert lanes.lane_for_gm_note(n) is None


def test_unknown_note_is_none():
    assert lanes.lane_for_gm_note(99) is None
    assert lanes.lane_for_gm_note(60) is None


def test_non_kit_percussion_maps_to_negative_lane():
    # clap/tambourine/cowbell + latin/aux perc -> the catch-all negative lane `x`
    for n in (39, 54, 56, 58, 60, 63, 75, 81):
        assert lanes.negative_lane_for_gm_note(n) == "x"
    # output-lane notes never shadow into the negative map
    for n in (36, 38, 42, 51, 49):
        assert lanes.negative_lane_for_gm_note(n) is None
    # truly out-of-everything notes stay None (not even a negative)
    assert lanes.negative_lane_for_gm_note(34) is None
    assert lanes.negative_lane_for_gm_note(120) is None


def test_negative_sibling_matrix_marks_every_lane():
    import numpy as np

    Sneg = np.asarray(lanes.negative_sibling_matrix())
    assert Sneg.shape == (len(lanes.LANES), len(lanes.NEGATIVE_LANES))
    assert Sneg.all()  # the catch-all `x` is a hard negative for every output lane
    assert lanes.WEIGHT_LANES == lanes.LANES + lanes.NEGATIVE_LANES
