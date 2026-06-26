"""Pure (host-testable) tom sub-classification clustering."""
import numpy as np

import drumjot_training.tom_subclassify as ts


def _pop(*groups):
    """Build a pitch population: each (centre_st, n) is n points spread +/-0.3 st."""
    out = []
    for c, n in groups:
        out += list(np.linspace(c - 0.3, c + 0.3, n))
    return out


def test_keys_for_k():
    assert ts._keys_for_k(1) == ["t"]
    assert ts._keys_for_k(2) == ["f", "t"]
    assert ts._keys_for_k(3) == ["f", "tl", "t"]
    assert ts._keys_for_k(4) == ["f", "tl", "tm", "t"]


def test_single_tom_one_cluster():
    centers = ts.cluster_pitches(_pop((80, 30)))
    assert len(centers) == 1
    assert ts.assign_keys(_pop((80, 30))) == ["t"] * 30


def test_two_well_separated_toms_split():
    keys = ts.assign_keys(_pop((74, 20), (86, 20)))
    assert set(keys) == {"f", "t"}
    assert keys[:20] == ["f"] * 20   # low cluster -> floor
    assert keys[20:] == ["t"] * 20   # high cluster -> high tom


def test_sparse_floor_tom_recovered():
    # only 4 floor-tom onsets, well below a dense rack-tom cluster -> still split
    centers = ts.cluster_pitches(_pop((74, 4), (86, 30)))
    assert len(centers) == 2
    keys = ts.assign_keys(_pop((74, 4), (86, 30)))
    assert keys[:4] == ["f"] * 4


def test_closely_tuned_not_split():
    # peaks < MIN_SEP (2.0 st) apart must NOT split (avoid phantom toms)
    assert len(ts.cluster_pitches(_pop((80, 15), (81.5, 15)))) == 1
    assert set(ts.assign_keys(_pop((80, 15), (81.5, 15)))) == {"t"}


def test_degenerate_single_pitch_not_split():
    assert len(ts.cluster_pitches(_pop((80, 40)))) == 1


def test_too_few_onsets_fall_back_to_merged():
    assert ts.assign_keys(_pop((74, 3), (86, 3))) == ["t"] * 6  # 6 < MIN_FIT


def test_three_tom_kit():
    centers = ts.cluster_pitches(_pop((70, 15), (80, 15), (90, 15)))
    assert len(centers) == 3
    keys = ts.assign_keys(_pop((70, 15), (80, 15), (90, 15)))
    assert keys[:15] == ["f"] * 15
    assert keys[15:30] == ["tl"] * 15
    assert keys[30:] == ["t"] * 15


def test_unvoiced_onsets_get_modal_tier():
    # two voiced clusters (low sparse, high dense) + some unvoiced -> unvoiced
    # follow the modal (most-populated = high) tier, voiced keep their own.
    pitches = _pop((74, 4), (86, 20)) + [None, None]
    keys = ts.assign_keys(pitches)
    assert keys[:4] == ["f"] * 4
    assert keys[-2:] == ["t", "t"]   # unvoiced -> modal (high) tier


def test_empty_input():
    assert ts.assign_keys([]) == []
    assert ts.cluster_pitches([]) == []
