"""Unit tests for the pure geometric onset-snap DP (`geometric_snap.py`).

Numbers in, numbers out: no audio, no I/O. Each onset is given as its
unrounded fractional slot position (ascending by time); `snap_lane`
returns the assigned integer slot per onset, or None when the onset was
band-rejected (left off-grid).
"""
from __future__ import annotations

from app.pipeline.geometric_snap import snap_lane


def test_empty_returns_empty() -> None:
    assert snap_lane([], band=2, off_grid_penalty=9.0) == []


def test_single_onset_snaps_to_nearest_slot() -> None:
    assert snap_lane([0.1], band=2, off_grid_penalty=9.0) == [0]
    assert snap_lane([2.9], band=2, off_grid_penalty=9.0) == [3]


def test_well_separated_onsets_stay_on_their_slots() -> None:
    assert snap_lane([0.0, 12.0, 24.0], band=2, off_grid_penalty=9.0) == [0, 12, 24]


def test_two_onsets_rounding_to_one_slot_get_distinct_slots() -> None:
    # Both near slot 5; the globally cheapest distinct, increasing pair is
    # (5, 6): cost 0 + 0.9^2 = 0.81, beating (4, 5) = 1 + 0.01 = 1.01.
    assert snap_lane([5.0, 5.1], band=2, off_grid_penalty=9.0) == [5, 6]


def test_injectivity_picks_globally_optimal_pair_not_greedy() -> None:
    # Both at 2.6: greedy nearest would put the first at 3, forcing the
    # second to 4 (cost 0.16 + 1.96). The optimal distinct pair is (2, 3)
    # at cost 0.36 + 0.16 = 0.52.
    assert snap_lane([2.6, 2.6], band=2, off_grid_penalty=9.0) == [2, 3]


def test_three_way_contention_centres_on_the_cluster() -> None:
    # Three onsets at slot 5; cheapest distinct increasing triple is
    # (4, 5, 6) at cost 1 + 0 + 1 = 2.
    assert snap_lane([5.0, 5.0, 5.0], band=2, off_grid_penalty=9.0) == [4, 5, 6]


def test_assignments_are_strictly_increasing() -> None:
    out = snap_lane([3.0, 3.0, 3.0], band=2, off_grid_penalty=9.0)
    placed = [s for s in out if s is not None]
    assert placed == sorted(set(placed))  # strictly increasing, no repeats


def test_placed_slots_stay_within_band() -> None:
    naturals = [1.0, 1.0, 1.0]
    band = 2
    out = snap_lane(naturals, band=band, off_grid_penalty=100.0)
    for nat, slot in zip(naturals, out, strict=True):
        if slot is not None:
            assert abs(slot - round(nat)) <= band


def test_band_rejection_when_cluster_exceeds_available_slots() -> None:
    # band=1 -> a 3-slot window {4,5,6}; four onsets can't all fit, so with
    # a high off-grid penalty exactly one is rejected and the other three
    # take the whole window.
    out = snap_lane([5.0, 5.0, 5.0, 5.0], band=1, off_grid_penalty=100.0)
    assert out.count(None) == 1
    assert sorted(s for s in out if s is not None) == [4, 5, 6]


def test_band_zero_forces_collisions_off_grid() -> None:
    # band=0 -> the only feasible slot is round(natural); two onsets that
    # both want slot 3 collide, so one is placed and the other rejected.
    out = snap_lane([3.0, 3.0], band=0, off_grid_penalty=5.0)
    assert out.count(None) == 1
    assert [s for s in out if s is not None] == [3]


def test_min_slot_clamps_the_low_edge_of_the_window() -> None:
    # natural 1, band 2 -> raw window [-1, 3]; min_slot=0 clamps to {0..3}
    # and the onset still snaps to its nearest in-range slot, 1.
    assert snap_lane([1.0], band=2, off_grid_penalty=9.0, min_slot=0, max_slot=10) == [1]


def test_window_clamped_empty_forces_off_grid() -> None:
    # natural 5, band 2 -> window [3, 7]; max_slot=2 leaves nothing in
    # range, so the onset is rejected.
    assert snap_lane([5.0], band=2, off_grid_penalty=9.0, min_slot=0, max_slot=2) == [None]


def test_bounds_force_collision_off_grid() -> None:
    # Both onsets can only reach slot 0 (single-slot range), so one is
    # placed and the other rejected.
    out = snap_lane([0.0, 0.0], band=2, off_grid_penalty=9.0, min_slot=0, max_slot=0)
    assert out.count(None) == 1
    assert [s for s in out if s is not None] == [0]


def test_cheap_penalty_prefers_rejection_over_a_costly_shift() -> None:
    # Two onsets at slot 5, band=2. Placing both costs >= (5,6)=0.81. With
    # a tiny penalty, leaving one off-grid (0 + 0.5) is cheaper, so the
    # second is rejected and the first sits exactly on 5.
    out = snap_lane([5.0, 5.0], band=2, off_grid_penalty=0.5)
    assert out.count(None) == 1
    assert [s for s in out if s is not None] == [5]
