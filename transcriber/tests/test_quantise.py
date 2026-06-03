"""Quantise-stage unit tests: no LLM, no transcriber service.

Covers the pure pieces of `pipeline/quantise.py`:
  - the geometric per-(lane, bar) snap (`_geometric_snap`): placed onsets
    get an exact slot time, same-lane onsets never share a slot
    (injectivity), cross-lane onsets are independent (no cluster pull),
    and an over-full cluster leaves the overflow off-grid,
  - LLM tool-result extraction + clamping.

The DP itself is exercised directly in `test_geometric_snap.py`; here we
test the orchestration that maps onsets <-> slots and mutates candidates.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.models import OnsetCandidate
from app.pipeline.envelope import OnsetEnvelope
from app.pipeline.quantise import (
    _MAX_LLM_SHIFT,
    _QUANTISE_TOOL,
    SLOTS_PER_BEAT,
    _apply_llm_shifts,
    _build_windows,
    _candidate_grids,
    _envelope_snap,
    _extract_shifts,
    _format_window,
    _geometric_snap,
    _infer_grid,
    _LlmEntry,
    _musical_grid_snap,
    _nearest_grid_slot,
    _resolve_cross_bar_target,
    _slot_label,
    quantise_kept_onsets,
)


def _bar(index, start_time, end_time, ts=(4, 4), bpm=120.0, feel="straight16"):
    return SimpleNamespace(
        index=index,
        start_time=start_time,
        end_time=end_time,
        time_signature=ts,
        tempo_bpm=bpm,
        feel=feel,
        beats=[],
    )


def _structure(bars):
    return SimpleNamespace(
        bars=bars,
        initial_tempo=120.0,
        initial_time_signature=(4, 4),
    )


def test_geometric_snap_sets_exact_slot_time_for_a_placed_onset() -> None:
    # 4/4 @ 120 BPM, 2.0s per bar; 48 slots, slot_span = 2/48 s.
    bar = _bar(0, 0.0, 2.0)
    structure = _structure([bar])
    # Kick on beat 1 exactly -> slot 0 -> quantised_time == bar start.
    k = OnsetCandidate(time=0.005, strength=5.0, bar=0, beat_in_bar=1.0)
    kept = {"k": [k]}
    shifts = _geometric_snap(kept, structure, slots_per_beat=12)  # type: ignore[arg-type]
    assert shifts == {}  # zero shift -> not in the summary map
    assert k.quantised_time == 0.0
    assert k.quantised_shift_slots == 0
    assert k.off_grid is False


def test_geometric_snap_is_injective_within_a_lane_and_bar() -> None:
    # Two kicks both detected on beat 1 (slot 0) can't share the slot; the
    # cheaper distinct pair is (0, 1).
    bar = _bar(0, 0.0, 2.0)
    structure = _structure([bar])
    a = OnsetCandidate(time=0.000, strength=5.0, bar=0, beat_in_bar=1.0)
    b = OnsetCandidate(time=0.010, strength=5.0, bar=0, beat_in_bar=1.0)
    kept = {"k": [a, b]}
    _geometric_snap(kept, structure, slots_per_beat=12)  # type: ignore[arg-type]
    slot_span = 2.0 / 48
    assert a.quantised_time == 0.0          # slot 0
    assert b.quantised_time == slot_span    # slot 1
    assert a.off_grid is False and b.off_grid is False


def test_geometric_snap_does_not_pull_cross_lane_onsets_together() -> None:
    # A kick and a snare both on beat 1 stay on slot 0 each, the old
    # cross-instrument cluster pull is gone, lanes are independent.
    bar = _bar(0, 0.0, 2.0)
    structure = _structure([bar])
    k = OnsetCandidate(time=0.000, strength=5.0, bar=0, beat_in_bar=1.0)
    s = OnsetCandidate(time=0.008, strength=5.0, bar=0, beat_in_bar=1.083)  # ~slot 1
    kept = {"k": [k], "s": [s]}
    _geometric_snap(kept, structure, slots_per_beat=12)  # type: ignore[arg-type]
    # Each lane snaps to its own nearest slot; the snare is NOT pulled onto
    # the kick's downbeat.
    assert k.quantised_time == 0.0           # slot 0
    assert s.quantised_time == 2.0 / 48      # slot 1
    assert k.quantised_shift_slots == 0
    assert s.quantised_shift_slots == 0


def test_geometric_snap_band_rejects_overflow_to_off_grid() -> None:
    # Six kicks all on beat 2 (slot 12): the band-2 window {10..14} holds
    # only five, so one is left off-grid.
    bar = _bar(0, 0.0, 2.0)
    structure = _structure([bar])
    kicks = [
        OnsetCandidate(time=0.5 + i * 0.001, strength=5.0, bar=0, beat_in_bar=2.0)
        for i in range(6)
    ]
    kept = {"k": kicks}
    _geometric_snap(kept, structure, slots_per_beat=12)  # type: ignore[arg-type]
    off = [c for c in kicks if c.off_grid]
    placed = [c for c in kicks if not c.off_grid]
    assert len(off) == 1
    assert len(placed) == 5
    # Off-grid onset keeps quantised_time None so the emitter uses raw time.
    assert off[0].quantised_time is None
    # The five placed onsets occupy distinct slots within the band window.
    placed_slots = sorted(round((c.quantised_time or 0.0) / (2.0 / 48)) for c in placed)
    assert placed_slots == [10, 11, 12, 13, 14]


def test_geometric_snap_reassigns_overflow_onset_to_next_bar_downbeat() -> None:
    # Two bars of 4/4 @ 120 BPM (2.0 s each, 48 slots, slot_span = 2/48 s).
    # The detector placed a kick in bar 0 just before bar 1's downbeat:
    # beat_in_bar = 4.99 -> natural slot = 47.88 -> rounds to 48 = past
    # bar 0's last slot (47). Without the cross-bar pre-pass it'd clamp
    # to slot 47 of bar 0 ("bar 1, 48/48" in 1-indexed display); with it,
    # it moves to bar 1's downbeat (slot 0) instead.
    bars = [_bar(0, 0.0, 2.0), _bar(1, 2.0, 4.0)]
    structure = _structure(bars)
    k = OnsetCandidate(time=1.995, strength=5.0, bar=0, beat_in_bar=4.99)
    kept = {"k": [k]}
    _geometric_snap(kept, structure, slots_per_beat=12)  # type: ignore[arg-type]
    assert k.bar == 1
    assert k.beat_in_bar == 1.0
    assert k.quantised_time == 2.0          # bar 1's start = slot 0
    assert k.off_grid is False


def test_geometric_snap_does_not_reassign_overflow_in_final_bar() -> None:
    # No bar 1 to receive the overflow: the onset falls through to the
    # existing per-bar clamp.
    bars = [_bar(0, 0.0, 2.0)]
    structure = _structure(bars)
    k = OnsetCandidate(time=1.995, strength=5.0, bar=0, beat_in_bar=4.99)
    kept = {"k": [k]}
    _geometric_snap(kept, structure, slots_per_beat=12)  # type: ignore[arg-type]
    assert k.bar == 0
    # Clamped to bar 0's last slot (47).
    assert k.quantised_time == 47 * (2.0 / 48)
    assert k.off_grid is False


def test_geometric_snap_ignores_out_of_range_bars() -> None:
    bar = _bar(0, 0.0, 2.0)
    structure = _structure([bar])
    c = OnsetCandidate(time=9.9, strength=1.0, bar=-1, beat_in_bar=-1.0)
    kept = {"k": [c]}
    shifts = _geometric_snap(kept, structure, slots_per_beat=12)  # type: ignore[arg-type]
    assert shifts == {}
    assert c.quantised_time is None
    assert c.off_grid is False


def _two_kicks_at_slots_12_13():
    # 4/4 @ 120 BPM, 48 slots, slot_span = 2/48 s. Beat 2.0 -> slot 12,
    # beat 2.0833 -> slot 13.
    bar = _bar(0, 0.0, 2.0)
    structure = _structure([bar])
    k0 = OnsetCandidate(time=0.50, strength=5.0, bar=0, beat_in_bar=2.0)
    k1 = OnsetCandidate(time=0.54, strength=5.0, bar=0, beat_in_bar=2.0 + 1 / 12)
    kept = {"k": [k0, k1]}
    _geometric_snap(kept, structure, slots_per_beat=12)  # type: ignore[arg-type]
    assert round((k0.quantised_time or 0.0) / (2.0 / 48)) == 12
    assert round((k1.quantised_time or 0.0) / (2.0 / 48)) == 13
    return kept, structure


def test_apply_llm_shifts_rejects_a_colliding_group() -> None:
    # Shifting k0 (+1) onto k1's slot 13 would break injectivity; the whole
    # group keeps its geometric placement instead.
    kept, structure = _two_kicks_at_slots_12_13()
    slot_span = 2.0 / 48
    _apply_llm_shifts(kept, structure, {("k", 0): 1}, slots_per_beat=12)  # type: ignore[arg-type]
    assert kept["k"][0].quantised_time == 12 * slot_span  # unchanged
    assert kept["k"][1].quantised_time == 13 * slot_span  # unchanged


def test_apply_llm_shifts_applies_a_valid_shift() -> None:
    # Shifting k0 (-1) to slot 11 keeps the group strictly increasing, so
    # it's applied.
    kept, structure = _two_kicks_at_slots_12_13()
    slot_span = 2.0 / 48
    _apply_llm_shifts(kept, structure, {("k", 0): -1}, slots_per_beat=12)  # type: ignore[arg-type]
    assert kept["k"][0].quantised_time == 11 * slot_span
    assert kept["k"][0].quantised_shift_slots == -1
    assert kept["k"][1].quantised_time == 13 * slot_span  # unchanged


def test_apply_llm_shifts_rejects_out_of_bar_shift() -> None:
    # A shift past the bar's last slot is dropped (no re-barring).
    bar = _bar(0, 0.0, 2.0)
    structure = _structure([bar])
    k = OnsetCandidate(time=1.95, strength=5.0, bar=0, beat_in_bar=4.92)  # ~slot 47
    kept = {"k": [k]}
    _geometric_snap(kept, structure, slots_per_beat=12)  # type: ignore[arg-type]
    before = kept["k"][0].quantised_time
    _apply_llm_shifts(kept, structure, {("k", 0): 2}, slots_per_beat=12)  # type: ignore[arg-type]
    assert kept["k"][0].quantised_time == before  # unchanged (would cross bar)


def _resp(shifts):
    block = SimpleNamespace(
        type="tool_use",
        name=_QUANTISE_TOOL["name"],
        input={"shifts": shifts},
    )
    return SimpleNamespace(content=[block])


def test_extract_shifts_filters_invalid_ids() -> None:
    out = _extract_shifts(
        _resp([
            {"id": 0, "shift": 1},
            {"id": 2, "shift": -2},
            {"id": 9, "shift": 1},   # out of range
            {"id": -1, "shift": 0},  # negative id
            {"shift": 1},            # missing id
            "not a dict",
        ]),
        n=3,
    )
    assert out == {0: 1, 2: -2}


def test_extract_shifts_preserves_oversized_for_caller_clamp() -> None:
    # The schema constrains the LLM to ±_MAX_LLM_SHIFT, but a misbehaving
    # model that returns 7 still gets surfaced; the caller applies the
    # clamp so a future change to where clamping lives doesn't silently
    # change behaviour.
    out = _extract_shifts(_resp([{"id": 0, "shift": 7}]), n=1)
    assert out == {0: 7}
    assert _MAX_LLM_SHIFT < 7  # sanity: clamp would do something


def test_extract_shifts_no_tool_block_means_no_shifts() -> None:
    empty = SimpleNamespace(content=[SimpleNamespace(type="text", text="hi")])
    assert _extract_shifts(empty, n=3) == {}


def test_slot_label_uses_the_given_grid_density() -> None:
    # At the default 12 slots/beat: slot 0 = downbeat, 6 = "&", 3 = "e".
    assert _slot_label(0, 12) == "(beat 1)"
    assert _slot_label(6, 12) == "(& of 1)"
    assert _slot_label(3, 12) == "(e of 1)"
    assert _slot_label(12, 12) == "(beat 2)"
    # At a denser 24 slots/beat the same musical positions scale: the "&"
    # is now slot 12, and slot 24 is beat 2, labels track the parameter,
    # not a hardcoded 12.
    assert _slot_label(0, 24) == "(beat 1)"
    assert _slot_label(12, 24) == "(& of 1)"
    assert _slot_label(24, 24) == "(beat 2)"


def test_summary_records_the_grid_density_actually_used() -> None:
    # The shifts.json summary must reflect the `slots_per_beat` the run
    # used, not the module default, so a consumer can read back the grid.
    bar = _bar(0, 0.0, 2.0)
    structure = _structure([bar])
    kept = {"k": [OnsetCandidate(time=0.0, strength=5.0, bar=0, beat_in_bar=1.0)]}
    summary = quantise_kept_onsets(
        kept, structure, use_llm=False, slots_per_beat=24  # type: ignore[arg-type]
    )
    assert summary["slots_per_beat"] == 24


def test_slots_per_beat_matches_frontend_default() -> None:
    # `src/midi/from_midi.ts::gridDivision` defaults to 48
    # (1/48-of-whole-note), which is 12 slots per quarter-note beat.
    assert SLOTS_PER_BEAT == 12


# ---------- LLM window chunking ----------

def _indexed(*specs):
    """Build a sorted `[_LlmEntry]` the way `_index_for_llm` would.

    Each spec is `(pitch, idx, bar, slot)`. Entries are returned in the
    same `(bar, slot, pitch)` order `_index_for_llm` guarantees, so the
    global ids `_build_windows` assigns line up with real runs.
    """
    entries = [_LlmEntry(pitch=p, idx=i, bar=b, slot=s) for (p, i, b, s) in specs]
    entries.sort(key=lambda e: (e.bar, e.slot, e.pitch))
    return entries


def test_build_windows_never_splits_a_dense_bar() -> None:
    # A single bar whose onset count exceeds the target is still one window
    # (the target is a soft cap; we never split a bar).
    entries = _indexed(*[("k", i, 0, i) for i in range(200)])
    windows = _build_windows(
        entries, _structure([_bar(0, 0.0, 2.0)]),  # type: ignore[arg-type]
        target_onsets=150, max_bars=8, context_bars=0,
    )
    assert len(windows) == 1
    assert windows[0].core_set == {0}
    assert len(windows[0].local_to_global) == 200


def test_build_windows_breaks_on_onset_target() -> None:
    # Bars of 60 onsets each, target 150: bars 0+1 fit (120), bar 2 would
    # overflow (180) so it starts a new window.
    specs = [("k", i, b, i % 60) for b in range(3) for i in range(60)]
    entries = _indexed(*specs)
    bars = [_bar(b, b * 2.0, b * 2.0 + 2.0) for b in range(3)]
    windows = _build_windows(
        entries, _structure(bars),  # type: ignore[arg-type]
        target_onsets=150, max_bars=8, context_bars=0,
    )
    assert [sorted(w.core_set) for w in windows] == [[0, 1], [2]]


def test_build_windows_breaks_on_bar_span() -> None:
    # Sparse bars far apart: the bar-span cap (8) forces a break even though
    # the onset target is nowhere near.
    entries = _indexed(("k", 0, 0, 0), ("k", 1, 10, 0))
    bars = [_bar(b, b * 2.0, b * 2.0 + 2.0) for b in range(11)]
    windows = _build_windows(
        entries, _structure(bars),  # type: ignore[arg-type]
        target_onsets=150, max_bars=8, context_bars=0,
    )
    assert [sorted(w.core_set) for w in windows] == [[0], [10]]


def test_build_windows_local_ids_map_back_in_global_order() -> None:
    # Local ids are assigned in (bar, slot, pitch) order across the window's
    # core bars; local_to_global must invert that exactly.
    entries = _indexed(
        ("s", 0, 0, 5), ("k", 0, 0, 0), ("h", 0, 0, 0), ("k", 1, 1, 3),
    )
    bars = [_bar(0, 0.0, 2.0), _bar(1, 2.0, 4.0)]
    windows = _build_windows(
        entries, _structure(bars),  # type: ignore[arg-type]
        target_onsets=150, max_bars=8, context_bars=0,
    )
    assert len(windows) == 1
    w = windows[0]
    # Global sort order: bar0 slot0 h, bar0 slot0 k, bar0 slot5 s, bar1 slot3 k
    assert w.local_to_global == [("h", 0), ("k", 0), ("s", 0), ("k", 1)]


def test_build_windows_adds_readonly_context_bars() -> None:
    # With context_bars=1 each window renders its neighbours, but they are
    # not in core_set (not shiftable).
    specs = [("k", i, b, i % 60) for b in range(3) for i in range(60)]
    entries = _indexed(*specs)
    bars = [_bar(b, b * 2.0, b * 2.0 + 2.0) for b in range(3)]
    windows = _build_windows(
        entries, _structure(bars),  # type: ignore[arg-type]
        target_onsets=150, max_bars=8, context_bars=1,
    )
    # Windows core = [0,1] and [2]. The first renders bar 2 as context; the
    # second renders bar 1 as context. Neither core_set grows.
    assert sorted(windows[0].core_set) == [0, 1]
    assert windows[0].render_bars == [0, 1, 2]
    assert sorted(windows[1].core_set) == [2]
    assert windows[1].render_bars == [1, 2]  # bar 3 doesn't exist


def test_format_window_tags_context_bars_and_drops_their_ids() -> None:
    entries = _indexed(("k", 0, 0, 0), ("s", 0, 1, 12))
    bars = [_bar(0, 0.0, 2.0), _bar(1, 2.0, 4.0)]
    # Window 0 owns bar 0; bar 1 is read-only context.
    windows = _build_windows(
        entries, _structure(bars),  # type: ignore[arg-type]
        target_onsets=1, max_bars=8, context_bars=1,
    )
    rendered = _format_window(
        _structure(bars), windows[0], slots_per_beat=12,  # type: ignore[arg-type]
    )
    # Core bar 0: shiftable onset carries a #id.
    assert "#0(k)" in rendered
    # Context bar 1: tagged read-only, its onset shown WITHOUT an id.
    assert "[context - read-only]" in rendered
    assert "(s)" in rendered and "#0(s)" not in rendered and "#1(s)" not in rendered


# ---------- Deterministic musical-grid pass ----------

_SLOT_SPAN = 2.0 / 48  # 4/4 @ 120 BPM, 12 slots/beat


_BAR_DURATION = 48 * _SLOT_SPAN  # 4/4 @ 120 BPM, 48 slots → 2.0 s


def _at_slots(slots, bar=0):
    """Candidates placed (post-geometric) at the given absolute slots.

    `quantised_time` is set directly so `_musical_grid_snap` sees a known
    current slot without us having to reverse-engineer `beat_in_bar`.
    Times are offset by `bar * _BAR_DURATION` so a candidate's `bar`
    index agrees with its `quantised_time` (lets `_current_slot` resolve
    correctly without relying on bounds clamping).
    """
    cands = []
    bar_start = bar * _BAR_DURATION
    for s in slots:
        c = OnsetCandidate(
            time=bar_start + s * _SLOT_SPAN, strength=5.0, bar=bar,
            beat_in_bar=1.0 + s / 12,
        )
        c.quantised_time = bar_start + s * _SLOT_SPAN
        c.off_grid = False
        cands.append(c)
    return cands


def _slot_of(c):
    return round((c.quantised_time or 0.0) / _SLOT_SPAN)


def test_infer_grid_picks_the_supported_subdivision() -> None:
    g = _candidate_grids(12)

    def name(folded):
        grid = _infer_grid(folded, g, 12)
        assert grid is not None
        return grid[0]

    assert name([0, 3, 6, 9, 0, 3, 6, 9]) == "straight_16"
    assert name([0, 4, 8, 0, 4, 8]) == "triplet_8"
    # Two jittered 8ths (1, 7) still read as straight 8ths, not promoted.
    assert name([0, 6, 0, 6, 1, 7, 0, 6]) == "straight_8"
    # Too little evidence -> defer (None).
    assert _infer_grid([0, 3], g, 12) is None


def test_grid_snap_pulls_a_stray_note_onto_the_voted_grid() -> None:
    # A straight-16 hat lane (0,6,9 across two beats) with one stray note on
    # slot 4 (a triplet-only position): the lane votes straight-16, so the
    # stray snaps to slot 3.
    structure = _structure([_bar(0, 0.0, 2.0)])
    cands = _at_slots([0, 4, 6, 9, 12, 18, 21])
    kept = {"h": cands}
    stray = cands[1]  # the slot-4 note
    _musical_grid_snap(kept, structure, slots_per_beat=12)  # type: ignore[arg-type]
    assert _slot_of(stray) == 3
    # The on-grid notes are untouched.
    assert _slot_of(cands[0]) == 0 and _slot_of(cands[2]) == 6


def test_grid_snap_leaves_a_genuine_triplet_lane_alone() -> None:
    # A clean 8th-triplet lane (0,4,8 per beat) must NOT be squared.
    structure = _structure([_bar(0, 0.0, 2.0)])
    cands = _at_slots([0, 4, 8, 12, 16, 20])
    kept = {"k": cands}
    shifts = _musical_grid_snap(kept, structure, slots_per_beat=12)  # type: ignore[arg-type]
    assert shifts == {}
    assert [_slot_of(c) for c in cands] == [0, 4, 8, 12, 16, 20]


def test_grid_snap_is_per_lane_so_polyrhythm_survives() -> None:
    # Same bar: straight-16 hats over a triplet kick. Each lane keeps its own
    # grid; the kick's slot-4 note is NOT squared to 3.
    structure = _structure([_bar(0, 0.0, 2.0)])
    hats = _at_slots([0, 6, 9, 12, 18, 21])
    kick = _at_slots([0, 4, 8, 12, 16, 20])
    kept = {"h": hats, "k": kick}
    _musical_grid_snap(kept, structure, slots_per_beat=12)  # type: ignore[arg-type]
    assert [_slot_of(c) for c in kick] == [0, 4, 8, 12, 16, 20]  # triplet kept


def test_grid_snap_sparse_lane_inherits_the_bar_aggregate() -> None:
    # A crash with only two hits can't vote its own grid; it inherits the
    # bar aggregate (dominated by the straight-16 hat lane) and its stray
    # slot-4 hit snaps to 3.
    structure = _structure([_bar(0, 0.0, 2.0)])
    hats = _at_slots([0, 3, 6, 9, 12, 15, 18, 21])
    crash = _at_slots([4, 24])
    kept = {"h": hats, "cr": crash}
    _musical_grid_snap(kept, structure, slots_per_beat=12)  # type: ignore[arg-type]
    assert _slot_of(crash[0]) == 3  # 4 -> 3 via inherited straight-16 grid


def test_grid_snap_respects_the_injectivity_guard() -> None:
    # The stray slot-4 note wants slot 3, but slot 3 is already occupied;
    # the monotonic-injective guard rejects the group, keeping placement.
    structure = _structure([_bar(0, 0.0, 2.0)])
    cands = _at_slots([0, 3, 4, 6])
    kept = {"h": cands}
    _musical_grid_snap(kept, structure, slots_per_beat=12)  # type: ignore[arg-type]
    assert [_slot_of(c) for c in cands] == [0, 3, 4, 6]  # unchanged


def test_nearest_grid_slot_wraps_to_the_next_downbeat() -> None:
    # Slot 11 of 12 is one slot before the next beat, not 11 after this one.
    assert _nearest_grid_slot(11, (0, 3, 6, 9), 12) == 12
    assert _nearest_grid_slot(4, (0, 3, 6, 9), 12) == 3


def test_grid_snap_moves_a_last_slot_onset_to_next_bar_downbeat() -> None:
    # 4/4 bar: kicks on the four downbeats plus a stray hit on slot 47
    # (the very last 48th of the bar, 1/48 short of bar 1's downbeat).
    # The lane votes straight-16 from the on-grid hits; the stray's
    # nearest grid position is slot 48, which is bar 1's downbeat.
    # Pre-fix the per-bar bounds check rejected the +1 shift, so the
    # stray stayed on slot 47.
    structure = _structure([_bar(0, 0.0, 2.0), _bar(1, 2.0, 4.0)])
    cands = _at_slots([0, 12, 24, 36, 47])
    kept = {"k": cands}
    stray = cands[4]
    _musical_grid_snap(kept, structure, slots_per_beat=12)  # type: ignore[arg-type]
    assert stray.bar == 1
    assert stray.beat_in_bar == 1.0
    assert stray.quantised_time == 2.0  # bar 1's start_time
    # The four downbeats stay put.
    assert [_slot_of(c) for c in cands[:4]] == [0, 12, 24, 36]


def test_grid_snap_skips_cross_bar_move_when_next_downbeat_is_occupied() -> None:
    # Same setup but bar 1 already has a kick on slot 0, moving the
    # stray would collide and break the per-bar injectivity invariant,
    # so the grid pass leaves it on slot 47.
    structure = _structure([_bar(0, 0.0, 2.0), _bar(1, 2.0, 4.0)])
    bar0_cands = _at_slots([0, 12, 24, 36, 47])
    bar1_cands = _at_slots([0, 12, 24, 36], bar=1)
    kept = {"k": bar0_cands + bar1_cands}
    stray = bar0_cands[4]
    _musical_grid_snap(kept, structure, slots_per_beat=12)  # type: ignore[arg-type]
    assert stray.bar == 0
    assert _slot_of(stray) == 47


def test_grid_snap_skips_cross_bar_move_when_no_next_bar() -> None:
    # Last bar's last-slot stray can't cross-bar (no next bar) so it
    # stays put rather than crashing on the out-of-range index.
    structure = _structure([_bar(0, 0.0, 2.0)])
    cands = _at_slots([0, 12, 24, 36, 47])
    kept = {"k": cands}
    stray = cands[4]
    _musical_grid_snap(kept, structure, slots_per_beat=12)  # type: ignore[arg-type]
    assert stray.bar == 0
    assert _slot_of(stray) == 47


def test_geometric_snap_records_the_sub_slot_residual() -> None:
    # A hit whose natural slot is 3.4 rounds to slot 3 with residual +0.4.
    structure = _structure([_bar(0, 0.0, 2.0)])
    c = OnsetCandidate(
        time=0.0, strength=5.0, bar=0, beat_in_bar=1.0 + 3.4 / 12
    )
    _geometric_snap({"k": [c]}, structure, slots_per_beat=12)  # type: ignore[arg-type]
    assert c.quantised_residual_slots is not None
    assert abs(c.quantised_residual_slots - 0.4) < 1e-6


def test_geometric_snap_leaves_residual_none_for_off_grid() -> None:
    # Six kicks crammed on beat 2: the band-rejected overflow is off-grid and
    # gets no residual.
    structure = _structure([_bar(0, 0.0, 2.0)])
    kicks = [
        OnsetCandidate(time=0.5 + i * 0.001, strength=5.0, bar=0, beat_in_bar=2.0)
        for i in range(6)
    ]
    _geometric_snap({"k": kicks}, structure, slots_per_beat=12)  # type: ignore[arg-type]
    off = [c for c in kicks if c.off_grid]
    assert len(off) == 1
    assert off[0].quantised_residual_slots is None


# ---------- per-note envelope re-snap ----------

def _env_with_pulses(slot_centers, *, height=10.0, width=0.004):
    """An OnsetEnvelope with sharp transients centred on the given slots."""
    ft = np.arange(0.0, 2.0, 0.001)
    env = np.zeros_like(ft)
    for s in slot_centers:
        env += height * np.exp(-((ft - s * _SLOT_SPAN) ** 2) / (2 * width**2))
    ref = float(np.percentile(env, 99)) if np.any(env) else 0.0
    return OnsetEnvelope(frame_times=ft, env=env, ref=ref)


def test_envelope_snap_moves_a_note_onto_its_transient() -> None:
    # Note geometrically mis-snapped to slot 11; the real transient is on
    # slot 12. The envelope re-snap pulls it over.
    structure = _structure([_bar(0, 0.0, 2.0)])
    cands = _at_slots([11])
    env = _env_with_pulses([12])
    shifts = _envelope_snap(
        {"s": cands}, structure, {"s": env}, slots_per_beat=12,  # type: ignore[arg-type]
    )
    assert shifts == {("s", 0): 1}
    assert _slot_of(cands[0]) == 12


def test_envelope_snap_leaves_a_note_already_on_its_transient() -> None:
    structure = _structure([_bar(0, 0.0, 2.0)])
    cands = _at_slots([12])
    env = _env_with_pulses([12])
    shifts = _envelope_snap(
        {"s": cands}, structure, {"s": env}, slots_per_beat=12,  # type: ignore[arg-type]
    )
    assert shifts == {}
    assert _slot_of(cands[0]) == 12


def test_envelope_snap_does_nothing_on_a_flat_envelope() -> None:
    # No dominant transient anywhere -> dominance gate keeps the note put.
    structure = _structure([_bar(0, 0.0, 2.0)])
    cands = _at_slots([11])
    ft = np.arange(0.0, 2.0, 0.001)
    env = OnsetEnvelope(frame_times=ft, env=np.full_like(ft, 0.05), ref=0.05)
    shifts = _envelope_snap(
        {"s": cands}, structure, {"s": env}, slots_per_beat=12,  # type: ignore[arg-type]
    )
    assert shifts == {}
    assert _slot_of(cands[0]) == 11


def test_envelope_snap_respects_the_injectivity_guard() -> None:
    # Two notes (slots 11 and 13) flanking a single transient on slot 12 would
    # both want to move onto 12; the guard rejects the colliding group.
    structure = _structure([_bar(0, 0.0, 2.0)])
    cands = _at_slots([11, 13])
    env = _env_with_pulses([12])
    _envelope_snap(
        {"s": cands}, structure, {"s": env}, slots_per_beat=12,  # type: ignore[arg-type]
    )
    assert [_slot_of(c) for c in cands] == [11, 13]  # unchanged


# ---------- _resolve_cross_bar_target ----------

def test_resolve_cross_bar_target_identity_for_zero_shift() -> None:
    structure = _structure([_bar(0, 0.0, 2.0), _bar(1, 2.0, 4.0)])
    assert _resolve_cross_bar_target(0, 12, 0, structure, 12) == (0, 12)  # type: ignore[arg-type]


def test_resolve_cross_bar_target_keeps_in_bar_shifts_in_place() -> None:
    structure = _structure([_bar(0, 0.0, 2.0), _bar(1, 2.0, 4.0)])
    assert _resolve_cross_bar_target(0, 10, 2, structure, 12) == (0, 12)  # type: ignore[arg-type]
    assert _resolve_cross_bar_target(0, 5, -3, structure, 12) == (0, 2)  # type: ignore[arg-type]


def test_resolve_cross_bar_target_walks_forward_one_bar() -> None:
    structure = _structure([_bar(0, 0.0, 2.0), _bar(1, 2.0, 4.0)])
    # Slot 47 + 1 = first slot of bar 1.
    assert _resolve_cross_bar_target(0, 47, 1, structure, 12) == (1, 0)  # type: ignore[arg-type]
    # Slot 44 + 8 = bar 1 slot 4.
    assert _resolve_cross_bar_target(0, 44, 8, structure, 12) == (1, 4)  # type: ignore[arg-type]


def test_resolve_cross_bar_target_walks_backward_one_bar() -> None:
    structure = _structure([_bar(0, 0.0, 2.0), _bar(1, 2.0, 4.0)])
    # Slot 0 of bar 1 - 1 = last slot of bar 0.
    assert _resolve_cross_bar_target(1, 0, -1, structure, 12) == (0, 47)  # type: ignore[arg-type]
    # Slot 3 of bar 1 - 5 = bar 0 slot 46.
    assert _resolve_cross_bar_target(1, 3, -5, structure, 12) == (0, 46)  # type: ignore[arg-type]


def test_resolve_cross_bar_target_walks_through_multiple_bars() -> None:
    bars = [_bar(i, i * 2.0, (i + 1) * 2.0) for i in range(4)]
    structure = _structure(bars)
    # Slot 0 of bar 0 + 100 = bar 2 slot 4 (skips bars 0 and 1 entirely).
    assert _resolve_cross_bar_target(0, 0, 100, structure, 12) == (2, 4)  # type: ignore[arg-type]
    # Slot 0 of bar 3 - 100 = bar 0 slot 44.
    assert _resolve_cross_bar_target(3, 0, -100, structure, 12) == (0, 44)  # type: ignore[arg-type]


def test_resolve_cross_bar_target_honours_time_signature_changes() -> None:
    # bar 0 = 4/4 (48 slots), bar 1 = 3/4 (36 slots), bar 2 = 4/4.
    bars = [
        _bar(0, 0.0, 2.0, ts=(4, 4)),
        _bar(1, 2.0, 3.5, ts=(3, 4)),
        _bar(2, 3.5, 5.5, ts=(4, 4)),
    ]
    structure = _structure(bars)
    # +1 from bar 0 slot 47 walks into bar 1's slot 0.
    assert _resolve_cross_bar_target(0, 47, 1, structure, 12) == (1, 0)  # type: ignore[arg-type]
    # +37 from bar 1 slot 0 walks past bar 1 (36 slots) into bar 2 slot 1.
    assert _resolve_cross_bar_target(1, 0, 37, structure, 12) == (2, 1)  # type: ignore[arg-type]
    # -1 from bar 2 slot 0 walks back to bar 1 slot 35 (last slot of 3/4).
    assert _resolve_cross_bar_target(2, 0, -1, structure, 12) == (1, 35)  # type: ignore[arg-type]


def test_resolve_cross_bar_target_returns_none_off_song_end() -> None:
    structure = _structure([_bar(0, 0.0, 2.0), _bar(1, 2.0, 4.0)])
    # Walking off the last bar forward.
    assert _resolve_cross_bar_target(1, 47, 1, structure, 12) is None  # type: ignore[arg-type]
    # Walking off the first bar backward.
    assert _resolve_cross_bar_target(0, 0, -1, structure, 12) is None  # type: ignore[arg-type]


# ---------- _apply_llm_shifts cross-bar ----------

def test_apply_shifts_moves_onset_forward_across_bar_boundary() -> None:
    structure = _structure([_bar(0, 0.0, 2.0), _bar(1, 2.0, 4.0)])
    c = _at_slots([47])[0]
    kept = {"k": [c]}
    _apply_llm_shifts(kept, structure, {("k", 0): 1}, slots_per_beat=12)  # type: ignore[arg-type]
    assert c.bar == 1
    assert c.beat_in_bar == 1.0
    assert abs(c.quantised_time - 2.0) < 1e-9
    assert c.quantised_shift_slots == 1


def test_apply_shifts_moves_onset_backward_across_bar_boundary() -> None:
    structure = _structure([_bar(0, 0.0, 2.0), _bar(1, 2.0, 4.0)])
    c = _at_slots([0], bar=1)[0]
    kept = {"k": [c]}
    _apply_llm_shifts(kept, structure, {("k", 0): -1}, slots_per_beat=12)  # type: ignore[arg-type]
    assert c.bar == 0
    assert abs(c.beat_in_bar - (1.0 + 47 / 12)) < 1e-9
    assert abs(c.quantised_time - 47 * _SLOT_SPAN) < 1e-9
    assert c.quantised_shift_slots == -1


def test_apply_shifts_walks_multi_slot_cross_bar_forward() -> None:
    # Shift +5 from slot 44 (4/4 bar) → bar 1 slot 1.
    structure = _structure([_bar(0, 0.0, 2.0), _bar(1, 2.0, 4.0)])
    c = _at_slots([44])[0]
    kept = {"k": [c]}
    _apply_llm_shifts(kept, structure, {("k", 0): 5}, slots_per_beat=12)  # type: ignore[arg-type]
    assert c.bar == 1
    assert abs(c.quantised_time - (2.0 + 1 * _SLOT_SPAN)) < 1e-9
    assert c.quantised_shift_slots == 5


def test_apply_shifts_rejects_cross_bar_move_when_destination_occupied() -> None:
    # Bar 0 has stray on slot 47 wanting to move to bar 1 slot 0; bar 1
    # already has a kick on slot 0 in the same lane, collision, whole
    # group rejected, the stray stays put.
    structure = _structure([_bar(0, 0.0, 2.0), _bar(1, 2.0, 4.0)])
    bar0 = _at_slots([47])
    bar1 = _at_slots([0], bar=1)
    kept = {"k": bar0 + bar1}
    _apply_llm_shifts(kept, structure, {("k", 0): 1}, slots_per_beat=12)  # type: ignore[arg-type]
    assert bar0[0].bar == 0
    assert _slot_of(bar0[0]) == 47


def test_apply_shifts_rejects_when_walk_runs_off_the_song_end() -> None:
    # Only one bar; shifting bar 0 slot 47 by +1 walks off the song end.
    structure = _structure([_bar(0, 0.0, 2.0)])
    c = _at_slots([47])[0]
    kept = {"k": [c]}
    _apply_llm_shifts(kept, structure, {("k", 0): 1}, slots_per_beat=12)  # type: ignore[arg-type]
    assert c.bar == 0
    assert _slot_of(c) == 47
    # Shift was not applied, so quantised_shift_slots stays at its initial value.
    assert c.quantised_shift_slots is None or c.quantised_shift_slots == 0


def test_apply_shifts_mixes_in_bar_and_cross_bar_moves_in_one_group() -> None:
    # Three onsets in bar 0: one stays put, one shifts in-bar, one crosses
    # into bar 1's downbeat. All apply atomically.
    structure = _structure([_bar(0, 0.0, 2.0), _bar(1, 2.0, 4.0)])
    cands = _at_slots([5, 20, 47])
    kept = {"k": cands}
    _apply_llm_shifts(
        kept, structure,  # type: ignore[arg-type]
        {("k", 0): 0, ("k", 1): -1, ("k", 2): 1},
        slots_per_beat=12,
    )
    assert cands[0].bar == 0 and _slot_of(cands[0]) == 5
    assert cands[1].bar == 0 and _slot_of(cands[1]) == 19
    # `_slot_of` is bar-0-relative (time / _SLOT_SPAN), so the cross-bar
    # candidate at bar 1's downbeat reads as absolute slot 48.
    assert cands[2].bar == 1 and cands[2].beat_in_bar == 1.0
    assert abs(cands[2].quantised_time - 2.0) < 1e-9


# ---------- _envelope_snap cross-bar ----------

def _env_with_pulses_two_bars(slot_centers, *, height=10.0, width=0.004):
    """`_env_with_pulses` extended to span bars 0 and 1 (0..4s).

    `slot_centers` are absolute slot numbers in a continuous 0..96 frame
    (48 slots/bar × 2 bars at 4/4 @ 120 BPM), so e.g. slot 48 is bar 1's
    downbeat.
    """
    ft = np.arange(0.0, 4.0, 0.001)
    env = np.zeros_like(ft)
    for s in slot_centers:
        env += height * np.exp(-((ft - s * _SLOT_SPAN) ** 2) / (2 * width**2))
    ref = float(np.percentile(env, 99)) if np.any(env) else 0.0
    return OnsetEnvelope(frame_times=ft, env=env, ref=ref)


def test_envelope_snap_moves_a_note_across_a_bar_forward() -> None:
    # Note geometrically placed on bar 0 slot 47; the real transient is on
    # bar 1 slot 0 (just past the boundary). The cross-bar walk in the
    # envelope re-snap finds it and migrates the note into bar 1.
    structure = _structure([_bar(0, 0.0, 2.0), _bar(1, 2.0, 4.0)])
    c = _at_slots([47])[0]
    env = _env_with_pulses_two_bars([48])
    _envelope_snap(
        {"k": [c]}, structure, {"k": env}, slots_per_beat=12,  # type: ignore[arg-type]
    )
    assert c.bar == 1
    assert c.beat_in_bar == 1.0
    assert abs(c.quantised_time - 2.0) < 1e-9


def test_envelope_snap_moves_a_note_across_a_bar_backward() -> None:
    # Note geometrically on bar 1 slot 0; the real transient is on bar 0
    # slot 47 (just before the boundary). Cross-bar walk pulls it back.
    structure = _structure([_bar(0, 0.0, 2.0), _bar(1, 2.0, 4.0)])
    c = _at_slots([0], bar=1)[0]
    env = _env_with_pulses_two_bars([47])
    _envelope_snap(
        {"k": [c]}, structure, {"k": env}, slots_per_beat=12,  # type: ignore[arg-type]
    )
    assert c.bar == 0
    assert abs(c.beat_in_bar - (1.0 + 47 / 12)) < 1e-9
    assert abs(c.quantised_time - 47 * _SLOT_SPAN) < 1e-9
