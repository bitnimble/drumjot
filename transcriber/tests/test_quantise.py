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

from app.models import OnsetCandidate
from app.pipeline.quantise import (
    _MAX_LLM_SHIFT,
    _QUANTISE_TOOL,
    SLOTS_PER_BEAT,
    _apply_llm_shifts,
    _extract_shifts,
    _geometric_snap,
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
