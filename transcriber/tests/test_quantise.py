"""Quantise-stage unit tests: no LLM, no transcriber service.

Covers the deterministic pieces of `pipeline/quantise.py`:
  - 1/48 slot computation from `beat_in_bar`,
  - beat-hierarchy weighting (downbeat > beat > 8th > triplet > 16th
    > arbitrary 48th),
  - cross-instrument cluster snap (kick + snare that landed one slot
    apart get pulled to the stronger slot),
  - shift bound enforcement (a cluster spanning more than ±2 slots
    leaves the outliers alone),
  - LLM tool-result extraction + clamping.
"""
from __future__ import annotations

from types import SimpleNamespace

from app.models import OnsetCandidate
from app.pipeline.quantise import (
    _MAX_LLM_SHIFT,
    _QUANTISE_TOOL,
    SLOTS_PER_BEAT,
    _deterministic_joint_snap,
    _extract_shifts,
    _initial_slot_for,
    _slot_weight,
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


def test_initial_slot_for_round_to_nearest_grid() -> None:
    bar = _bar(0, 0.0, 2.0)  # 4/4 @ 120 BPM => 2.0s per bar
    # beat_in_bar = 1.0 => slot 0 (downbeat)
    c = OnsetCandidate(time=0.0, strength=1.0, bar=0, beat_in_bar=1.0)
    slot, slot_s = _initial_slot_for(c, bar)
    assert slot == 0
    # 4 beats * 12 slots = 48 per bar => 2.0/48 seconds per slot.
    assert abs(slot_s - 2.0 / 48) < 1e-9

    # beat_in_bar = 2.5 => slot 18 ("& of beat 2" in 4/4)
    c = OnsetCandidate(time=0.0, strength=1.0, bar=0, beat_in_bar=2.5)
    slot, _ = _initial_slot_for(c, bar)
    assert slot == 18

    # Slight jitter early: beat_in_bar = 1.99 => snap to slot 12 (beat 2)
    c = OnsetCandidate(time=0.0, strength=1.0, bar=0, beat_in_bar=1.99)
    slot, _ = _initial_slot_for(c, bar)
    assert slot == 12

    # Clamp at end of bar: beat 5.0 in 4/4 is the next downbeat;
    # within-bar slot caps at num_beats*12 - 1 = 47.
    c = OnsetCandidate(time=0.0, strength=1.0, bar=0, beat_in_bar=5.0)
    slot, _ = _initial_slot_for(c, bar)
    assert slot == 47


def test_slot_weight_hierarchy() -> None:
    # Downbeat strongest, then other beats, then offbeat 8ths, then
    # triplets, then 16ths, then arbitrary 48ths.
    assert _slot_weight(0, 4) > _slot_weight(12, 4)  # downbeat > beat 2
    assert _slot_weight(12, 4) > _slot_weight(6, 4)  # beat > offbeat 8th
    assert _slot_weight(6, 4) > _slot_weight(4, 4)   # offbeat 8th > triplet
    assert _slot_weight(4, 4) > _slot_weight(3, 4)   # triplet > 16th
    assert _slot_weight(3, 4) > _slot_weight(1, 4)   # 16th > arbitrary 48th


def test_deterministic_snap_pulls_cluster_to_stronger_slot() -> None:
    # 4/4 @ 120 BPM, 2.0s per bar; one slot ~= 41.67 ms.
    bar = _bar(0, 0.0, 2.0)
    structure = _structure([bar])

    # Two onsets that fired ~25 ms apart but rounded to different slots:
    #   kick at beat 1 exactly (slot 0)
    #   snare jitter: beat_in_bar 1.083 => slot 1 (about 41 ms late)
    # The deterministic snap should pull the snare to slot 0 (downbeat,
    # higher weight) since they're within the cluster window.
    k = OnsetCandidate(time=0.000, strength=5.0, bar=0, beat_in_bar=1.000)
    s = OnsetCandidate(time=0.025, strength=5.0, bar=0, beat_in_bar=1.083)
    kept = {"k": [k], "s": [s]}
    shifts = _deterministic_joint_snap(kept, structure)  # type: ignore[arg-type]

    # The snare moves by -1 slot (slot 1 -> slot 0).
    assert shifts == {("s", 0): -1}
    assert s.quantised_time == 0.0
    assert s.quantised_shift_slots == -1
    # Kick is unchanged.
    assert k.quantised_time is None
    assert k.quantised_shift_slots is None


def test_deterministic_snap_respects_cluster_window() -> None:
    # Same setup but the snare is ~80 ms late (outside the 60 ms
    # cluster window), so it should NOT be pulled to the downbeat.
    bar = _bar(0, 0.0, 2.0)
    structure = _structure([bar])
    k = OnsetCandidate(time=0.000, strength=5.0, bar=0, beat_in_bar=1.000)
    s = OnsetCandidate(time=0.080, strength=5.0, bar=0, beat_in_bar=1.192)
    kept = {"k": [k], "s": [s]}
    shifts = _deterministic_joint_snap(kept, structure)  # type: ignore[arg-type]
    assert shifts == {}
    assert s.quantised_time is None


def test_deterministic_snap_skips_oversized_shifts() -> None:
    # Three onsets in the same 60 ms cluster but spread across slots 0,
    # 1 and 4: pulling slot-4 onto slot 0 would be a 4-slot shift,
    # exceeding the deterministic cap (=2). That onset stays put;
    # slot-1 still pulls to slot 0.
    bar = _bar(0, 0.0, 2.0)
    structure = _structure([bar])
    a = OnsetCandidate(time=0.000, strength=5.0, bar=0, beat_in_bar=1.000)
    b = OnsetCandidate(time=0.025, strength=5.0, bar=0, beat_in_bar=1.083)
    c = OnsetCandidate(time=0.050, strength=5.0, bar=0, beat_in_bar=1.333)
    kept = {"k": [a], "s": [b], "h": [c]}
    shifts = _deterministic_joint_snap(kept, structure)  # type: ignore[arg-type]
    assert shifts == {("s", 0): -1}
    assert c.quantised_time is None


def test_deterministic_snap_out_of_range_ignored() -> None:
    bar = _bar(0, 0.0, 2.0)
    structure = _structure([bar])
    c = OnsetCandidate(time=9.9, strength=1.0, bar=-1, beat_in_bar=-1.0)
    kept = {"k": [c]}
    shifts = _deterministic_joint_snap(kept, structure)  # type: ignore[arg-type]
    assert shifts == {}
    assert c.quantised_time is None


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
    # The schema constrains the LLM to ±_MAX_LLM_SHIFT, but a
    # misbehaving model that returns 7 still gets surfaced; the
    # caller applies the clamp so a future change to where clamping
    # lives doesn't silently change behaviour.
    out = _extract_shifts(_resp([{"id": 0, "shift": 7}]), n=1)
    assert out == {0: 7}
    assert _MAX_LLM_SHIFT < 7  # sanity: clamp would do something


def test_extract_shifts_no_tool_block_means_no_shifts() -> None:
    empty = SimpleNamespace(content=[SimpleNamespace(type="text", text="hi")])
    assert _extract_shifts(empty, n=3) == {}


def test_slots_per_beat_matches_frontend_default() -> None:
    # `src/midi/from_midi.ts::gridDivision` defaults to 48
    # (1/48-of-whole-note), which is 12 slots per quarter-note beat.
    assert SLOTS_PER_BEAT == 12
