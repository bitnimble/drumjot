"""Tests for the time-signature heuristic and trailing-bar padding in beats.py.

These exercise pure-Python branches that don't require madmom or audio
I/O - we construct a BeatStructure by hand or call the small helpers
directly.
"""
from __future__ import annotations

from app.pipeline.beats import (
    BarInfo,
    BeatStructure,
    BeatTick,
    _choose_time_signature,
    _pad_trailing_bars,
    align_beats_to_onsets,
)

# ---------- _choose_time_signature ----------


def test_simple_meters_keep_quarter_denominator() -> None:
    assert _choose_time_signature(4, 120.0) == (4, 4)
    assert _choose_time_signature(3, 80.0) == (3, 4)
    assert _choose_time_signature(5, 140.0) == (5, 4)
    assert _choose_time_signature(7, 100.0) == (7, 4)


def test_six_beats_at_slow_tempo_stays_6_4() -> None:
    # Slow rock waltz feel.
    assert _choose_time_signature(6, 70.0) == (6, 4)
    assert _choose_time_signature(6, 95.0) == (6, 4)


def test_six_beats_at_fast_tempo_becomes_6_8() -> None:
    # Jazz waltz, jig: compound meter assumed once we hit ~100 BPM.
    assert _choose_time_signature(6, 100.0) == (6, 8)
    assert _choose_time_signature(6, 160.0) == (6, 8)


def test_twelve_beats_at_fast_tempo_becomes_12_8() -> None:
    assert _choose_time_signature(12, 130.0) == (12, 8)
    assert _choose_time_signature(12, 80.0) == (12, 4)


# ---------- _pad_trailing_bars ----------


def _make_bar(index: int, start: float, beat_gap: float, count: int = 4) -> BarInfo:
    beats = [
        BeatTick(time=start + i * beat_gap, beat_in_bar=i + 1, bar_index=index)
        for i in range(count)
    ]
    return BarInfo(
        index=index,
        start_time=start,
        end_time=start + count * beat_gap,
        beats=beats,
        time_signature=(count, 4),
        tempo_bpm=60.0 / beat_gap,
    )


def test_padding_extends_to_audio_duration() -> None:
    # One detected 4/4 bar covering 0.0 - 2.0 s (beat_gap = 0.5 s, 120 BPM).
    # Audio duration is 10 s, so we should get four synthetic bars after.
    bar = _make_bar(0, start=0.0, beat_gap=0.5)
    structure = BeatStructure(
        beats=list(bar.beats),
        bars=[bar],
        initial_tempo=120.0,
        initial_time_signature=(4, 4),
    )
    _pad_trailing_bars(structure, duration_seconds=10.0)
    # Original bar + four pads = 5 bars total covering 0..10s.
    assert len(structure.bars) == 5
    assert structure.bars[-1].end_time >= 10.0
    # The pads inherit the original tempo and time signature.
    for b in structure.bars[1:]:
        assert b.time_signature == (4, 4)
        assert abs(b.tempo_bpm - 120.0) < 0.01


def test_padding_lets_position_resolve_late_onsets() -> None:
    bar = _make_bar(0, start=0.0, beat_gap=0.5)
    structure = BeatStructure(
        beats=list(bar.beats),
        bars=[bar],
        initial_tempo=120.0,
        initial_time_signature=(4, 4),
    )
    _pad_trailing_bars(structure, duration_seconds=10.0)
    # Onset at t=5.25 s should fall in a synthetic bar; structure.position
    # returns a (bar, beat_in_bar) with the bar's index from the pads.
    pos = structure.position(5.25)
    assert pos is not None
    bar_idx, beat_in_bar = pos
    assert bar_idx > 0
    # `beat_in_bar` must stay sensible (inside the bar's beat count).
    assert 1.0 <= beat_in_bar < 5.0


def test_padding_skips_when_already_long_enough() -> None:
    bar = _make_bar(0, start=0.0, beat_gap=0.5)
    structure = BeatStructure(
        beats=list(bar.beats),
        bars=[bar],
    )
    original = list(structure.bars)
    _pad_trailing_bars(structure, duration_seconds=1.0)
    assert structure.bars == original


def test_position_outside_padded_range_returns_none() -> None:
    bar = _make_bar(0, start=0.0, beat_gap=0.5)
    structure = BeatStructure(
        beats=list(bar.beats),
        bars=[bar],
    )
    _pad_trailing_bars(structure, duration_seconds=3.0)
    # Onset past the audio duration: dropped.
    assert structure.position(50.0) is None


# ---------- align_beats_to_onsets ----------


def test_align_snaps_late_beats_back_to_onsets() -> None:
    # 160 BPM 4/4 bar: ideal beat times are 0.000, 0.375, 0.750, 1.125.
    # Simulate BT's ~30 ms lag: every detected beat is 0.030 s late.
    beat_gap = 60.0 / 160.0
    detected_offsets = [0.030] * 4
    beats = [
        BeatTick(
            time=i * beat_gap + detected_offsets[i],
            beat_in_bar=i + 1,
            bar_index=0,
        )
        for i in range(4)
    ]
    bar = BarInfo(
        index=0,
        start_time=beats[0].time,
        end_time=beats[-1].time + beat_gap,
        beats=list(beats),
        time_signature=(4, 4),
        tempo_bpm=60.0 / beat_gap,
    )
    structure = BeatStructure(beats=list(beats), bars=[bar])

    # Strong drum onsets sit on the true beat positions.
    onsets = [(i * beat_gap, 10.0) for i in range(4)]
    align_beats_to_onsets(structure, onsets, max_distance=0.05)

    for i, b in enumerate(structure.beats):
        assert abs(b.time - i * beat_gap) < 1e-9
    # Bar start/end + tempo recomputed from snapped beats.
    assert structure.bars[0].start_time == 0.0
    assert abs(structure.bars[0].tempo_bpm - 160.0) < 0.1


def test_align_picks_strongest_onset_in_window_not_closest() -> None:
    # Beat at t=1.000. A weak onset at t=0.995 (5 ms away, strength 0.1)
    # and a strong onset at t=1.020 (20 ms away, strength 5.0). The
    # strong one should win.
    beat = BeatTick(time=1.000, beat_in_bar=1, bar_index=0)
    bar = BarInfo(
        index=0,
        start_time=1.000,
        end_time=1.500,
        beats=[beat],
        time_signature=(4, 4),
        tempo_bpm=120.0,
    )
    structure = BeatStructure(beats=[beat], bars=[bar])

    align_beats_to_onsets(
        structure, [(0.995, 0.1), (1.020, 5.0)], max_distance=0.05,
    )
    assert abs(structure.beats[0].time - 1.020) < 1e-9


def test_align_leaves_beats_with_no_nearby_onset_alone() -> None:
    beat = BeatTick(time=2.000, beat_in_bar=1, bar_index=0)
    bar = BarInfo(
        index=0,
        start_time=2.000,
        end_time=2.500,
        beats=[beat],
        time_signature=(4, 4),
        tempo_bpm=120.0,
    )
    structure = BeatStructure(beats=[beat], bars=[bar])

    # All onsets sit > 50 ms from the beat.
    align_beats_to_onsets(
        structure, [(1.800, 5.0), (2.200, 5.0)], max_distance=0.05,
    )
    assert structure.beats[0].time == 2.000


def test_align_no_op_when_no_onsets() -> None:
    bar = _make_bar(0, start=0.0, beat_gap=0.5)
    structure = BeatStructure(beats=list(bar.beats), bars=[bar])
    align_beats_to_onsets(structure, [])
    for i, b in enumerate(structure.beats):
        assert b.time == i * 0.5


def test_align_preserves_per_bar_tempo_under_humanized_onsets() -> None:
    # Regression for the per-bar BPM wobble. The DBN grid is perfectly
    # regular at 160 BPM over two 4/4 bars. The drummer's actual onsets
    # are humanized — each sits a few ms off its grid position by a
    # *different* amount. Per-beat snapping would chase that jitter and
    # make every bar's tempo swing; the global-offset alignment must
    # keep the grid regular so per-bar tempo stays put.
    beat_gap = 60.0 / 160.0
    bars: list[BarInfo] = []
    beats: list[BeatTick] = []
    for i in range(8):
        tick = BeatTick(
            time=i * beat_gap, beat_in_bar=(i % 4) + 1, bar_index=i // 4
        )
        beats.append(tick)
    for b_idx in range(2):
        bb = beats[b_idx * 4:(b_idx + 1) * 4]
        bars.append(
            BarInfo(
                index=b_idx,
                start_time=bb[0].time,
                end_time=bb[-1].time + beat_gap,
                beats=list(bb),
                time_signature=(4, 4),
                tempo_bpm=160.0,
            )
        )
    structure = BeatStructure(beats=list(beats), bars=bars)

    # Per-beat, zero-mean-ish humanization (different offset each beat).
    jitter = [0.012, -0.009, 0.014, -0.011, 0.010, -0.013, 0.008, -0.012]
    onsets = [(i * beat_gap + jitter[i], 5.0) for i in range(8)]
    align_beats_to_onsets(structure, onsets, max_distance=0.05)

    # Inter-beat gaps stay uniform => no manufactured tempo wobble.
    gaps = [
        structure.beats[i + 1].time - structure.beats[i].time
        for i in range(len(structure.beats) - 1)
    ]
    assert max(gaps) - min(gaps) < 1e-9
    assert abs(structure.bars[0].tempo_bpm - 160.0) < 0.05
    assert abs(structure.bars[1].tempo_bpm - 160.0) < 0.05
    assert structure.bars[0].tempo_bpm == structure.bars[1].tempo_bpm
