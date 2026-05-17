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
