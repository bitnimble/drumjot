"""Filter-pathway unit tests: no transcriber service, no LLM, no datasets.

Covers the two deterministic pieces the `filter` mode rests on:
  1. kept onsets -> MIDI (`onsets_to_midi_bytes`) parses back through the
     benchmark's shared mido helper (`midi_bytes_to_events`) to exactly
     the expected 3-class events at the original times. This is the
     "scored directly on the MIDI, no Jot" contract.
  2. the pure index/parse helpers in `filter_llm` (stable ordering,
     out-of-range exclusion, tool-result clamping).
"""
from __future__ import annotations

import io
import math
from types import SimpleNamespace

import mido

from app.pipeline.filter_llm import (
    _FILTER_TOOL,
    _extract_rejected,
    _index_in_range,
)
from app.pipeline.onsets_midi import onsets_to_midi_bytes
from benchmarks.core.classes import DrumClass
from benchmarks.core.midi_events import midi_bytes_to_events


def _c(time: float, strength: float = 5.0, bar: int = 0, beat: float = 1.0):
    """A duck-typed onset candidate (onsets_to_midi_bytes / filter_llm
    only read .time/.strength/.bar/.beat_in_bar via getattr)."""
    return SimpleNamespace(
        time=time, strength=strength, bar=bar, beat_in_bar=beat
    )


def test_onsets_midi_roundtrip_to_3class_events() -> None:
    onsets = {
        "k": [_c(0.000), _c(0.500)],
        "s": [_c(0.250), _c(0.750)],
        "h": [_c(0.000), _c(0.250), _c(0.500), _c(0.750)],
        # ride is real but outside the 3-class metric -> must drop out,
        # exactly like the E-GMD ground-truth path.
        "d": [_c(0.125)],
    }
    midi = onsets_to_midi_bytes(onsets, initial_tempo_bpm=120.0)
    events = midi_bytes_to_events(midi)

    got = sorted((round(e.time, 3), e.drum_class) for e in events)
    expected = sorted(
        [
            (0.000, DrumClass.KD), (0.500, DrumClass.KD),
            (0.250, DrumClass.SD), (0.750, DrumClass.SD),
            (0.000, DrumClass.HH), (0.250, DrumClass.HH),
            (0.500, DrumClass.HH), (0.750, DrumClass.HH),
        ]
    )
    assert len(got) == len(expected)
    for (gt, gc), (et, ec) in zip(got, expected):
        assert gc == ec
        # 480 PPQ @ 120 BPM => 1 tick ≈ 1.04 ms of integer-rounding error.
        assert math.isclose(gt, et, abs_tol=0.003)


def _bar(index, start_time, ts=(4, 4), bpm=120.0):
    return SimpleNamespace(
        index=index, start_time=start_time, time_signature=ts, tempo_bpm=bpm
    )


def test_structure_path_writes_meta_and_roundtrips_times() -> None:
    # 2 bars, 4/4 @ 120 BPM: each bar = 4 beats = 2.0 s.
    structure = SimpleNamespace(
        bars=[_bar(0, 0.0), _bar(1, 2.0)],
        initial_tempo=120.0,
        initial_time_signature=(4, 4),
    )
    onsets = {
        "k": [_c(0.000, bar=0), _c(1.000, bar=0)],
        "h": [_c(2.500, bar=1)],
        # out-of-range -> dropped by the structure path
        "s": [_c(9.9, bar=-1)],
    }
    midi = onsets_to_midi_bytes(onsets, initial_tempo_bpm=120.0, structure=structure)

    mid = mido.MidiFile(file=io.BytesIO(midi))
    metas = [m.type for tr in mid.tracks for m in tr]
    assert "time_signature" in metas
    assert "set_tempo" in metas
    ts = next(
        m for tr in mid.tracks for m in tr if m.type == "time_signature"
    )
    assert (ts.numerator, ts.denominator) == (4, 4)

    events = midi_bytes_to_events(midi)
    got = sorted((round(e.time, 3), e.drum_class) for e in events)
    expected = sorted([
        (0.000, DrumClass.KD), (1.000, DrumClass.KD), (2.500, DrumClass.HH),
    ])
    assert len(got) == len(expected)  # bar=-1 snare dropped
    for (gt, gc), (et, ec) in zip(got, expected):
        assert gc == ec
        assert math.isclose(gt, et, abs_tol=0.005)


def test_index_in_range_orders_and_drops_out_of_range() -> None:
    cands = [
        _c(9.9, bar=1, beat=2.0),
        _c(0.1, bar=-1, beat=-1.0),   # out of tracked range -> excluded
        _c(0.5, bar=0, beat=3.0),
        _c(0.4, bar=0, beat=1.0),
    ]
    indexed = _index_in_range(cands)
    # Sorted by (bar, beat_in_bar); the bar=-1 one is gone.
    assert [c.beat_in_bar for _, c in indexed] == [1.0, 3.0, 2.0]
    assert [i for i, _ in indexed] == [0, 1, 2]


def _resp(rejected):
    block = SimpleNamespace(
        type="tool_use", name=_FILTER_TOOL["name"],
        input={"rejected_indices": rejected},
    )
    return SimpleNamespace(content=[block])


def test_extract_rejected_clamps_and_dedupes() -> None:
    # 5 onsets; out-of-range / duplicate / non-int entries are ignored.
    out = _extract_rejected(_resp([0, 2, 2, 4, 9, -1, "x"]), n=5)
    assert out == {0, 2, 4}


def test_extract_rejected_no_tool_block_means_keep_all() -> None:
    empty = SimpleNamespace(content=[SimpleNamespace(type="text", text="hi")])
    assert _extract_rejected(empty, n=3) == set()
