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
import pytest

from app.pipeline.filter_llm import (
    _FILTER_TOOL,
    _apply_refractory_guardrail,
    _extract_rejected,
    _index_in_range,
    _refractory_window_s,
    filter_onsets_all_instruments,
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
    for (gt, gc), (et, ec) in zip(got, expected, strict=True):
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
    for (gt, gc), (et, ec) in zip(got, expected, strict=True):
        assert gc == ec
        assert math.isclose(gt, et, abs_tol=0.005)


def test_structure_path_excludes_bar_drift_from_note_tick() -> None:
    # Bar 0 drifted +0.1s: a hit the detector placed at 1.000s sits 0.1s past
    # the bar's REAL downbeat, so the note tick must encode only 0.9s of
    # sub-grid micro-timing. The drift rides separately (transcription.json
    # barDrift) and the editor re-applies it per bar; baking it into the tick
    # too would double-count. Same hit on a non-drifted bar still reads 1.0s.
    bar0 = _bar(0, 0.0)
    bar0.drift_sec = 0.1
    structure = SimpleNamespace(
        bars=[bar0, _bar(1, 2.0)],
        initial_tempo=120.0,
        initial_time_signature=(4, 4),
    )
    onsets = {"k": [_c(1.000, bar=0)]}
    midi = onsets_to_midi_bytes(onsets, initial_tempo_bpm=120.0, structure=structure)
    events = midi_bytes_to_events(midi)
    assert len(events) == 1
    assert math.isclose(events[0].time, 0.9, abs_tol=0.005)


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
    """Build a tool-use response with `rejected_onsets` items.

    Accepts either bare ints (auto-wrapped with reason=`noise`) for tests
    that only care about the index, or pre-shaped dicts for tests that
    exercise the reason path.
    """
    items = [
        {"index": r, "reason": "noise"} if isinstance(r, int) else r
        for r in rejected
    ]
    block = SimpleNamespace(
        type="tool_use", name=_FILTER_TOOL["name"],
        input={"rejected_onsets": items},
    )
    return SimpleNamespace(content=[block])


def test_extract_rejected_dedupes_valid_indices() -> None:
    # 5 onsets; the duplicate `2` collapses (last-wins by dict key), but
    # every index is in range. Now returns a dict of {index: info}.
    out = _extract_rejected(_resp([0, 2, 2, 4]), n=5)
    assert set(out.keys()) == {0, 2, 4}
    assert all(info["reason"] == "noise" for info in out.values())


def test_extract_rejected_parses_double_of_for_double_trigger() -> None:
    out = _extract_rejected(
        _resp([{"index": 1, "reason": "double_trigger", "double_of": 0}]),
        n=3,
    )
    assert out[1] == {"reason": "double_trigger", "reason_text": None, "double_of": 0}


def test_extract_rejected_double_trigger_without_double_of_is_allowed() -> None:
    # A missing double_of is not a hard error; the refractory guardrail
    # resolves an unverifiable double_trigger by keeping the onset.
    out = _extract_rejected(
        _resp([{"index": 1, "reason": "double_trigger"}]), n=3,
    )
    assert out[1] == {"reason": "double_trigger", "reason_text": None}
    assert "double_of" not in out[1]


def test_extract_rejected_double_of_must_be_integer() -> None:
    with pytest.raises(RuntimeError, match="double_of.*integer"):
        _extract_rejected(
            _resp([{"index": 1, "reason": "double_trigger", "double_of": "x"}]),
            n=3,
        )


def test_extract_rejected_ignores_double_of_on_non_double_trigger() -> None:
    out = _extract_rejected(
        _resp([{"index": 0, "reason": "bleed", "double_of": 1}]), n=3,
    )
    assert out[0] == {"reason": "bleed", "reason_text": None}


def _dt(double_of: int | None = None):
    info: dict = {"reason": "double_trigger", "reason_text": None}
    if double_of is not None:
        info["double_of"] = double_of
    return info


def test_guardrail_overturns_widely_spaced_snare_double() -> None:
    # The real-world case: a snare hit ~217 ms after a stronger one, which
    # the LLM mislabelled a double_trigger. Snare's window is 30 ms, so the
    # rejection is overturned and the hit is kept.
    ordered = [_c(143.152), _c(143.369)]
    rejected = {1: _dt(double_of=0)}
    after, overturned = _apply_refractory_guardrail(ordered, rejected, "s")
    assert after == {}
    assert set(overturned) == {1}


def test_guardrail_keeps_genuine_close_snare_double() -> None:
    # 25 ms apart, below the 30 ms snare window: a plausible re-fire, so
    # the LLM's rejection stands.
    ordered = [_c(1.000), _c(1.025)]
    rejected = {1: _dt(double_of=0)}
    after, overturned = _apply_refractory_guardrail(ordered, rejected, "s")
    assert set(after) == {1}
    assert overturned == {}


def test_guardrail_kick_window_is_wider_than_default() -> None:
    assert _refractory_window_s("k") > _refractory_window_s("s")
    # 40 ms apart: stands for kick (window 55 ms) but would be overturned
    # on snare (window 30 ms).
    ordered = [_c(2.000), _c(2.040)]
    rejected = {1: _dt(double_of=0)}
    kick_after, _ = _apply_refractory_guardrail(ordered, rejected, "k")
    snare_after, _ = _apply_refractory_guardrail(ordered, dict(rejected), "s")
    assert set(kick_after) == {1}
    assert snare_after == {}


def test_guardrail_overturns_when_double_of_missing_or_invalid() -> None:
    ordered = [_c(1.000), _c(1.005), _c(1.010)]
    # No double_of, out-of-range, and self-reference are all unverifiable.
    rejected = {0: _dt(), 1: _dt(double_of=9), 2: _dt(double_of=2)}
    after, overturned = _apply_refractory_guardrail(ordered, rejected, "s")
    assert after == {}
    assert set(overturned) == {0, 1, 2}


def test_guardrail_leaves_non_double_trigger_reasons_untouched() -> None:
    ordered = [_c(1.000), _c(5.000)]
    rejected = {
        0: {"reason": "bleed", "reason_text": None},
        1: {"reason": "noise", "reason_text": None},
    }
    after, overturned = _apply_refractory_guardrail(ordered, rejected, "s")
    assert after == rejected
    assert overturned == {}


def test_extract_rejected_returns_reason_codes() -> None:
    out = _extract_rejected(
        _resp([
            {"index": 0, "reason": "bleed"},
            {"index": 1, "reason": "double_trigger"},
            {"index": 2, "reason": "noise"},
            {
                "index": 3,
                "reason": "custom",
                "reason_text": "looks like a stick click on the rim",
            },
        ]),
        n=4,
    )
    assert out[0] == {"reason": "bleed", "reason_text": None}
    assert out[1] == {"reason": "double_trigger", "reason_text": None}
    assert out[2] == {"reason": "noise", "reason_text": None}
    assert out[3] == {
        "reason": "custom",
        "reason_text": "looks like a stick click on the rim",
    }


def test_extract_rejected_optional_text_for_standard_reason() -> None:
    out = _extract_rejected(
        _resp([{"index": 0, "reason": "bleed", "reason_text": "from snare"}]),
        n=2,
    )
    assert out[0] == {"reason": "bleed", "reason_text": "from snare"}


def test_extract_rejected_raises_on_unknown_reason() -> None:
    with pytest.raises(RuntimeError, match="invalid.*reason"):
        _extract_rejected(
            _resp([{"index": 0, "reason": "weird"}]), n=3,
        )


def test_extract_rejected_raises_when_custom_missing_text() -> None:
    with pytest.raises(RuntimeError, match="custom.*reason_text"):
        _extract_rejected(
            _resp([{"index": 0, "reason": "custom"}]), n=3,
        )


def test_extract_rejected_raises_when_custom_text_is_blank() -> None:
    with pytest.raises(RuntimeError, match="custom.*reason_text"):
        _extract_rejected(
            _resp([{"index": 0, "reason": "custom", "reason_text": "   "}]),
            n=3,
        )


def test_extract_rejected_raises_on_out_of_range() -> None:
    # Per CLEANROOM_SPEC §11.14: malformed tool responses are surfaced
    # (HTTP 502 via StageError) rather than silently corrected, so the
    # operator notices when the model emits invalid indices instead of
    # receiving a quietly-degraded filter pass.
    with pytest.raises(RuntimeError, match="out-of-range"):
        _extract_rejected(_resp([0, 9]), n=5)


def test_extract_rejected_raises_on_negative_index() -> None:
    with pytest.raises(RuntimeError, match="out-of-range"):
        _extract_rejected(_resp([-1]), n=5)


def test_extract_rejected_raises_on_non_integer_item() -> None:
    with pytest.raises(RuntimeError, match="invalid `index`"):
        _extract_rejected(
            _resp([{"index": "x", "reason": "noise"}]), n=5,
        )


def test_extract_rejected_raises_when_no_tool_block() -> None:
    # The filter LLM is forced to use the tool channel; the absence of a
    # tool_use block means the model refused or otherwise misbehaved.
    # Treat that as a model failure rather than "no rejections."
    empty = SimpleNamespace(content=[SimpleNamespace(type="text", text="hi")])
    with pytest.raises(RuntimeError, match="no tool_use block"):
        _extract_rejected(empty, n=3)


def test_skip_pitches_short_circuits_when_all_pitches_skipped() -> None:
    """`hihat_split` LLM-vets `h`/`H` upstream of the filter pass; if those
    were the only instruments with onsets, the filter pool should not
    even try to submit work (which would require an API key)."""
    cands = {
        "h": [_c(0.0, bar=0)],
        "H": [_c(0.5, bar=0)],
    }
    structure = SimpleNamespace(
        bars=[_bar(0, 0.0)],
        initial_tempo=120.0,
        initial_time_signature=(4, 4),
    )
    kept, reasons = filter_onsets_all_instruments(
        cands,
        structure,  # type: ignore[arg-type]
        skip_pitches={"h", "H"},
    )
    assert kept == {}
    assert reasons == {}
