"""Tests for the time-signature heuristic and trailing-bar padding in beats.py.

These exercise pure-Python branches that don't require Beat This! or audio
I/O - we construct a BeatStructure by hand or call the small helpers
directly.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.beats import (
    _DRIFT_DEADBAND_SEC,
    BarInfo,
    BeatStructure,
    BeatTick,
    _beats_downbeats_to_raw,
    _choose_time_signature,
    _coarse_offset_from_envelope,
    _finalize_bar,
    _finalize_bar_tempos,
    _pad_trailing_bars,
    _raw_to_structure,
    _summarize,
    align_beats_to_onsets,
)
from app.pipeline.onsets_midi import compute_bar_tick_grid


def _ramp_beat_times(b0: float, b1: float, n_beats: int) -> list[float]:
    """Absolute times of `n_beats + 1` beats for a linear-in-time tempo
    ramp from `b0` to `b1` BPM, matching `src/schema/dsl/tempo.ts`.

    Tempo is linear in time, so `bpm² is linear in beat`:
    `bpm(d) = sqrt(b0² + (b1²-b0²)·d/L)` over `L = n_beats` beats, and the
    time to reach beat `d` is `120·L·(bpm(d)-b0)/(b1²-b0²)` seconds (the
    closed-form integral; the constant case `b0==b1` reduces to `d·60/b0`).
    """
    L = float(n_beats)
    times: list[float] = []
    for d in range(n_beats + 1):
        if abs(b1 - b0) < 1e-9:
            times.append(d * 60.0 / b0)
        else:
            bpm_d = (b0 * b0 + (b1 * b1 - b0 * b0) * (d / L)) ** 0.5
            times.append(120.0 * L * (bpm_d - b0) / (b1 * b1 - b0 * b0))
    return times

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


# ---------- coarse envelope phase alignment ----------


def _pulse_env(frame_times, centers, width=0.01, height=1.0):
    env = np.zeros_like(frame_times)
    for c in centers:
        env += height * np.exp(-((frame_times - c) ** 2) / (2 * width**2))
    return env


def test_coarse_offset_recovers_a_known_phase_error() -> None:
    # Grid beats at 0.5, 1.0, ... but the real drum hits land 0.1 s later
    # (a ~2-3 slot systematic error). The coarse search should recover +0.1.
    ft = np.arange(0.0, 10.0, 0.002)
    beats = np.array([0.5 * k for k in range(1, 18)], dtype=float)
    env = _pulse_env(ft, beats + 0.1)
    off = _coarse_offset_from_envelope(
        beats, env, ft, max_shift=1.0, step=0.002,
        center_penalty=0.15, prominence=1.10,
    )
    assert off == pytest.approx(0.1, abs=0.005)


def test_coarse_offset_prefers_the_smaller_competing_shift() -> None:
    # Two equally-tall alignment peaks, one near 0 and one far. The centre
    # taper must pick the near one (don't yank the grid a long way for a
    # tie), exercising the small-|δ| bias.
    ft = np.arange(0.0, 3.0, 0.002)
    env = _pulse_env(ft, [1.05, 1.45])  # beat at 1.0 -> δ=+0.05 or +0.45
    off = _coarse_offset_from_envelope(
        np.array([1.0]), env, ft, max_shift=0.5, step=0.002,
        center_penalty=0.15, prominence=1.10,
    )
    assert off == pytest.approx(0.05, abs=0.005)


def test_coarse_offset_returns_zero_on_a_flat_envelope() -> None:
    # No clear pulse -> the prominence gate rejects any shift.
    ft = np.arange(0.0, 5.0, 0.002)
    env = np.ones_like(ft)
    off = _coarse_offset_from_envelope(
        np.array([0.5 * k for k in range(1, 10)]), env, ft,
        max_shift=1.0, step=0.002, center_penalty=0.15, prominence=1.10,
    )
    assert off == 0.0


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
    assert structure.bars[0].start_time == pytest.approx(0.0, abs=1e-9)
    assert structure.bars[0].tempo_bpm == pytest.approx(160.0, abs=0.1)


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


# ---------- constant-tempo grid denoising (_finalize_bar_tempos) ----------


def _frame_quantized_constant_structure(
    *, tempo_bpm: float, fps: float, n_bars: int, count: int = 4
) -> BeatStructure:
    """Build a BeatStructure for a perfectly constant-tempo song whose
    beat times are frame-quantized to `fps`, mimicking a DBN tracker.

    Frame quantization is the real-world jitter source: each beat lands on
    the nearest `1/fps` frame, so consecutive bars' downbeat-to-downbeat
    spans differ by up to ±1 frame even though the underlying tempo never
    changes. That ±1-frame span wobble is what `_bar_duration_tempo_bpm`
    (the MIDI tempo source) turns into a 2-3 BPM swing per bar.
    """
    beat_gap = 60.0 / tempo_bpm
    beats: list[BeatTick] = []
    for i in range(n_bars * count):
        true_t = i * beat_gap
        q_t = round(true_t * fps) / fps  # snap to the frame grid
        beats.append(
            BeatTick(time=q_t, beat_in_bar=(i % count) + 1, bar_index=i // count)
        )
    bars: list[BarInfo] = []
    for b_idx in range(n_bars):
        bb = beats[b_idx * count:(b_idx + 1) * count]
        bars.append(_finalize_bar(b_idx, bb))
    structure = _summarize(beats, bars)
    return structure


def test_constant_tempo_midi_grid_is_flat_despite_frame_jitter() -> None:
    # Regression for the BPM jitter seen in the editor: a metronomic song
    # whose DBN beat times are frame-quantized must emit ONE flat MIDI
    # tempo across every bar, not a 2-3 BPM swing per bar. The MIDI tempo
    # the editor reads comes from `compute_bar_tick_grid`'s per-bar
    # `_bar_duration_tempo_bpm` (downbeat-to-downbeat span), so it's the
    # value we assert on.
    structure = _frame_quantized_constant_structure(
        tempo_bpm=120.0, fps=43.07, n_bars=16
    )

    # Sanity: the raw per-bar spans really do jitter before finalization,
    # otherwise the test wouldn't be exercising the bug.
    _, raw_tempos, _, _ = compute_bar_tick_grid(structure, 120.0)
    assert max(raw_tempos) - min(raw_tempos) > 0.5  # multi-BPM swing present

    _finalize_bar_tempos(structure)

    assert structure.has_tempo_changes is False
    _, midi_tempos, _, _ = compute_bar_tick_grid(structure, structure.initial_tempo)
    # Every bar must now carry the same MIDI tempo (no per-bar set_tempo
    # churn => no tempoEvents => flat BPM in the editor).
    assert max(midi_tempos) - min(midi_tempos) < 1e-6
    # ...and it must be the true tempo, not drifted by the quantization.
    assert midi_tempos[0] == pytest.approx(120.0, abs=0.2)


def test_constant_tempo_resamples_beats_to_uniform_pulse() -> None:
    # The denoising works by snapping the beat grid itself to a uniform
    # pulse, so onset attribution stays self-consistent with the flat
    # tempo map. Inter-beat gaps must all be equal afterward.
    structure = _frame_quantized_constant_structure(
        tempo_bpm=100.0, fps=43.07, n_bars=12
    )
    _finalize_bar_tempos(structure)

    gaps = np.diff([b.time for b in structure.beats])
    assert float(gaps.max() - gaps.min()) < 1e-9
    assert float(np.mean(gaps)) == pytest.approx(60.0 / 100.0, abs=1e-3)


def test_genuine_tempo_change_is_not_flattened() -> None:
    # A real per-bar accelerando must keep its rising tempo; the constant
    # denoising only fires when a single flat line fits the whole song.
    beats: list[BeatTick] = []
    bars: list[BarInfo] = []
    t = 0.0
    n_bars = 12
    for b_idx in range(n_bars):
        # Ramp from 100 -> ~150 BPM over the song.
        bpm = 100.0 + b_idx * 5.0
        gap = 60.0 / bpm
        bb = [
            BeatTick(time=t + j * gap, beat_in_bar=j + 1, bar_index=b_idx)
            for j in range(4)
        ]
        beats.extend(bb)
        bars.append(_finalize_bar(b_idx, bb))
        t += 4 * gap
    structure = _summarize(beats, bars)
    _finalize_bar_tempos(structure)

    assert structure.has_tempo_changes is True
    # Tempo still rises across the song (contour preserved, not pinned flat).
    assert structure.bars[-1].tempo_bpm - structure.bars[1].tempo_bpm > 20.0


# ---------- tempo map / segmentation (TempoSegment) ----------


def _structure_from_beat_times(times: list[float], count: int = 4) -> BeatStructure:
    beats = [
        BeatTick(time=t, beat_in_bar=(i % count) + 1, bar_index=i // count)
        for i, t in enumerate(times)
    ]
    n_bars = (len(times) + count - 1) // count
    bars = [
        _finalize_bar(b, beats[b * count:(b + 1) * count])
        for b in range(n_bars)
        if beats[b * count:(b + 1) * count]
    ]
    return _summarize(beats, bars)


def test_constant_song_is_one_constant_segment() -> None:
    structure = _frame_quantized_constant_structure(
        tempo_bpm=120.0, fps=43.07, n_bars=16
    )
    _finalize_bar_tempos(structure)

    assert len(structure.tempo_segments) == 1
    seg = structure.tempo_segments[0]
    assert seg.is_ramp() is False
    assert seg.start_bpm == pytest.approx(120.0, abs=0.2)
    assert seg.end_bpm == pytest.approx(seg.start_bpm, abs=1e-9)
    assert structure.has_tempo_changes is False


def test_single_accelerando_is_one_ramp_segment() -> None:
    # A clean linear-in-time accelerando 110 -> 140 BPM over 64 beats, with
    # realistic frame quantization on top. It must collapse to ONE ramp
    # segment whose endpoints recover the true tempi, not a staircase.
    times = _ramp_beat_times(110.0, 140.0, 64)
    fps = 43.07
    times = [round(t * fps) / fps for t in times]  # frame-quantize
    structure = _structure_from_beat_times(times)
    _finalize_bar_tempos(structure)

    assert len(structure.tempo_segments) == 1
    seg = structure.tempo_segments[0]
    assert seg.is_ramp() is True
    assert seg.start_bpm == pytest.approx(110.0, abs=2.0)
    assert seg.end_bpm == pytest.approx(140.0, abs=2.0)
    assert structure.has_tempo_changes is True


def test_ramp_regularized_beats_are_monotonic_and_accelerating() -> None:
    times = _ramp_beat_times(100.0, 150.0, 48)
    fps = 43.07
    times = [round(t * fps) / fps for t in times]
    structure = _structure_from_beat_times(times)
    _finalize_bar_tempos(structure)

    gaps = np.diff([b.time for b in structure.beats])
    # Strictly accelerating => every inter-beat gap shorter than the last,
    # with no frame-quantization wobble surviving.
    assert np.all(gaps > 0)
    assert np.all(np.diff(gaps) < 1e-9)


def _concat_ramp_blocks(blocks: list[tuple[float, float, int]]) -> list[float]:
    """Stitch `(start_bpm, end_bpm, n_beats)` blocks into one continuous,
    phase-consistent beat-time list (each block's first beat coincides with
    the previous block's last)."""
    times = [0.0]
    cursor = 0.0
    for b0, b1, nb in blocks:
        seg = _ramp_beat_times(b0, b1, nb)  # nb + 1 points, seg[0] == 0
        for t in seg[1:]:
            times.append(cursor + t)
        cursor = times[-1]
    return times


def _quantize(times: list[float], fps: float = 43.07) -> list[float]:
    return [round(t * fps) / fps for t in times]


def test_localized_ramp_between_steady_sections() -> None:
    # Steady verse -> push into the chorus -> steady chorus. The old
    # single-global-ramp fit could not represent flat-curved-flat and fell
    # back to a staircase; multi-segment must surface a real ramp in the
    # middle with constant sections around it.
    times = _quantize(
        _concat_ramp_blocks([(120.0, 120.0, 24), (120.0, 140.0, 12), (140.0, 140.0, 24)])
    )
    structure = _structure_from_beat_times(times)
    _finalize_bar_tempos(structure)

    segs = structure.tempo_segments
    assert structure.has_tempo_changes is True
    assert any(s.is_ramp() for s in segs)
    # Overall the map climbs from ~120 to ~140.
    assert min(s.start_bpm for s in segs) == pytest.approx(120.0, abs=4.0)
    assert max(s.end_bpm for s in segs) == pytest.approx(140.0, abs=4.0)
    # Beats stay strictly monotonic across the segment seams.
    gaps = np.diff([b.time for b in structure.beats])
    assert np.all(gaps > 0)


def test_two_separate_accelerandos() -> None:
    # Two distinct gradual events separated by a long steady stretch. The
    # dominant plateau means a single straight tempo line can't span both
    # ramps (unlike a near-collinear up-flat-up), so the map must keep them
    # as separate ramps with a constant section between.
    times = _quantize(
        _concat_ramp_blocks(
            [
                (112.0, 132.0, 8),
                (132.0, 132.0, 40),
                (132.0, 152.0, 8),
            ]
        )
    )
    structure = _structure_from_beat_times(times)
    _finalize_bar_tempos(structure)

    ramps = [s for s in structure.tempo_segments if s.is_ramp()]
    assert len(ramps) >= 2
    assert structure.has_tempo_changes is True


def test_hard_tempo_step_splits_into_two_constants() -> None:
    # An abrupt 120 -> 140 jump (no gradual change) must split into two
    # constant segments at the step, not be smeared into one ramp.
    times = _quantize(
        _concat_ramp_blocks([(120.0, 120.0, 24), (140.0, 140.0, 24)])
    )
    structure = _structure_from_beat_times(times)
    _finalize_bar_tempos(structure)

    segs = structure.tempo_segments
    assert len(segs) >= 2
    assert structure.has_tempo_changes is True
    # The map spans from ~120 to ~140 with the bulk of each side flat.
    assert min(s.start_bpm for s in segs) == pytest.approx(120.0, abs=5.0)
    assert max(s.end_bpm for s in segs) == pytest.approx(140.0, abs=5.0)


def test_step_boundary_refined_to_the_change_beat() -> None:
    # 16 beats at 120 then 16 at 150, a sudden step at beat ~16. Greedy
    # growth overshoots the split into the new tempo; the change-point
    # refinement must pull the boundary back onto the beat the tempo
    # actually changed, so the faster segment starts at ~beat 16/17 (not the
    # late greedy split ~20) and reads a clean ~150 with no contamination.
    times = _concat_ramp_blocks([(120.0, 120.0, 16), (150.0, 150.0, 16)])
    structure = _structure_from_beat_times(times)
    _finalize_bar_tempos(structure)

    fast = [s for s in structure.tempo_segments if s.start_bpm > 135.0]
    assert len(fast) == 1
    assert fast[0].start_beat == pytest.approx(16, abs=1)
    assert fast[0].start_bpm == pytest.approx(150.0, abs=2.0)
    # And the slow side stays a clean ~120 right up to the step.
    slow = [s for s in structure.tempo_segments if s.start_bpm < 135.0]
    assert slow[0].start_bpm == pytest.approx(120.0, abs=2.0)


def test_constant_song_has_negligible_drift() -> None:
    # A metronomic song: the per-bar drift channel must stay ~0 (the
    # smoothing + deadband strip the DBN frame-quantization noise), so the
    # waveform isn't spuriously stretched.
    structure = _frame_quantized_constant_structure(
        tempo_bpm=120.0, fps=100.0, n_bars=16
    )
    _finalize_bar_tempos(structure)
    drift = [b.drift_sec for b in structure.bars]
    assert max(abs(d) for d in drift) < 0.012


def _wandering_beat_times(amp: float, period: int, n: int) -> list[float]:
    """120-BPM beats whose times wander sinusoidally by ±`amp` seconds, a
    pre-click recording that smoothly speeds up and slows down. The wander is
    sub-threshold, so it's judged one segment; it survives as drift, not as a
    tempo change."""
    return [
        0.5 * i + amp * float(np.sin(2.0 * np.pi * i / period)) for i in range(n)
    ]


def test_sustained_drift_is_captured() -> None:
    # A smoothly wandering (drifting) tempo: the model flattens it to one
    # tempo, but the drift channel must record the ±wander so the editor can
    # still align the bar lines + waveform to the recording.
    structure = _structure_from_beat_times(_wandering_beat_times(0.06, 48, 96))
    _finalize_bar_tempos(structure)
    drift = [b.drift_sec for b in structure.bars]
    # The wander (well above the deadband) is preserved, not smoothed away.
    assert max(drift) - min(drift) > 0.03


def test_sub_threshold_long_bar_drift_not_zeroed() -> None:
    # One bar runs 60 ms long (the drummer held it); the tempo model never
    # emits a BPM change (the drift stays sub-threshold), but the drift
    # channel must NOT zero it - `model + drift` has to still reconstruct the
    # real bar lines so the waveform + notes stay aligned to the recording.
    times: list[float] = []
    t = 0.0
    for i in range(48):
        times.append(t)
        t += 0.5
        if i == 23:  # extra 60 ms before bar 6's downbeat
            t += 0.06
    structure = _structure_from_beat_times(times)
    _finalize_bar_tempos(structure)

    assert structure.has_tempo_changes is False  # still one flat tempo
    # Not zeroed: the drift carries real signal.
    assert max(abs(b.drift_sec) for b in structure.bars) > _DRIFT_DEADBAND_SEC
    # And `regularized_start + drift` reconstructs the real grid, so the long
    # bar shows up as clearly the longest reconstructed bar (the LS fit
    # absorbs some of it into the tempo, but the drift recovers the rest).
    recon = np.array([b.start_time + b.drift_sec for b in structure.bars])
    durs = np.diff(recon)
    assert float(durs.max() - np.median(durs)) > 0.025


def test_transcription_carries_bar_drift() -> None:
    from app.pipeline.transcription import build_transcription

    structure = _structure_from_beat_times(_wandering_beat_times(0.06, 48, 96))
    _finalize_bar_tempos(structure)
    payload = build_transcription(structure)

    bar_drift = payload["barDrift"]
    assert len(bar_drift) == len(structure.bars)
    assert max(bar_drift) - min(bar_drift) > 0.03


def test_step_refinement_survives_frame_quantization() -> None:
    times = _quantize(_concat_ramp_blocks([(126.0, 126.0, 20), (96.0, 96.0, 20)]))
    structure = _structure_from_beat_times(times)
    _finalize_bar_tempos(structure)

    fast = [s for s in structure.tempo_segments if s.start_bpm > 110.0]
    slow = [s for s in structure.tempo_segments if s.start_bpm < 110.0]
    assert fast and slow
    # The step (beat ~20) is recovered within a beat despite ±1-frame jitter.
    assert min(s.start_beat for s in slow) == pytest.approx(20, abs=2)


# ---------- Phase 2: ramp-aware stepwise MIDI set_tempo ----------


def _render_set_tempos(structure: BeatStructure) -> list[tuple[int, float]]:
    """Render the structure to onsets MIDI and read back `(abs_tick, bpm)`
    for every `set_tempo` event."""
    import io

    import mido

    from app.pipeline.onsets_midi import onsets_to_midi_bytes

    class _Onset:
        def __init__(self, t: float, bar: int) -> None:
            self.time = t
            self.strength = 1.0
            self.bar = bar

    onsets = {"k": [_Onset(b.start_time, b.index) for b in structure.bars]}
    raw = onsets_to_midi_bytes(
        onsets, initial_tempo_bpm=structure.initial_tempo, structure=structure
    )
    mid = mido.MidiFile(file=io.BytesIO(raw))
    out: list[tuple[int, float]] = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                out.append((abs_tick, 60_000_000 / msg.tempo))
    return out


def test_constant_song_emits_single_midi_tempo() -> None:
    # Regression guard: the per-beat tempo emission must still collapse a
    # metronomic song to exactly one tempo value (no per-bar churn).
    structure = _frame_quantized_constant_structure(
        tempo_bpm=120.0, fps=43.07, n_bars=16
    )
    _finalize_bar_tempos(structure)
    tempos = _render_set_tempos(structure)
    distinct = {round(bpm, 2) for _, bpm in tempos}
    assert len(distinct) == 1
    assert next(iter(distinct)) == pytest.approx(120.0, abs=0.3)


def test_ramp_emits_stepwise_increasing_midi_tempo() -> None:
    # A ramp must survive the MIDI as >= RAMP_MIN_POINTS (4) monotonically
    # increasing set_tempo steps so `detectRampRuns` can rebuild it on the
    # MIDI-only path.
    times = _quantize(_ramp_beat_times(110.0, 150.0, 64))
    structure = _structure_from_beat_times(times)
    _finalize_bar_tempos(structure)
    tempos = _render_set_tempos(structure)

    # Exclude the tick-0 lead-in event: it's the pre-roll tempo (`from_midi`
    # consumes it as the lead-in duration, not part of the song's contour),
    # and a sub-frame regularization pre-roll can back-solve it to a very
    # high BPM by design (see compute_bar_tick_grid).
    bpms = [bpm for tick, bpm in tempos if tick > 0]
    assert len(bpms) >= 4  # enough monotonic steps for detectRampRuns
    assert all(b >= a - 1e-6 for a, b in zip(bpms, bpms[1:], strict=False))
    assert bpms[0] == pytest.approx(110.0, abs=4.0)
    assert bpms[-1] == pytest.approx(150.0, abs=4.0)


# ---------- Phase 3: transcription.json tempoMap ----------


def test_transcription_constant_has_no_tempo_events() -> None:
    from app.pipeline.transcription import TRANSCRIPTION_FORMAT, build_transcription

    structure = _frame_quantized_constant_structure(
        tempo_bpm=120.0, fps=43.07, n_bars=16
    )
    _finalize_bar_tempos(structure)
    payload = build_transcription(structure)

    assert payload["format"] == TRANSCRIPTION_FORMAT
    tempo_map = payload["tempoMap"]
    assert tempo_map["initial_bpm"] == pytest.approx(120.0, abs=0.3)
    assert tempo_map["events"] == []  # frontend -> one bpm, zero tempoEvents


def test_transcription_ramp_is_one_linear_event() -> None:
    from app.pipeline.transcription import build_transcription

    times = _quantize(_ramp_beat_times(110.0, 150.0, 64))
    structure = _structure_from_beat_times(times)
    _finalize_bar_tempos(structure)
    tempo_map = build_transcription(structure)["tempoMap"]

    events = tempo_map["events"]
    assert len(events) == 1
    ev = events[0]
    assert ev["shape"] == "linear"
    assert isinstance(ev["bpm"], dict)
    assert ev["bpm"]["start"] == pytest.approx(110.0, abs=4.0)
    assert ev["bpm"]["end"] == pytest.approx(150.0, abs=4.0)
    assert ev["bpm"]["end_tick"] > ev["tick"]
    assert tempo_map["initial_bpm"] == pytest.approx(110.0, abs=4.0)


def test_transcription_localized_ramp_has_ramp_then_step() -> None:
    from app.pipeline.transcription import build_transcription

    times = _quantize(
        _concat_ramp_blocks([(120.0, 120.0, 24), (120.0, 140.0, 12), (140.0, 140.0, 24)])
    )
    structure = _structure_from_beat_times(times)
    _finalize_bar_tempos(structure)
    events = build_transcription(structure)["tempoMap"]["events"]

    # A linear ramp event somewhere, and the map climbs to ~140.
    assert any(isinstance(e["bpm"], dict) for e in events)
    ticks = [e["tick"] for e in events]
    assert ticks == sorted(ticks)  # tick-ascending


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
    assert structure.bars[0].tempo_bpm == pytest.approx(structure.bars[1].tempo_bpm, abs=1e-9)


# ---------- downbeat smoothing guards (Beat This! mis-detections) ----------

def _structure_from_bars(bar_lengths: list[int], spb: float = 0.5) -> BeatStructure:
    """Build beats + downbeats from an as-detected list of bar lengths (beats
    per bar at `spb` s/beat, downbeat at each bar start), run them through the
    production conversion (smoothing included), return the structure."""
    beats: list[float] = []
    downbeats: list[float] = []
    t = 0.0
    for n in bar_lengths:
        downbeats.append(round(t, 3))
        for _ in range(n):
            beats.append(round(t, 3))
            t += spb
    return _raw_to_structure(_beats_downbeats_to_raw(beats, downbeats))


def _sigs(s: BeatStructure) -> list[tuple[int, int]]:
    return [b.time_signature for b in s.bars]


def test_smooth_splits_merged_bar_no_multiples():
    # A missed downbeat merges two 4/4 bars into an 8-beat bar -> split back.
    s = _structure_from_bars([4, 4, 8, 4, 4])
    assert _sigs(s) == [(4, 4)] * 6
    assert s.has_time_sig_changes is False


def test_smooth_splits_multiple_of_three_base():
    # 6 beats against a 3/4 majority is a merged pair, not 6/8.
    s = _structure_from_bars([3, 3, 6, 3, 3])
    assert _sigs(s) == [(3, 4)] * 6
    assert s.has_time_sig_changes is False


def test_smooth_merges_fragmented_bar_two_plus_two():
    # An extra downbeat fragments one 4/4 bar into 2+2 -> merge back.
    s = _structure_from_bars([4, 4, 2, 2, 4, 4])
    assert _sigs(s) == [(4, 4)] * 5
    assert s.has_time_sig_changes is False


def test_smooth_merges_fragmented_bar_one_plus_three():
    s = _structure_from_bars([4, 4, 1, 3, 4, 4])
    assert _sigs(s) == [(4, 4)] * 5
    assert s.has_time_sig_changes is False


def test_smooth_preserves_sustained_odd_meter():
    # A real, sustained 3/4 section in a 4/4 song must survive untouched.
    s = _structure_from_bars([4, 4, 3, 3, 3, 4, 4])
    assert _sigs(s) == [(4, 4), (4, 4), (3, 4), (3, 4), (3, 4), (4, 4), (4, 4)]
    assert s.has_time_sig_changes is True


def test_smooth_preserves_genuine_six_eight():
    # A lone 6-beat bar is NOT a multiple of the 4/4 base -> the bar is kept as
    # 6/8 (fast), but a single off-meter bar is not a *sustained* change, so the
    # song-level flag stays False (robust-summary semantics).
    s = _structure_from_bars([4, 4, 6, 4, 4])
    assert (6, 8) in _sigs(s)
    assert s.has_time_sig_changes is False


def test_smooth_noop_without_majority_meter():
    # No meter holds a majority -> leave the grid exactly as detected.
    s = _structure_from_bars([4, 3, 5, 4, 3])
    assert _sigs(s) == [(4, 4), (3, 4), (5, 4), (4, 4), (3, 4)]


def _structure_from_bars_dur(bars: list[tuple[int, float]]) -> BeatStructure:
    """Like _structure_from_bars but each bar is (n_beats, duration_seconds),
    so a bar can be densely subdivided (local tempo flip) without changing its
    span. Beats are evenly spaced within each bar."""
    beats: list[float] = []
    downbeats: list[float] = []
    t = 0.0
    for n, dur in bars:
        downbeats.append(round(t, 4))
        step = dur / n
        for _ in range(n):
            beats.append(round(t, 4))
            t += step
    return _raw_to_structure(_beats_downbeats_to_raw(beats, downbeats))


def test_smooth_decimates_doubled_tempo_bar_to_three():
    # 3/4 song; one busy bar tracked at 2x tempo (6 beats in the SAME 1.5 s
    # span) -> must become a single 3/4 bar (decimate), NOT split, NOT 6/8.
    s = _structure_from_bars_dur(
        [(3, 1.5), (3, 1.5), (6, 1.5), (3, 1.5), (3, 1.5)]
    )
    assert _sigs(s) == [(3, 4)] * 5     # 5 bars, not 6 -> not split
    assert s.has_time_sig_changes is False


def test_smooth_splits_merged_bar_when_duration_is_doubled():
    # Same beat count (6) but TWICE the span -> genuinely two merged bars,
    # so split into 3+3 (six bars), the opposite of the decimate case.
    s = _structure_from_bars_dur(
        [(3, 1.5), (3, 1.5), (6, 3.0), (3, 1.5), (3, 1.5)]
    )
    assert _sigs(s) == [(3, 4)] * 6     # 6 bars -> split
    assert s.has_time_sig_changes is False


def test_smooth_decimates_doubled_tempo_bar_in_four_four():
    s = _structure_from_bars_dur(
        [(4, 2.0), (4, 2.0), (8, 2.0), (4, 2.0), (4, 2.0)]
    )
    assert _sigs(s) == [(4, 4)] * 5
    assert s.has_time_sig_changes is False


# ---------- robust global summary (modal meter, median tempo) ----------

def _ts_bars(counts: list[int], beat_gap: float = 0.5) -> list[BarInfo]:
    bars: list[BarInfo] = []
    t = 0.0
    for i, c in enumerate(counts):
        bars.append(_make_bar(i, t, beat_gap, count=c))
        t += c * beat_gap
    return bars


def test_summary_meter_is_modal_not_a_single_bar():
    # bar 1 (past the anacrusis) is a lone 2-beat glitch; song is 4/4.
    s = _summarize([], _ts_bars([4, 2, 4, 4, 4, 4]))
    assert s.initial_time_signature == (4, 4)
    assert s.has_time_sig_changes is False  # one off-meter bar is noise


def test_summary_flags_only_sustained_meter_change():
    s = _summarize([], _ts_bars([4, 4, 4, 3, 3, 3, 4, 4]))  # real 3/4 run
    assert s.initial_time_signature == (4, 4)
    assert s.has_time_sig_changes is True


def test_summary_initial_tempo_robust_to_glitch_bar():
    bars = _ts_bars([4, 4, 4, 4, 4])  # all 120 BPM (beat_gap 0.5)
    bars[1].tempo_bpm = 600.0          # a fragmented bar's wild BPM
    s = _summarize([], bars)
    assert 100.0 < s.initial_tempo < 140.0  # median ignores the outlier


def test_beats_downbeats_raw_handles_leading_pickup():
    # Beats start before the first downbeat (db[0] > 0): a 2-beat pickup, then
    # 4/4 bars. Must not crash; pickup beats land in bar 0.
    beats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
    downbeats = [1.0, 3.0, 5.0]  # first two beats are an upbeat pickup
    s = _raw_to_structure(_beats_downbeats_to_raw(beats, downbeats))
    assert len(s.beats) == len(beats)            # no beats lost
    assert s.bars[0].beats[0].time == 0.0        # pickup kept in bar 0
    downbeat_times = [b.time for b in s.beats if b.beat_in_bar == 1]
    assert 1.0 in downbeat_times and 3.0 in downbeat_times


def test_beats_downbeats_raw_drops_far_downbeat():
    # A downbeat 60 ms from the nearest beat (> tol 0.05) is ignored, not
    # snapped onto an unrelated beat.
    beats = [0.0, 0.5, 1.0, 1.5]
    downbeats = [0.0, 1.06]
    raw = _beats_downbeats_to_raw(beats, downbeats)
    assert [int(r[1]) for r in raw] == [1, 2, 3, 4]  # one bar; far downbeat dropped
