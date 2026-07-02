"""Tempo map building + beat-grid regularization + trailing-bar padding.

`_finalize_bar_tempos` is the entry point: it partitions the beats into
maximal constant / linear-ramp `TempoSegment`s (`_segment_beats` +
`_greedy_segment_ranges` + `_fit_range` + boundary refinement), snaps each
run's beats onto its fitted curve to kill the DBN's frame-quantization wobble,
derives the per-bar `drift_sec` channel, and projects the summary tempo fields.
`_pad_trailing_bars` extends the bar timeline past the last detected beat so
onsets in a fadeout still land in a bar.

Depends on `beats_types` + `beats_structure`.
"""
from __future__ import annotations

import numpy as np

from app.pipeline.beats_structure import (
    _rebuild_bar_fields,
    _reference_bars,
    _robust_initial_tempo,
)
from app.pipeline.beats_types import (
    _RAMP_EPS_BPM,
    BarInfo,
    BeatStructure,
    BeatTick,
    TempoSegment,
)

# Largest per-beat residual (in beats) a single constant- or ramp-tempo
# fit may leave before a segment is split. Frame-quantization noise sits
# well under this (≤ ~0.05 beat even at BT's ~43 fps); a real sectional
# tempo step or a second ramp blows past it because its ordinal error
# accumulates across the section rather than cancelling.
_SEGMENT_FIT_TOL_BEATS = 0.2

# A trailing segment shorter than this many beats is folded back into its
# predecessor, so every segment has enough support for a stable ramp fit
# (~one bar). Only affects the tail; interior splits are tempo-driven.
_MIN_SEG_BEATS = 4

# Median-filter width (in bars) for the per-bar drift channel. The median
# rejects an isolated per-bar frame-quantization spike while preserving a
# sustained drift level (a one-bar step keeps every later bar shifted), so a
# real sub-threshold drift survives where a moving average would blur it.
_DRIFT_SMOOTHING_WINDOW = 3

# Below this much drift (peak abs, seconds) a song is treated as on-grid and
# `barDrift` is zeroed, so the metronomic case ships a clean all-zero channel
# instead of the couple-ms residue the median leaves. Genuine drift clears it.
_DRIFT_DEADBAND_SEC = 0.003

# Beats averaged on each side of a candidate boundary when locating the
# exact beat a sudden tempo step happened (the greedy split lands a beat or
# two late). Averaging a few beats smooths the ±1-frame quantization noise
# on the raw inter-beat gaps so the change-point search isn't fooled by it.
_BOUNDARY_REFINE_HALF = 3

# A relocated step's mean gap-jump must exceed the within-side gap noise by
# this factor, else the greedy boundary stands (don't chase noise on a
# spurious split or a near-flat transition).
_BOUNDARY_REFINE_PROMINENCE = 2.0


def _resample_beats_uniform(
    structure: BeatStructure, ref_start: int, tempo_bpm: float
) -> None:
    """Snap every beat onto a uniform pulse at `tempo_bpm`, in place.

    Used only once a song is judged constant-tempo. The DBN emits
    frame-quantized beat times, so the detected pulse jitters ±1 frame
    around the true (even) grid; deriving anything per-bar from those
    spans manufactures tempo wobble. Replacing the times with a single
    regular pulse removes that jitter while preserving the grid's phase.

    Phase is taken from the reliable reference beats (`bar_index >=
    ref_start`, i.e. excluding a short anacrusis bar 0 whose beats are
    unreliable, mirroring `_fit_constant_or_ramp`): we hold the slope
    fixed at `60 / tempo_bpm` and solve only the intercept that
    least-squares-fits those beats, then lay the uniform grid over *all*
    beats by their contiguous ordinal (extrapolating back over bar 0).
    A uniform shift the alignment passes already applied is preserved,
    since the intercept is fit to the post-alignment times.

    `structure.beats` and each `bar.beats` hold the same `BeatTick`
    objects, so mutating `beat.time` here updates both; the caller then
    runs `_rebuild_bar_fields` to refresh per-bar start/end times.
    """
    beats = structure.beats
    if len(beats) < 2 or tempo_bpm <= 0:
        return
    period = 60.0 / tempo_bpm
    idx = np.arange(len(beats), dtype=np.float64)
    times = np.asarray([b.time for b in beats], dtype=np.float64)
    ref_mask = np.asarray(
        [b.bar_index >= ref_start for b in beats], dtype=bool
    )
    if not ref_mask.any():
        ref_mask[:] = True
    # Least-squares intercept with the slope pinned to `period`.
    intercept = float(np.mean(times[ref_mask] - period * idx[ref_mask]))
    for i, beat in enumerate(beats):
        beat.time = intercept + period * i


def _finalize_bar_tempos(structure: BeatStructure) -> None:
    """Build the tempo map (`structure.tempo_segments`) and regularize the
    beat grid to it, in place.

    The DBN emits frame-quantized beat times, so the detected pulse
    jitters ±1 frame around the true grid; anything derived per bar from
    those spans (notably `onsets_midi._bar_duration_tempo_bpm`, the MIDI
    tempo the editor reads) manufactures a 2-3 BPM wobble between nearly
    every bar even on a dead-steady song.

    `_segment_beats` partitions the beats into maximal runs each described
    by one tempo law, **constant** (ordinal linear in time) or a
    **linear-in-time ramp** (ordinal quadratic in time, so `bpm²` is linear
    in beat, matching `src/schema/dsl/tempo.ts`), and snaps each run's
    beats onto its fitted curve. So a metronomic song collapses to one
    constant segment (jitter gone), a single accelerando to one ramp, and a
    "steady verse → push into the chorus → steady" shape to
    constant/ramp/constant, the localized ramp the old single-global fit
    missed.

    `bar.tempo_bpm` / `initial_tempo` / `has_tempo_changes` become
    projections of the resulting `tempo_segments`. Bar 0 is often a short
    anacrusis whose beats are unreliable, so the dominant single-constant
    case excludes it from the phase fit (`ref_start`) while still extending
    the regularized grid back over it.
    """
    bars = structure.bars
    if not bars:
        return
    ref_start = 1 if len(bars) >= 2 else 0

    # The real (post-alignment) downbeat time of each bar, captured before
    # regularization snaps the grid uniform, so we can record how far the
    # performance drifted from the clean tempo (`bar.drift_sec`).
    real_downbeat = {
        id(bar): float(bar.beats[0].time) if bar.beats else bar.start_time
        for bar in bars
    }

    if len(structure.beats) >= 2:
        segments = _segment_beats(structure, ref_start)
        # Clean beats → refresh per-bar start/end/tempo from them.
        _rebuild_bar_fields(structure)
    else:
        # Degenerate (0-1 beats): one flat segment at the bar's tempo.
        bpm = float(bars[min(ref_start, len(bars) - 1)].tempo_bpm)
        segments = [TempoSegment(0, max(0, len(structure.beats) - 1), bpm, bpm)]

    # Per-bar drift = real downbeat - regularized downbeat (a cumulative
    # offset). The raw residual mixes the genuine performance drift we want
    # to keep, including a sub-threshold one-bar step the tempo model never
    # turned into a BPM change, with the DBN's per-beat frame-quantization
    # noise (±~half a frame). A *median* filter separates them: a real drift
    # is a *sustained* level (every bar after a long bar stays shifted), which
    # the median preserves, while an isolated per-bar quantization spike is an
    # outlier the median rejects. So `model + drift` faithfully reconstructs
    # the recording's bar lines without the constant-song bars wobbling.
    raw_drift = np.asarray(
        [
            real_downbeat.get(id(bar), bar.start_time)
            - (float(bar.beats[0].time) if bar.beats else bar.start_time)
            for bar in bars
        ],
        dtype=np.float64,
    )
    half = _DRIFT_SMOOTHING_WINDOW // 2
    denoised = np.array(
        [
            float(np.median(raw_drift[max(0, i - half):min(len(bars), i + half + 1)]))
            for i in range(len(bars))
        ]
    )
    # Only the residual detection noise (a couple ms after the median) sits
    # below the deadband; zero it so a metronomic song ships a clean all-zero
    # `barDrift`. Genuine drift comfortably clears it and is kept verbatim.
    if denoised.size and float(np.max(np.abs(denoised))) < _DRIFT_DEADBAND_SEC:
        denoised[:] = 0.0
    for bar, d in zip(bars, denoised, strict=True):
        bar.drift_sec = float(d)

    structure.tempo_segments = segments
    # initial_tempo from the robust first-bars median, not segments[0], a
    # glitchy early bar's wild BPM otherwise poisons the start tempo.
    structure.initial_tempo = (
        _robust_initial_tempo(_reference_bars(structure.bars))
        if structure.bars else (segments[0].start_bpm if segments else 120.0)
    )
    structure.has_tempo_changes = len(segments) > 1 or any(
        s.is_ramp() for s in segments
    )


def _segment_beats(
    structure: BeatStructure, ref_start: int
) -> list[TempoSegment]:
    """Partition the beats into constant/ramp segments, regularizing in place.

    Greedy left-to-right growth: a segment extends while a single quadratic
    (ordinal vs time) still fits within `_SEGMENT_FIT_TOL_BEATS`; when the
    next beat would break the fit, the segment closes and a new one starts.
    A constant region is the zero-curvature special case (the quadratic's
    `t²` term ≈ 0), a ramp the curved case, and a hard tempo step forces a
    split because no single quadratic spans both sides. Each segment's beats
    are then snapped onto its own fitted curve, removing the DBN's
    frame-quantization wobble while preserving real motion.
    """
    beats = structure.beats
    n = len(beats)
    times = np.asarray([b.time for b in beats], dtype=np.float64)
    ranges = _greedy_segment_ranges(times)
    fits = [_fit_range(times, a, b) for (a, b) in ranges]
    if len(ranges) >= 2:
        ranges, fits = _refine_step_boundaries(times, ranges, fits)

    # Whole song is one constant segment: use the anacrusis-aware uniform
    # resample (period + phase from the reliable reference beats) for the
    # maximum-precision pin the metronomic case deserves.
    if len(ranges) == 1 and not fits[0][0]:
        ref = np.asarray(
            [bt.bar_index >= ref_start for bt in beats], dtype=bool
        )
        if ref.sum() < 3:
            ref[:] = True
        slope = float(np.polyfit(times[ref], np.arange(n)[ref], 1)[0])
        tempo = 60.0 * slope if slope > 0 else fits[0][1]
        _resample_beats_uniform(structure, ref_start, tempo)
        return [TempoSegment(0, n - 1, tempo, tempo)]

    segments: list[TempoSegment] = []
    for (a, b), (_is_ramp, b_start, b_end, c2, c1, c0) in zip(
        ranges, fits, strict=True
    ):
        _regularize_beats_quadratic(beats, a, b, c2, c1, c0)
        segments.append(TempoSegment(a, b, b_start, b_end))
    _enforce_monotonic_times(beats)
    return segments


def _greedy_segment_ranges(times: np.ndarray) -> list[tuple[int, int]]:
    """Greedy breakpoints: maximal `(start, end)` ordinal ranges (inclusive)
    each fit by one quadratic within `_SEGMENT_FIT_TOL_BEATS`.

    Ranges of < 3 beats fit any quadratic exactly, so growth is only gated
    once a range is long enough to expose curvature. A too-short trailing
    range is folded back into its predecessor so a ramp/constant fit always
    has enough support.
    """
    n = times.size
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = start
        while end + 1 < n:
            a, b = start, end + 1
            if b - a >= 2:
                resid = _quad_resid(
                    times[a:b + 1], np.arange(a, b + 1, dtype=np.float64)
                )
                if resid > _SEGMENT_FIT_TOL_BEATS:
                    break
            end += 1
        ranges.append((start, end))
        start = end + 1
    if len(ranges) >= 2 and ranges[-1][1] - ranges[-1][0] + 1 < _MIN_SEG_BEATS:
        a_prev, _ = ranges[-2]
        ranges[-2] = (a_prev, ranges[-1][1])
        ranges.pop()
    return ranges


def _quad_resid(t_slice: np.ndarray, n_slice: np.ndarray) -> float:
    """Max abs residual (in beats) of an ordinal-vs-time quadratic fit."""
    if t_slice.size < 3:
        return 0.0
    coeffs = np.polyfit(t_slice, n_slice, 2)
    return float(np.max(np.abs(n_slice - np.polyval(coeffs, t_slice))))


def _refine_step_boundaries(
    times: np.ndarray,
    ranges: list[tuple[int, int]],
    fits: list[tuple[bool, float, float, float, float, float]],
) -> tuple[list[tuple[int, int]], list[tuple[bool, float, float, float, float, float]]]:
    """Nudge each sudden-step boundary onto the beat the tempo actually changed.

    Greedy growth stops a beat or two *after* a step (the segment absorbs
    some of the new tempo before the single-quadratic residual breaks
    tolerance), so the raw boundary sits late. We slide a candidate split
    across a window around it and score each by the change in delta,
    `|mean(gap_right) - mean(gap_left)|` over `_BOUNDARY_REFINE_HALF` beats
    each side (the smoothed derivative of the inter-beat gap), then move the
    boundary to where that peaks.

    Two guards keep it from misfiring. A genuine step shows an *interior*
    peak (the score rises onto the step and falls past it); a constant->ramp
    transition's score instead climbs monotonically into the ramp, peaking
    at the window edge, so we relocate only on an interior peak (this also
    means we needn't scope by fit class, which is good because the late
    greedy segment swallowed the step and is fit as a faux-ramp). A
    prominence gate vs the within-side gap noise is the second guard.

    Returns refreshed `(ranges, fits)`; the two affected segments are re-fit.
    """
    gaps = np.diff(times)
    half = _BOUNDARY_REFINE_HALF
    out_ranges = [list(r) for r in ranges]
    out_fits = list(fits)
    for j in range(len(out_ranges) - 1):
        a = out_ranges[j][0]
        right_b = out_ranges[j + 1][1]
        cur = out_ranges[j + 1][0]
        lo = a + _MIN_SEG_BEATS
        hi = right_b - _MIN_SEG_BEATS + 1
        if hi - lo < 3:
            continue
        cands = list(range(lo, hi))
        scores: list[float] = []
        for c in cands:
            left = gaps[max(a, c - half):c]
            right = gaps[c:min(right_b, c + half)]
            if left.size < 2 or right.size < 2:
                scores.append(-1.0)
            else:
                scores.append(abs(float(right.mean()) - float(left.mean())))
        best_i = int(np.argmax(scores))
        best_c = cands[best_i]
        best_score = scores[best_i]
        # Interior peak only (a monotonic rise to an edge is a ramp, not a
        # step), and the step must clear the within-side gap noise.
        interior = 0 < best_i < len(cands) - 1
        local = gaps[max(a, cur - 2 * half):min(right_b, cur + 2 * half)]
        noise = float(np.std(np.diff(local))) if local.size > 2 else 0.0
        if (
            interior
            and best_c != cur
            and best_score > _BOUNDARY_REFINE_PROMINENCE * (noise + 1e-9)
        ):
            out_ranges[j] = [a, best_c - 1]
            out_ranges[j + 1] = [best_c, right_b]
            out_fits[j] = _fit_range(times, a, best_c - 1)
            out_fits[j + 1] = _fit_range(times, best_c, right_b)
    return [(r[0], r[1]) for r in out_ranges], out_fits


def _fit_range(
    times: np.ndarray, a: int, b: int
) -> tuple[bool, float, float, float, float, float]:
    """Fit one range and classify it.

    Returns `(is_ramp, start_bpm, end_bpm, c2, c1, c0)` for the quadratic
    `ordinal = c2·t² + c1·t + c0`. A range is a ramp only when its endpoint
    tempi differ by ≥ `_RAMP_EPS_BPM`, the implied tempo stays positive, and
    it doesn't reverse inside the span (the parabola's vertex lies outside
    `[t0, t1]`); otherwise it's pinned constant (clean line refit) so
    frame-quantization curvature never manufactures a hairline ramp.
    """
    seg_t = times[a:b + 1]
    seg_n = np.arange(a, b + 1, dtype=np.float64)
    m = b - a + 1
    if m >= 4:
        c2, c1, c0 = (float(c) for c in np.polyfit(seg_t, seg_n, 2))
    elif m >= 2:
        c1, c0 = (float(c) for c in np.polyfit(seg_t, seg_n, 1))
        c2 = 0.0
    else:
        # Single-beat range (only via a degenerate input); treat as 120 BPM.
        c2, c1, c0 = 0.0, 2.0, -2.0 * a
    t0, t1 = float(seg_t[0]), float(seg_t[-1])
    b_start = 60.0 * (2 * c2 * t0 + c1)
    b_end = 60.0 * (2 * c2 * t1 + c1)
    vertex = -c1 / (2 * c2) if c2 != 0 else float("inf")
    monotonic = not (min(t0, t1) < vertex < max(t0, t1))
    is_ramp = (
        abs(b_end - b_start) >= _RAMP_EPS_BPM
        and monotonic
        and b_start > 0
        and b_end > 0
    )
    if not is_ramp:
        if m >= 2:
            c1, c0 = (float(c) for c in np.polyfit(seg_t, seg_n, 1))
            c2 = 0.0
        bpm = 60.0 * c1
        return (False, bpm, bpm, c2, c1, c0)
    return (True, b_start, b_end, c2, c1, c0)


def _regularize_beats_quadratic(
    beats: list[BeatTick], a: int, b: int, c2: float, c1: float, c0: float
) -> None:
    """Snap beats `[a, b]` onto the curve `ordinal = c2·t² + c1·t + c0`.

    Inverts the quadratic per beat: the regularized time for beat `k` is the
    root of `c2·t² + c1·t + (c0 − k)` nearest the beat's detected time (so
    phase + monotonic order are preserved). Degenerates to the linear solve
    when `c2` is ~0 (a constant segment → a uniform pulse).
    """
    for k in range(a, b + 1):
        beat = beats[k]
        target = float(k)
        if abs(c2) < 1e-12:
            beat.time = (target - c0) / c1
            continue
        disc = max(c1 * c1 - 4 * c2 * (c0 - target), 0.0)
        root = disc**0.5
        r1 = (-c1 + root) / (2 * c2)
        r2 = (-c1 - root) / (2 * c2)
        beat.time = r1 if abs(r1 - beat.time) <= abs(r2 - beat.time) else r2


def _enforce_monotonic_times(beats: list[BeatTick]) -> None:
    """Guarantee strictly increasing beat times after per-segment fits.

    Independent per-segment regularization can in principle leave a seam
    where a segment boundary's two beats touch or invert; nudge any such
    beat just past its predecessor so onset attribution stays well-defined.
    With sub-tolerance fits this is a no-op in practice.
    """
    for k in range(1, len(beats)):
        if beats[k].time <= beats[k - 1].time:
            beats[k].time = beats[k - 1].time + 1e-6


def _pad_trailing_bars(structure: BeatStructure, duration_seconds: float) -> None:
    """Extend the bar timeline forward to cover `duration_seconds`.

    The beat tracker tends to cut off well before the audio's
    actual end (it stops emitting confident beats in silence / fadeouts).
    Without padding, onsets after the last detected beat get attributed
    to the last real bar with arbitrarily large `beat_in_bar` values
    (e.g. 4.9 or 6.2 inside a 4/4 bar). Instead, synthesise additional
    bars after the last one that reuse its tempo and time signature, so
    those onsets land in dedicated bars with a sensible position.

    The synthetic bars share `time_signature` and `tempo_bpm` with the
    last detected bar; their beats are placed at regular intervals so
    `position()` can interpolate normally.
    """
    if not structure.bars:
        return
    last = structure.bars[-1]
    if not last.beats:
        return
    # Estimate beat spacing from the last bar; fall back to 60/tempo.
    if len(last.beats) >= 2:
        beat_gap = (last.beats[-1].time - last.beats[0].time) / max(
            len(last.beats) - 1, 1
        )
    else:
        beat_gap = 60.0 / max(last.tempo_bpm, 1.0)
    if beat_gap <= 0:
        return

    count = last.time_signature[0]
    unit = last.time_signature[1]
    bar_duration = beat_gap * count

    # Synthesise bars until we've covered `duration_seconds`. Cap the
    # count to prevent runaway loops on broken inputs.
    cursor = last.end_time
    next_index = last.index + 1
    MAX_PAD_BARS = 256
    while cursor < duration_seconds and next_index - last.index <= MAX_PAD_BARS:
        new_beats: list[BeatTick] = []
        for i in range(count):
            new_beats.append(
                BeatTick(
                    time=cursor + i * beat_gap,
                    beat_in_bar=i + 1,
                    bar_index=next_index,
                )
            )
        new_bar = BarInfo(
            index=next_index,
            start_time=cursor,
            end_time=cursor + bar_duration,
            beats=new_beats,
            time_signature=(count, unit),
            tempo_bpm=last.tempo_bpm,
            feel=last.feel,
        )
        structure.bars.append(new_bar)
        structure.beats.extend(new_beats)
        cursor += bar_duration
        next_index += 1
