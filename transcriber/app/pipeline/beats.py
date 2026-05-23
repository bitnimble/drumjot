"""Beat tracking + downbeat detection + per-bar feel analysis.

This module replaces the older `tempo.py` + `quantize.py` pair. Rather
than assuming a constant tempo + fixed grid (1/16 by default), we use
madmom's RNN+DBN downbeat tracker to extract per-beat anchors from the
audio. Everything downstream then works in **beat-relative** time -
that is, "1/3 of the way through beat 2 of bar 3" rather than
"slot 47 of a 1/16 grid".

Why it matters:

- Tempo changes are handled naturally: each beat has its own absolute
  time, so a song that accelerates or hits a click drop still maps
  onsets onto the right beats.
- Time-signature changes are detected from the downbeat-classifier
  output (gaps of 4 beats between downbeats -> 4/4; 7 beats -> 7/4 or
  7/8 depending on tempo).
- Triplet and swing feel become a property of intra-beat fractions and
  are encoded as the bar's `feel` field; the LLM can then emit
  `(...)_N` groups vs straight grids per bar.
- The grid is implicit and per-bar, not a global constant - the LLM
  decides how to position each onset in DSL space based on the bar's
  feel and the onset's beat fraction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from app.config import settings

log = logging.getLogger(__name__)

# Common beat-counts the downbeat tracker should consider. Covers all
# popular-music time signatures (and a few odd-meter ones).
BEATS_PER_BAR_CANDIDATES = [2, 3, 4, 5, 6, 7]

# Feel classification labels emitted on each `BarInfo`.
Feel = Literal["straight16", "straight8", "triplet", "shuffle", "sparse", "mixed"]


@dataclass
class BeatTick:
    """One beat detected by the tracker."""

    time: float
    beat_in_bar: int  # 1-indexed; 1 = downbeat
    bar_index: int    # 0-indexed bar this beat belongs to


@dataclass
class BarInfo:
    """A single bar after downbeat classification."""

    index: int                          # 0-indexed
    start_time: float
    end_time: float
    beats: list[BeatTick]               # at least 1
    time_signature: tuple[int, int]     # (count, unit) - unit is always 4 by default
    tempo_bpm: float                    # avg over this bar's beats
    feel: Feel = "straight16"           # filled in once onsets are known


@dataclass
class BeatStructure:
    """Sequence of beats + bar groupings + global summary fields."""

    beats: list[BeatTick] = field(default_factory=list)
    bars: list[BarInfo] = field(default_factory=list)
    initial_tempo: float = 120.0
    initial_time_signature: tuple[int, int] = (4, 4)
    has_tempo_changes: bool = False
    has_time_sig_changes: bool = False

    def position(self, t: float) -> tuple[int, float] | None:
        """Map an absolute time `t` (seconds) to `(bar_index, beat_in_bar)`.

        The returned `beat_in_bar` is a float in `[1.0, num_beats + 1)`
        that linearly maps the bar's audio span to its meter — i.e.
        position is uniform across the bar, not anchored to the
        tracker's individual beat times. The downbeat is exactly `1.0`,
        the next bar's downbeat is exactly `num_beats + 1` (and reported
        on that next bar as `1.0`); in 4/4, `2.0` is the position 1/4 of
        the way through the bar regardless of micro-timing wobbles in
        the detected beat 2. Using the bar's own audio span rather than
        the bracketing inter-beat gap means the value is stable against
        per-beat detection noise and reads on the same scale as the
        meter ("2.5" is always the midpoint between beat 2 and beat 3,
        across the bar uniformly).

        Returns None if `t` falls outside the tracked range. Trailing
        audio after the last detected beat is handled upstream
        (`analyze_beats` pads synthetic bars to cover the full duration),
        so the in-bar branch picks them up. Onsets fractionally before
        the first beat get extrapolated onto bar 0 with a value < 1.0.
        """
        if not self.beats or not self.bars:
            return None
        first_t = self.beats[0].time
        last_t = self.beats[-1].time
        if t > last_t:
            # With trailing-bar padding this only fires for onsets PAST
            # the audio duration - drop them rather than guessing.
            return None
        if t < first_t:
            # Pre-roll: extrapolate using bar 0's expected beat spacing
            # so the value reads as a fraction below 1.0 on bar 0.
            bar0 = self.bars[0]
            span = bar0.end_time - bar0.start_time
            if span <= 0:
                return None
            num_beats = max(int(bar0.time_signature[0]), 1)
            return (bar0.index, 1.0 + (t - bar0.start_time) / span * num_beats)

        # Binary search for the enclosing beat to find the bar.
        lo, hi = 0, len(self.beats) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.beats[mid].time <= t:
                lo = mid
            else:
                hi = mid - 1
        bar_idx = self.beats[lo].bar_index
        if bar_idx < 0 or bar_idx >= len(self.bars):
            return None
        bar = self.bars[bar_idx]
        span = bar.end_time - bar.start_time
        if span <= 0:
            return (bar.index, float(self.beats[lo].beat_in_bar))
        num_beats = max(int(bar.time_signature[0]), 1)
        return (bar.index, 1.0 + (t - bar.start_time) / span * num_beats)


# ---------- Detection ----------

def analyze_beats(
    audio_path: Path,
    duration_seconds: float | None = None,
    align_onsets: list[tuple[float, float]] | None = None,
) -> BeatStructure:
    """Run beat + downbeat detection on the audio at `audio_path`.

    Uses madmom (RNN + DBN) for the actual ML; falls back to a
    librosa-based heuristic if madmom isn't importable, so the rest of
    the pipeline degrades gracefully rather than failing outright.

    When `duration_seconds` is supplied, the returned structure is padded
    with synthetic bars after the last detected beat so that every
    timestamp within the audio falls inside some bar. This stops onsets
    that occur after the final detected beat from piling up at an
    out-of-range `beat_in_bar` value on the last real bar (see
    `position`).

    When `align_onsets` is supplied (list of `(time, strength)` tuples),
    each detected beat is snapped to the strongest nearby drum onset
    before bars are padded. Neural beat trackers (BT especially) tend
    to report beat times ~50 ms past the actual transient because the
    activation peak lags the strike; snapping to onsets removes that
    systematic lag without changing bar phase or beat count.
    """
    try:
        tracker = settings.beat_tracker
        if tracker == "beat_transformer":
            structure = _beat_transformer_beats(audio_path)
        else:
            structure = _madmom_beats(audio_path)
    except Exception as exc:
        log.warning(
            "%s beat tracking failed (%s); falling back to librosa "
            "(no downbeat classification - assuming 4/4 throughout)",
            settings.beat_tracker,
            exc,
        )
        structure = _librosa_fallback(audio_path)
    if align_onsets:
        align_beats_to_onsets(structure, align_onsets)
    _finalize_bar_tempos(structure)
    if duration_seconds is not None and duration_seconds > 0:
        _pad_trailing_bars(structure, duration_seconds)
    return structure


# When BT's tempo head supplies a prediction, we narrow the DBN's
# tempo search to a window around it rather than letting it pick any
# tempo in [55, 215]. ±15% leaves headroom for the head's own error
# (it's classification with int-bin quantization so off-by-a-few BPM
# is normal) while still killing the half-time / double-time
# alternatives that share the same beat positions.
BT_TEMPO_WINDOW = 0.15


def _madmom_beats(audio_path: Path) -> BeatStructure:
    # Imports kept local so a madmom install issue doesn't block the
    # rest of the service from starting up.
    from madmom.features.downbeats import RNNDownBeatProcessor

    log.info("madmom: extracting downbeat activations from %s", audio_path.name)
    activations = RNNDownBeatProcessor()(str(audio_path))
    return _decode_activations(activations, fps=100)


def _beat_transformer_beats(audio_path: Path) -> BeatStructure:
    """Beat Transformer activations -> shared DBN postprocessor.

    The DBN is reused from madmom; only the source of per-frame
    activations + an optional BPM constraint changes. See
    `pipeline/beat_transformer.py` for the preprocessing / model /
    activation details. When BT's tempo head returns a trusted BPM
    estimate, we narrow the DBN's tempo search around it to break
    half/double-time ambiguities that the activations alone can't
    resolve (the DBN can't distinguish 80 BPM 2/4 from 160 BPM 4/4
    from the peak positions — they're identical).
    """
    from app.pipeline.beat_transformer import FPS, extract_activations

    activations, predicted_bpm = extract_activations(audio_path)
    if predicted_bpm is not None:
        min_bpm = predicted_bpm * (1 - BT_TEMPO_WINDOW)
        max_bpm = predicted_bpm * (1 + BT_TEMPO_WINDOW)
        log.info(
            "DBN tempo search narrowed to [%.1f, %.1f] BPM by BT tempo head (predicted=%.1f)",
            min_bpm, max_bpm, predicted_bpm,
        )
    else:
        min_bpm = None
        max_bpm = None
    return _decode_activations(activations, fps=FPS, min_bpm=min_bpm, max_bpm=max_bpm)


def _decode_activations(
    activations: np.ndarray,
    fps: float,
    min_bpm: float | None = None,
    max_bpm: float | None = None,
) -> BeatStructure:
    """Decode an Nx2 (beat, downbeat) activation array into a BeatStructure.

    The DBN's job is to pick the most-likely beat positions + bar phase
    given the per-frame activations. It doesn't care which network
    produced them, so swapping in BT activations here is a drop-in.
    `fps` must match the frame rate at which the activations were
    sampled (madmom RNN = 100, BT = ~43.07). `min_bpm` / `max_bpm` are
    optional tempo constraints; when omitted, madmom's defaults
    (~55 / ~215) apply.
    """
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor

    dbn_kwargs: dict = dict(
        beats_per_bar=BEATS_PER_BAR_CANDIDATES,
        fps=fps,
    )
    if min_bpm is not None:
        dbn_kwargs["min_bpm"] = min_bpm
    if max_bpm is not None:
        dbn_kwargs["max_bpm"] = max_bpm
    tracker = DBNDownBeatTrackingProcessor(**dbn_kwargs)
    raw = tracker(activations)  # Nx2 array: (time, beat_pos_in_bar)
    if raw.size == 0:
        log.warning("beat tracker returned no beats (fps=%s)", fps)
        return BeatStructure()
    return _from_madmom_raw(raw)


def _from_madmom_raw(raw: np.ndarray) -> BeatStructure:
    """Convert madmom's Nx2 output into our typed BeatStructure."""
    beats: list[BeatTick] = []
    bars: list[BarInfo] = []
    current_bar_beats: list[BeatTick] = []
    current_bar_index = -1

    for row in raw:
        time = float(row[0])
        beat_pos = int(round(float(row[1])))  # 1-indexed in bar
        if beat_pos == 1:
            # Close out the prior bar (if any)
            if current_bar_beats:
                bars.append(_finalize_bar(current_bar_index, current_bar_beats))
            current_bar_index += 1
            current_bar_beats = []
        if current_bar_index < 0:
            # Beats before the first detected downbeat - park them in bar 0.
            current_bar_index = 0
        tick = BeatTick(
            time=time, beat_in_bar=max(1, beat_pos), bar_index=current_bar_index
        )
        beats.append(tick)
        current_bar_beats.append(tick)

    if current_bar_beats:
        bars.append(_finalize_bar(current_bar_index, current_bar_beats))

    return _summarize(beats, bars)


def _finalize_bar(index: int, beats: list[BeatTick]) -> BarInfo:
    """Build a BarInfo from the beats falling inside it."""
    if not beats:
        return BarInfo(
            index=index,
            start_time=0.0,
            end_time=0.0,
            beats=[],
            time_signature=(4, 4),
            tempo_bpm=120.0,
        )
    start = beats[0].time
    # `end_time` is the start of the next bar; approximate with the last
    # beat's time + average gap. Will be overwritten by `_summarize` once
    # we know the next bar's actual start.
    gap = beats[-1].time - beats[-2].time if len(beats) >= 2 else 60.0 / 120.0
    end = beats[-1].time + max(gap, 0.0)
    count = len(beats)
    # Average tempo over this bar's beats
    if len(beats) >= 2:
        gaps = np.diff([b.time for b in beats])
        tempo_bpm = float(60.0 / np.mean(gaps))
    else:
        tempo_bpm = 120.0
    time_sig = _choose_time_signature(count, tempo_bpm)
    return BarInfo(
        index=index,
        start_time=start,
        end_time=end,
        beats=list(beats),
        time_signature=time_sig,
        tempo_bpm=tempo_bpm,
    )


def _choose_time_signature(count: int, tempo_bpm: float) -> tuple[int, int]:
    """Pick a `(numerator, denominator)` for a bar with `count` beats.

    Madmom only reports a count of beats per bar; it doesn't tell us
    whether they're quarter notes (simple meter) or dotted quarters
    (compound meter). For 4-, 5- and 7-beat bars 'quarter notes' is
    nearly always right in popular music. For 6 beats it's genuinely
    ambiguous: a slow rock waltz is 6/4, but a fast jazz waltz or
    Irish jig is 6/8 with the same beat count.

    ASSUMPTION (refine when we have ground-truth data): treat 6-beat
    bars as 6/8 whenever the detected tempo is on the fast side
    (>= 100 BPM in the beat-tracker's "quarter-note" sense). Anything
    slower stays at 6/4. 12-beat bars get the same split for 12/8 vs
    12/4.

    Returns `(count, unit)` where `unit` is the note value of the
    denominator (4 = quarter, 8 = eighth).
    """
    if count == 6 and tempo_bpm >= 100.0:
        return (6, 8)
    if count == 12 and tempo_bpm >= 100.0:
        return (12, 8)
    return (count, 4)


def _summarize(beats: list[BeatTick], bars: list[BarInfo]) -> BeatStructure:
    """Fill end_time of each bar from the next bar's start, and compute
    global summary fields (initial tempo, change flags).

    Bar 0 is excluded from the global summary when ≥2 bars are present:
    madmom often starts counting partway through the first bar (anacrusis
    / pickup), producing a leading bar with fewer beats than the rest.
    Reading `initial_time_signature` off such a bar mislabels the whole
    song (e.g. 3/4 for a 4/4 song with a 3-beat pickup) AND falsely flips
    `has_time_sig_changes` to true because bar 0's `(3,4)` differs from
    every subsequent `(4,4)`. The per-bar `time_signature` on `bars[0]`
    is left alone — it accurately describes the 3 beats that bar holds —
    so anacrusis-aware DSL emission can still distinguish it downstream.
    """
    for i in range(len(bars) - 1):
        bars[i].end_time = bars[i + 1].start_time
    reference_bars = bars[1:] if len(bars) >= 2 else bars
    if reference_bars:
        initial_tempo = reference_bars[0].tempo_bpm
        initial_ts = reference_bars[0].time_signature
    else:
        initial_tempo = 120.0
        initial_ts = (4, 4)

    tempo_set = {round(b.tempo_bpm, 1) for b in reference_bars}
    sig_set = {b.time_signature for b in reference_bars}
    return BeatStructure(
        beats=beats,
        bars=bars,
        initial_tempo=float(initial_tempo),
        initial_time_signature=initial_ts,
        has_tempo_changes=len(tempo_set) > 1,
        has_time_sig_changes=len(sig_set) > 1,
    )


# Width (in bars) of the median filter used to test for *sustained*
# tempo motion. Each bar's raw `60 / mean(gaps)` estimate uses only 3-4
# beat gaps, so 10-20 ms of beat-time variance (normal even after onset
# alignment) translates into 5-10 BPM swings. The median over 5 bars
# strips that high-frequency wobble so what's left is real drift.
TEMPO_SMOOTHING_WINDOW = 5

# A song is treated as having real tempo changes only if its *smoothed*
# bar tempos span more than this much. A 5-bar median of constant-tempo
# audio still wobbles ~3-6 BPM (the variance doesn't fully cancel), so
# the threshold sits above that band; genuine sectional accelerandos
# (band pushing into a chorus, etc.) comfortably exceed it. When the
# song is judged constant, every bar is pinned to one global tempo so
# beats.json and the prompt show a flat line instead of residual jitter.
SUSTAINED_TEMPO_CHANGE_BPM = 8.0


def _global_tempo_from_beats(
    structure: BeatStructure, ref_start: int
) -> float | None:
    """Constant tempo from the *full* detected-beat baseline.

    `_finalize_bar_tempos` otherwise pins a constant song to
    `median(per-bar 60/mean(gaps))` — a median of short, 3-4-gap window
    estimates. Those windows throw away almost all the available
    baseline, and `median(60/mean(x))` is a biased estimator of the true
    rate, so the pinned tempo lands a few tenths of a BPM off. The
    frontend plays that single value at a constant rate, so the error
    becomes a fixed fractional offset and the drum grid drifts linearly
    away from the recording over the track.

    For a genuinely constant tempo the maximum-precision estimate is the
    slope of beat time vs beat ordinal across every detected beat: the
    long baseline shrinks the residual error to ~1/N of a per-bar
    estimate, collapsing the cumulative drift. Bar 0 (often a short
    anacrusis whose beats are unreliable) is excluded for the same
    reason it's excluded from `ref_start`.

    Returns None when there aren't enough clean beats to fit, so the
    caller falls back to the per-bar median (never a regression).
    """
    times = np.asarray(
        [b.time for b in structure.beats if b.bar_index >= ref_start],
        dtype=np.float64,
    )
    if times.size < 2:
        times = np.asarray(
            [b.time for b in structure.beats], dtype=np.float64
        )
    if times.size < 2:
        return None
    # Least-squares slope (seconds per beat) over contiguous beat
    # ordinals. The pipeline already builds bars from a contiguous,
    # regular beat sequence, so ordinal == beat number here.
    idx = np.arange(times.size, dtype=np.float64)
    slope = float(np.polyfit(idx, times, 1)[0])
    if not np.isfinite(slope) or slope <= 0.0:
        return None
    return 60.0 / slope


def _finalize_bar_tempos(structure: BeatStructure) -> None:
    """Pin or smooth per-bar tempos in place and rederive the change flag.

    `_finalize_bar` / `_rebuild_bar_fields` set each `bar.tempo_bpm` to
    `60 / mean(gaps)` over the bar's own 3-4 beat gaps; that estimate
    wobbles 5-10 BPM at typical tempos even on a constant-tempo song.
    The transcribe prompt asks the LLM to emit `{{ bpm: N }}` whenever
    consecutive bars differ by >2 BPM, so the raw wobble produces a
    bpm change between essentially every bar.

    A median filter alone only *reduces* the wobble (a 5-bar median of
    constant-tempo audio still swings ~5 BPM). So instead we use the
    smoothed series only to *decide* whether the song is constant: if
    its span is under `SUSTAINED_TEMPO_CHANGE_BPM`, every bar is pinned
    to a single global tempo (flat beats.json, no spurious `{{ bpm }}`);
    otherwise the smoothed contour is kept so genuine motion survives.

    Replaces the old `len({round(bpm,1) for ...}) > 1` test, which
    flipped `has_tempo_changes` to True on any 0.1 BPM jitter.
    """
    bars = structure.bars
    if not bars:
        return

    # Bar 0 frequently has fewer beats (anacrusis / pickup), so its
    # tempo estimate is unreliable — exclude it from the reference set
    # used for the global tempo + change decision when possible.
    ref_start = 1 if len(bars) >= 2 else 0
    raw = np.asarray([b.tempo_bpm for b in bars], dtype=np.float64)
    global_tempo = float(np.median(raw[ref_start:]))

    half = TEMPO_SMOOTHING_WINDOW // 2
    smoothed = np.empty_like(raw)
    for i in range(len(bars)):
        lo = max(0, i - half)
        hi = min(len(bars), i + half + 1)
        smoothed[i] = float(np.median(raw[lo:hi]))

    ref_smoothed = smoothed[ref_start:]
    span = (
        float(ref_smoothed.max() - ref_smoothed.min())
        if ref_smoothed.size
        else 0.0
    )

    if span < SUSTAINED_TEMPO_CHANGE_BPM:
        # Constant song: prefer the long-baseline fit over the per-bar
        # median so the single emitted bpm matches the recording's true
        # rate and the drum grid doesn't drift against it. Fall back to
        # the median only if there aren't enough beats to fit.
        fitted = _global_tempo_from_beats(structure, ref_start)
        constant_tempo = global_tempo if fitted is None else fitted
        for bar in bars:
            bar.tempo_bpm = constant_tempo
        structure.initial_tempo = constant_tempo
        structure.has_tempo_changes = False
    else:
        for bar, s in zip(bars, smoothed, strict=True):
            bar.tempo_bpm = float(s)
        structure.initial_tempo = float(smoothed[ref_start])
        structure.has_tempo_changes = True


def _pad_trailing_bars(structure: BeatStructure, duration_seconds: float) -> None:
    """Extend the bar timeline forward to cover `duration_seconds`.

    Madmom's downbeat tracker tends to cut off well before the audio's
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


# Minimum fraction of beats that must have a strong drum onset within
# the alignment window before we trust the median offset enough to
# shift the whole grid. Drum stems normally hit this easily (most beats
# coincide with a kick/snare/hat); sparse or heavily-syncopated material
# that falls below it keeps the raw tracker grid rather than risk a
# bogus global shift.
MIN_ALIGN_COVERAGE = 0.30

# Wider search window for the very first detected beat. The DBN has no
# preceding-beat context at the song start, so beat 0 is the noisiest
# tick in the structure -- its delta to the first audible drum hit can
# easily exceed the regular `max_distance` even when the rest of the
# song aligns cleanly. Catching that hit anchors the global shift to
# the song's intended downbeat, fixing the "kick visibly off the first
# beat after import" symptom.
FIRST_BEAT_ANCHOR_WINDOW = 0.10

# How much weight the first-beat-to-kick delta gets when folded into
# the global shift. 0.5 = equal influence with the all-beat median;
# weaker values keep more of the existing behaviour, stronger values
# risk over-correcting songs whose first beat happens to be unusual.
FIRST_BEAT_ANCHOR_WEIGHT = 0.5


def align_beats_to_onsets(
    structure: BeatStructure,
    onsets: list[tuple[float, float]],
    max_distance: float = 0.05,
) -> None:
    """Shift the whole beat grid by the tracker's *systematic* lag.

    Neural beat trackers (Beat Transformer especially) report each beat
    ~30-50 ms after the transient, because the activation peak lags the
    strike. We still want to correct that.

    The previous implementation snapped **each beat independently** to
    the strongest drum onset within ±`max_distance`. That removed the
    lag but also absorbed the drummer's natural micro-timing into the
    grid: every beat's gap to its neighbours changed, so the per-bar
    tempo (`60 / mean(gap)` over a bar's 3-4 gaps) wobbled 5-10 BPM
    even on a dead-steady song — the LLM then emitted a `{{ bpm }}`
    change between nearly every bar.

    Instead, estimate ONE offset — the median over all beats of
    `(nearest strong onset − beat time)` — and shift every beat by it.
    The grid stays exactly as metrically regular as the DBN produced
    it (per-bar tempo is therefore stable), while the systematic lag is
    still removed. A uniform shift leaves inter-beat gaps unchanged, so
    a genuine accelerando the DBN tracked is preserved untouched.

    The first detected beat gets special treatment: we additionally
    search a wider ±`FIRST_BEAT_ANCHOR_WINDOW` for the strongest drum
    hit (the "first kick" in the typical kick-on-1 song). When found,
    its delta is folded into the global shift with
    `FIRST_BEAT_ANCHOR_WEIGHT`. Rationale: the DBN has no preceding
    context for beat 0, so its position is the noisiest in the song
    and a hit 60-90 ms away (outside the regular `max_distance`)
    won't even register as a deltas-list entry; without this anchor the
    median shift leaves beat 0 visibly off the first kick after import.
    Subsequent beats are constrained by the periodic grid so their
    local deltas stay close to the median and don't need this fixup.

    The offset is only applied when enough beats actually had a nearby
    onset (`MIN_ALIGN_COVERAGE`); a handful of coincidental matches
    shouldn't drag the whole grid. `_rebuild_bar_fields` then refreshes
    per-bar `start_time` / `end_time` (which the shift moved) and the
    global tempo fields.
    """
    if not structure.beats or not onsets:
        return
    times = np.asarray([t for t, _ in onsets], dtype=np.float64)
    strengths = np.asarray([s for _, s in onsets], dtype=np.float64)
    order = np.argsort(times)
    times = times[order]
    strengths = strengths[order]

    # Strongest-not-closest: a quiet ghost hi-hat sitting nearer the
    # activation peak shouldn't outrank a louder kick/snare transient
    # slightly further out — the strong transient is the strike that
    # defines the beat.
    deltas: list[float] = []
    for beat in structure.beats:
        lo = int(np.searchsorted(times, beat.time - max_distance, side="left"))
        hi = int(np.searchsorted(times, beat.time + max_distance, side="right"))
        if lo >= hi:
            continue
        j = lo + int(np.argmax(strengths[lo:hi]))
        deltas.append(float(times[j] - beat.time))

    if not deltas:
        log.info(
            "beat alignment: no beats had an onset within ±%.0f ms; "
            "grid left unchanged",
            max_distance * 1000,
        )
        return

    coverage = len(deltas) / len(structure.beats)
    median_offset = float(np.median(deltas))

    # Wider-window anchor for the first detected beat. Fold its delta
    # into the global shift only when the search actually finds a hit;
    # otherwise behave exactly like the median-only path.
    first_beat = structure.beats[0]
    lo_a = int(np.searchsorted(
        times, first_beat.time - FIRST_BEAT_ANCHOR_WINDOW, side="left"
    ))
    hi_a = int(np.searchsorted(
        times, first_beat.time + FIRST_BEAT_ANCHOR_WINDOW, side="right"
    ))
    first_beat_delta: float | None = None
    if hi_a > lo_a:
        j_a = lo_a + int(np.argmax(strengths[lo_a:hi_a]))
        first_beat_delta = float(times[j_a] - first_beat.time)
        offset = (
            (1.0 - FIRST_BEAT_ANCHOR_WEIGHT) * median_offset
            + FIRST_BEAT_ANCHOR_WEIGHT * first_beat_delta
        )
    else:
        offset = median_offset

    if coverage < MIN_ALIGN_COVERAGE:
        log.info(
            "beat alignment: only %.0f%% of beats had a nearby onset "
            "(< %.0f%% required); offset %+.1f ms rejected, grid unchanged",
            coverage * 100, MIN_ALIGN_COVERAGE * 100, offset * 1000,
        )
        return

    for beat in structure.beats:
        beat.time += offset
    if first_beat_delta is not None:
        log.info(
            "beat alignment: shifted all %d beats by %+.1f ms "
            "(median %+.1f ms over %d beat→onset deltas, "
            "anchored to first-beat delta %+.1f ms at weight %.2f, "
            "coverage %.0f%%)",
            len(structure.beats), offset * 1000,
            median_offset * 1000, len(deltas),
            first_beat_delta * 1000, FIRST_BEAT_ANCHOR_WEIGHT,
            coverage * 100,
        )
    else:
        log.info(
            "beat alignment: shifted all %d beats by %+.1f ms "
            "(median of %d beat→onset deltas, coverage %.0f%%)",
            len(structure.beats), offset * 1000, len(deltas), coverage * 100,
        )
    _rebuild_bar_fields(structure)


def _rebuild_bar_fields(structure: BeatStructure) -> None:
    """Recompute per-bar start/end/tempo + global summary after beats
    have been re-timed in place."""
    for i, bar in enumerate(structure.bars):
        if not bar.beats:
            continue
        bar.start_time = bar.beats[0].time
        if i + 1 < len(structure.bars) and structure.bars[i + 1].beats:
            bar.end_time = structure.bars[i + 1].beats[0].time
        elif len(bar.beats) >= 2:
            gap = bar.beats[-1].time - bar.beats[-2].time
            bar.end_time = bar.beats[-1].time + max(gap, 0.0)
        if len(bar.beats) >= 2:
            gaps = np.diff([b.time for b in bar.beats])
            bar.tempo_bpm = float(60.0 / np.mean(gaps))

    reference_bars = (
        structure.bars[1:] if len(structure.bars) >= 2 else structure.bars
    )
    if reference_bars:
        structure.initial_tempo = float(reference_bars[0].tempo_bpm)
        tempo_set = {round(b.tempo_bpm, 1) for b in reference_bars}
        structure.has_tempo_changes = len(tempo_set) > 1


def _librosa_fallback(audio_path: Path) -> BeatStructure:
    """Plain librosa beat tracking, no downbeat classification.

    Used only when madmom is unavailable. Produces a 4/4 structure with
    constant time signature; tempo follows whatever librosa returned.
    """
    import librosa

    log.info("librosa fallback: tracking beats on %s", audio_path.name)
    audio, sr = librosa.load(str(audio_path), sr=44100, mono=True)
    tempo, beat_times = librosa.beat.beat_track(y=audio, sr=sr, units="time", trim=False)
    tempo = float(np.atleast_1d(tempo)[0]) if np.isfinite(tempo) else 120.0
    if not isinstance(beat_times, np.ndarray) or beat_times.size == 0:
        return BeatStructure(initial_tempo=tempo)

    # Group every 4 beats into a 4/4 bar.
    beats: list[BeatTick] = []
    bars: list[BarInfo] = []
    cur_bar_beats: list[BeatTick] = []
    bar_index = 0
    for i, t in enumerate(beat_times):
        beat_in_bar = (i % 4) + 1
        if beat_in_bar == 1 and cur_bar_beats:
            bars.append(_finalize_bar(bar_index, cur_bar_beats))
            bar_index += 1
            cur_bar_beats = []
        tick = BeatTick(time=float(t), beat_in_bar=beat_in_bar, bar_index=bar_index)
        beats.append(tick)
        cur_bar_beats.append(tick)
    if cur_bar_beats:
        bars.append(_finalize_bar(bar_index, cur_bar_beats))

    return _summarize(beats, bars)


# ---------- Feel detection ----------

# Reference fraction sets for each feel. We score a bar against each
# hypothesis by summing the minimum distance from each onset's intra-beat
# fraction to the nearest reference fraction. Whichever set has the
# smallest total residual wins.
_FEEL_GRIDS: dict[Feel, list[float]] = {
    "straight16": [0.0, 0.25, 0.5, 0.75],
    "straight8": [0.0, 0.5],
    "triplet": [0.0, 1.0 / 3.0, 2.0 / 3.0],
    "shuffle": [0.0, 2.0 / 3.0],
}


def detect_feel_for_bars(
    structure: BeatStructure, onset_times: list[float]
) -> None:
    """Annotate each `BarInfo` in-place with a detected `feel`.

    `onset_times` is a flat list of onset times across all instruments
    in seconds. Per-instrument granularity isn't needed for feel detection
    - we just need to see where the strikes fall within each beat.
    """
    if not structure.bars or not onset_times:
        return
    # Bucket onsets per bar
    onset_arr = np.asarray(onset_times)
    for bar in structure.bars:
        in_bar = onset_arr[(onset_arr >= bar.start_time) & (onset_arr < bar.end_time)]
        if in_bar.size < 4:
            # Not enough hits to classify; mark sparse and skip.
            bar.feel = "sparse"
            continue
        # Convert onset times to intra-beat fractions in [0, 1)
        fractions: list[float] = []
        for t in in_bar:
            pos = _intra_beat_fraction(bar, float(t))
            if pos is None:
                continue
            fractions.append(pos)
        if len(fractions) < 4:
            bar.feel = "sparse"
            continue
        bar.feel = _score_feel(fractions)


def _intra_beat_fraction(bar: BarInfo, t: float) -> float | None:
    # Find enclosing beat in this bar.
    for i, beat in enumerate(bar.beats):
        next_t = (
            bar.beats[i + 1].time if i + 1 < len(bar.beats) else bar.end_time
        )
        if beat.time <= t < next_t:
            gap = max(next_t - beat.time, 1e-6)
            return (t - beat.time) / gap
    return None


def _score_feel(fractions: list[float]) -> Feel:
    """Pick the feel whose reference fractions best match the data."""
    best: tuple[Feel, float] = ("mixed", float("inf"))
    for label, grid in _FEEL_GRIDS.items():
        total = 0.0
        for f in fractions:
            # closest distance considering wrap-around at 1.0
            total += min(
                min(abs(f - g), 1.0 - abs(f - g)) for g in grid
            )
        if total < best[1]:
            best = (label, total)
    # If even the best is too noisy, call it mixed (LLM should treat as
    # a hint to fall back to a free grid).
    avg_residual = best[1] / max(len(fractions), 1)
    if avg_residual > 0.10:
        return "mixed"
    return best[0]


# ---------- LLM-friendly summary ----------

@dataclass
class CandidateOnset:
    """One onset attached to its beat-relative position."""

    pitch: str
    time: float
    strength: float
    bar: int
    beat_in_bar: float  # 1-indexed, uniform across the bar; see BeatStructure.position


def candidates_with_beat_positions(
    onsets_by_pitch: dict[str, list[tuple[float, float]]],
    structure: BeatStructure,
) -> dict[str, list[CandidateOnset]]:
    """For each (time, strength) per pitch, compute (bar, beat_in_bar).

    Onsets outside the tracked range are dropped.
    """
    out: dict[str, list[CandidateOnset]] = {}
    for pitch, items in onsets_by_pitch.items():
        kept: list[CandidateOnset] = []
        for time, strength in items:
            pos = structure.position(time)
            if pos is None:
                continue
            bar, beat = pos
            kept.append(
                CandidateOnset(
                    pitch=pitch,
                    time=float(time),
                    strength=float(strength),
                    bar=int(bar),
                    beat_in_bar=float(beat),
                )
            )
        if kept:
            out[pitch] = kept
    return out


def summarize_bar_for_prompt(bar: BarInfo) -> dict[str, object]:
    """Render `BarInfo` as a compact dict the prompt template can embed.

    Stable key order so the prompt is reproducible across runs.
    """
    return {
        "bar": bar.index,
        "time_signature": f"{bar.time_signature[0]}/{bar.time_signature[1]}",
        "tempo_bpm": round(bar.tempo_bpm, 1),
        "feel": bar.feel,
        "start_time": round(bar.start_time, 3),
    }
