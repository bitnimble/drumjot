"""Beat tracking + downbeat detection + per-bar feel analysis.

This module replaces the older `tempo.py` + `quantize.py` pair. Rather
than assuming a constant tempo + fixed grid (1/16 by default), we use
Beat This! (ISMIR 2024; DBN-free, meter-agnostic) to extract per-beat
anchors + downbeats from the audio. Everything downstream then works in
**beat-relative** time -
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
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

log = logging.getLogger(__name__)

# Common beat-counts the downbeat tracker should consider. Covers all
# popular-music time signatures (and a few odd-meter ones).
BEATS_PER_BAR_CANDIDATES = [2, 3, 4, 5, 6, 7]

# Feel classification labels emitted on each `BarInfo`.
Feel = Literal["straight16", "straight8", "triplet", "shuffle", "sparse", "mixed"]

# A tempo segment counts as a ramp (rather than constant) only when its
# endpoints differ by more than this. Below it the endpoints are snapped
# equal so frame-quantization residue never manufactures a hairline ramp.
_RAMP_EPS_BPM = 0.5


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
    # Seconds the bar's *real* (detected) downbeat sits past where the
    # regularized/uniform tempo grid puts it (real - model), i.e. the
    # performance drift `_finalize_bar_tempos` smoothed out of `start_time`.
    # `start_time + drift_sec` reconstructs the true audio downbeat, so the
    # editor can keep a clean uniform tempo display yet still align bar lines
    # + the waveform to the recording. 0 for a perfectly on-grid bar (and for
    # synthetic trailing/padded bars). See `_finalize_bar_tempos`.
    drift_sec: float = 0.0


@dataclass
class TempoSegment:
    """One maximal run of beats sharing a single tempo behaviour.

    A segment is either **constant** (`start_bpm == end_bpm`) or a
    **linear-in-time ramp** (tempo rises/falls at a constant BPM-per-second
    rate, so `bpm²` varies linearly with beat). This matches the runtime
    semantics of `src/schema/dsl/tempo.ts` (`BpmTransition` /
    `segmentBeatToSec`) exactly, so a ramp here round-trips to a frontend
    `BpmTransition` without re-deriving its shape.

    `start_beat` / `end_beat` index into `BeatStructure.beats` (inclusive).
    The segment spans `end_beat - start_beat` beats; for a ramp that count
    is the `duration` (in quarter notes) of the equivalent `BpmTransition`.
    The sequence of segments is the transcriber's first-class tempo map, the source of truth that drives `bar.tempo_bpm`, the MIDI tempo track,
    and the `transcription.json` sidecar.
    """

    start_beat: int
    end_beat: int
    start_bpm: float
    end_bpm: float

    def is_ramp(self, eps: float = _RAMP_EPS_BPM) -> bool:
        return abs(self.end_bpm - self.start_bpm) > eps


@dataclass
class BeatStructure:
    """Sequence of beats + bar groupings + global summary fields."""

    beats: list[BeatTick] = field(default_factory=list)
    bars: list[BarInfo] = field(default_factory=list)
    initial_tempo: float = 120.0
    initial_time_signature: tuple[int, int] = (4, 4)
    has_tempo_changes: bool = False
    has_time_sig_changes: bool = False
    # First-class tempo map: an ordered, gap-free list of constant/ramp
    # segments covering every beat. Source of truth for tempo downstream
    # (`bar.tempo_bpm` is a derived projection of this). Empty only for a
    # degenerate (beatless) structure. Built by `_finalize_bar_tempos`.
    tempo_segments: list[TempoSegment] = field(default_factory=list)
    # The seconds-space shift `align_beats_to_onsets` applied uniformly
    # to every beat (added to `beat.time`). Defaults to `0.0` and stays
    # that way when alignment didn't run, found no matches, or was
    # rejected by the coverage gate; so the value is always a number,
    # which lets the frontend's Beat control and the per-note "Global
    # beat alignment" row render `+0.000s` consistently instead of
    # disappearing on rejection.
    align_offset_sec: float = 0.0
    # Per-pass breakdown of `align_offset_sec`. `align_beats_to_envelope`
    # (coarse, ±2 quarter-notes) writes `align_coarse_offset_sec`;
    # `align_beats_to_onsets` (fine, ±50 ms median snap) writes
    # `align_fine_offset_sec`. Their sum always equals `align_offset_sec`,
    # which downstream code keeps reading as the combined shift. The split
    # is surfaced in `note_provenance.json` so the per-note debug popup
    # can show "coarse +12 ms · fine +5 ms" instead of one collapsed
    # number, useful when diagnosing whether a residual lag survived the
    # coarse pass.
    align_coarse_offset_sec: float = 0.0
    align_fine_offset_sec: float = 0.0

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

    Uses Beat This! (DBN-free, meter-agnostic) for the actual ML; falls
    back to a librosa-based heuristic if it isn't importable, so the rest
    of the pipeline degrades gracefully rather than failing outright.

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
        structure = _beat_this_beats(audio_path)
    except Exception as exc:
        log.warning(
            "Beat This! beat tracking failed (%s); falling back to librosa "
            "(no downbeat classification - assuming 4/4 throughout)",
            exc,
        )
        structure = _librosa_fallback(audio_path)
    if align_onsets:
        # Coarse envelope phase-align first (wide ±half-bar search) so the
        # fine onset snap below isn't starved by a multi-slot phase error
        # outside its ±50 ms window, then remove the residual lag.
        align_beats_to_envelope(structure, audio_path)
        align_beats_to_onsets(structure, align_onsets)
    _finalize_bar_tempos(structure)
    if duration_seconds is not None and duration_seconds > 0:
        _pad_trailing_bars(structure, duration_seconds)
    return structure


_BEAT_THIS_MODEL = None


def _beat_this_model():
    """Lazily build the cached Beat This! inference wrapper.

    Beat This! (Foscarin et al., ISMIR 2024) is DBN-free, so it tracks
    beats + downbeats jointly with no fixed beats-per-bar prior, which is
    why it handles odd/compound meters where the old madmom-DBN path
    (madmom RNN and Beat Transformer both fed it) collapsed. Weights
    auto-download to the torch hub cache on first use. Lazy so an import/
    download hiccup doesn't block service startup.
    """
    global _BEAT_THIS_MODEL
    if _BEAT_THIS_MODEL is None:
        import torch
        from beat_this.inference import File2Beats

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading Beat This! (final0) onto %s", device)
        _BEAT_THIS_MODEL = File2Beats(device=device, dbn=False)
    return _BEAT_THIS_MODEL


def park_model() -> None:
    """Drop the cached Beat This! model so /lyrics/align gets a clean GPU.

    Reloaded lazily on the next transcribe; the model is tiny (~78 MB) and
    its weights are disk-cached, so the reload cost is negligible. Mirrors
    the `park_model`/`unpark_model` pair on `adtof_onsets`."""
    global _BEAT_THIS_MODEL
    _BEAT_THIS_MODEL = None


def unpark_model() -> None:
    """No-op: `_beat_this_model` reloads lazily on next use."""


def _beat_this_beats(audio_path: Path) -> BeatStructure:
    """Beat This! beats + downbeats -> typed BeatStructure.

    Runs on the full mix (Beat This!'s training distribution; the
    `beat_input=full_mix` default). The downstream grid (alignment,
    tempo finalisation, padding) is unchanged: we convert the native
    (beats, downbeats) into the same Nx2 `(time, beat_pos_in_bar)` shape
    the old DBN emitted and reuse `_raw_to_structure`.
    """
    log.info("Beat This!: tracking beats/downbeats in %s", audio_path.name)
    beats, downbeats = _beat_this_model()(str(audio_path))
    raw = _beats_downbeats_to_raw(beats, downbeats)
    if raw.size == 0:
        log.warning("Beat This! returned no beats for %s", audio_path.name)
        return BeatStructure()
    return _raw_to_structure(raw)


def _beats_downbeats_to_raw(beats, downbeats, tol: float = 0.05) -> np.ndarray:
    """(beat times, downbeat times) -> Nx2 `(time, beat_pos_in_bar)`.

    `beat_pos_in_bar` is 1 at each downbeat and increments for the beats
    in between, matching the convention `_raw_to_structure` expects.
    Downbeats are a subset of the beat times, matched within `tol`.
    """
    beats = np.asarray(sorted(float(b) for b in beats), dtype=np.float64)
    if beats.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    db = np.asarray(sorted(float(d) for d in downbeats), dtype=np.float64)
    is_downbeat = np.zeros(beats.size, dtype=bool)
    for d in db:
        j = int(np.searchsorted(beats, d))
        nearest = min((k for k in (j - 1, j) if 0 <= k < beats.size),
                      key=lambda k: abs(beats[k] - d), default=None)
        if nearest is not None and abs(beats[nearest] - d) <= tol:
            is_downbeat[nearest] = True

    is_downbeat = _smooth_downbeats(is_downbeat)

    rows = np.empty((beats.size, 2), dtype=np.float64)
    pos = 0
    for k in range(beats.size):
        pos = 1 if (bool(is_downbeat[k]) or pos == 0) else pos + 1
        rows[k] = (beats[k], pos)
    return rows


def _smooth_downbeats(is_downbeat: np.ndarray) -> np.ndarray:
    """Repair downbeat mis-detections that would fake a time-signature change.

    Beat This! is DBN-free (no fixed-meter prior), so a single mis-placed
    downbeat shows up as a one-off odd bar. Against the *prevailing* meter P
    (the majority bar length) two corrections are applied, both purely by
    inserting/removing downbeats (never inventing or dropping beats):

    1. **No multiples** (missed downbeat merged two bars): a bar whose length
       is an exact multiple ``k·P`` (k≥2) is split back into k bars of P, so
       4/4 never reads as 8/4 and 3/4 never as 6/4.
    2. **2-bar persistence** (extra downbeat fragmented one bar): a run of
       consecutive sub-P bars whose lengths sum to exactly one P bar is merged
       back into a single bar (e.g. 2+2 or 1+3 → 4). A *sustained* odd meter
       (≥2 bars that don't sum to one P bar, e.g. a real 3/4 or 6/8 section) is
       left untouched, so genuine mid-song changes survive.

    No-ops unless one meter holds a clear majority of the interior bars (so a
    genuinely meter-varied song isn't forced onto a single grid). A truly
    dropped/added *beat* (not downbeat) still yields a lone odd bar; we can't
    fix that without inventing beats, so it's preserved.
    """
    db = [int(i) for i in np.flatnonzero(is_downbeat)]
    if len(db) < 3:
        return is_downbeat
    counts = [db[k + 1] - db[k] for k in range(len(db) - 1)]  # interior bar lengths
    p, freq = Counter(counts).most_common(1)[0]
    if p < 2 or freq * 2 <= len(counts):  # no clear majority meter -> don't touch
        return is_downbeat

    # 1. split exact multiples of P
    split: list[int] = [db[0]]
    for k in range(len(db) - 1):
        c = db[k + 1] - db[k]
        if c >= 2 * p and c % p == 0:
            split.extend(db[k] + m * p for m in range(1, c // p))
        split.append(db[k + 1])
    split = sorted(set(split))

    # 2. merge runs of consecutive sub-P bars that sum to exactly one P bar
    kept = [split[0]]
    k = 0
    while k < len(split) - 1:
        if split[k + 1] - split[k] < p:
            run_sum, j = 0, k
            while j < len(split) - 1 and (split[j + 1] - split[j]) < p and run_sum < p:
                run_sum += split[j + 1] - split[j]
                j += 1
            if run_sum == p and j - k >= 2:  # fragments of one bar -> collapse
                kept.append(split[j])
                k = j
                continue
        kept.append(split[k + 1])
        k += 1

    out = np.zeros(is_downbeat.shape[0], dtype=bool)
    out[kept] = True
    return out


def _raw_to_structure(raw: np.ndarray) -> BeatStructure:
    """Convert an Nx2 `(time, beat_pos_in_bar)` array into a BeatStructure."""
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

    The tracker only reports a count of beats per bar; it doesn't tell us
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
    the tracker often starts counting partway through the first bar (anacrusis
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
    structure.initial_tempo = segments[0].start_bpm if segments else 120.0
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


# Minimum fraction of beats that must have a strong drum onset within
# the alignment window before we trust the median offset enough to
# shift the whole grid. Drum stems normally hit this easily (most beats
# coincide with a kick/snare/hat); sparse or heavily-syncopated material
# that falls below it keeps the raw tracker grid rather than risk a
# bogus global shift.
MIN_ALIGN_COVERAGE = 0.30

# ---- Coarse envelope phase alignment (runs before the fine onset snap) ----
#
# The fine `align_beats_to_onsets` only searches ±50 ms (~1 slot). When the
# beat tracker locks onto a phase that's a few slots off, the true hits sit
# outside that window, so the fine pass no-ops or misfires and a constant
# multi-slot offset survives into the score. This coarse pass first finds a
# single global shift (up to ±`COARSE_MAX_SHIFT_BEATS` quarter notes) that
# best seats the beat grid on the drum-stem onset-strength envelope, then
# hands a well-phased grid to the fine pass for sub-frame lag removal.
COARSE_MAX_SHIFT_BEATS = 2.0   # cap: ± two quarter notes (half a 4/4 bar)
COARSE_SEARCH_STEP_SEC = 0.002  # offset search resolution (~0.05 slot @120)
COARSE_ENV_HOP = 256            # onset-strength hop (~5.8 ms @ 44.1 kHz)
# Multiplicative taper favouring small shifts: a shift at the ±cap must beat
# the zero-shift score by >this fraction to win, so we never lock onto a
# louder backbeat a full beat away when the grid is already close.
COARSE_CENTER_PENALTY = 0.15
# The winning comb score must exceed the mean comb score over the whole
# search range by this factor, else there's no clear pulse and we shift
# nothing (envelope too flat / not enough drum energy).
COARSE_PROMINENCE = 1.10


def align_beats_to_envelope(
    structure: BeatStructure,
    audio_path: Path,
) -> None:
    """Coarse global phase align: seat the beat grid on the drum envelope.

    Computes the onset-strength envelope of `audio_path` (the drum stem
    when available) and finds the single global time shift `δ`, within
    ±`COARSE_MAX_SHIFT_BEATS` quarter notes, that maximises the envelope
    energy summed over all beat positions shifted by `δ`. A multiplicative
    centre taper (`COARSE_CENTER_PENALTY`) biases the search toward small
    shifts so a grid that's already close isn't dragged a full beat onto a
    louder backbeat, and a prominence gate (`COARSE_PROMINENCE`) leaves the
    grid untouched when there's no clear pulse to lock onto.

    The shift is applied uniformly to every beat (inter-beat gaps and hence
    per-bar tempo are preserved) and accumulated into `align_offset_sec`.
    This runs *before* `align_beats_to_onsets`: it kills a multi-slot phase
    error the fine pass's ±50 ms window can't see, leaving the fine pass a
    well-phased grid on which to do precise lag removal.
    """
    if len(structure.beats) < 2:
        return
    import librosa

    beat_times = np.asarray([b.time for b in structure.beats], dtype=np.float64)
    beat_period = float(np.median(np.diff(np.sort(beat_times))))
    if not np.isfinite(beat_period) or beat_period <= 0:
        return
    max_shift = COARSE_MAX_SHIFT_BEATS * beat_period

    try:
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    except Exception as exc:
        log.warning("coarse align: could not load %s (%s); skipping", audio_path, exc)
        return
    if y.size == 0:
        return
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=COARSE_ENV_HOP)
    if env.size == 0 or not np.any(env):
        return
    frame_times = librosa.frames_to_time(
        np.arange(env.size), sr=sr, hop_length=COARSE_ENV_HOP
    )

    offset = _coarse_offset_from_envelope(
        beat_times, env.astype(np.float64), frame_times,
        max_shift=max_shift, step=COARSE_SEARCH_STEP_SEC,
        center_penalty=COARSE_CENTER_PENALTY, prominence=COARSE_PROMINENCE,
    )
    if offset == 0.0:
        log.info(
            "coarse align: no confident global phase shift found; grid unchanged"
        )
        return
    for beat in structure.beats:
        beat.time += offset
    structure.align_offset_sec += offset
    structure.align_coarse_offset_sec += offset
    log.info(
        "coarse align: shifted all %d beats by %+.1f ms (%.2f slot @ %.1f BPM, "
        "search ±%.0f ms)",
        len(structure.beats), offset * 1000.0,
        offset / (beat_period / 12.0), 60.0 / beat_period, max_shift * 1000.0,
    )
    _rebuild_bar_fields(structure)


def _coarse_offset_from_envelope(
    beat_times: np.ndarray,
    env: np.ndarray,
    frame_times: np.ndarray,
    *,
    max_shift: float,
    step: float,
    center_penalty: float,
    prominence: float,
) -> float:
    """Global shift `δ` maximising Σ env(beat + δ), centre-biased and gated.

    Pure (no audio I/O) so it's unit-testable: sweeps `δ` over
    [-max_shift, +max_shift] in `step` increments, sampling the envelope at
    each shifted beat position by linear interpolation. The raw comb score
    is tapered by `(1 - center_penalty·|δ|/max_shift)` to favour small
    shifts, and the winner is accepted only if its raw score clears
    `prominence × mean(raw scores)`. Returns 0.0 when nothing is confident.
    """
    deltas = np.arange(-max_shift, max_shift + step, step)
    if deltas.size == 0:
        return 0.0
    raw = np.array(
        [float(np.interp(beat_times + d, frame_times, env).sum()) for d in deltas]
    )
    if not np.any(raw > 0):
        return 0.0
    taper = 1.0 - center_penalty * (np.abs(deltas) / max_shift)
    best = int(np.argmax(raw * taper))
    if raw[best] < prominence * float(raw.mean()):
        return 0.0
    return float(deltas[best])


def align_beats_to_onsets(
    structure: BeatStructure,
    onsets: list[tuple[float, float]],
    max_distance: float = 0.05,
) -> None:
    """Shift the whole beat grid by the tracker's *systematic* lag.

    Neural beat trackers report each beat
    ~30-50 ms after the transient, because the activation peak lags the
    strike. We still want to correct that.

    The previous implementation snapped **each beat independently** to
    the strongest drum onset within ±`max_distance`. That removed the
    lag but also absorbed the drummer's natural micro-timing into the
    grid: every beat's gap to its neighbours changed, so the per-bar
    tempo (`60 / mean(gap)` over a bar's 3-4 gaps) wobbled 5-10 BPM
    even on a dead-steady song — the LLM then emitted a `{{ bpm }}`
    change between nearly every bar.

    Instead, estimate ONE offset, the median over all beats of
    `(nearest strong onset − beat time)`, and shift every beat by it.
    The grid stays exactly as metrically regular as the DBN produced
    it (per-bar tempo is therefore stable), while the systematic lag is
    still removed. A uniform shift leaves inter-beat gaps unchanged, so
    a genuine accelerando the DBN tracked is preserved untouched.

    This is the FINE pass: it only searches ±`max_distance` (~50 ms), so
    it assumes the grid is already within ~1 slot of the true phase.
    `align_beats_to_envelope` runs first to guarantee that, it kills any
    larger multi-slot phase error this window can't see. The shift here is
    *added* to whatever the coarse pass already applied (`align_offset_sec`
    accumulates).

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
    offset = float(np.median(deltas))

    if coverage < MIN_ALIGN_COVERAGE:
        log.info(
            "beat alignment: only %.0f%% of beats had a nearby onset "
            "(< %.0f%% required); offset %+.1f ms rejected, grid unchanged",
            coverage * 100, MIN_ALIGN_COVERAGE * 100, offset * 1000,
        )
        return

    for beat in structure.beats:
        beat.time += offset
    structure.align_offset_sec += offset
    structure.align_fine_offset_sec += offset
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

    Used only when Beat This! is unavailable. Produces a 4/4 structure with
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
