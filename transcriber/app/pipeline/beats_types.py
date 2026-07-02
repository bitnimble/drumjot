"""Core beat-tracking data types + shared constants.

The typed vocabulary every other `beats_*` module shares: `BeatTick`,
`BarInfo`, `TempoSegment`, `BeatStructure` (and the `Feel` label + the small
constants that describe them). Kept dependency-free (numpy only) so it sits at
the bottom of the beats module graph with nothing importing back into it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

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
        that linearly maps the bar's audio span to its meter, i.e.
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
