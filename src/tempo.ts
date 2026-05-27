/**
 * Tempo resolution helpers shared by every consumer that needs to map a
 * jot's bar/beat positions to wall-clock time.
 *
 * The single runtime source of truth for tempo is `Jot.tempoEvents`. The
 * pre-first-event span uses `Jot.globalMetadata.bpm` (else 120). Inline
 * `{{bpm}}` blocks and `{bpm}` modifiers on notes/groups get hoisted into
 * `tempoEvents` by the parser; other producers (from_midi, rlrr_to_jot)
 * populate `tempoEvents` directly. No runtime path reads
 * `bar.metadata.bpm` / `note.metadata.bpm` any more.
 *
 * The hot path is per-bar `BarTempos` computation: every consumer that
 * walks bars to compute timings (playback timeline, midi writer,
 * waveform chunker, score header) calls {@link buildBarTempos} and reads
 * the resulting `durationSec` / `segments`. A within-bar `(beat) -> sec`
 * lookup uses {@link beatToSecWithinBar}.
 */
import { BpmTransition, Jot, TempoEvent } from './dsl';

/** Default tempo when neither tempoEvents nor globalMetadata.bpm is set. */
export const DEFAULT_BPM = 120;

/**
 * Resolve a `Metadata.bpm` field (a number, a {@link BpmTransition}, or
 * absent) to a positive BPM, falling back when missing or non-positive.
 * Transitions are taken as `start` (else `end`), no interpolation;
 * mirrors how volume transitions are handled.
 */
export function resolveBpm(
  field: number | BpmTransition | undefined,
  fallback: number,
): number {
  if (typeof field === 'number') return field > 0 ? field : fallback;
  if (field && typeof field === 'object') {
    const v = field.start ?? field.end;
    return typeof v === 'number' && v > 0 ? v : fallback;
  }
  return fallback;
}

/**
 * The initial tempo before any `tempoEvents` fire. Equals
 * `globalMetadata.bpm` if set, else `DEFAULT_BPM`.
 */
export function initialBpm(jot: Jot): number {
  return resolveBpm(jot.globalMetadata.bpm, DEFAULT_BPM);
}

/**
 * An intra-bar tempo span. `[startBeat, endBeat)` plays at constant
 * `bpm`. Concatenating a bar's segments tiles `[0, bar.beats)`. Always
 * at least one segment per bar.
 */
export type TempoSegment = {
  startBeat: number;
  endBeat: number;
  bpm: number;
};

/**
 * Per-bar tempo layout: total duration in seconds plus the ordered list
 * of constant-tempo segments inside the bar. The `bpm` carried into the
 * bar (= tempo at the bar's downbeat) is `segments[0].bpm`; the tempo
 * carried out (= tempo at the bar's last beat) is
 * `segments[segments.length - 1].bpm`.
 */
export type BarTempos = {
  durationSec: number;
  segments: TempoSegment[];
};

/**
 * Sort tempoEvents canonically. Multiple events at the same anchor are
 * kept in their source order (the parser emits them in encounter order).
 */
function sortedEvents(jot: Jot): TempoEvent[] {
  const events = jot.tempoEvents ?? [];
  if (events.length <= 1) return events;
  return events
    .map((e, i) => ({ e, i }))
    .sort((a, b) => {
      if (a.e.barIndex !== b.e.barIndex) return a.e.barIndex - b.e.barIndex;
      if (a.e.beat !== b.e.beat) return a.e.beat - b.e.beat;
      return a.i - b.i;
    })
    .map(({ e }) => e);
}

/**
 * Build the per-bar tempo layout for a sequence of bars (typically
 * `voices[0].bars` lengths in quarter-note beats). Walks `tempoEvents`
 * once forward, switching tempos at each event's `(barIndex, beat)`
 * anchor and accumulating per-bar durations.
 *
 * `bars[i].beats` is the bar's length in quarter notes (matches
 * `StructuralBar.beats`). The returned array has the same length as
 * `bars`. An empty bar (beats === 0) gets `durationSec: 0` and a single
 * zero-width segment carrying the current tempo.
 */
export function buildBarTempos(
  jot: Jot,
  bars: { beats: number }[],
): BarTempos[] {
  const events = sortedEvents(jot);
  let evIdx = 0;
  let currentBpm = initialBpm(jot);

  // Any events anchored before bar 0 (defensive: shouldn't happen from
  // the parser, but a future producer could emit them) collapse onto
  // bar 0's initial tempo.
  while (evIdx < events.length && events[evIdx].barIndex < 0) {
    currentBpm = resolveBpm(events[evIdx].bpm, currentBpm);
    evIdx++;
  }

  const out: BarTempos[] = new Array(bars.length);
  for (let i = 0; i < bars.length; i++) {
    const barBeats = bars[i].beats;
    const segments: TempoSegment[] = [];
    let cursor = 0;
    let durationSec = 0;

    while (
      evIdx < events.length &&
      events[evIdx].barIndex === i &&
      events[evIdx].beat <= barBeats
    ) {
      const ev = events[evIdx];
      const clampedBeat = ev.beat < 0 ? 0 : ev.beat;
      if (clampedBeat > cursor) {
        segments.push({ startBeat: cursor, endBeat: clampedBeat, bpm: currentBpm });
        durationSec += (clampedBeat - cursor) * (60 / currentBpm);
        cursor = clampedBeat;
      }
      currentBpm = resolveBpm(ev.bpm, currentBpm);
      evIdx++;
    }

    if (cursor < barBeats || segments.length === 0) {
      segments.push({ startBeat: cursor, endBeat: barBeats, bpm: currentBpm });
      durationSec += (barBeats - cursor) * (60 / currentBpm);
    }

    out[i] = { durationSec, segments };

    // Advance past any events anchored past the bar's end (defensive;
    // they should have landed in a later bar; but a producer that
    // anchored at exactly bar.beats lands here on this bar). They roll
    // into the next bar's downbeat.
    while (
      evIdx < events.length &&
      events[evIdx].barIndex === i &&
      events[evIdx].beat > barBeats
    ) {
      currentBpm = resolveBpm(events[evIdx].bpm, currentBpm);
      evIdx++;
    }
  }

  return out;
}

/**
 * Map a beat-within-bar to a seconds offset from that bar's start,
 * walking the bar's tempo segments. Beats past the bar's last segment
 * clamp to the bar end (segments tile `[0, beats)`); beats before 0
 * clamp to 0.
 */
export function beatToSecWithinBar(tempos: BarTempos, beat: number): number {
  if (beat <= 0) return 0;
  let sec = 0;
  for (const seg of tempos.segments) {
    if (beat <= seg.startBeat) return sec;
    if (beat < seg.endBeat) {
      return sec + (beat - seg.startBeat) * (60 / seg.bpm);
    }
    sec += (seg.endBeat - seg.startBeat) * (60 / seg.bpm);
  }
  return sec;
}

/**
 * Effective tempo at a given (barIndex, beat) position. Used by readers
 * that need a single number (subtitle formatters, score timeline
 * headers, ad-hoc conversions). For timeline construction prefer
 * {@link buildBarTempos} so intra-bar segments are visible.
 */
export function tempoAt(jot: Jot, barIndex: number, beat: number): number {
  let bpm = initialBpm(jot);
  for (const ev of sortedEvents(jot)) {
    if (
      ev.barIndex < barIndex ||
      (ev.barIndex === barIndex && ev.beat <= beat)
    ) {
      bpm = resolveBpm(ev.bpm, bpm);
    } else break;
  }
  return bpm;
}
