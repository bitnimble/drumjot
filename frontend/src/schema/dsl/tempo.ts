/**
 * Tempo resolution helpers shared by every consumer that needs to map a
 * jot's bar/beat positions to wall-clock time.
 *
 * The single runtime source of truth for tempo is `Jot.tempoEvents`. The
 * pre-first-event span uses `Jot.globalMetadata.bpm` (else 120). Inline
 * `{{bpm}}` blocks and `{bpm}` modifiers on notes/groups get hoisted into
 * `tempoEvents` by the parser; other producers (from_midi, the rlrr parser)
 * populate `tempoEvents` directly. No runtime path reads
 * `bar.metadata.bpm` / `note.metadata.bpm` any more.
 *
 * The hot path is per-bar `BarTempos` computation: every consumer that
 * walks bars to compute timings (playback timeline, midi writer,
 * waveform chunker, score header) calls {@link buildBarTempos} and reads
 * the resulting `durationSec` / `segments`. A within-bar `(beat) -> sec`
 * lookup uses {@link beatToSecWithinBar}.
 */
import { BpmTransition, TempoEvent } from 'src/schema/dsl/dsl';

/** Default tempo when neither tempoEvents nor globalMetadata.bpm is set. */
export const DEFAULT_BPM = 120;

/**
 * The slice of a jot the tempo maths reads: the barIndex-anchored sticky
 * tempo events plus the initial/global bpm. A DSL `Jot` satisfies this
 * structurally (load/export paths), as does the live reactive→barIndex
 * projection the editor builds from the reactive model
 * (`StructuralPresenter.tempoSource`). Keeping these pure helpers on this
 * minimal shape (rather than the DSL `Jot`) is what lets the editor read
 * tempo straight off the reactive document, with no frozen DSL snapshot in
 * the runtime path.
 */
export type TempoJot = {
  tempoEvents?: readonly TempoEvent[];
  globalMetadata: { bpm?: number | BpmTransition };
};

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
export function initialBpm(jot: TempoJot): number {
  return resolveBpm(jot.globalMetadata.bpm, DEFAULT_BPM);
}

/**
 * An intra-bar tempo span over `[startBeat, endBeat)`. Concatenating a
 * bar's segments tiles `[0, bar.beats)`; always at least one per bar.
 *
 * `bpm` is the tempo at `startBeat`. When `endBpm` is absent (or equal to
 * `bpm`) the span is constant. When `endBpm` differs it's a gradual tempo
 * change with **linear-in-time** semantics (the tempo rises/falls at a
 * constant rate per second, the everyday "steady accelerando"), which makes
 * `bpm²` vary linearly with beat. See {@link segmentBeatToSec}.
 */
export type TempoSegment = {
  startBeat: number;
  endBeat: number;
  bpm: number;
  /** Tempo at `endBeat` for a gradual ramp; absent/equal ⇒ constant. */
  endBpm?: number;
};

/**
 * Seconds to traverse a whole tempo segment. For a linear-in-time ramp the
 * mean rate is the arithmetic mean of the endpoint BPMs, so the duration is
 * `L beats ÷ mean-bpm`; this unifies the constant case (`endBpm === bpm`).
 */
export function segmentDurationSec(seg: TempoSegment): number {
  const L = seg.endBeat - seg.startBeat;
  if (L <= 0) return 0;
  return (120 * L) / (seg.bpm + (seg.endBpm ?? seg.bpm));
}

/**
 * Seconds from a segment's `startBeat` to `beat` (assumed inside the
 * segment). Constant segments integrate linearly; ramp segments use the
 * closed-form linear-in-time integral. With `bpm(b) = √(b0² + (b1²−b0²)·d/L)`
 * (d = beat − startBeat), the time is `120·L·(bpm(b) − b0)/(b1² − b0²)`.
 */
function segmentBeatToSec(seg: TempoSegment, beat: number): number {
  const L = seg.endBeat - seg.startBeat;
  const d = beat - seg.startBeat;
  if (L <= 0 || d <= 0) return 0;
  const b0 = seg.bpm;
  const b1 = seg.endBpm ?? seg.bpm;
  if (b1 === b0) return d * (60 / b0);
  const bpmAt = Math.sqrt(b0 * b0 + (b1 * b1 - b0 * b0) * (d / L));
  return (120 * L * (bpmAt - b0)) / (b1 * b1 - b0 * b0);
}

/**
 * Inverse of {@link segmentBeatToSec}: the beat reached `sec` seconds after
 * a segment's `startBeat` (assumed inside the segment). For a ramp the tempo
 * is linear in time, `bpm(t) = b0 + a·t` with `a = (b1²−b0²)/(120·L)`, and
 * the beat is `startBeat + (bpm(t)² − b0²)·L/(b1² − b0²)`.
 */
function segmentSecToBeat(seg: TempoSegment, sec: number): number {
  const L = seg.endBeat - seg.startBeat;
  if (L <= 0 || sec <= 0) return seg.startBeat;
  const b0 = seg.bpm;
  const b1 = seg.endBpm ?? seg.bpm;
  if (b1 === b0) return seg.startBeat + sec * (b0 / 60);
  const a = (b1 * b1 - b0 * b0) / (120 * L);
  const bpmAt = b0 + a * sec;
  return seg.startBeat + ((bpmAt * bpmAt - b0 * b0) * L) / (b1 * b1 - b0 * b0);
}

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
function sortedEvents(jot: TempoJot): readonly TempoEvent[] {
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

/** A tempo span in GLOBAL beat space (cumulative over all bars). `bpm` /
 *  `endBpm` are the tempo at `startBeat` / `endBeat`; a ramp follows the
 *  linear-in-time law (`bpm²` linear in beat). */
type GlobalSeg = { startBeat: number; endBeat: number; bpm: number; endBpm: number };

/** Tempo at global beat `g` inside a linear-in-time ramp from `(g0, bpm0)`
 *  to `(g1, bpm1)`; clamps outside `[g0, g1]`. `bpm²` is linear in beat. */
function bpmInRamp(g0: number, bpm0: number, g1: number, bpm1: number, g: number): number {
  if (g <= g0 || g1 <= g0) return bpm0;
  if (g >= g1) return bpm1;
  const f = (g - g0) / (g1 - g0);
  return Math.sqrt(bpm0 * bpm0 + (bpm1 * bpm1 - bpm0 * bpm0) * f);
}

/** Tempo at global beat `x` from the tiled global segments, falling back to
 *  `trailing` (the constant tempo past the last segment). */
function globalBpmAt(segs: GlobalSeg[], trailing: number, x: number): number {
  for (const s of segs) {
    if (x >= s.startBeat && x <= s.endBeat) {
      return bpmInRamp(s.startBeat, s.bpm, s.endBeat, s.endBpm, x);
    }
  }
  return trailing;
}

/**
 * Build the per-bar tempo layout for a sequence of bars (typically
 * `layers[0].bars` lengths in quarter-note beats). Resolves `tempoEvents`
 * (flat changes AND gradual `BpmTransition` ramps) into a single tempo
 * function over global beats, then slices it per bar.
 *
 * A ramp produces one (possibly bar-spanning) span carrying its endpoint
 * BPMs; the slicer splits it at bar boundaries, interpolating each bar's
 * endpoint tempos so every bar's `durationSec` and intra-bar `segments`
 * stay exact (see {@link segmentDurationSec} / {@link segmentBeatToSec}).
 *
 * `bars[i].beats` is the bar's length in quarter notes (matches
 * `StructBar.beats`). The returned array has the same length as `bars`. An
 * empty bar (beats === 0) gets `durationSec: 0` and a single zero-width
 * segment carrying the tempo in force there.
 */
export function buildBarTempos(
  jot: TempoJot,
  // `synthetic` bars (e.g. the view-only virtual lead-in) carry no
  // tempo-event anchors of their own, `jot.tempoEvents` are indexed against
  // the SOURCE bars, which don't include them, so they're skipped for
  // event anchoring and just inherit the tempo in force. Plain `{beats}`
  // bars (the musical structure) all count as non-synthetic.
  bars: { beats: number; synthetic?: boolean }[],
): BarTempos[] {
  // Global start beat of every bar (synthetic included), plus the global
  // start of every SOURCE bar (what `event.barIndex` references).
  const barStart: number[] = new Array(bars.length);
  const sourceBarStart: number[] = [];
  let totalBeats = 0;
  for (let i = 0; i < bars.length; i++) {
    barStart[i] = totalBeats;
    if (!bars[i].synthetic) sourceBarStart.push(totalBeats);
    totalBeats += bars[i].beats;
  }

  // Anchor every tempo event to a global beat; drop unplaceable ones.
  const events = sortedEvents(jot)
    .map((e) => {
      const base = e.barIndex < 0 ? 0 : sourceBarStart[e.barIndex];
      return base === undefined ? undefined : { g: base + Math.max(0, e.beat), bpm: e.bpm };
    })
    .filter((e): e is { g: number; bpm: number | BpmTransition } => e !== undefined)
    .sort((a, b) => a.g - b.g); // stable: preserves source order at equal g

  // Walk the events, emitting global tempo spans. An active ramp ends at its
  // own `g1` OR earlier if a later event preempts it.
  const segs: GlobalSeg[] = [];
  let cur = initialBpm(jot);
  let ramp: { g0: number; bpm0: number; g1: number; bpm1: number } | null = null;
  let cursor = 0;
  let ei = 0;
  const emit = (a: number, b: number) => {
    if (b <= a) return;
    if (ramp) {
      segs.push({
        startBeat: a,
        endBeat: b,
        bpm: bpmInRamp(ramp.g0, ramp.bpm0, ramp.g1, ramp.bpm1, a),
        endBpm: bpmInRamp(ramp.g0, ramp.bpm0, ramp.g1, ramp.bpm1, b),
      });
    } else {
      segs.push({ startBeat: a, endBeat: b, bpm: cur, endBpm: cur });
    }
  };
  while (ei < events.length || ramp) {
    const nextEvent = ei < events.length ? events[ei].g : Infinity;
    const rampEnd = ramp ? ramp.g1 : Infinity;
    if (rampEnd <= nextEvent) {
      // The active ramp completes (possibly exactly at the next event).
      emit(cursor, rampEnd);
      cursor = rampEnd;
      cur = ramp!.bpm1;
      ramp = null;
      continue;
    }
    emit(cursor, nextEvent);
    cursor = nextEvent;
    const ev = events[ei++];
    if (typeof ev.bpm === 'object') {
      const start = ev.bpm.start ?? cur;
      ramp = { g0: cursor, bpm0: start, g1: cursor + ev.bpm.duration, bpm1: ev.bpm.end };
      cur = start;
    } else {
      cur = resolveBpm(ev.bpm, cur);
      ramp = null;
    }
  }
  if (cursor < totalBeats) emit(cursor, totalBeats); // trailing constant (ramp is null here)

  // Slice the global spans into bar-local segments.
  const out: BarTempos[] = new Array(bars.length);
  for (let i = 0; i < bars.length; i++) {
    const bStart = barStart[i];
    const beats = bars[i].beats;
    const bEnd = bStart + beats;
    const segments: TempoSegment[] = [];
    let durationSec = 0;
    for (const s of segs) {
      if (s.endBeat <= bStart || s.startBeat >= bEnd) continue;
      const a = Math.max(s.startBeat, bStart);
      const b = Math.min(s.endBeat, bEnd);
      if (b <= a) continue;
      const bpm0 = bpmInRamp(s.startBeat, s.bpm, s.endBeat, s.endBpm, a);
      const bpm1 = bpmInRamp(s.startBeat, s.bpm, s.endBeat, s.endBpm, b);
      const seg: TempoSegment =
        bpm0 === bpm1
          ? { startBeat: a - bStart, endBeat: b - bStart, bpm: bpm0 }
          : { startBeat: a - bStart, endBeat: b - bStart, bpm: bpm0, endBpm: bpm1 };
      segments.push(seg);
      durationSec += segmentDurationSec(seg);
    }
    if (segments.length === 0) {
      // Zero-width bar (beats === 0), or an uncovered gap: one segment at
      // the tempo in force at the bar's downbeat.
      const seg = { startBeat: 0, endBeat: beats, bpm: globalBpmAt(segs, cur, bStart) };
      segments.push(seg);
      durationSec += segmentDurationSec(seg);
    }
    out[i] = { durationSec, segments };
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
      return sec + segmentBeatToSec(seg, beat);
    }
    sec += segmentDurationSec(seg);
  }
  return sec;
}

/**
 * Inverse of {@link beatToSecWithinBar}: the beat-within-bar reached `sec`
 * seconds after the bar's start, walking the bar's tempo segments. Seconds
 * past the bar's total duration clamp to the last segment's `endBeat`;
 * non-positive seconds clamp to the first segment's `startBeat` (0).
 */
export function secToBeatWithinBar(tempos: BarTempos, sec: number): number {
  const segments = tempos.segments;
  if (segments.length === 0) return 0;
  if (sec <= 0) return segments[0].startBeat;
  let acc = 0;
  for (const seg of segments) {
    const dur = segmentDurationSec(seg);
    if (sec < acc + dur) return segmentSecToBeat(seg, sec - acc);
    acc += dur;
  }
  return segments[segments.length - 1].endBeat;
}

/**
 * Convert a sub-slot timing offset in milliseconds to a fraction of a
 * quarter-note beat, given the bar's local seconds-per-beat. Used by the
 * score renderer to shift an off-grid note's glyph to where it actually
 * plays (the score's x-axis is notational beats, so a real-time ms offset
 * must be divided by the local tempo). Returns 0 for a non-positive
 * `secPerBeat` (degenerate / zero-length bar).
 */
export function msOffsetToBeats(offsetMs: number, secPerBeat: number): number {
  if (!(secPerBeat > 0)) return 0;
  return offsetMs / 1000 / secPerBeat;
}

/**
 * Effective tempo at a given (barIndex, beat) position. Used by readers
 * that need a single number (subtitle formatters, score timeline
 * headers, ad-hoc conversions). For timeline construction prefer
 * {@link buildBarTempos} so intra-bar segments are visible.
 */
export function tempoAt(jot: TempoJot, barIndex: number, beat: number): number {
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
