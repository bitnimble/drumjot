/**
 * Drumjot -> MIDI conversion.
 *
 * Design decisions / assumptions (each tagged [B#] inline below):
 *
 *  [B1] We always emit a single Format-1 track on the drum channel (default
 *       10). All layers in the source Jot are merged onto that track because
 *       `||` global simultaneity in the DSL is about timing, not about
 *       distinct MIDI tracks.
 *
 *  [B2] PPQN is fixed at 480 ticks per beat, a common high-resolution choice
 *       that avoids rounding noise for sixteenth-note and triplet grids.
 *
 *  [B3] Notes are emitted with a fixed 1-tick gate (noteOff immediately after
 *       noteOn). Drum samples are typically triggered by note-on only, so a
 *       longer gate carries no useful information and only confuses some
 *       hosts.
 *
 *  [B4] Per-note `metadata.midi.note` and `metadata.midi.velocity` win when
 *       present (this is the channel the reader uses to preserve fidelity).
 *       Otherwise we fall back to `instrumentMapping[lane].midi` and, for
 *       unknown lanes, to a heuristic in `defaultMidiNote`.
 *
 *  [B5] When no velocity is supplied we map `vol` buckets to GM-ish values
 *       and apply `:a` / `:g` adjustments. Volume transitions
 *       (`VolTransition`) are NOT interpolated; we use only the bucket's
 *       baseline (`vol.start` if present, else `vol.end`).
 *
 *  [B6] Roll / buzz (`~`) is rendered as a single note-on. We do not yet
 *       emit a tremolo or expand it into multiple strikes; doing so well
 *       depends on tempo/genre and is left for a future pass.
 *
 *  [B7] Initial BPM (`globalMetadata.bpm`) and initial time signature meta
 *       events are emitted at tick 0. Subsequent tempo changes come from
 *       `jot.tempoEvents` and are emitted as `setTempo` at the precise
 *       tick (bar offset + beat * TICKS_PER_BEAT), so mid-bar tempo
 *       changes survive the export. Per-bar time-signature overrides are
 *       still honoured from `bar.metadata.time`.
 *
 *  [B8] Modifier-only effects that have no MIDI analogue (e.g. `:x`
 *       cross-stick on a snare) are realised by remapping the note number
 *       (e.g. cross-stick -> 37). The mapping is exactly the inverse of the
 *       read path so a round trip preserves the visible note.
 */
import { writeMidi, MidiEvent } from 'midi-file';
import { Instrument, Jot, Modifier, Volume } from 'src/schema/dsl/dsl';
import { buildStructural } from 'src/editing/jot_editor_store';
import type { StructNote } from 'src/editing/structure/structure_store';
import {
  ACCENT_BOOST,
  DEFAULT_VELOCITY,
  GHOST_REDUCTION,
  VOLUME_TO_VELOCITY,
} from 'src/dynamics/dynamics';
import { resolveBpm } from 'src/schema/dsl/tempo';
import { defaultMidiNote } from './gm';

export type ToMidiOptions = {
  drumChannel?: number;
  defaultVelocity?: number;
  accentBoost?: number;
  ghostReduction?: number;
};

const DEFAULTS: Required<ToMidiOptions> = {
  drumChannel: 10,
  // Velocity defaults come from the shared `src/dynamics.ts`; see
  // ACCENT_BOOST's note there for why an accent must clear `ff` (96) to
  // round-trip through `from_midi`.
  defaultVelocity: DEFAULT_VELOCITY,
  accentBoost: ACCENT_BOOST,
  ghostReduction: GHOST_REDUCTION,
};

/** [B2] */
export const TICKS_PER_BEAT = 480;

/** Convert a Drumjot `Jot` into a MIDI byte buffer (Standard MIDI File). */
export function toMidi(jot: Jot, options: ToMidiOptions = {}): Uint8Array {
  const opts = { ...DEFAULTS, ...options };
  // Musical structure only, the view-only virtual lead-in is never exported.
  const layers = buildStructural(jot).musicalLayers;
  const instrumentFor = (lane: string): Instrument =>
    jot.globalMetadata.instrumentMapping?.[lane] ?? { kind: 'custom' };

  type AbsEvent = {
    tick: number;
    kind: 'noteOn' | 'noteOff';
    note: number;
    velocity: number;
  };
  const abs: AbsEvent[] = [];
  const channel = opts.drumChannel - 1;

  type TsChange = { tick: number; count: number; unit: number };
  const tsChanges: TsChange[] = [];

  type TempoChange = { tick: number; bpm: number };
  const tempoChanges: TempoChange[] = [];
  const globalBpm = resolveBpm(jot.globalMetadata.bpm, 120);

  // Compute each layer-0 bar's absolute tick offset; we'll resolve
  // `jot.tempoEvents` anchors against this. Bars are uniform in length
  // across layers (same time-signature sequence) so layer 0 is canonical.
  const layer0 = layers[0];
  const barTickStart: number[] = [];
  if (layer0) {
    let cursor = 0;
    for (const bar of layer0.bars) {
      barTickStart.push(cursor);
      cursor += Math.max(1, Math.round(bar.beats * TICKS_PER_BEAT));
    }
  }

  // [B7] Tempo events: walk `jot.tempoEvents` and emit setTempo at the
  // exact tick of each anchor. Dedup no-ops (parser already dedupes,
  // but defensive). Skipped events whose barIndex is out of range
  // collapse onto the last valid bar.
  {
    let currentBpm = globalBpm;
    for (const ev of jot.tempoEvents ?? []) {
      const bpm = resolveBpm(ev.bpm, currentBpm);
      if (bpm === currentBpm) continue;
      const idx = Math.min(Math.max(0, ev.barIndex), barTickStart.length - 1);
      const barStart = barTickStart[idx] ?? 0;
      const tick = barStart + Math.round(ev.beat * TICKS_PER_BEAT);
      tempoChanges.push({ tick, bpm });
      currentBpm = bpm;
    }
  }

  // Tempo (bpm) in force at an absolute tick, for converting a note's
  // ms `offset` into ticks. `tempoChanges` is tick-ascending (tempoEvents
  // are sorted by (barIndex, beat)).
  const bpmAtTick = (tick: number): number => {
    let result = globalBpm;
    for (const tc of tempoChanges) {
      if (tc.tick <= tick) result = tc.bpm;
      else break;
    }
    return result > 0 ? result : 120;
  };

  // [B1] Merge all layers onto one MIDI stream.
  // tsChanges are collected on the first layer only; bar tick offsets
  // are identical across layers (same time-signature sequence and bar
  // count), and emitting the same meta event from every layer would
  // duplicate it on the merged track.
  let firstLayer = true;
  for (const layer of layers) {
    let barOffset = 0;
    let prevTime = jot.globalMetadata.time ?? { count: 4, unit: 4 };
    for (let bi = 0; bi < layer.bars.length; bi++) {
      const bar = layer.bars[bi];
      const barTicks = Math.max(1, Math.round(bar.beats * TICKS_PER_BEAT));

      if (firstLayer && bi > 0 && (prevTime.count !== bar.tsCount || prevTime.unit !== bar.tsUnit)) {
        tsChanges.push({
          tick: barOffset,
          count: bar.tsCount,
          unit: bar.tsUnit,
        });
      }
      prevTime = { count: bar.tsCount, unit: bar.tsUnit };

      for (const lane of layer.lanes) {
        const track = bar.tracks[lane];
        if (!track) continue;
        const instrument = instrumentFor(lane);
        for (const note of track.notes) {
          const midiNote = resolveMidiNote(note, instrument);
          if (midiNote === undefined) continue;
          const velocity = clampVelocity(resolveVelocity(note, opts));
          const baseTick = barOffset + Math.round((note.beat / bar.beats) * barTicks);
          // Sub-slot timing offset (ms): nudge the emitted tick so swing /
          // off-grid feel survives the round trip. Absent offset = on-slot.
          const offsetMs = note.offsetMs;
          const offsetTicks =
            offsetMs !== undefined
              ? Math.round((offsetMs * TICKS_PER_BEAT * bpmAtTick(baseTick)) / 60_000)
              : 0;
          // Clamp to a non-negative tick: MIDI can't represent an event
          // before tick 0, so a negative offset on the very first slot
          // (bar 1 beat 1, baseTick 0) collapses to on-grid and the
          // sub-slot nuance is lost on round-trip. Unavoidable, and
          // limited to that one position.
          const tick = Math.max(0, baseTick + offsetTicks);
          abs.push({ tick, kind: 'noteOn', note: midiNote, velocity });
          // [B3] One-tick gate.
          abs.push({ tick: tick + 1, kind: 'noteOff', note: midiNote, velocity: 0 });
        }
      }
      barOffset += barTicks;
    }
    firstLayer = false;
  }

  // Sort: by tick, noteOff before noteOn when ticks collide so any same-tick
  // retrigger comes out cleanly.
  abs.sort((a, b) => {
    if (a.tick !== b.tick) return a.tick - b.tick;
    if (a.kind === b.kind) return a.note - b.note;
    return a.kind === 'noteOff' ? -1 : 1;
  });

  // Build the MIDI track event list with deltaTime values.
  const trackEvents: MidiEvent[] = [];

  // [B7] Initial tempo & time signature.
  trackEvents.push({
    deltaTime: 0,
    meta: true,
    type: 'setTempo',
    microsecondsPerBeat: Math.max(1, Math.round(60_000_000 / globalBpm)),
  });

  const initTs = jot.globalMetadata.time ?? { count: 4, unit: 4 };
  trackEvents.push({
    deltaTime: 0,
    meta: true,
    type: 'timeSignature',
    numerator: initTs.count,
    denominator: initTs.unit,
    metronome: 24,
    thirtyseconds: 8,
  });

  if (jot.title) {
    trackEvents.push({
      deltaTime: 0,
      meta: true,
      type: 'trackName',
      text: jot.title,
    });
  }

  // Merge time-sig + tempo change events into the timeline along with note events.
  type Pending =
    | { tick: number; kind: 'noteOn' | 'noteOff'; note: number; velocity: number }
    | { tick: number; kind: 'timeSig'; count: number; unit: number }
    | { tick: number; kind: 'tempo'; bpm: number };

  const merged: Pending[] = [
    ...abs.map<Pending>((e) => ({ ...e })),
    ...tsChanges.map<Pending>((t) => ({
      tick: t.tick,
      kind: 'timeSig',
      count: t.count,
      unit: t.unit,
    })),
    ...tempoChanges.map<Pending>((t) => ({
      tick: t.tick,
      kind: 'tempo',
      bpm: t.bpm,
    })),
  ];
  merged.sort((a, b) => {
    if (a.tick !== b.tick) return a.tick - b.tick;
    // Meta events first at a tick boundary; tempo before time-sig is
    // arbitrary (both are meta) but stable so output is deterministic.
    const metaRank = (k: Pending['kind']) =>
      k === 'tempo' ? 0 : k === 'timeSig' ? 1 : k === 'noteOff' ? 2 : 3;
    return metaRank(a.kind) - metaRank(b.kind);
  });

  let lastTick = 0;
  for (const ev of merged) {
    const dt = ev.tick - lastTick;
    if (ev.kind === 'tempo') {
      trackEvents.push({
        deltaTime: dt,
        meta: true,
        type: 'setTempo',
        microsecondsPerBeat: Math.max(1, Math.round(60_000_000 / ev.bpm)),
      });
    } else if (ev.kind === 'timeSig') {
      trackEvents.push({
        deltaTime: dt,
        meta: true,
        type: 'timeSignature',
        numerator: ev.count,
        denominator: ev.unit,
        metronome: 24,
        thirtyseconds: 8,
      });
    } else if (ev.kind === 'noteOn') {
      trackEvents.push({
        deltaTime: dt,
        type: 'noteOn',
        noteNumber: ev.note,
        velocity: ev.velocity,
        channel,
      });
    } else {
      trackEvents.push({
        deltaTime: dt,
        type: 'noteOff',
        noteNumber: ev.note,
        velocity: ev.velocity,
        channel,
      });
    }
    lastTick = ev.tick;
  }

  trackEvents.push({ deltaTime: 0, meta: true, type: 'endOfTrack' });

  const bytes = writeMidi({
    header: { format: 1, numTracks: 1, ticksPerBeat: TICKS_PER_BEAT },
    tracks: [trackEvents],
  });
  return new Uint8Array(bytes);
}

// ---------- Helpers ----------

function resolveMidiNote(note: StructNote, instrument: Instrument): number | undefined {
  // [B4] Per-note override wins.
  if (note.midiNote !== undefined) return note.midiNote;
  if (instrument.midi?.note !== undefined) return instrument.midi.note;
  return defaultMidiNote(note.lane, new Set(note.modifiers as Modifier[]));
}

function resolveVelocity(note: StructNote, opts: Required<ToMidiOptions>): number {
  if (typeof note.velocity === 'number') return note.velocity;

  let baseline = opts.defaultVelocity;
  const vol = note.vol as Volume | undefined;
  if (typeof vol === 'string') {
    baseline = VOLUME_TO_VELOCITY[vol] ?? baseline;
  }

  if (note.modifiers.includes('a')) baseline += opts.accentBoost;
  if (note.modifiers.includes('g')) baseline -= opts.ghostReduction;

  return baseline;
}

function clampVelocity(v: number): number {
  if (!Number.isFinite(v)) return 1;
  if (v < 1) return 1;
  if (v > 127) return 127;
  return Math.round(v);
}
