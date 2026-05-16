/**
 * Paradiddle `.rlrr` -> MIDI conversion. This direction is NOT present in the
 * Python tool; it's the natural inverse of `midiToRlrr` and gives us a way
 * to take a fully-authored Paradiddle song back into a standard DAW.
 *
 * Assumptions (tagged [T#] inline):
 *  [T1] PPQN fixed at 480, matching `src/midi/to_midi.ts`.
 *  [T2] Drum hits land on MIDI channel 10 (index 9).
 *  [T3] Each event becomes a noteOn followed by a noteOff one tick later
 *       (drums are one-shot strikes).
 *  [T4] Note number resolution prefers `event.midi` (which `midiToRlrr`
 *       always writes), then a lookup via the drum class on the matching
 *       `instruments[].class`, falling back to GM defaults from
 *       `CLASS_TO_DRUM`. Events that can't resolve are dropped.
 *  [T5] Tempo changes from `bpmEvents` are emitted at their corresponding
 *       tick. The first bpm event always lands at tick 0 even when its
 *       source `time` is non-zero (we synthesize a tick-0 anchor at that
 *       BPM).
 *  [T6] An initial 4/4 time signature meta event is emitted at tick 0
 *       because Standard MIDI Files conventionally include one. RLRR has
 *       no time signature so this is the only sensible default.
 */
import { MidiEvent, writeMidi } from 'midi-file';
import { CLASS_TO_DRUM, describeDrum } from './drums';
import {
  RlrrBpmEvent,
  RlrrFile,
  RlrrInstrument,
  eventTimeSeconds,
} from './schema';

export const TICKS_PER_BEAT = 480; // [T1]

export type RlrrToMidiOptions = {
  drumChannel?: number; // 1-based
  /** Trim the produced MIDI to the last note's tick. Default: true. */
  trimTrailing?: boolean;
};

const DEFAULTS: Required<RlrrToMidiOptions> = {
  drumChannel: 10,
  trimTrailing: true,
};

type TempoSegment = { seconds: number; ticks: number; microsPerBeat: number };

export function rlrrToMidi(rlrr: RlrrFile, options: RlrrToMidiOptions = {}): Uint8Array {
  const opts = { ...DEFAULTS, ...options };
  const channel = opts.drumChannel - 1;

  // Build tempo timeline as {seconds, ticks, microsPerBeat} segments. The
  // tick column is what we'll convert seconds -> ticks with.
  const timeline = buildTempoTimeline(rlrr.bpmEvents);

  // Pre-build a lookup table from instrument name -> MIDI note.
  const nameToMidi = new Map<string, number>();
  for (const inst of rlrr.instruments) {
    const note = midiForInstrument(inst);
    if (note !== undefined) nameToMidi.set(inst.name, note);
  }

  type Pending =
    | { tick: number; kind: 'noteOn' | 'noteOff'; note: number; velocity: number }
    | { tick: number; kind: 'tempo'; microsPerBeat: number };

  const pending: Pending[] = [];

  // [T5] Initial tempo: always at tick 0.
  pending.push({
    tick: 0,
    kind: 'tempo',
    microsPerBeat: timeline[0].microsPerBeat,
  });
  // Subsequent tempo changes.
  for (let i = 1; i < timeline.length; i++) {
    pending.push({
      tick: timeline[i].ticks,
      kind: 'tempo',
      microsPerBeat: timeline[i].microsPerBeat,
    });
  }

  // [T4] Convert each RLRR event into a noteOn/noteOff pair.
  for (const ev of rlrr.events) {
    const seconds = eventTimeSeconds(ev);
    const tick = secondsToTicks(seconds, timeline);
    const note =
      ev.midi !== undefined
        ? ev.midi
        : nameToMidi.get(ev.name);
    if (note === undefined) continue;
    pending.push({ tick, kind: 'noteOn', note, velocity: clampVelocity(ev.vel) });
    pending.push({ tick: tick + 1, kind: 'noteOff', note, velocity: 0 }); // [T3]
  }

  pending.sort((a, b) => {
    if (a.tick !== b.tick) return a.tick - b.tick;
    // Tempo before any note at the same tick; off before on.
    const rank = (p: Pending) =>
      p.kind === 'tempo' ? 0 : p.kind === 'noteOff' ? 1 : 2;
    return rank(a) - rank(b);
  });

  // Assemble the track.
  const track: MidiEvent[] = [];
  // [T6] Always start with a 4/4 time signature.
  track.push({
    deltaTime: 0,
    meta: true,
    type: 'timeSignature',
    numerator: 4,
    denominator: 4,
    metronome: 24,
    thirtyseconds: 8,
  });

  let lastTick = 0;
  for (const p of pending) {
    const dt = p.tick - lastTick;
    if (p.kind === 'tempo') {
      track.push({
        deltaTime: dt,
        meta: true,
        type: 'setTempo',
        microsecondsPerBeat: p.microsPerBeat,
      });
    } else if (p.kind === 'noteOn') {
      track.push({
        deltaTime: dt,
        type: 'noteOn',
        noteNumber: p.note,
        velocity: p.velocity,
        channel,
      });
    } else {
      track.push({
        deltaTime: dt,
        type: 'noteOff',
        noteNumber: p.note,
        velocity: p.velocity,
        channel,
      });
    }
    lastTick = p.tick;
  }
  track.push({ deltaTime: 0, meta: true, type: 'endOfTrack' });

  const bytes = writeMidi({
    header: { format: 1, numTracks: 1, ticksPerBeat: TICKS_PER_BEAT },
    tracks: [track],
  });
  return new Uint8Array(bytes);
}

// ---------- helpers ----------

function buildTempoTimeline(bpmEvents: RlrrBpmEvent[]): TempoSegment[] {
  const sorted = [...(bpmEvents ?? [])].sort((a, b) => a.time - b.time);
  if (sorted.length === 0) {
    return [{ seconds: 0, ticks: 0, microsPerBeat: 500_000 }];
  }
  // The first bpm event anchors at tick 0 even if its source time is non-zero.
  const out: TempoSegment[] = [];
  let micros = bpmToMicros(sorted[0].bpm);
  out.push({ seconds: sorted[0].time, ticks: 0, microsPerBeat: micros });

  for (let i = 1; i < sorted.length; i++) {
    const prev = out[out.length - 1];
    const dt = sorted[i].time - prev.seconds;
    const dTicks = (dt * 1_000_000 * TICKS_PER_BEAT) / prev.microsPerBeat;
    micros = bpmToMicros(sorted[i].bpm);
    out.push({
      seconds: sorted[i].time,
      ticks: Math.round(prev.ticks + dTicks),
      microsPerBeat: micros,
    });
  }

  return out;
}

function secondsToTicks(seconds: number, timeline: TempoSegment[]): number {
  let i = 0;
  while (i + 1 < timeline.length && timeline[i + 1].seconds <= seconds) i++;
  const seg = timeline[i];
  const dt = Math.max(0, seconds - seg.seconds);
  return Math.round(seg.ticks + (dt * 1_000_000 * TICKS_PER_BEAT) / seg.microsPerBeat);
}

function midiForInstrument(inst: RlrrInstrument): number | undefined {
  const descriptor = describeDrum(inst.name) ?? CLASS_TO_DRUM[inst.class];
  return descriptor?.midi;
}

function bpmToMicros(bpm: number): number {
  return Math.max(1, Math.round(60_000_000 / bpm));
}

function clampVelocity(v: number): number {
  if (!Number.isFinite(v)) return 1;
  if (v < 1) return 1;
  if (v > 127) return 127;
  return Math.round(v);
}
