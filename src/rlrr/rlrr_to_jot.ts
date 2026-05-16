/**
 * RLRR -> Drumjot Jot conversion.
 *
 * Assumptions (tagged [R#] inline):
 *  [R1] RLRR has no notion of time signature; we default to 4/4 (configurable).
 *  [R2] Events are quantized onto a configurable grid (default sixteenths)
 *       relative to the first bpm event's tempo. Tempo CHANGES at runtime
 *       are honoured for second->beat conversion, but the grid is uniform
 *       in beat-space.
 *  [R3] Multiple events at the same quantized slot collapse into a `simul`.
 *  [R4] Each event preserves its original RLRR `name`, `vel` and `loc` via
 *       a custom `metadata.rlrr` field on the Note so a round trip back to
 *       RLRR is lossless.
 *  [R5] Drums whose class is unknown to our `CLASS_TO_DRUM` table get a
 *       deterministic letter (last 3 chars of the class name's hash, mapped
 *       into a-z) and an `instrumentMapping` entry naming them after the
 *       original drum class.
 *  [R6] `audioFileData`, `recordingMetadata` and the original `instruments`
 *       array are preserved verbatim on `jot.globalMetadata.rlrr` so a
 *       subsequent `jotToRlrr` round-trips them.
 */
import {
  Bar,
  Element,
  Instrument,
  Jot,
  Metadata,
  Modifier,
  Note,
  Simultaneity,
  TimeSignature,
  Voice,
} from 'src/dsl';
import { CLASS_TO_DRUM, describeDrum, instanceNameToClass } from './drums';
import {
  RlrrFile,
  eventTimeSeconds,
} from './schema';

export type RlrrToJotOptions = {
  /** Quantization grid (sub-divisions per quarter note); 16 = sixteenths. */
  gridDivision?: number;
  /** Time signature to assume (RLRR doesn't store one). */
  timeSignature?: TimeSignature;
};

const DEFAULTS: Required<RlrrToJotOptions> = {
  gridDivision: 16,
  timeSignature: { count: 4, unit: 4 },
};

type TempoSegment = {
  startSeconds: number;
  startBeats: number;
  bpm: number;
};

export function rlrrToJot(rlrr: RlrrFile, options: RlrrToJotOptions = {}): Jot {
  const opts = { ...DEFAULTS, ...options };

  const tempoTimeline = buildTempoTimeline(rlrr);
  const initialBpm = tempoTimeline[0]?.bpm ?? 120;

  const time = opts.timeSignature;
  const barBeats = (time.count * 4) / time.unit;
  // grid spacing in beats (quarter-note units). gridDivision=16 -> 0.25 beats.
  const gridBeats = 4 / opts.gridDivision;
  if (!Number.isFinite(gridBeats) || gridBeats <= 0) {
    throw new Error(`Invalid gridDivision: ${opts.gridDivision}`);
  }

  type SlotNote = {
    note: Note;
    /** Lower drum names (e.g. snare) sort below cymbals - stable order in output. */
    sortKey: string;
  };
  // bar index -> slot index -> notes at that slot
  const slots = new Map<number, Map<number, SlotNote[]>>();
  const usedClasses = new Set<string>();

  for (const event of rlrr.events) {
    const seconds = eventTimeSeconds(event);
    const beats = secondsToBeats(seconds, tempoTimeline);
    const snapped = Math.round(beats / gridBeats) * gridBeats;
    const barIdx = Math.max(0, Math.floor(snapped / barBeats));
    const slotIdx = Math.round((snapped - barIdx * barBeats) / gridBeats);

    const cls = instanceNameToClass(event.name);
    if (cls) usedClasses.add(cls);
    const note = buildNote(event);
    const sortKey = `${cls ?? 'zzz'}:${event.name}`;

    let bucket = slots.get(barIdx);
    if (!bucket) {
      bucket = new Map();
      slots.set(barIdx, bucket);
    }
    let slot = bucket.get(slotIdx);
    if (!slot) {
      slot = [];
      bucket.set(slotIdx, slot);
    }
    slot.push({ note, sortKey });
  }

  // Determine bar count: at least one bar; otherwise enough to fit the latest event.
  const maxBar = slots.size > 0 ? Math.max(...slots.keys()) : 0;
  const barCount = Math.max(1, maxBar + 1);
  const slotsPerBar = Math.max(1, Math.round(barBeats / gridBeats));

  const bars: Bar[] = [];
  for (let bi = 0; bi < barCount; bi++) {
    const bucket = slots.get(bi);
    const elements: Element[] = [];
    for (let si = 0; si < slotsPerBar; si++) {
      const slot = bucket?.get(si);
      if (!slot || slot.length === 0) {
        elements.push({ kind: 'rest' });
        continue;
      }
      slot.sort((a, b) => a.sortKey.localeCompare(b.sortKey));
      if (slot.length === 1) {
        elements.push(slot[0].note);
      } else {
        const simul: Simultaneity = { kind: 'simul', elements: slot.map((s) => s.note) };
        elements.push(simul);
      }
    }
    bars.push({ elements });
  }

  // Strip trailing all-rest bars.
  while (bars.length > 1 && bars[bars.length - 1].elements.every((e) => e.kind === 'rest')) {
    bars.pop();
  }

  // Build an instrument mapping for every unique class we saw.
  const instrumentMapping: Record<string, Instrument> = {};
  for (const cls of usedClasses) {
    const descriptor = CLASS_TO_DRUM[cls];
    if (descriptor && !instrumentMapping[descriptor.pitch]) {
      instrumentMapping[descriptor.pitch] = {
        name: descriptor.name,
        midi: { note: descriptor.midi },
      };
    }
  }

  // [R6] Preserve original RLRR sidecar data on global metadata.
  const rlrrSidecar: Record<string, unknown> = {
    version: rlrr.version,
    authoringTool: rlrr.authoringTool,
    instruments: rlrr.instruments,
  };
  if (rlrr.audioFileData) rlrrSidecar.audioFileData = rlrr.audioFileData;
  if (rlrr.recordingMetadata) rlrrSidecar.recordingMetadata = rlrr.recordingMetadata;

  const globalMetadata: Metadata = {
    bpm: initialBpm,
    time,
    instrumentMapping,
    rlrr: rlrrSidecar,
  };

  const voice: Voice = { bars };
  return {
    title: rlrr.recordingMetadata?.title ?? '',
    globalMetadata,
    voices: [voice],
  };
}

// ---------- helpers ----------

function buildTempoTimeline(rlrr: RlrrFile): TempoSegment[] {
  const out: TempoSegment[] = [];
  let beats = 0;
  let lastSeconds = 0;
  let lastBpm = 120;
  let first = true;
  const events = [...(rlrr.bpmEvents ?? [])].sort((a, b) => a.time - b.time);
  if (events.length === 0 || events[0].time > 0) {
    events.unshift({ bpm: 120, time: 0 });
  }
  for (const ev of events) {
    if (first) {
      out.push({ startSeconds: ev.time, startBeats: 0, bpm: ev.bpm });
      lastSeconds = ev.time;
      lastBpm = ev.bpm;
      first = false;
      continue;
    }
    const dt = ev.time - lastSeconds;
    beats += (dt * lastBpm) / 60;
    out.push({ startSeconds: ev.time, startBeats: beats, bpm: ev.bpm });
    lastSeconds = ev.time;
    lastBpm = ev.bpm;
  }
  return out;
}

function secondsToBeats(seconds: number, timeline: TempoSegment[]): number {
  let i = 0;
  while (i + 1 < timeline.length && timeline[i + 1].startSeconds <= seconds) i++;
  const seg = timeline[i];
  const dt = Math.max(0, seconds - seg.startSeconds);
  return seg.startBeats + (dt * seg.bpm) / 60;
}

function buildNote(event: { name: string; vel: number; loc: number; midi?: number }): Note {
  const descriptor = describeDrum(event.name);
  const pitch = descriptor?.pitch ?? deriveFallbackPitch(event.name);
  const modifiers: Modifier[] = descriptor?.modifiers ? [...descriptor.modifiers] : [];

  // [R7] Velocity-driven accents/ghosts, mirroring the MIDI converter's policy.
  if (event.vel >= 100 && !modifiers.includes('a')) modifiers.push('a');
  else if (event.vel < 40 && !modifiers.includes('g')) modifiers.push('g');

  const note: Note = { kind: 'note', pitch };
  if (modifiers.length > 0) note.modifiers = modifiers;
  // Preserve everything we'd need to reconstruct an identical RLRR event.
  note.metadata = {
    rlrr: { name: event.name, vel: event.vel, loc: event.loc },
    ...(event.midi !== undefined ? { midi: { note: event.midi, velocity: event.vel } } : {
      midi: { note: descriptor?.midi ?? 38, velocity: event.vel },
    }),
  } as Metadata;
  return note;
}

/**
 * [R5] Deterministically derive a letter for an unknown drum class so we can
 * still produce a valid Jot. Letters from the end of the alphabet are used
 * to avoid clashes with the standard kit-letter assignments.
 */
function deriveFallbackPitch(name: string): string {
  let h = 0;
  for (let i = 0; i < name.length; i++) {
    h = (h * 31 + name.charCodeAt(i)) | 0;
  }
  const slot = Math.abs(h) % 26;
  return String.fromCharCode('z'.charCodeAt(0) - slot);
}
