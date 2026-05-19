/**
 * RLRR -> Drumjot Jot conversion.
 *
 * Assumptions (tagged [R#] inline):
 *  [R1] RLRR has no notion of time signature; we default to 4/4 (configurable).
 *  [R2] Events are quantized onto a configurable grid (default sixteenths)
 *       relative to the first bpm event's tempo. Tempo CHANGES at runtime
 *       are honoured for second->beat conversion (the grid is uniform in
 *       beat-space) AND surfaced as per-bar `{{ bpm: ... }}` metadata so
 *       playback / the playhead / the waveform follow the song's real
 *       tempo instead of running the whole chart at the initial bpm.
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
import { defaultKindForPitch } from 'src/instruments';
import { CLASS_TO_DRUM, describeDrum, instanceNameToClass } from './drums';
import { allocateFallbackLetters } from './fallback';
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

  // Pre-real-tempo intro -> startOffset lead-in (see computeLeadInSeconds).
  const leadInSec = computeLeadInSeconds(rlrr);
  const tempoTimeline = buildTempoTimeline(rlrr, leadInSec);
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

  // Per-song deterministic allocation of `instanceName -> pitch`. Unknown
  // drum classes get unique fallback letters that don't collide with
  // CLASS_TO_DRUM or with one another.
  const pitchByName = allocateFallbackLetters(rlrr.events.map((e) => e.name));

  for (const event of rlrr.events) {
    // Times are rebased so the real-tempo downbeat sits at beat 0; the
    // dropped intro is reinstated as `startOffset` on global metadata below.
    const seconds = Math.max(0, eventTimeSeconds(event) - leadInSec);
    const beats = secondsToBeats(seconds, tempoTimeline);
    const snapped = Math.round(beats / gridBeats) * gridBeats;
    const barIdx = Math.max(0, Math.floor(snapped / barBeats));
    const slotIdx = Math.round((snapped - barIdx * barBeats) / gridBeats);

    const cls = instanceNameToClass(event.name);
    if (cls) usedClasses.add(cls);
    const note = buildNote(event, pitchByName.get(event.name));
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
  // Effective tempo at each bar's start beat, emitted as a sticky per-bar
  // `{{ bpm }}` override whenever it changes. Bar 0 is covered by
  // `globalMetadata.bpm` (= initialBpm = the tempo at beat 0), so it never
  // needs its own override; later bars only carry one on a change.
  let prevBpm = initialBpm;
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
    const bar: Bar = { elements };
    const barBpm = bpmAtBeat(bi * barBeats, tempoTimeline);
    if (bi > 0 && barBpm !== prevBpm) {
      bar.metadata = { bpm: barBpm };
    }
    prevBpm = barBpm;
    bars.push(bar);
  }

  // Strip trailing all-rest bars.
  while (bars.length > 1 && bars[bars.length - 1].elements.every((e) => e.kind === 'rest')) {
    bars.pop();
  }

  // Build an instrument mapping. Canonical CLASS_TO_DRUM entries win; any
  // unknown instances get a friendly fallback entry that reuses the
  // allocated letter (so the inline pitches and the mapping agree).
  const instrumentMapping: Record<string, Instrument> = {};
  for (const cls of usedClasses) {
    const descriptor = CLASS_TO_DRUM[cls];
    if (descriptor && !instrumentMapping[descriptor.pitch]) {
      instrumentMapping[descriptor.pitch] = {
        kind: defaultKindForPitch(descriptor.pitch),
        name: descriptor.name,
        midi: { note: descriptor.midi },
      };
    }
  }
  for (const [name, pitch] of pitchByName) {
    if (instrumentMapping[pitch]) continue;
    const cls = instanceNameToClass(name);
    instrumentMapping[pitch] = {
      kind: defaultKindForPitch(pitch),
      name: cls ?? name,
    };
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
    // Recording's intro before the real-tempo downbeat: playback delays its
    // schedule by this so the drums hit at the same wall-clock offset as the
    // audio (and the audio plays its own intro during the lead-in).
    ...(leadInSec > 0 ? { startOffset: leadInSec } : {}),
  };

  const voice: Voice = { bars };
  return {
    title: rlrr.recordingMetadata?.title ?? '',
    globalMetadata,
    voices: [voice],
  };
}

// ---------- helpers ----------

/**
 * Paradiddle charts frequently carry a placeholder `{ bpm: 120, time: 0 }`
 * left by the authoring tool, with the song's real tempo appearing as a
 * later bpm event at the first downbeat — often *exactly* the first note's
 * time. Treating that pre-real-tempo region as 120-bpm music shifts the
 * whole drum grid against the audio, because playback can only change tempo
 * at bar boundaries (per-bar `{{ bpm }}`) and so cannot reproduce a mid-bar
 * tempo change. The fix: the lead-in is the time of the last bpm event at or
 * before the first drum onset (everything before it is intro). It is dropped
 * from the tempo timeline and surfaced as `globalMetadata.startOffset`, so
 * playback delays its schedule by it and the audio plays its own intro while
 * the grid waits. A chart whose real tempo already starts at time 0 yields
 * `leadInSec === 0` and is unaffected. A note that precedes the later bpm
 * event (a genuine pickup) also keeps `leadInSec` at 0, since that bpm event
 * is then *after* the first onset.
 */
function computeLeadInSeconds(rlrr: RlrrFile): number {
  const events = rlrr.events ?? [];
  if (events.length === 0) return 0;
  let firstEventSec = Number.POSITIVE_INFINITY;
  for (const ev of events) {
    const t = eventTimeSeconds(ev);
    if (t < firstEventSec) firstEventSec = t;
  }
  if (!Number.isFinite(firstEventSec)) return 0;
  let leadIn = 0;
  for (const ev of rlrr.bpmEvents ?? []) {
    if (ev.time <= firstEventSec + 1e-6 && ev.time > leadIn) leadIn = ev.time;
  }
  return leadIn;
}

function buildTempoTimeline(rlrr: RlrrFile, leadInSec: number): TempoSegment[] {
  const out: TempoSegment[] = [];
  let beats = 0;
  let lastSeconds = 0;
  let lastBpm = 120;
  let first = true;
  // Drop bpm events before the lead-in and rebase the rest so the governing
  // real-tempo event sits at time 0. With leadInSec === 0 this is a no-op.
  const events = [...(rlrr.bpmEvents ?? [])]
    .filter((e) => e.time >= leadInSec - 1e-6)
    .map((e) => ({ bpm: e.bpm, time: Math.max(0, e.time - leadInSec) }))
    .sort((a, b) => a.time - b.time);
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

/**
 * Effective tempo at a given beat position: the bpm of the last tempo
 * segment whose `startBeats` is at or before `beat`. Beat-space analogue
 * of {@link secondsToBeats}'s segment scan; used to stamp each bar with
 * the tempo in force at its start.
 */
function bpmAtBeat(beat: number, timeline: TempoSegment[]): number {
  let i = 0;
  while (i + 1 < timeline.length && timeline[i + 1].startBeats <= beat) i++;
  return timeline[i].bpm;
}

function secondsToBeats(seconds: number, timeline: TempoSegment[]): number {
  let i = 0;
  while (i + 1 < timeline.length && timeline[i + 1].startSeconds <= seconds) i++;
  const seg = timeline[i];
  const dt = Math.max(0, seconds - seg.startSeconds);
  return seg.startBeats + (dt * seg.bpm) / 60;
}

function buildNote(
  event: { name: string; vel: number; loc: number; midi?: number },
  allocatedPitch: string | undefined
): Note {
  const descriptor = describeDrum(event.name);
  // [R5] If neither the canonical map nor the per-song allocator has a
  // pitch for this instrument we fall back to `z`. The allocator should
  // always have an entry though - it's seeded from the event list - so
  // this is purely defensive.
  const pitch = descriptor?.pitch ?? allocatedPitch ?? 'z';
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
