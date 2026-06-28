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
 *       subsequent `writeRlrr` round-trips them.
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
  TempoEvent,
  TimeSignature,
  Layer,
} from 'src/schema/dsl/dsl';
import { defaultKindForLane } from 'src/instruments/instruments';
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

export function parseRlrr(rlrr: RlrrFile, options: RlrrToJotOptions = {}): Jot {
  const opts = { ...DEFAULTS, ...options };

  // preRollSec is literally the audio time of the first drum onset —
  // the three-epoch model the rest of the codebase agrees on. The
  // initial bpm is the most recent tempo event at or before that point;
  // that's what governs bar 1's downbeat. See `computePreRollSec` and
  // `chooseInitialBpm` for the detailed rationale (a placeholder 120-bpm
  // event during a guitar intro must not become the drum-grid anchor).
  const preRollSec = computePreRollSec(rlrr);
  const initialBpm = chooseInitialBpm(rlrr, preRollSec);
  const tempoTimeline = buildTempoTimeline(rlrr, preRollSec, initialBpm);

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

  // Per-song deterministic allocation of `instanceName -> lane`. Unknown
  // drum classes get unique fallback letters that don't collide with
  // CLASS_TO_DRUM or with one another.
  const laneByName = allocateFallbackLetters(rlrr.events.map((e) => e.name));

  for (const event of rlrr.events) {
    // Times are rebased so the real-tempo downbeat sits at beat 0; the
    // dropped intro is reinstated as `preRollSec` on global metadata below.
    const seconds = Math.max(0, eventTimeSeconds(event) - preRollSec);
    const beats = secondsToBeats(seconds, tempoTimeline);
    const snapped = Math.round(beats / gridBeats) * gridBeats;
    const barIdx = Math.max(0, Math.floor(snapped / barBeats));
    const slotIdx = Math.round((snapped - barIdx * barBeats) / gridBeats);

    const cls = instanceNameToClass(event.name);
    if (cls) usedClasses.add(cls);
    const note = buildNote(event, laneByName.get(event.name));
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

  // Emit tempo events at each RLRR `bpmEvent` (post-drumsT0 rebase),
  // anchored at (barIndex, beat-within-bar) so mid-bar tempo changes
  // survive into the runtime tempo timeline. The initial tempo is the
  // first event, at the drums-enter downbeat (bar 0, beat 0); there is no
  // `globalMetadata.bpm` any more. tempoTimeline[1..] are the genuine
  // later changes.
  const tempoEvents: TempoEvent[] = [];
  // Materialise the initial tempo as the first event only when it differs
  // from the 120 default (matching the DSL hoist + from_midi); a default
  // chart relies on `tempo.initialBpm`'s 120 fallback.
  if (Math.abs(initialBpm - 120) > 1e-6) {
    tempoEvents.push({ barIndex: 0, beat: 0, bpm: initialBpm });
  }
  for (let i = 1; i < tempoTimeline.length; i++) {
    const seg = tempoTimeline[i];
    const barIdx = Math.max(0, Math.floor(seg.startBeats / barBeats));
    const beat = Math.max(0, seg.startBeats - barIdx * barBeats);
    tempoEvents.push({ barIndex: barIdx, beat, bpm: seg.bpm });
  }

  // Strip trailing all-rest bars.
  while (bars.length > 1 && bars[bars.length - 1].elements.every((e) => e.kind === 'rest')) {
    bars.pop();
  }

  // Build an instrument mapping. Canonical CLASS_TO_DRUM entries win; any
  // unknown instances get a friendly fallback entry that reuses the
  // allocated letter (so the inline lanes and the mapping agree).
  const instrumentMapping: Record<string, Instrument> = {};
  for (const cls of usedClasses) {
    const descriptor = CLASS_TO_DRUM[cls];
    if (descriptor && !instrumentMapping[descriptor.lane]) {
      instrumentMapping[descriptor.lane] = {
        kind: defaultKindForLane(descriptor.lane),
        name: descriptor.name,
        midi: { note: descriptor.midi },
      };
    }
  }
  for (const [name, lane] of laneByName) {
    if (instrumentMapping[lane]) continue;
    const cls = instanceNameToClass(name);
    instrumentMapping[lane] = {
      kind: defaultKindForLane(lane),
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
    time,
    instrumentMapping,
    rlrr: rlrrSidecar,
    // Recording's intro before the real-tempo downbeat, as a jot-time audio
    // start (negative): playback offsets by this so the drums hit at the same
    // wall-clock offset as the audio (and the audio plays its own intro
    // during the lead-in).
    ...(preRollSec > 0 ? { songLeadIn: -preRollSec } : {}),
  };

  const layer: Layer = { bars };
  const jot: Jot = {
    title: rlrr.recordingMetadata?.title ?? '',
    globalMetadata,
    layers: [layer],
  };
  if (tempoEvents.length > 0) jot.tempoEvents = tempoEvents;
  return jot;
}

// ---------- helpers ----------

/**
 * `preRollSec` per the three-epoch model: literally the audio time of
 * the first drum onset, period. RLRR has no separate "drum start"
 * signal, so we infer it from the earliest drum event in the file.
 *
 * This replaces the older "last bpm event at or before first onset"
 * heuristic, which conflated two distinct things: the audio-time when
 * the drum grid begins (drumsT0) and the audio-time when the song's
 * real tempo took effect (which can sit during a guitar/vocal intro
 * before drums enter — that's a different epoch). The bpm-event time
 * is still used as the bar-1 starting tempo via `chooseInitialBpm`; only
 * the time-origin moved.
 *
 * A chart whose first drum sits at time 0 returns 0 — no lead-in.
 */
function computePreRollSec(rlrr: RlrrFile): number {
  const events = rlrr.events ?? [];
  if (events.length === 0) return 0;
  let firstEventSec = Number.POSITIVE_INFINITY;
  for (const ev of events) {
    const t = eventTimeSeconds(ev);
    if (t < firstEventSec) firstEventSec = t;
  }
  return Number.isFinite(firstEventSec) ? Math.max(0, firstEventSec) : 0;
}

/**
 * The bpm in effect at bar 1's downbeat — the bpm of the latest tempo
 * event at or before `preRollSec`. Paradiddle charts commonly carry a
 * placeholder `{ bpm: 120, time: 0 }` left by the authoring tool with
 * the song's real tempo appearing as a later bpm event at (or just
 * before) the first drum. Choosing the latest-at-or-before-drums event
 * picks the real tempo and ignores the placeholder.
 *
 * Falls back to 120 if no bpm event exists or none is at-or-before
 * preRollSec (which means the first bpm event is itself a pickup after
 * the first drum — rare, but treat as standard 120-bpm until that event
 * lands).
 */
function chooseInitialBpm(rlrr: RlrrFile, preRollSec: number): number {
  let bpm = 120;
  let bestTime = Number.NEGATIVE_INFINITY;
  for (const ev of rlrr.bpmEvents ?? []) {
    if (ev.time <= preRollSec + 1e-6 && ev.time > bestTime) {
      bestTime = ev.time;
      bpm = ev.bpm;
    }
  }
  return bpm;
}

function buildTempoTimeline(
  rlrr: RlrrFile,
  preRollSec: number,
  initialBpm: number,
): TempoSegment[] {
  const out: TempoSegment[] = [];
  let beats = 0;
  let lastSeconds = 0;
  let lastBpm = initialBpm;
  // Drop bpm events strictly before preRollSec — those govern the
  // pre-drum lead-in audio that isn't represented in the bar grid. The
  // event AT preRollSec (if any) collapses with the synthesised t=0
  // bootstrap below. Subsequent bpm events get rebased so the timeline's
  // origin is bar 1's downbeat.
  const events = [...(rlrr.bpmEvents ?? [])]
    .filter((e) => e.time > preRollSec + 1e-6)
    .map((e) => ({ bpm: e.bpm, time: Math.max(0, e.time - preRollSec) }))
    .sort((a, b) => a.time - b.time);
  // Synthesise a t=0 segment carrying `initialBpm` so secondsToBeats /
  // bpmAtBeat always have a starting tempo to anchor against, even
  // when no original bpm event sat at-or-before preRollSec.
  out.push({ startSeconds: 0, startBeats: 0, bpm: initialBpm });
  for (const ev of events) {
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

function buildNote(
  event: { name: string; vel: number; loc: number; midi?: number },
  allocatedLane: string | undefined
): Note {
  const descriptor = describeDrum(event.name);
  // [R5] If neither the canonical map nor the per-song allocator has a
  // lane for this instrument we fall back to `z`. The allocator should
  // always have an entry though - it's seeded from the event list - so
  // this is purely defensive.
  const lane = descriptor?.lane ?? allocatedLane ?? 'z';
  const modifiers: Modifier[] = descriptor?.modifiers ? [...descriptor.modifiers] : [];

  // [R7] Velocity-driven accents/ghosts, mirroring the MIDI converter's policy.
  if (event.vel >= 100 && !modifiers.includes('a')) modifiers.push('a');
  else if (event.vel < 40 && !modifiers.includes('g')) modifiers.push('g');

  const note: Note = { kind: 'note', lane };
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
