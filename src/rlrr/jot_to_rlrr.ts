/**
 * Drumjot Jot -> Paradiddle `.rlrr` conversion.
 *
 * Assumptions (tagged [S#] inline):
 *  [S1] Tempo is taken from `globalMetadata.bpm` (number form only). If a
 *       `BpmTransition` is set, only its `start` (else `end`) is used as a
 *       single tempo; we don't yet emit a multi-segment bpm timeline.
 *  [S2] The drum kit (`instruments`) is taken from `globalMetadata.rlrr
 *       .instruments` if present (preserves a round trip), else falls back
 *       to `DEFAULT_INSTRUMENTS`.
 *  [S3] For each note we resolve a target drum instance name in this
 *       precedence order:
 *         (a) `note.metadata.rlrr.name`                (exact round-trip),
 *         (b) `track.instrument.midi.note` matched to an instrument in the
 *             kit via the GM table,
 *         (c) `pitchToClass(pitch, modifiers)` then pick the first
 *             instrument of that class in the kit.
 *       Notes that don't resolve to any kit instrument are dropped.
 *  [S4] Velocity comes from `metadata.rlrr.vel` if present, else
 *       `metadata.midi.velocity`, else the `vol`-bucket mapping with
 *       `:a` / `:g` adjustments.
 *  [S5] All voices in the Jot collapse onto a single RLRR event stream;
 *       Paradiddle has no concept of independent voices.
 *  [S6] `audioFileData` and `recordingMetadata` are taken from
 *       `globalMetadata.rlrr` when present; the Jot's `title` always
 *       overrides `recordingMetadata.title`.
 */
import { Jot, Volume } from 'src/dsl';
import { RenderedJot, ResolvedNote, ResolvedTrack } from 'src/jot';
import {
  CLASS_TO_DRUM,
  describeDrum,
  pitchToClass,
} from './drums';
import {
  DEFAULT_INSTRUMENTS,
  RLRR_AUTHORING_TOOL,
  RLRR_VERSION,
  RlrrAudioFileData,
  RlrrEvent,
  RlrrFile,
  RlrrInstrument,
  RlrrRecordingMetadata,
  formatEventTime,
} from './schema';

export type JotToRlrrOptions = {
  defaultVelocity?: number;
  accentBoost?: number;
  ghostReduction?: number;
  /** Override the kit; if omitted, prefer `globalMetadata.rlrr.instruments` then `DEFAULT_INSTRUMENTS`. */
  instruments?: RlrrInstrument[];
  authoringTool?: string;
  /** Override `recordingMetadata` fields. Title falls back to `jot.title`. */
  recordingMetadata?: RlrrRecordingMetadata;
  audioFileData?: RlrrAudioFileData;
};

const DEFAULTS: Required<Pick<JotToRlrrOptions, 'defaultVelocity' | 'accentBoost' | 'ghostReduction' | 'authoringTool'>> = {
  defaultVelocity: 80,
  accentBoost: 24,
  ghostReduction: 32,
  authoringTool: RLRR_AUTHORING_TOOL,
};

const VOLUME_TO_VELOCITY: Record<Volume, number> = {
  pp: 16,
  p: 33,
  mp: 49,
  mf: 64,
  f: 80,
  ff: 96,
};

type Sidecar = {
  instruments?: RlrrInstrument[];
  audioFileData?: RlrrAudioFileData;
  recordingMetadata?: RlrrRecordingMetadata;
  authoringTool?: string;
};

export function jotToRlrr(jot: Jot, options: JotToRlrrOptions = {}): RlrrFile {
  const opts = { ...DEFAULTS, ...options };
  const rendered = new RenderedJot(jot);
  const resolved = rendered.resolved;

  const sidecar = (jot.globalMetadata.rlrr ?? {}) as Sidecar;
  const instruments: RlrrInstrument[] =
    options.instruments ?? sidecar.instruments ?? [...DEFAULT_INSTRUMENTS];

  const bpmVal = resolveBpm(jot);
  const events: RlrrEvent[] = [];

  // [S5] Merge all voices.
  for (const voice of resolved.voices) {
    let secondsAtBarStart = 0;
    for (const bar of voice.bars) {
      const barDurationSeconds = (bar.beats * 60) / bpmVal;
      for (const pitch of voice.pitches) {
        const track = bar.tracks[pitch];
        if (!track) continue;
        for (const note of track.notes) {
          const target = resolveInstrument(note, track, instruments);
          if (!target) continue;
          const seconds =
            secondsAtBarStart + (note.beat / Math.max(bar.beats, 1e-9)) * barDurationSeconds;
          const vel = clampVelocity(resolveVelocity(note, opts));
          const event: RlrrEvent = {
            name: target.name,
            vel,
            loc: resolveLocation(note),
            time: formatEventTime(seconds),
          };
          const midiNote = resolveMidiNote(note, target);
          if (midiNote !== undefined) event.midi = midiNote;
          events.push(event);
        }
      }
      secondsAtBarStart += barDurationSeconds;
    }
  }

  events.sort((a, b) => {
    const at = parseFloat(a.time as string);
    const bt = parseFloat(b.time as string);
    if (at !== bt) return at - bt;
    return a.name.localeCompare(b.name);
  });

  const lastEventSeconds =
    events.length > 0 ? parseFloat(events[events.length - 1].time as string) : 0;

  const recordingMetadata: RlrrRecordingMetadata = {
    ...(sidecar.recordingMetadata ?? {}),
    ...(options.recordingMetadata ?? {}),
  };
  if (jot.title) recordingMetadata.title = jot.title;
  if (recordingMetadata.length === undefined && lastEventSeconds > 0) {
    recordingMetadata.length = lastEventSeconds;
  }
  if (recordingMetadata.complexity === undefined) {
    recordingMetadata.complexity = 1;
  }

  const audioFileData: RlrrAudioFileData = {
    ...(sidecar.audioFileData ?? {}),
    ...(options.audioFileData ?? {}),
  };

  return {
    version: RLRR_VERSION,
    authoringTool: opts.authoringTool,
    recordingMetadata,
    audioFileData,
    instruments,
    events,
    bpmEvents: [{ bpm: bpmVal, time: 0 }],
  };
}

// ---------- helpers ----------

function resolveBpm(jot: Jot): number {
  const bpm = jot.globalMetadata.bpm;
  if (typeof bpm === 'number') return bpm;
  if (bpm && typeof bpm === 'object') return bpm.start ?? bpm.end ?? 120;
  return 120;
}

function resolveInstrument(
  note: ResolvedNote,
  track: ResolvedTrack,
  kit: RlrrInstrument[]
): RlrrInstrument | undefined {
  // [S3](a) explicit RLRR name in metadata wins.
  const meta = note.source.metadata as { rlrr?: { name?: string } } | undefined;
  const explicit = meta?.rlrr?.name;
  if (explicit) {
    const found = kit.find((i) => i.name === explicit);
    if (found) return found;
  }

  // [S3](b) map via the track's midi note if it matches a known kit class.
  const midiNote = track.instrument.midi?.note;
  if (midiNote !== undefined) {
    for (const inst of kit) {
      const descriptor = describeDrum(inst.name);
      if (descriptor && descriptor.midi === midiNote) return inst;
    }
  }

  // [S3](c) pitch+modifiers heuristic.
  const cls = pitchToClass(note.pitch, note.modifiers);
  if (cls) {
    const found = kit.find((i) => i.class === cls);
    if (found) return found;
  }

  return undefined;
}

function resolveVelocity(
  note: ResolvedNote,
  opts: Required<Pick<JotToRlrrOptions, 'defaultVelocity' | 'accentBoost' | 'ghostReduction'>>
): number {
  const meta = note.source.metadata as
    | {
        rlrr?: { vel?: number };
        midi?: { velocity?: number };
        vol?: Volume | { start?: Volume; end: Volume };
      }
    | undefined;
  if (typeof meta?.rlrr?.vel === 'number') return meta.rlrr.vel;
  if (typeof meta?.midi?.velocity === 'number') return meta.midi.velocity;

  let baseline = opts.defaultVelocity;
  const vol = meta?.vol;
  if (typeof vol === 'string') {
    baseline = VOLUME_TO_VELOCITY[vol] ?? baseline;
  } else if (vol && typeof vol === 'object') {
    const v = vol.start ?? vol.end;
    if (v) baseline = VOLUME_TO_VELOCITY[v] ?? baseline;
  }

  if (note.modifiers.has('a')) baseline += opts.accentBoost;
  if (note.modifiers.has('g')) baseline -= opts.ghostReduction;
  return baseline;
}

function resolveLocation(note: ResolvedNote): number {
  const meta = note.source.metadata as { rlrr?: { loc?: number } } | undefined;
  return typeof meta?.rlrr?.loc === 'number' ? meta.rlrr.loc : 0;
}

function resolveMidiNote(note: ResolvedNote, instrument: RlrrInstrument): number | undefined {
  const meta = note.source.metadata as { midi?: { note?: number } } | undefined;
  if (typeof meta?.midi?.note === 'number') return meta.midi.note;
  const descriptor =
    describeDrum(instrument.name) ?? CLASS_TO_DRUM[instrument.class];
  return descriptor?.midi;
}

function clampVelocity(v: number): number {
  if (!Number.isFinite(v)) return 1;
  if (v < 1) return 1;
  if (v > 127) return 127;
  return Math.round(v);
}
