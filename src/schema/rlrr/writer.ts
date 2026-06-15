/**
 * Drumjot Jot -> Paradiddle `.rlrr` conversion.
 *
 * Assumptions (tagged [S#] inline):
 *  [S1] Tempo is taken from `globalMetadata.bpm` (initial) plus
 *       `jot.tempoEvents` for sticky tempo changes. Each tempoEvent
 *       (mid-bar OK) becomes a separate RLRR `bpmEvent` at the event's
 *       wall-clock time; `BpmTransition` values are flattened to
 *       `start` (else `end`) since RLRR has no transition concept.
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
import { Instrument, Jot, Modifier, Volume } from 'src/schema/dsl/dsl';
import {
  ACCENT_BOOST,
  DEFAULT_VELOCITY,
  GHOST_REDUCTION,
  VOLUME_TO_VELOCITY,
} from 'src/dynamics/dynamics';
import { buildJotModel } from 'src/jot_view/jot_view_store';
import type { StructNote } from 'src/jot_view/structure/structure_store';
import { beatToSecWithinBar, buildBarTempos, initialBpm, resolveBpm } from 'src/schema/dsl/tempo';
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

// Velocity defaults come from the shared `src/dynamics.ts` so an accent
// plays back, exports to MIDI, and exports to RLRR at the same loudness.
// (Previously `accentBoost` here was 24 while MIDI export used 36.)
const DEFAULTS: Required<Pick<JotToRlrrOptions, 'defaultVelocity' | 'accentBoost' | 'ghostReduction' | 'authoringTool'>> = {
  defaultVelocity: DEFAULT_VELOCITY,
  accentBoost: ACCENT_BOOST,
  ghostReduction: GHOST_REDUCTION,
  authoringTool: RLRR_AUTHORING_TOOL,
};

type Sidecar = {
  instruments?: RlrrInstrument[];
  audioFileData?: RlrrAudioFileData;
  recordingMetadata?: RlrrRecordingMetadata;
  authoringTool?: string;
};

export function writeRlrr(jot: Jot, options: JotToRlrrOptions = {}): RlrrFile {
  const opts = { ...DEFAULTS, ...options };
  const voices = buildJotModel(jot).structural.voices;
  const instrumentFor = (pitch: string): Instrument =>
    jot.globalMetadata.instrumentMapping?.[pitch] ?? { kind: 'custom' };

  const sidecar = (jot.globalMetadata.rlrr ?? {}) as Sidecar;
  const instruments: RlrrInstrument[] =
    options.instruments ?? sidecar.instruments ?? [...DEFAULT_INSTRUMENTS];

  // Voice 0 is canonical for tempo (its bar grid is shared across voices).
  // `buildBarTempos` produces per-bar `durationSec` + within-bar segments
  // so a note at `note.beat` resolves via `beatToSecWithinBar` even when
  // its bar contains mid-bar tempo changes.
  const voice0 = voices[0];
  const barTempos = voice0 ? buildBarTempos(jot, voice0.bars) : [];
  const barStartSec: number[] = new Array(voice0?.bars.length ?? 0);
  {
    let cursor = 0;
    for (let i = 0; i < barStartSec.length; i++) {
      barStartSec[i] = cursor;
      cursor += barTempos[i]?.durationSec ?? 0;
    }
  }

  const events: RlrrEvent[] = [];
  // [S5] Merge all voices. All voices share voice 0's bar timing.
  for (const voice of voices) {
    for (let bi = 0; bi < voice.bars.length; bi++) {
      const bar = voice.bars[bi];
      const tempos = barTempos[bi];
      const startSec = barStartSec[bi] ?? 0;
      for (const pitch of voice.pitches) {
        const track = bar.tracks[pitch];
        if (!track) continue;
        const instrument = instrumentFor(pitch);
        for (const note of track.notes) {
          const target = resolveInstrument(note, instrument, instruments);
          if (!target) continue;
          // RLRR event times are real-time seconds, so a note's sub-slot
          // `offset` (ms) applies directly, a swung/off-grid hit charts at
          // the time it actually plays. (Re-importing snaps to RLRR's 1/16
          // grid, so the offset is lost on the way back in, by design.)
          const offsetSec = (note.offsetMs ?? 0) / 1000;
          const seconds =
            (tempos ? startSec + beatToSecWithinBar(tempos, note.beat) : startSec) + offsetSec;
          const vel = clampVelocity(resolveVelocity(note, opts));
          const event: RlrrEvent = {
            name: target.name,
            vel,
            // The hand (loc) lived only in per-note RLRR metadata, which the
            // reactive model doesn't carry, so charts export hand 0 (left).
            loc: 0,
            time: formatEventTime(seconds),
          };
          const midiNote = resolveMidiNote(note, target);
          if (midiNote !== undefined) event.midi = midiNote;
          events.push(event);
        }
      }
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

  // [S1] One bpmEvent per tempoEvent at its wall-clock time, prefixed by
  // the initial tempo at time 0. RLRR has no `BpmTransition`, so a
  // transition flattens to its `start` (else `end`).
  const bpmEvents = [{ bpm: initialBpm(jot), time: 0 }];
  for (const ev of jot.tempoEvents ?? []) {
    const tempos = barTempos[ev.barIndex];
    if (!tempos) continue;
    const startSec = barStartSec[ev.barIndex] ?? 0;
    const time = startSec + beatToSecWithinBar(tempos, ev.beat);
    const bpm = resolveBpm(ev.bpm, bpmEvents[bpmEvents.length - 1].bpm);
    if (bpm === bpmEvents[bpmEvents.length - 1].bpm) continue;
    bpmEvents.push({ bpm, time });
  }

  return {
    version: RLRR_VERSION,
    authoringTool: opts.authoringTool,
    recordingMetadata,
    audioFileData,
    instruments,
    events,
    bpmEvents,
  };
}

// ---------- helpers ----------

function resolveInstrument(
  note: StructNote,
  instrument: Instrument,
  kit: RlrrInstrument[]
): RlrrInstrument | undefined {
  // [S3](b) map via the instrument's midi note if it matches a known kit class.
  const midiNote = instrument.midi?.note;
  if (midiNote !== undefined) {
    for (const inst of kit) {
      const descriptor = describeDrum(inst.name);
      if (descriptor && descriptor.midi === midiNote) return inst;
    }
  }

  // [S3](c) pitch+modifiers heuristic.
  const cls = pitchToClass(note.pitch, new Set(note.modifiers as Modifier[]));
  if (cls) {
    const found = kit.find((i) => i.class === cls);
    if (found) return found;
  }

  return undefined;
}

function resolveVelocity(
  note: StructNote,
  opts: Required<Pick<JotToRlrrOptions, 'defaultVelocity' | 'accentBoost' | 'ghostReduction'>>
): number {
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

function resolveMidiNote(note: StructNote, instrument: RlrrInstrument): number | undefined {
  if (typeof note.midiNote === 'number') return note.midiNote;
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
