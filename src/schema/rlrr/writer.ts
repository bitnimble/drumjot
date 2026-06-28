/**
 * Drumjot Jot -> Paradiddle `.rlrr` conversion.
 *
 * Assumptions (tagged [S#] inline):
 *  [S1] Tempo is taken from `tempo.initialBpm` (the song-start tempo
 *       event) plus `jot.tempoEvents` for sticky tempo changes. Each tempoEvent
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
 *         (c) `laneToClass(lane, modifiers)` then pick the first
 *             instrument of that class in the kit.
 *       Notes that don't resolve to any kit instrument are dropped.
 *  [S4] Velocity comes from `metadata.rlrr.vel` if present, else
 *       `metadata.midi.velocity`, else the `vol`-bucket mapping with
 *       `:a` / `:g` adjustments.
 *  [S5] All layers in the Jot collapse onto a single RLRR event stream;
 *       Paradiddle has no concept of independent layers.
 *  [S6] `audioFileData` and `recordingMetadata` are taken from
 *       `globalMetadata.rlrr` when present; the Jot's `title` always
 *       overrides `recordingMetadata.title`.
 */
import { Instrument, Jot, Modifier } from 'src/schema/dsl/dsl';
import { DEFAULT_VELOCITY } from 'src/dynamics/dynamics';
import { buildStructural } from 'src/editing/jot_editor_store';
import type { StructNote } from 'src/editing/structure/structure_store';
import { beatToSecWithinBar, buildBarTempos, initialBpm, resolveBpm } from 'src/schema/dsl/tempo';
import {
  CLASS_TO_DRUM,
  describeDrum,
  laneToClass,
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
  /** Override the kit; if omitted, prefer `globalMetadata.rlrr.instruments` then `DEFAULT_INSTRUMENTS`. */
  instruments?: RlrrInstrument[];
  authoringTool?: string;
  /** Override `recordingMetadata` fields. Title falls back to `jot.title`. */
  recordingMetadata?: RlrrRecordingMetadata;
  audioFileData?: RlrrAudioFileData;
};

// Loudness for a note with no explicit velocity; from the shared dynamics so
// playback, MIDI export, and RLRR export agree.
const DEFAULTS: Required<Pick<JotToRlrrOptions, 'defaultVelocity' | 'authoringTool'>> = {
  defaultVelocity: DEFAULT_VELOCITY,
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
  // Musical structure only, the view-only virtual lead-in is never exported.
  const layers = buildStructural(jot).musicalLayers;
  const instrumentFor = (lane: string): Instrument =>
    jot.globalMetadata.instrumentMapping?.[lane] ?? { kind: 'custom' };

  const sidecar = (jot.globalMetadata.rlrr ?? {}) as Sidecar;
  const instruments: RlrrInstrument[] =
    options.instruments ?? sidecar.instruments ?? [...DEFAULT_INSTRUMENTS];

  // Layer 0 is canonical for tempo (its bar grid is shared across layers).
  // `buildBarTempos` produces per-bar `durationSec` + within-bar segments
  // so a note at `note.beat` resolves via `beatToSecWithinBar` even when
  // its bar contains mid-bar tempo changes.
  const layer0 = layers[0];
  const barTempos = layer0 ? buildBarTempos(jot, layer0.bars) : [];
  const barStartSec: number[] = new Array(layer0?.bars.length ?? 0);
  const barStartBeat: number[] = new Array(layer0?.bars.length ?? 0);
  {
    let secCursor = 0;
    let beatCursor = 0;
    for (let i = 0; i < barStartSec.length; i++) {
      barStartSec[i] = secCursor;
      barStartBeat[i] = beatCursor;
      secCursor += barTempos[i]?.durationSec ?? 0;
      beatCursor += layer0?.bars[i]?.beats ?? 0;
    }
  }
  // Wall-clock seconds at a global beat position (cumulative over bars),
  // used to time each step of a subdivided tempo ramp that may span bars.
  const globalBeatToSec = (g: number): number => {
    let j = 0;
    while (j + 1 < barStartBeat.length && barStartBeat[j + 1] <= g) j++;
    const tempos = barTempos[j];
    return (barStartSec[j] ?? 0) + (tempos ? beatToSecWithinBar(tempos, g - (barStartBeat[j] ?? 0)) : 0);
  };

  const events: RlrrEvent[] = [];
  // [S5] Merge all layers. All layers share layer 0's bar timing.
  for (const layer of layers) {
    for (let bi = 0; bi < layer.bars.length; bi++) {
      const bar = layer.bars[bi];
      const tempos = barTempos[bi];
      const startSec = barStartSec[bi] ?? 0;
      for (const lane of layer.lanes) {
        const track = bar.tracks[lane];
        if (!track) continue;
        const instrument = instrumentFor(lane);
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
  // the initial tempo at time 0. RLRR has no `BpmTransition`, so a gradual
  // ramp is subdivided into stepwise bpmEvents (1/32-beat resolution) along
  // the linear-in-time curve, with the final step landing exactly on `end`.
  const bpmEvents = [{ bpm: initialBpm(jot), time: 0 }];
  const pushBpm = (bpm: number, time: number) => {
    if (bpm > 0 && Math.abs(bpm - bpmEvents[bpmEvents.length - 1].bpm) > 1e-9) {
      bpmEvents.push({ bpm, time });
    }
  };
  for (const ev of jot.tempoEvents ?? []) {
    if (ev.barIndex < 0 || ev.barIndex >= barStartBeat.length) continue;
    const baseG = (barStartBeat[ev.barIndex] ?? 0) + ev.beat;
    if (typeof ev.bpm === 'object') {
      const start = ev.bpm.start ?? bpmEvents[bpmEvents.length - 1].bpm;
      const { end, duration } = ev.bpm;
      const steps = Math.max(1, Math.ceil(duration * 32));
      for (let s = 0; s <= steps; s++) {
        const f = s / steps;
        const bpm = s === steps ? end : Math.sqrt(start * start + (end * end - start * start) * f);
        pushBpm(bpm, globalBeatToSec(baseG + f * duration));
      }
    } else {
      pushBpm(resolveBpm(ev.bpm, bpmEvents[bpmEvents.length - 1].bpm), globalBeatToSec(baseG));
    }
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

  // [S3](c) lane+modifiers heuristic.
  const cls = laneToClass(note.lane, new Set(note.modifiers as Modifier[]));
  if (cls) {
    const found = kit.find((i) => i.class === cls);
    if (found) return found;
  }

  return undefined;
}

function resolveVelocity(
  note: StructNote,
  opts: Required<Pick<JotToRlrrOptions, 'defaultVelocity'>>
): number {
  return typeof note.velocity === 'number' ? note.velocity : opts.defaultVelocity;
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
