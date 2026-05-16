/**
 * MIDI -> Paradiddle `.rlrr` conversion. Ported from
 *   https://github.com/emretanirgan/ParadiddleUtilities/blob/master/PDUtilities/midiconvert.py
 *
 * The Python tool is one-way (MIDI -> RLRR). We faithfully replicate its
 * runtime semantics with a few intentional simplifications, each tagged
 * [M#] inline:
 *
 *  [M1] Track selection: format 0 picks track[0]; formats 1/2 prefer the
 *       first track whose name contains "drum" (case-insensitive), else
 *       track[1] if there is one, else track[0]. Matches the Python.
 *  [M2] Tempo timeline: built by scanning ALL tracks (not just the chosen
 *       drum track), because tempo events live in track 0 in format-1 MIDI
 *       files. This matches the Python.
 *  [M3] Per-event seconds are computed from the tempo timeline using the
 *       segment that contains the event's absolute tick.
 *  [M4] Note map: MIDI note -> drum class. Defaults from `midi_mapping.yaml`
 *       (Easy difficulty); callers may pass a custom map.
 *  [M5] We always use the first matching instrument of a given class in the
 *       supplied kit. Multiple-instance kits would need callers to provide
 *       a richer map (TODO).
 *  [M6] We do NOT yet support `toggle_notes` from the Python config; toggle
 *       handling is a larger rewrite and is left for a follow-up.
 *  [M7] note-off events and noteOn(velocity=0) are ignored, as in Python.
 *  [M8] The original MIDI note number is recorded on each event as `midi`
 *       so `rlrrToMidi` round-trips losslessly.
 */
import { parseMidi, MidiEvent } from 'midi-file';
import {
  CLASS_TO_DRUM,
  DEFAULT_NOTE_TO_CLASS,
} from './drums';
import {
  DEFAULT_INSTRUMENTS,
  RLRR_AUTHORING_TOOL,
  RLRR_VERSION,
  RlrrAudioFileData,
  RlrrBpmEvent,
  RlrrEvent,
  RlrrFile,
  RlrrInstrument,
  RlrrRecordingMetadata,
  formatEventTime,
} from './schema';

export type MidiToRlrrOptions = {
  /** Override the MIDI-note -> drum-class map. */
  noteToClass?: Record<number, string>;
  /** Override the kit; default = `DEFAULT_INSTRUMENTS`. */
  instruments?: RlrrInstrument[];
  authoringTool?: string;
  recordingMetadata?: RlrrRecordingMetadata;
  audioFileData?: RlrrAudioFileData;
};

type TempoPoint = { tick: number; seconds: number; microsPerBeat: number };

const DEFAULT_TEMPO_MICROS = 500_000; // 120 bpm

export function midiToRlrr(
  buffer: Uint8Array | ArrayBuffer | ArrayLike<number>,
  options: MidiToRlrrOptions = {}
): RlrrFile {
  const bytes = toByteArray(buffer);
  const midi = parseMidi(bytes);

  const ticksPerBeat = midi.header.ticksPerBeat;
  if (!ticksPerBeat || ticksPerBeat <= 0) {
    throw new Error('SMPTE-timed MIDI files are not supported');
  }

  const noteToClass = options.noteToClass ?? DEFAULT_NOTE_TO_CLASS;
  const instruments = options.instruments ?? [...DEFAULT_INSTRUMENTS];

  // [M2] Build tempo timeline across all tracks.
  const tempoTimeline = buildTempoTimeline(midi.tracks, ticksPerBeat);

  // [M1] Pick the drum track.
  const drumTrackIdx = pickDrumTrack(midi);
  const drumTrack = midi.tracks[drumTrackIdx];

  // [M3] Walk the drum track, emitting RLRR events.
  const events: RlrrEvent[] = [];
  let tick = 0;
  for (const ev of drumTrack) {
    tick += ev.deltaTime;
    if (ev.type !== 'noteOn') continue;
    if (ev.velocity === 0) continue; // [M7]
    const cls = noteToClass[ev.noteNumber];
    if (!cls) continue;
    const instrument = instruments.find((i) => i.class === cls);
    if (!instrument) continue; // [M5] no kit slot for this class -> drop.
    const seconds = ticksToSeconds(tick, tempoTimeline, ticksPerBeat);
    const event: RlrrEvent = {
      name: instrument.name,
      vel: ev.velocity,
      loc: 0,
      time: formatEventTime(seconds),
      midi: ev.noteNumber, // [M8]
    };
    events.push(event);
  }
  events.sort((a, b) => parseFloat(a.time as string) - parseFloat(b.time as string));

  const bpmEvents: RlrrBpmEvent[] = tempoTimeline.map((t) => ({
    bpm: Math.round(60_000_000 / t.microsPerBeat),
    time: t.seconds,
  }));
  if (bpmEvents.length === 0) {
    bpmEvents.push({ bpm: Math.round(60_000_000 / DEFAULT_TEMPO_MICROS), time: 0 });
  }

  const lastEventSeconds =
    events.length > 0 ? parseFloat(events[events.length - 1].time as string) : 0;
  const recordingMetadata: RlrrRecordingMetadata = {
    title: '',
    description: '',
    coverImagePath: '',
    artist: '',
    creator: '',
    length: lastEventSeconds,
    complexity: 1,
    ...(options.recordingMetadata ?? {}),
  };

  return {
    version: RLRR_VERSION,
    authoringTool: options.authoringTool ?? RLRR_AUTHORING_TOOL,
    recordingMetadata,
    audioFileData: options.audioFileData ?? {
      songTracks: [],
      drumTracks: [],
      songPreview: '',
      calibrationOffset: 0,
    },
    instruments,
    events,
    bpmEvents,
  };
}

// ---------- helpers ----------

function toByteArray(buffer: Uint8Array | ArrayBuffer | ArrayLike<number>): Uint8Array {
  if (buffer instanceof Uint8Array) return buffer;
  if (buffer instanceof ArrayBuffer) return new Uint8Array(buffer);
  return new Uint8Array(Array.from(buffer as ArrayLike<number>));
}

function pickDrumTrack(midi: { header: { format: 0 | 1 | 2 }; tracks: MidiEvent[][] }): number {
  if (midi.header.format === 0) return 0;
  // Look for a track named containing "drum".
  for (let i = 0; i < midi.tracks.length; i++) {
    const name = findTrackName(midi.tracks[i]);
    if (name && name.toLowerCase().includes('drum')) return i;
  }
  return midi.tracks.length > 1 ? 1 : 0;
}

function findTrackName(track: MidiEvent[]): string | undefined {
  for (const ev of track) {
    if (ev.type === 'trackName') return ev.text;
  }
  return undefined;
}

function buildTempoTimeline(tracks: MidiEvent[][], _ticksPerBeat: number): TempoPoint[] {
  type Abs = { tick: number; ev: MidiEvent };
  const flat: Abs[] = [];
  for (const track of tracks) {
    let t = 0;
    for (const ev of track) {
      t += ev.deltaTime;
      flat.push({ tick: t, ev });
    }
  }
  flat.sort((a, b) => a.tick - b.tick);

  const points: TempoPoint[] = [];
  let curMicros = DEFAULT_TEMPO_MICROS;
  let curSeconds = 0;
  let curTick = 0;
  points.push({ tick: 0, seconds: 0, microsPerBeat: curMicros });

  for (const { tick, ev } of flat) {
    if (ev.type !== 'setTempo') continue;
    if (tick === curTick) {
      // tempo at current position replaces previous (don't duplicate)
      curMicros = ev.microsecondsPerBeat;
      points[points.length - 1] = {
        tick: curTick,
        seconds: curSeconds,
        microsPerBeat: curMicros,
      };
      continue;
    }
    const dTicks = tick - curTick;
    curSeconds += (dTicks * curMicros) / (_ticksPerBeat * 1_000_000);
    curTick = tick;
    curMicros = ev.microsecondsPerBeat;
    points.push({ tick: curTick, seconds: curSeconds, microsPerBeat: curMicros });
  }

  return points;
}

function ticksToSeconds(tick: number, timeline: TempoPoint[], ticksPerBeat: number): number {
  let i = 0;
  while (i + 1 < timeline.length && timeline[i + 1].tick <= tick) i++;
  const seg = timeline[i];
  const dTicks = tick - seg.tick;
  return seg.seconds + (dTicks * seg.microsPerBeat) / (ticksPerBeat * 1_000_000);
}
