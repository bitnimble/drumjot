/**
 * Paradiddle `.rlrr` file schema. A `.rlrr` file is a JSON document with the
 * following top-level shape:
 *
 *   {
 *     "version": 0.7,
 *     "authoringTool": "...",
 *     "recordingMetadata": { ... },
 *     "audioFileData": { ... },
 *     "instruments": [ ... ],   // the drum kit definition
 *     "events": [ ... ],        // drum hits at absolute seconds
 *     "bpmEvents": [ ... ]      // tempo timeline
 *   }
 *
 * Source of truth: https://github.com/emretanirgan/ParadiddleUtilities
 */

export type Vec3 = [number, number, number];

export type RlrrInstrument = {
  /** Instance name, e.g. `BP_Snare_C_1`. Referenced by `RlrrEvent.name`. */
  name: string;
  /** Drum class, e.g. `BP_Snare_C`. The trailing `_<idx>` is dropped. */
  class: string;
  location: Vec3;
  rotation: Vec3;
  scale: Vec3;
};

export type RlrrEvent = {
  /** Instrument instance name, matches an entry in `instruments`. */
  name: string;
  /** Velocity 0-127. */
  vel: number;
  /** Hit zone / location index (always 0 in the reference Python tool). */
  loc: number;
  /**
   * Absolute time in seconds. The reference Python tool writes this as a
   * 4-decimal string (e.g. `"1.2345"`). We accept either form on read and
   * emit strings on write to maintain byte-level compatibility with the
   * Python tool.
   */
  time: number | string;
  /**
   * Custom additive extension used by this converter to preserve the source
   * MIDI note number across `MIDI -> RLRR -> MIDI` round trips. Paradiddle
   * itself ignores unknown fields; the reference Python tool does not emit
   * this field. Safe to drop if you need a strictly-canonical RLRR.
   */
  midi?: number;
};

export type RlrrBpmEvent = {
  bpm: number;
  /** Absolute time in seconds. Numeric, not string-formatted. */
  time: number;
};

export type RlrrRecordingMetadata = {
  title?: string;
  description?: string;
  coverImagePath?: string;
  artist?: string;
  creator?: string;
  /** Total song length in seconds. */
  length?: number;
  /** Difficulty index 1..4 (Easy, Medium, Hard, Expert). */
  complexity?: number;
};

export type RlrrAudioFileData = {
  songTracks?: string[];
  drumTracks?: string[];
  songPreview?: string;
  calibrationOffset?: number;
};

export type RlrrFile = {
  version: number;
  authoringTool?: string;
  recordingMetadata: RlrrRecordingMetadata;
  audioFileData?: RlrrAudioFileData;
  instruments: RlrrInstrument[];
  events: RlrrEvent[];
  bpmEvents: RlrrBpmEvent[];
};

export const RLRR_VERSION = 0.7;
export const RLRR_AUTHORING_TOOL = 'drumjot';

/**
 * Default Paradiddle drum kit. Positions/rotations are copied verbatim from
 * the reference `defaultset.rlrr` so the produced `.rlrr` opens with a
 * plausible 3D kit layout out of the box. Users who want a custom kit should
 * supply `instruments` via the `writeRlrr` options.
 */
export const DEFAULT_INSTRUMENTS: ReadonlyArray<RlrrInstrument> = [
  {
    name: 'BP_HiHat_C_1',
    class: 'BP_HiHat_C',
    location: [17.22353, -34.08699, 94.211975],
    rotation: [15.416171, 19.771341, 6.125736],
    scale: [1, 1, 1],
  },
  {
    name: 'BP_Kick_C_1',
    class: 'BP_Kick_C',
    location: [-69.758492, 60.319191, 34.294422],
    rotation: [-96.766594, 2.265444, 38.000126],
    scale: [1, 1, 1],
  },
  {
    name: 'BP_Crash15_C_1',
    class: 'BP_Crash15_C',
    location: [27.7938, -37.487892, 132.343323],
    rotation: [17.085356, 39.032425, -19.564053],
    scale: [1, 1, 1],
  },
  {
    name: 'BP_Crash17_C_1',
    class: 'BP_Crash17_C',
    location: [26.813931, 52.215088, 135.513931],
    rotation: [-10.405132, 44.075054, 25.513369],
    scale: [1, 1, 1],
  },
  {
    name: 'BP_FloorTom_C_1',
    class: 'BP_FloorTom_C',
    location: [-1.75137, 42.720551, 61.630424],
    rotation: [1.261108, 28.997799, 73.703827],
    scale: [1, 1, 1],
  },
  {
    name: 'BP_Ride17_C_1',
    class: 'BP_Ride17_C',
    location: [-14.226593, 61.129753, 105.472443],
    rotation: [-27.074295, 19.200071, 15.730481],
    scale: [1, 1, 1],
  },
  {
    name: 'BP_Ride20_C_1',
    class: 'BP_Ride20_C',
    location: [-13.356556, 77.818153, 138.67569],
    rotation: [18.895599, 42.784584, 101.073982],
    scale: [1, 1, 1],
  },
  {
    name: 'BP_Snare_C_1',
    class: 'BP_Snare_C',
    location: [17.106194, -1.370838, 68.553436],
    rotation: [3.194685, 15.114579, 1.082093],
    scale: [1, 1, 1],
  },
  {
    name: 'BP_Tom1_C_1',
    class: 'BP_Tom1_C',
    location: [40.340771, -2.559204, 95.154297],
    rotation: [-40.931034, 37.27087, -55.179005],
    scale: [1, 1, 1],
  },
  {
    name: 'BP_Tom2_C_2',
    class: 'BP_Tom2_C',
    location: [30.543972, 33.849445, 93.909325],
    rotation: [14.762558, 52.847816, 50.355915],
    scale: [1, 1, 1],
  },
];

/** Read an event's `time` as a number whether it arrives as string or number. */
export function eventTimeSeconds(ev: RlrrEvent): number {
  return typeof ev.time === 'number' ? ev.time : parseFloat(ev.time);
}

/** Format a time in seconds as the canonical RLRR 4-decimal string. */
export function formatEventTime(seconds: number): string {
  return seconds.toFixed(4);
}
