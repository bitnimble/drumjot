/**
 * Public API for the Drumjot <-> Paradiddle `.rlrr` converters.
 *
 *   import { rlrrToJot, jotToRlrr, midiToRlrr, rlrrToMidi } from 'src/rlrr';
 *
 * Pipelines:
 *
 *   .rlrr (JSON)  -- rlrrToJot -->     Jot
 *   Jot           -- jotToRlrr -->     .rlrr (JSON)
 *   MIDI bytes    -- midiToRlrr -->    .rlrr (JSON)
 *   .rlrr (JSON)  -- rlrrToMidi -->    MIDI bytes
 *
 * All RLRR <-> Jot fidelity is preserved via custom keys under
 * `note.metadata.rlrr` and `jot.globalMetadata.rlrr`; see the converter
 * source files for the full list of assumptions.
 */
export { rlrrToJot } from './rlrr_to_jot';
export type { RlrrToJotOptions } from './rlrr_to_jot';
export { loadParadbZip } from './paradb';
export type { ParadbMap, ParadbTrack, LoadParadbOptions } from './paradb';
export { jotToRlrr } from './jot_to_rlrr';
export type { JotToRlrrOptions } from './jot_to_rlrr';
export { midiToRlrr } from './midi_to_rlrr';
export type { MidiToRlrrOptions } from './midi_to_rlrr';
export { rlrrToMidi, TICKS_PER_BEAT } from './rlrr_to_midi';
export type { RlrrToMidiOptions } from './rlrr_to_midi';
export {
  DEFAULT_INSTRUMENTS,
  RLRR_VERSION,
  RLRR_AUTHORING_TOOL,
  eventTimeSeconds,
  formatEventTime,
} from './schema';
export type {
  RlrrFile,
  RlrrEvent,
  RlrrBpmEvent,
  RlrrInstrument,
  RlrrRecordingMetadata,
  RlrrAudioFileData,
  Vec3,
} from './schema';
export {
  CLASS_TO_DRUM,
  DEFAULT_NOTE_TO_CLASS,
  describeDrum,
  instanceNameToClass,
  pitchToClass,
} from './drums';
export type { DrumDescriptor } from './drums';
export { allocateFallbackLetters } from './fallback';
