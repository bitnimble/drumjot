/**
 * Public API for the Drumjot MIDI converter.
 *
 *   import { fromMidi, toMidi } from 'src/midi';
 *
 *   const jot = fromMidi(bytes);     // MIDI bytes -> Jot
 *   const out = toMidi(jot);         // Jot -> MIDI bytes
 *
 * See `from_midi.ts` and `to_midi.ts` for the documented assumptions about
 * how MIDI features map onto the DSL (and vice versa).
 */
export { fromMidi } from './from_midi';
export type { FromMidiOptions } from './from_midi';
export { toMidi, TICKS_PER_BEAT } from './to_midi';
export type { ToMidiOptions } from './to_midi';
export {
  GM_PERCUSSION,
  defaultMidiNote,
  deriveLetterFromMidi,
  allocatePitchesForMidi,
} from './gm';
export type { GmEntry } from './gm';
