import type { StructBar, StructNote } from 'src/editing/structure/structure_store';
import { TICKS_PER_BEAT } from 'src/midi/to_midi';

/**
 * Computes the rendered (post-quantization) beat position in the same
 * 1-indexed `beat_in_bar` convention the provenance entry uses, so the
 * "Detected beat" → "Quantized to" comparison reads natively.
 *
 * `note.beat` is in quarter-notes from bar start (0-indexed). The
 * provenance's `beat_in_bar` counts beats of the bar's time
 * signature (1-indexed, where downbeat = 1.0). The conversion is
 * `1 + note.beat × (time.count / bar.beats)`, equals
 * `1 + note.beat` in 4/4, scales correctly for 6/8, 7/8, etc.
 */
export function renderedBeatInBar(note: StructNote, bar: StructBar): number {
  if (bar.beats <= 0) return 1;
  return 1 + (note.beat / bar.beats) * bar.tsCount;
}

/** Grid step in MIDI ticks for a given grid division, matching
 * `from_midi.ts`'s `gridTicks = ticksPerBeat * 4 / gridDivision`. Used by
 * the per-note Snap-delta computation so the value depends only on the
 * immutable detected tick, not on the rendered note's current bar (which
 * moves under the Beat-offset slider). */
export function midiGridTicks(gridDivision: number): number {
  return (TICKS_PER_BEAT * 4) / gridDivision;
}

/**
 * Render a delta in whole-note-subdivision slots with a sign and a
 * `/${gridDivision}` denominator (matching {@link
 * NotePosition.formatBarBeat48ths}'s absolute-position format).
 * Integer-rounded when the slot count is effectively whole (within 0.05)
 * so jitter-class deltas read as `+1/48` rather than `+0.97/48`;
 * fractional otherwise so a sub-slot snap delta (e.g. `+0.3/48`) still
 * surfaces its magnitude.
 */
export function formatSignedSlots(slots: number, gridDivision: number): string {
  const sign = slots >= 0 ? '+' : '';
  const rounded = Math.round(slots);
  if (Math.abs(slots - rounded) < 0.05) return `${sign}${rounded}/${gridDivision}`;
  return `${sign}${slots.toFixed(1)}/${gridDivision}`;
}

/** Signed ms with one decimal and a ` ms` suffix (e.g. `+12.3 ms`). */
export function formatSignedMs(ms: number): string {
  return `${ms >= 0 ? '+' : ''}${ms.toFixed(1)} ms`;
}

/** Signed beats with three decimals and no unit suffix (e.g. `+0.123`). */
export function formatSignedBeats(beats: number): string {
  return `${beats >= 0 ? '+' : ''}${beats.toFixed(3)}`;
}
