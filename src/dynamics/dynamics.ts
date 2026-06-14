/**
 * Shared note-dynamics constants: the single source of truth for how a
 * note's volume marker (`pp`..`ff`), `:a` accent, and `:g` ghost map to a
 * MIDI velocity (1-127). Consumed by playback (`playback/events.ts`),
 * MIDI export (`midi/to_midi.ts`), MIDI import (`midi/from_midi.ts`), and
 * RLRR export (`rlrr/jot_to_rlrr.ts`).
 *
 * Keeping these in one place is load-bearing for round-trip fidelity: an
 * accent authored in the DSL must play back, export to MIDI, and export
 * to RLRR at the *same* loudness, and a `play -> export -> re-import`
 * cycle must preserve the accent. Previously each consumer redeclared its
 * own copy and the accent boost had drifted (playback/RLRR used 24, MIDI
 * used 36), so an accent sounded different from what it exported as.
 */
import { Volume } from 'src/dsl/dsl';

/** Velocity for a note with no explicit volume marker. */
export const DEFAULT_VELOCITY = 80;

/**
 * Velocity added to a note's baseline for an `:a` accent.
 *
 * Must push an accent above the loudest non-accent volume (`ff` = 96) so
 * the MIDI round-trip can tell them apart: `from_midi` tags a velocity
 * `>= ACCENT_THRESHOLD` (100) as `:a`. The worst case is an explicit
 * `mf:a` (baseline 64): 64 + 36 = 100 lands exactly on the threshold,
 * while a smaller boost (e.g. the old 24 -> 88) would sit *below* `ff`
 * (96) and be unrecoverable on import. 36 is therefore the smallest boost
 * that keeps every accent round-trippable.
 */
export const ACCENT_BOOST = 36;

/** Velocity subtracted from a note's baseline for a `:g` ghost note. */
export const GHOST_REDUCTION = 32;

/** Volume marker -> baseline MIDI velocity. */
export const VOLUME_TO_VELOCITY: Record<Volume, number> = {
  pp: 16,
  p: 33,
  mp: 49,
  mf: 64,
  f: 80,
  ff: 96,
};

/** Velocity at and above which `from_midi` decorates a note with `:a`.
 *  Paired with {@link ACCENT_BOOST} (see its note) so the export boost
 *  and the import threshold can't drift apart. */
export const ACCENT_THRESHOLD = 100;

/** Velocity below which `from_midi` decorates a note with `:g`. */
export const GHOST_THRESHOLD = 40;
