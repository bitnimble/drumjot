/**
 * Shared note-dynamics constants: the single source of truth for how an
 * authored volume marker (`pp`..`ff`) maps to a MIDI velocity, and how a
 * `:a` accent / `:g` ghost adjust a note's baseline velocity (1-127).
 *
 * A note's loudness lives in its numeric `velocity` field. Authored DSL
 * dynamics (`pp`..`ff`) are converted to a velocity once, at parse time, via
 * {@link VOLUME_TO_VELOCITY} (see `from_dsl.ts`). The accent/ghost adjustments
 * here only apply at export when a note has NO explicit velocity, and are
 * consumed by playback (`playback/events.ts`), MIDI export (`midi/to_midi.ts`),
 * MIDI import (`midi/from_midi.ts`), and RLRR export (`schema/rlrr/writer.ts`).
 *
 * Keeping these in one place is load-bearing for round-trip fidelity: an
 * accent must play back, export to MIDI, and export to RLRR at the *same*
 * loudness. Previously each consumer redeclared its own copy and the accent
 * boost had drifted (playback/RLRR used 24, MIDI used 36).
 */
import { Volume } from 'src/schema/dsl/dsl';

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

/**
 * Authored volume marker -> MIDI velocity, applied once at parse time
 * (`from_dsl.ts`) so the note carries a numeric `velocity` thereafter. The
 * values place each marker on the note-properties panel's 0-10 loudness scale
 * (velocity = round(step * 12.7)): pp=1, p=3, mp=5, mf=6, f=8, ff=10.
 */
export const VOLUME_TO_VELOCITY: Record<Volume, number> = {
  pp: 13,
  p: 38,
  mp: 64,
  mf: 76,
  f: 102,
  ff: 127,
};

/** Velocity at and above which `from_midi` decorates a note with `:a`.
 *  Paired with {@link ACCENT_BOOST} (see its note) so the export boost
 *  and the import threshold can't drift apart. */
export const ACCENT_THRESHOLD = 100;

/** Velocity below which `from_midi` decorates a note with `:g`. */
export const GHOST_THRESHOLD = 40;
