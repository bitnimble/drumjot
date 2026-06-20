/**
 * Shared note-dynamics constants. A note's loudness lives entirely in its
 * numeric `velocity` field (1-127); there is no stored accent/ghost modifier.
 *
 * Authored DSL loudness is just sugar over `velocity`, converted once at parse
 * time (`from_dsl.ts`): a `pp`..`ff` marker maps via {@link VOLUME_TO_VELOCITY},
 * and the `:a` / `:g` accent/ghost markers map to {@link ACCENT_VELOCITY} /
 * {@link GHOST_VELOCITY}. Accent/ghost NOTATION (the ring / dimmed glyph) is
 * then derived from velocity at render time via {@link ACCENT_THRESHOLD} /
 * {@link GHOST_THRESHOLD} (see `bar_view.tsx`); the same thresholds let
 * `from_midi` / the RLRR parser tag loud/soft imported hits with the `:a`/`:g`
 * sugar so they survive into the schema as the right velocity.
 */
import { Volume } from 'src/schema/dsl/dsl';

/** Velocity for a note with no explicit loudness. */
export const DEFAULT_VELOCITY = 80;

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

/** Velocity a bare `:a` accent marker maps to (kept clear of
 *  {@link ACCENT_THRESHOLD} so it renders as accented). */
export const ACCENT_VELOCITY = 112;

/** Velocity a bare `:g` ghost marker maps to (kept below
 *  {@link GHOST_THRESHOLD} so it renders as a ghost note). */
export const GHOST_VELOCITY = 28;

/** Velocity at and above which a note renders with the accent ring, and at
 *  which `from_midi` / the RLRR parser tag an imported hit with `:a`. */
export const ACCENT_THRESHOLD = 100;

/** Velocity below which a note renders dimmed as a ghost, and at which
 *  `from_midi` / the RLRR parser tag an imported hit with `:g`. */
export const GHOST_THRESHOLD = 40;
