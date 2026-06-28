/**
 * The Jot's first-class derived fields: the cross-domain computed state declared
 * on the document so any presenter can read it off the model (`jot.tempoTimeline`)
 * without depending on the presenter that produces it. The implementations are
 * installed by the owning feature presenters (see {@link TempoPresenter}).
 *
 * Each slot's type is the single source of truth, flowing into the model surface
 * ({@link MutableJot}) and the registry's `define`. The editing-layer value types
 * (`JotTimeline`) are imported type-only, so this module's runtime imports stay
 * within the schema layer (no schema → editing runtime cycle).
 */
import type { TimeSignature } from 'src/schema/dsl/dsl';
import type { JotTimeline } from 'src/editing/playback/timeline';
import { createDerivedRegistry, type DerivedRegistry, declareDerived, slot } from './derived_registry';

export const Derived = declareDerived({
  /** Audio-time timeline (per-bar start/duration + tempo segments) used by the
   *  playhead, minimap, lyrics and playback transport. Implemented by
   *  `TempoPresenter`. */
  tempoTimeline: slot<JotTimeline>(),
  /** The bpm + time signature the song spends the most audio time in (subtitle
   *  + default-tempo fallback). Implemented by `TempoPresenter`. */
  dominantBpmAndTime: slot<{
    dominantBpm: number | undefined;
    dominantTime: TimeSignature | undefined;
  }>(),
});

export type JotDerivedRegistry = DerivedRegistry<typeof Derived>;

export function createJotDerivedRegistry(): JotDerivedRegistry {
  return createDerivedRegistry(Derived);
}
