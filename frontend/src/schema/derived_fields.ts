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
import type { Instrument, TimeSignature } from 'src/schema/dsl/dsl';
import type { TempoJot } from 'src/schema/dsl/tempo';
import type { JotTimeline } from 'src/editing/playback/timeline';
import type { StructLayer } from 'src/editing/structure/structure_store';
import type { LaneBars } from 'src/editing/structure/structural_presenter';
import {
  createDerivedRegistry,
  type DerivedRegistry,
  declareDerived,
  fnSlot,
  slot,
} from './derived_registry';

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
  /** Ordered lane list across all `||` layers (lanes carrying a note). Read by
   *  the mixer row list. Implemented by `StructuralPresenter`. */
  lanes: slot<readonly string[]>(),
  /** The musical structural layers, OMITTING the view-only virtual lead-in
   *  (distinct from `renderedLayers`, which includes it) so the playback
   *  scheduler never fires drum events for the lead-in. Read by the playback
   *  event scheduler + editing geometry. Implemented by `StructuralPresenter`. */
  musicalLayers: slot<StructLayer[]>(),
  /** Per-lane row data (keyed by lane) the mixer's instrument rows read.
   *  Implemented by `StructuralPresenter`. */
  barsForLane: fnSlot<string, LaneBars>(),
  /** The barIndex-anchored tempo events + lead-in count the tempo maths read,
   *  projected off the reactive document. Read by playback / tempo-edit /
   *  waveform. Implemented by `StructuralPresenter`. */
  tempoSource: slot<TempoJot>(),
  /** Per-bar performance drift seconds (indexed by `layers[0]` bars, lead-in
   *  = 0); feeds the waveform stretch + playback `DriftMap`. Implemented by
   *  `StructuralPresenter`. */
  barDrift: slot<readonly number[]>(),
  /** Lane → instrument display/playback info (keyed). Read by the playback
   *  scheduler + mixer row list. Implemented by `StructuralPresenter`. */
  instrumentFor: fnSlot<string, Instrument>(),
  /** Id of the `||` layer that owns `lane` (keyed); `undefined` for a lane no
   *  layer carries. Read by editing placement. Implemented by
   *  `StructuralPresenter`. */
  ownerLayerFor: fnSlot<string, string | undefined>(),
  /** Rendered structural layers WITH the view-only virtual lead-in bar (the
   *  score's layout spine, distinct from `musicalLayers` which omits it). Read
   *  by the waveform layout + tempo-edit. Implemented by `StructuralPresenter`. */
  renderedLayers: slot<StructLayer[]>(),
});

export type JotDerivedRegistry = DerivedRegistry<typeof Derived>;

export function createJotDerivedRegistry(): JotDerivedRegistry {
  return createDerivedRegistry(Derived);
}
