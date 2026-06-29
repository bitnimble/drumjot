import { Instrument, Jot, Limb } from 'src/schema/dsl/dsl';
import { DrumInstrumentKind, INSTRUMENT_METADATA } from 'src/instruments/instruments';

/**
 * The default kit a blank jot starts with: a lighter stock rock/pop setup,
 * crash, ride, hi-hat, snare, kick. No toms (the user can add them). Listed
 * top-to-bottom in canonical drum-notation order (matches
 * `DEFAULT_MIXER_KIND_ORDER` minus toms), so `instrumentMapping`'s insertion
 * order, which is the rendered lane order, stacks cymbals over the snare over
 * the kick.
 *
 * Lane letters are the canonical Drumjot ones (`defaultKindForLane`), so the
 * blank jot round-trips through MIDI / RLRR / save like any other.
 */
const BLANK_JOT_LANES: ReadonlyArray<{ lane: string; kind: DrumInstrumentKind; limb: Limb }> = [
  { lane: 'c', kind: 'crash', limb: 'rh' },
  { lane: 'd', kind: 'ride', limb: 'rh' },
  { lane: 'h', kind: 'hihat', limb: 'rh' },
  { lane: 's', kind: 'snare', limb: 'lh' },
  { lane: 'k', kind: 'kick', limb: 'rf' },
];

/**
 * Build a fresh, empty jot: the default drum kit declared as lanes (so the
 * score renders the kit's rows ready to receive notes, see the
 * declared-but-empty-lane handling in `StructureStore.lanesForLayer`) but with
 * no notes, no audio tracks, one empty 4/4 bar at 120 bpm.
 *
 * Returned as a DSL {@link Jot} so it loads through the same
 * `JotEditorStore.loadSource` path as every other song.
 */
export function createBlankJot(title = 'New Jot'): Jot {
  const instrumentMapping: Record<string, Instrument> = {};
  for (const { lane, kind, limb } of BLANK_JOT_LANES) {
    instrumentMapping[lane] = { kind, name: INSTRUMENT_METADATA[kind].label, limb };
  }
  return {
    title,
    globalMetadata: {
      time: { count: 4, unit: 4 },
      instrumentMapping,
    },
    layers: [{ bars: [{ elements: [] }] }],
  };
}
