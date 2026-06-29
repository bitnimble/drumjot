import { Instrument } from 'src/schema/dsl/dsl';
import { DrumInstrumentKind, defaultKindForLane } from 'src/instruments/instruments';

/**
 * Default top-to-bottom row ordering for drum-instrument kinds when the user
 * hasn't manually reordered: top-of-kit cymbals first, then drums high to low,
 * kick last; `custom` falls to the very bottom. Shared by the mixer's default
 * lane order and the converter's default `ordering` so a freshly-loaded jot
 * and the mixer agree.
 */
export const DEFAULT_MIXER_KIND_ORDER: readonly DrumInstrumentKind[] = [
  'crash',
  'ride',
  'hihat',
  'tom',
  'snare',
  'kick',
  'custom',
];

/**
 * Best-effort `DrumInstrumentKind` from an instrument's display name. Recovers
 * a sensible position for rows whose loader stamped `kind: 'custom'` despite a
 * recognisable name. Substring-based; mirrors the RLRR / MIDI / transcriber
 * loader names.
 */
export function inferKindFromInstrumentName(name: string | undefined): DrumInstrumentKind | undefined {
  if (!name) return undefined;
  const n = name.toLowerCase();
  if (/\bkick\b|\bbass\s*drum\b/.test(n)) return 'kick';
  if (/\bsnare\b/.test(n)) return 'snare';
  if (/hi.?hat/.test(n)) return 'hihat';
  if (/\bride\b/.test(n)) return 'ride';
  if (/\bcrash\b|\bchina\b|\bsplash\b/.test(n)) return 'crash';
  if (/\bfloor\s*tom\b|\btom\b/.test(n)) return 'tom';
  return undefined;
}

/**
 * Floor toms render below regular toms within the tom group. Detected from the
 * instrument name; lane letter `f` is the GM importer's floor-tom convention so
 * it counts even with no display name.
 */
export function isFloorTom(instrument: Instrument | undefined, lane: string): boolean {
  if (instrument?.name && /floor/i.test(instrument.name)) return true;
  return lane === 'f';
}

/**
 * Sort tuple for the default mixer order: [kind rank, intra-kind rank, lane].
 * Kind comes from the parsed `Instrument` when available; `custom` falls back
 * to a name heuristic, then the lane letter's default kind. Intra-kind rank
 * only matters for toms today (regular before floor).
 */
export function defaultMixerSortKey(
  lane: string,
  instrument: Instrument | undefined
): [number, number, string] {
  let kind: DrumInstrumentKind = instrument?.kind ?? 'custom';
  if (kind === 'custom') {
    const fromName = inferKindFromInstrumentName(instrument?.name);
    if (fromName) kind = fromName;
  }
  if (kind === 'custom') {
    const fromLetter = defaultKindForLane(lane);
    if (fromLetter !== 'custom') kind = fromLetter;
  }
  const kindRank = DEFAULT_MIXER_KIND_ORDER.indexOf(kind);
  const subRank = kind === 'tom' && isFloorTom(instrument, lane) ? 1 : 0;
  return [kindRank === -1 ? DEFAULT_MIXER_KIND_ORDER.length : kindRank, subRank, lane];
}

/** Compare two lanes by the default mixer order (kind rank, then lane). */
export function compareLanesByDefaultMixerOrder(
  a: string,
  b: string,
  instrumentFor: (lane: string) => Instrument | undefined
): number {
  const ka = defaultMixerSortKey(a, instrumentFor(a));
  const kb = defaultMixerSortKey(b, instrumentFor(b));
  if (ka[0] !== kb[0]) return ka[0] - kb[0];
  if (ka[1] !== kb[1]) return ka[1] - kb[1];
  return ka[2].localeCompare(kb[2]);
}
