/**
 * General MIDI percussion mapping (MIDI note number <-> Drumjot DSL lane).
 *
 * The DSL only allows single-letter lanes `a`-`z`, so several MIDI notes
 * share a letter and are disambiguated via:
 *   - modifiers (e.g. `:o` for open hi-hat),
 *   - `instrumentMapping[lane].midi.note` (declares the default MIDI note
 *     for a letter),
 *   - per-note `metadata.midi.note` overrides (used by the converter to preserve
 *     the exact source note for round-trip fidelity).
 *
 * The choice of letters here is deliberate:
 *   - `r`/`x`/`o`/`c`/`h`/`f`/`a`/`g` are reserved as DSL modifiers; using them
 *     as lanes would still parse, but to avoid visual clashes we prefer
 *     other letters for cymbals and toms.
 *   - `d` (instead of `r`) for Ride to avoid clashing with the rim-shot
 *     modifier when rendered as `s:r`.
 *   - `p` for hand-clap, `b` for "bells/percussion" (tambourine, cowbell).
 */
import { Modifier } from 'src/schema/dsl/dsl';
import { DrumInstrumentKind } from 'src/instruments/instruments';

export type Limb = 'lh' | 'rh' | 'lf' | 'rf';

export type GmEntry = {
  lane: string;
  /** First-class instrument kind for the linter / lint-aware tooling. */
  kind: DrumInstrumentKind;
  modifiers?: Modifier[];
  name: string;
  limb?: Limb;
};

/** Read-side mapping: MIDI note number -> DSL lane + display data. */
export const GM_PERCUSSION: Readonly<Record<number, GmEntry>> = {
  35: { lane: 'k', kind: 'kick', name: 'Acoustic Bass Drum', limb: 'rf' },
  36: { lane: 'k', kind: 'kick', name: 'Kick', limb: 'rf' },
  37: { lane: 's', kind: 'snare', modifiers: ['x'], name: 'Side Stick', limb: 'lh' },
  38: { lane: 's', kind: 'snare', name: 'Snare', limb: 'lh' },
  39: { lane: 'p', kind: 'custom', name: 'Hand Clap', limb: 'lh' },
  40: { lane: 's', kind: 'snare', name: 'Electric Snare', limb: 'lh' },
  41: { lane: 'f', kind: 'tom', name: 'Low Floor Tom' },
  42: { lane: 'h', kind: 'hihat', modifiers: ['c'], name: 'Closed Hi-Hat', limb: 'rh' },
  43: { lane: 'f', kind: 'tom', name: 'High Floor Tom' },
  44: { lane: 'h', kind: 'hihat', modifiers: ['f'], name: 'Pedal Hi-Hat', limb: 'lf' },
  45: { lane: 't', kind: 'tom', name: 'Low Tom' },
  46: { lane: 'h', kind: 'hihat', modifiers: ['o'], name: 'Open Hi-Hat', limb: 'rh' },
  47: { lane: 't', kind: 'tom', name: 'Low-Mid Tom' },
  48: { lane: 't', kind: 'tom', name: 'Hi-Mid Tom' },
  49: { lane: 'c', kind: 'crash', name: 'Crash Cymbal 1', limb: 'rh' },
  50: { lane: 't', kind: 'tom', name: 'High Tom' },
  51: { lane: 'd', kind: 'ride', name: 'Ride Cymbal 1', limb: 'rh' },
  52: { lane: 'c', kind: 'crash', name: 'Chinese Cymbal', limb: 'rh' },
  53: { lane: 'd', kind: 'ride', name: 'Ride Bell', limb: 'rh' },
  54: { lane: 'b', kind: 'custom', name: 'Tambourine' },
  55: { lane: 'c', kind: 'crash', name: 'Splash Cymbal', limb: 'rh' },
  56: { lane: 'b', kind: 'custom', name: 'Cowbell' },
  57: { lane: 'c', kind: 'crash', name: 'Crash Cymbal 2', limb: 'rh' },
  59: { lane: 'd', kind: 'ride', name: 'Ride Cymbal 2', limb: 'rh' },
};

/**
 * Generic display name to use when multiple GM entries with different
 * `modifiers` map to the same DSL lane. Drumjot's instrument row holds
 * every variant of a lane together (notes carry `:o` / `:c` / `:f`
 * modifiers per-hit), so a variant-specific name like "Closed Hi-Hat"
 * reads wrong on a row that also contains open hits. `buildInstrumentMap`
 * substitutes from this map when the source MIDI has more than one
 * `GM_PERCUSSION` entry on the same lane; single-variant rows keep the
 * GM entry's specific name (a row with only MIDI 42 still reads "Closed
 * Hi-Hat"; the generic is only for the mixed case).
 */
export const GENERIC_INSTRUMENT_NAME_BY_PITCH: Readonly<Record<string, string>> = {
  k: 'Kick',
  s: 'Snare',
  f: 'Floor Tom',
  h: 'Hi-Hat',
  t: 'Tom',
  c: 'Crash Cymbal',
  d: 'Ride Cymbal',
  b: 'Percussion',
};

/**
 * Pick a default MIDI note for a (lane, modifiers) combination. Used when
 * writing a Drumjot Note to MIDI and neither the note's own
 * `metadata.midi.note` nor the instrument mapping's `midi.note` is set.
 *
 * Returns `undefined` for unknown lanes; callers must decide whether to
 * skip the note or substitute another value.
 */
export function defaultMidiNote(
  lane: string,
  modifiers: ReadonlySet<Modifier>
): number | undefined {
  switch (lane) {
    case 'k':
      return 36;
    case 's':
      if (modifiers.has('x')) return 37; // cross-stick
      return 38;
    case 'h':
      if (modifiers.has('o')) return 46; // open hi-hat
      if (modifiers.has('f')) return 44; // pedal hi-hat
      // ':h' (half-open) currently has no distinct GM note; fall back to closed.
      return 42;
    case 'c':
      return 49;
    case 'd':
      return 51;
    case 't':
      return 50;
    case 'f':
      return 41;
    case 'p':
      return 39;
    case 'b':
      return 56;
    default:
      return undefined;
  }
}

/**
 * Fallback letter assignment for MIDI notes outside the canonical GM map.
 *
 * Two callers need this: the note-emission loop and the instrument-map
 * builder. They must agree on the letter assigned to each MIDI number,
 * and no two unknown MIDI numbers in the same song may share a letter
 * (otherwise the resulting `instrumentMapping` is ambiguous and one of
 * the drums silently disappears).
 *
 * Strategy: deterministic per-song allocation, not per-note. We use a
 * starting "hint" derived from the MIDI number (so identical files keep
 * stable letters across runs), but skip any letter that's already taken
 * either by GM_PERCUSSION or by a previous fallback in the same song.
 * Letters are walked from the end of the alphabet (`z`, `y`, ...) to
 * minimise clashes with conventional kit letters.
 */
export function deriveLetterFromMidi(
  midiNote: number,
  claimed: ReadonlySet<string> = new Set()
): string {
  const hint = String.fromCharCode('z'.charCodeAt(0) - (midiNote % 26));
  if (!claimed.has(hint)) return hint;
  // Walk z -> a looking for the first free slot. 26 letters total, so
  // this terminates trivially for any plausible drum kit.
  for (let i = 25; i >= 0; i--) {
    const c = String.fromCharCode('a'.charCodeAt(0) + i);
    if (!claimed.has(c)) return c;
  }
  // Exhausted: more than 26 distinct unknown drum classes in one song.
  // Pick `z` and accept the collision; in practice this never happens.
  return 'z';
}

/**
 * Build a per-song MIDI-note -> DSL-letter map. The canonical
 * `GM_PERCUSSION` entries win for any MIDI note they cover; everything
 * else gets a fallback letter that doesn't collide with the canonical
 * mappings or with any previously-assigned fallback.
 */
export function allocateLanesForMidi(
  used: Iterable<number>
): Map<number, string> {
  const out = new Map<number, string>();
  const claimed = new Set<string>();
  const uniqueSorted = Array.from(new Set(used)).sort((a, b) => a - b);

  for (const midi of uniqueSorted) {
    const entry = GM_PERCUSSION[midi];
    if (entry) {
      out.set(midi, entry.lane);
      claimed.add(entry.lane);
    }
  }
  for (const midi of uniqueSorted) {
    if (out.has(midi)) continue;
    const letter = deriveLetterFromMidi(midi, claimed);
    out.set(midi, letter);
    claimed.add(letter);
  }
  return out;
}
