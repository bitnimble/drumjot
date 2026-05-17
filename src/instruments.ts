/**
 * First-class drum-instrument taxonomy.
 *
 * Drumjot used to identify instruments purely by user-supplied string names
 * (`Instrument.name`). That made it impossible to reason deterministically
 * about which modifiers / sticking / simultaneity patterns are valid, since
 * we never knew whether "Snare" was actually a snare or some user-defined
 * percussion. This module introduces a finite enum of `DrumInstrumentKind`s
 * plus a metadata table describing each one's constraints.
 *
 * Scope intentionally matches the six stems split out by the MDX23C
 * drum-piece separator we use in the transcriber pipeline (kick, snare,
 * hi-hat, ride, crash, tom) plus a `custom` catch-all for anything else.
 * `custom` opts out of instrument-tier lint checks — users can still attach
 * a display label via `Instrument.name`.
 */
import { Modifier } from 'src/dsl';

export type DrumInstrumentKind =
  | 'kick'
  | 'snare'
  | 'hihat'
  | 'ride'
  | 'crash'
  | 'tom'
  | 'custom';

export const ALL_DRUM_INSTRUMENT_KINDS: readonly DrumInstrumentKind[] = [
  'kick',
  'snare',
  'hihat',
  'ride',
  'crash',
  'tom',
  'custom',
] as const;

/**
 * Whether an instrument is naturally played with hands or feet. Hi-hat is the
 * notable swing — closed/open hits are hand strokes but the `:f` (foot/chick)
 * and `:s` (foot-splash) modifiers redirect to the left foot.
 */
export type LimbCategory = 'hand' | 'foot' | 'either';

export type InstrumentKindMetadata = {
  kind: DrumInstrumentKind;
  /** Display label used when the user-supplied `Instrument.name` is absent. */
  label: string;
  /** Default limb category when no modifier overrides it. */
  defaultLimb: LimbCategory;
  /**
   * Set of modifiers valid for this instrument. `null` means "any modifier"
   * (used by `custom` to opt out of validity checks).
   */
  validModifiers: ReadonlySet<Modifier> | null;
  /**
   * Modifiers that flip this instrument from hand to foot. Used by the
   * sticking inferrer + the "too many hands" performance rule to exclude
   * hi-hat foot strikes from the hand count.
   */
  footModifiers: ReadonlySet<Modifier>;
  /** Severity bumps for specific modifiers (e.g. ':o' on crash is a warning). */
  modifierWarnings: ReadonlySet<Modifier>;
  /** Canonical GM MIDI note number; undefined for `custom`. */
  defaultMidi: number | undefined;
};

// ---------- Modifier sets ----------

/** All modifiers from SPEC.md — used for `custom` (no restriction). */
const allMods: ReadonlySet<Modifier> = new Set<Modifier>([
  'a', 'g', 'c', 'h', 'o', 'f', 's', 'r', 'x', 'z', 'k', 'm', 'l', 'fl', 'dr', 'rf',
]);
// silence unused: kept for documentation / future use
void allMods;

const drumHeadMods: ReadonlySet<Modifier> = new Set<Modifier>([
  'a', 'g', 'r', 'x', 'z', 'fl', 'dr', 'rf', 'm',
]);

const kickMods: ReadonlySet<Modifier> = new Set<Modifier>([
  'a', 'g',
]);

const hihatMods: ReadonlySet<Modifier> = new Set<Modifier>([
  'a', 'g', 'c', 'h', 'o', 'f', 's', 'k', 'm',
]);

const rideMods: ReadonlySet<Modifier> = new Set<Modifier>([
  'a', 'g', 'k', 'm', 'l',
]);

const crashMods: ReadonlySet<Modifier> = new Set<Modifier>([
  'a', 'g', 'k', 'm', 'l', 'o',
]);

const tomMods: ReadonlySet<Modifier> = new Set<Modifier>([
  'a', 'g', 'r', 'x', 'z', 'fl', 'dr', 'rf', 'm',
]);

// ---------- Metadata table ----------

export const INSTRUMENT_METADATA: Readonly<
  Record<DrumInstrumentKind, InstrumentKindMetadata>
> = {
  kick: {
    kind: 'kick',
    label: 'Kick',
    defaultLimb: 'foot',
    validModifiers: kickMods,
    footModifiers: new Set<Modifier>(),
    modifierWarnings: new Set<Modifier>(),
    defaultMidi: 36,
  },
  snare: {
    kind: 'snare',
    label: 'Snare',
    defaultLimb: 'hand',
    validModifiers: drumHeadMods,
    footModifiers: new Set<Modifier>(),
    modifierWarnings: new Set<Modifier>(),
    defaultMidi: 38,
  },
  hihat: {
    kind: 'hihat',
    label: 'Hi-hat',
    defaultLimb: 'hand',
    validModifiers: hihatMods,
    footModifiers: new Set<Modifier>(['f', 's']),
    modifierWarnings: new Set<Modifier>(),
    defaultMidi: 42,
  },
  ride: {
    kind: 'ride',
    label: 'Ride',
    defaultLimb: 'hand',
    validModifiers: rideMods,
    footModifiers: new Set<Modifier>(),
    modifierWarnings: new Set<Modifier>(),
    defaultMidi: 51,
  },
  crash: {
    kind: 'crash',
    label: 'Crash',
    defaultLimb: 'hand',
    validModifiers: crashMods,
    footModifiers: new Set<Modifier>(),
    // ':o' on crash/china is technically a thing (open hit, no choke) but
    // most charts don't use it and the LLM tends to over-emit it; flag as
    // a warning rather than a hard error.
    modifierWarnings: new Set<Modifier>(['o']),
    defaultMidi: 49,
  },
  tom: {
    kind: 'tom',
    label: 'Tom',
    defaultLimb: 'hand',
    validModifiers: tomMods,
    footModifiers: new Set<Modifier>(),
    modifierWarnings: new Set<Modifier>(),
    defaultMidi: 47,
  },
  custom: {
    kind: 'custom',
    label: 'Custom',
    defaultLimb: 'either',
    validModifiers: null,
    footModifiers: new Set<Modifier>(),
    modifierWarnings: new Set<Modifier>(),
    defaultMidi: undefined,
  },
};

// ---------- Pitch-letter defaults ----------

/**
 * Auto-fill mapping from the canonical Drumjot pitch letter to its instrument
 * kind. This matches the transcriber pipeline's `STEM_NAME_TO_PITCH` so
 * round-trips through MIDI / RLRR / transcription land on consistent kinds.
 *
 * Unknown letters fall back to `custom` (callers can still preserve a
 * human-readable `name`).
 */
const DEFAULT_PITCH_TO_KIND: Readonly<Record<string, DrumInstrumentKind>> = {
  k: 'kick',
  s: 'snare',
  h: 'hihat',
  d: 'ride',
  c: 'crash',
  t: 'tom',
  // legacy/extended letters used by the MIDI importer
  f: 'tom',     // floor tom letter in gm.ts
};

export function defaultKindForPitch(pitch: string): DrumInstrumentKind {
  return DEFAULT_PITCH_TO_KIND[pitch] ?? 'custom';
}

// ---------- Helpers ----------

export function getInstrumentMetadata(
  kind: DrumInstrumentKind
): InstrumentKindMetadata {
  return INSTRUMENT_METADATA[kind];
}

export function isValidModifier(
  kind: DrumInstrumentKind,
  modifier: Modifier
): boolean {
  const meta = INSTRUMENT_METADATA[kind];
  if (meta.validModifiers === null) return true;
  return meta.validModifiers.has(modifier);
}

export function isWarningModifier(
  kind: DrumInstrumentKind,
  modifier: Modifier
): boolean {
  return INSTRUMENT_METADATA[kind].modifierWarnings.has(modifier);
}

export function isFootModifier(
  kind: DrumInstrumentKind,
  modifier: Modifier
): boolean {
  return INSTRUMENT_METADATA[kind].footModifiers.has(modifier);
}

/**
 * Compute the effective limb category for a single note: starts from the
 * instrument's default, then flips to `foot` if any of the note's modifiers
 * is a foot-modifier for that instrument (e.g. `:f` / `:s` on hi-hat).
 */
export function effectiveLimbCategory(
  kind: DrumInstrumentKind,
  modifiers: readonly Modifier[] | ReadonlySet<Modifier>
): LimbCategory {
  const meta = INSTRUMENT_METADATA[kind];
  const mods =
    modifiers instanceof Set ? modifiers : new Set<Modifier>(modifiers);
  for (const m of mods) {
    if (meta.footModifiers.has(m)) return 'foot';
  }
  return meta.defaultLimb;
}
