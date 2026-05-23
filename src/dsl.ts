/**
 * DSL data structures matching the Drumming DSL spec (SPEC.md).
 *
 * These types describe a jot at the level the DSL describes it: a tree of
 * notes, rests, simultaneities, groups, bar separators, pattern definitions
 * and references, plus metadata. No parser is included; a Jot is authored
 * either directly as data or (in future) produced by a parser.
 */

// ---------- Metadata ----------

export type Volume = 'pp' | 'p' | 'mp' | 'mf' | 'f' | 'ff';

/**
 * Time signature, e.g. 4/4 = { count: 4, unit: 4 }, 7/8 = { count: 7, unit: 8 }.
 * `unit` is the note value of the denominator (4 = quarter, 8 = eighth, ...).
 */
export type TimeSignature = {
  count: number;
  unit: number;
};

export type BpmTransition = {
  start?: number;
  end: number;
  /** duration of the transition, in bars */
  duration: number;
};

export type VolTransition = {
  start?: Volume;
  end: Volume;
  /** duration of the transition, in bars */
  duration: number;
};

export type Limb = 'lh' | 'rh' | 'lf' | 'rf';

/**
 * One drum-kit instrument (kick, snare, hi-hat, ...). The DSL's
 * `instrumentMapping` is a `Record<pitch, Instrument>` that resolves each
 * single-letter pitch to an instrument with a first-class `kind` (used by
 * the linter), an optional human-readable display label, optional limb
 * assignment, and an optional default MIDI note.
 *
 * `kind` is required — see `src/instruments.ts` for the enum and defaults.
 * For pitches outside the canonical kit, set `kind: 'custom'`; the linter
 * treats those as unrestricted.
 */
export type Instrument = {
  kind: import('./instruments').DrumInstrumentKind;
  name?: string;
  limb?: Limb;
  midi?: { note: number; vol?: Volume };
};

/**
 * Metadata as it appears in `{ ... }` or `{{ ... }}`. Per-note > per-group >
 * global > instrumentMapping precedence is the consumer's responsibility.
 */
export type Metadata = {
  bpm?: number | BpmTransition;
  vol?: Volume | VolTransition;
  time?: TimeSignature;
  /** Maps each pitch letter to an Instrument. Order is the rendered lane order. */
  instrumentMapping?: Record<string, Instrument>;
  comment?: string;
  /**
   * Optional song title. When present in `globalMetadata` it is lifted out
   * to `Jot.title` by the parser, but we declare it explicitly here so
   * direct authors get autocomplete + typing without falling back to the
   * `[key: string]: unknown` index signature.
   */
  title?: string;
  /**
   * Three timeline epochs the playback / score / waveform code coordinates
   * around. All are seconds, measured forward from t=0 of the loaded audio
   * file (`audioT0`, which has no field because it's the origin by
   * definition). The expected ordering is
   * `audioT0 (=0) <= signalT0Sec <= drumsT0Sec`.
   *
   * - `drumsT0Sec` — audio time of the first drum onset. Bar 1 of the
   *   score sits exactly here; the player delays its schedule by this much
   *   so rendered drums hit at the same wall-clock offset as in the source
   *   audio. Replaces the previous `startOffset` field. Optional —
   *   undefined / 0 mean drums start at the file head.
   * - `signalT0Sec` — audio time of the first non-silent sample (e.g. a
   *   vocal pickup, a guitar intro). Strictly informational today; the
   *   waveform + debug overlays may use it to show "music starts here"
   *   distinct from "drums start here". Optional.
   *
   * Pre-drum bars (audio that exists before the first drum hit) get
   * negative `bar.index` in the rendered jot. `leadBars` counts how many
   * such bars precede bar 1; carried here so consumers (mixer, debug
   * provenance, etc.) don't have to recount the leading rest bars
   * themselves.
   */
  drumsT0Sec?: number;
  signalT0Sec?: number;
  leadBars?: number;
  [key: string]: unknown;
};

// ---------- Modifiers & sticking ----------

/** Single- and multi-character modifiers from the spec. */
export type Modifier =
  | 'a' // accent
  | 'g' // ghost
  | 'c' // closed (hi-hat)
  | 'h' // half-open
  | 'o' // open
  | 'f' // foot / chick
  | 's' // splash
  | 'r' // rim shot
  | 'x' // cross-stick
  | 'z' // buzz / press
  | 'k' // choke
  | 'm' // mute
  | 'l' // let-ring
  | 'fl' // flam
  | 'dr' // drag
  | 'rf'; // ruff

export type Sticking = 'r' | 'l' | 'rf' | 'lf';

// ---------- Source ranges ----------

/**
 * Half-open byte range `[start, end)` into the original DSL source string.
 * Populated by the parser for elements where the linter needs to point at a
 * specific span (`Note`, `Group`, `Bar`, `Voice`). Manually-authored Jot
 * objects (e.g. in `src/fakes.ts`) leave it undefined; the linter degrades
 * gracefully when ranges are absent (it surfaces diagnostics without
 * position info rather than crashing).
 */
export type SourceRange = {
  start: number;
  end: number;
};

// ---------- Elements ----------

export type Note = {
  kind: 'note';
  /** Single lowercase letter a-z; resolved via the active `instrumentMapping`. */
  pitch: string;
  modifiers?: Modifier[];
  sticking?: Sticking;
  /** Roll/buzz fill (`~`). */
  roll?: boolean;
  /** Duration weight from `_N`; defaults to 1. */
  weight?: number;
  /** Repeat count from `*N`; defaults to 1. */
  repeat?: number;
  metadata?: Metadata;
  /** Source range; populated by the parser, omitted by hand-built jots. */
  range?: SourceRange;
};

export type Rest = {
  kind: 'rest';
  weight?: number;
  repeat?: number;
};

/** Onset-aligned simultaneity: `a+b+c`. All inner elements share a single onset. */
export type Simultaneity = {
  kind: 'simul';
  elements: Element[];
  weight?: number;
};

/**
 * Parenthesised group `(...)`. A top-level group spans one bar.
 * Inner elements are distributed evenly across the group's duration unless
 * individual elements override with `_N` weights.
 */
export type Group = {
  kind: 'group';
  elements: Element[];
  weight?: number;
  repeat?: number;
  roll?: boolean;
  modifiers?: Modifier[];
  metadata?: Metadata;
  /**
   * Set during expansion (`RenderedJot.expandElements`) when this group
   * originated from a `patternRef`. The renderer uses it to draw outlines
   * around pattern usages and link them back to their definition.
   */
  patternSource?: { name: string };
  /** Source range; populated by the parser, omitted by hand-built jots. */
  range?: SourceRange;
};

/**
 * Substitution within a pattern reference, e.g. `[Name#3=(...)]` or `[Name#4-8=(...)]`.
 * Positions are 1-based; ranges are inclusive on both ends. Path entries
 * support descent into nested groups: `[Name#3#2=(...)]` -> path: [3, 2].
 */
export type PatternSubstitution = {
  path: Array<number | [number, number]>;
  replacement: Element;
};

/**
 * Reference to a previously-defined pattern. With substitutions, this is the
 * `[Name#3=(...)]` style position replacement; the original pattern is not
 * mutated unless the result is re-assigned via `=`.
 */
export type PatternRef = {
  kind: 'patternRef';
  name: string;
  substitutions?: PatternSubstitution[];
  weight?: number;
  repeat?: number;
};

export type Element = Note | Rest | Simultaneity | Group | PatternRef;

// ---------- Bars, voices, patterns, jot ----------

/**
 * One bar's worth of elements (between two `|`). The sum of element weights
 * must equal the current time signature's bar length (rendering enforces).
 */
export type Bar = {
  elements: Element[];
  /** Inline metadata override for this bar (rare; usually metadata is global). */
  metadata?: Metadata;
  /** Source range covering the bar's content (`|` to `|`). */
  range?: SourceRange;
};

/**
 * One side of a global simultaneity (`||`). Multiple voices play in parallel
 * from a common start; the track length equals the longest voice. The DSL
 * surface syntax for joining voices is the `||` operator; nothing else in the
 * codebase should use the term "voice" for any other concept.
 *
 * `name` is purely a display hint (the renderer uses it instead of
 * "Voice 1/2/..." when present); it has no spec representation since DSL
 * source can't name a `||` side.
 */
export type Voice = {
  /** Optional display label, e.g. "Hands" / "Feet". Not parseable from DSL. */
  name?: string;
  /** Anacrusis/pickup elements before the first `|`. Not length-checked. */
  anacrusis?: Element[];
  bars: Bar[];
  /** Source range spanning the voice (from start to next `||` or EOF). */
  range?: SourceRange;
};

export type Pattern = {
  name: string;
  elements: Element[];
};

/** Top-level jot. */
export type Jot = {
  title: string;
  /** Effective from the start; per-group/per-note metadata overrides. */
  globalMetadata: Metadata;
  /** Named patterns keyed by identifier (without the `[]`). */
  patterns?: Record<string, Pattern>;
  /** Voices laid out in parallel via `||`. A single-voice jot has length 1. */
  voices: Voice[];
};

// ---------- Convenience helpers ----------

export const note = (pitch: string, opts: Omit<Note, 'kind' | 'pitch'> = {}): Note => ({
  kind: 'note',
  pitch,
  ...opts,
});

export const rest = (opts: Omit<Rest, 'kind'> = {}): Rest => ({ kind: 'rest', ...opts });

export const simul = (...elements: Element[]): Simultaneity => ({ kind: 'simul', elements });

export const group = (elements: Element[], opts: Omit<Group, 'kind' | 'elements'> = {}): Group => ({
  kind: 'group',
  elements,
  ...opts,
});

export const bar = (...elements: Element[]): Bar => ({ elements });

export const patternRef = (
  name: string,
  opts: Omit<PatternRef, 'kind' | 'name'> = {}
): PatternRef => ({ kind: 'patternRef', name, ...opts });
