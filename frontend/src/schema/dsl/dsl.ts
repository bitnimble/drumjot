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
  /** ramp length, in quarter-note beats (may be fractional) */
  duration: number;
};

export type VolTransition = {
  start?: Volume;
  end: Volume;
  /** ramp length, in quarter-note beats (may be fractional) */
  duration: number;
};

export type Limb = 'lh' | 'rh' | 'lf' | 'rf';

/**
 * One drum-kit instrument (kick, snare, hi-hat, ...). The DSL's
 * `instrumentMapping` is a `Record<lane, Instrument>` that resolves each
 * single-letter lane to an instrument with a first-class `kind` (used by
 * the linter), an optional human-readable display label, optional limb
 * assignment, and an optional default MIDI note.
 *
 * `kind` is required — see `src/instruments.ts` for the enum and defaults.
 * For lanes outside the canonical kit, set `kind: 'custom'`; the linter
 * treats those as unrestricted.
 */
export type Instrument = {
  kind: import('src/instruments/instruments').DrumInstrumentKind;
  name?: string;
  limb?: Limb;
  midi?: { note: number; vol?: Volume };
};

/**
 * Sticky tempo change anchored to a precise (bar, beat) position within
 * layer 0's bar timeline. The single source of truth for tempo at
 * runtime, including the **initial** tempo, which is just the event
 * anchored at the song's start (the drums-enter downbeat, bar `leadBars`).
 * Readers walk `Jot.tempoEvents` forward and the tempo at any (bar, beat)
 * is the most recent event at or before that position; the span before
 * the first event defaults to 120 (see `tempo.initialBpm`). There is no
 * `globalMetadata.bpm`.
 *
 * Anchored by array index into `layers[0].bars[]` (NOT the renderer's
 * 1-based `index` field) so the position survives lead-in synthesis,
 * anacrusis insertion, and drum-offset shifts. Tempo is jot-global (a
 * single MIDI tempo track on output), so anchoring against layer 0 is
 * sufficient; additional layers share the same bar grid.
 */
export type TempoEvent = {
  /** 0-based index into layers[0].bars. */
  barIndex: number;
  /** Beat-within-bar offset of the anchor (quarter notes from bar start). 0 = downbeat. */
  beat: number;
  bpm: number | BpmTransition;
};

/**
 * Metadata as it appears in `{ ... }` or `{{ ... }}`. Per-note > per-group >
 * global > instrumentMapping precedence is the consumer's responsibility.
 *
 * NOTE: `bpm` is a parse-time field only; it carries an authored `{{bpm}}`
 * / `{bpm}` value out of the DSL text. The parser hoists EVERY occurrence
 * (global, bar, group, note) into `Jot.tempoEvents` and strips `bpm` from
 * all metadata, including `globalMetadata`; no runtime path reads
 * `metadata.bpm`. The song's initial tempo is the first `tempoEvent` (at
 * the drums-enter downbeat), not a `globalMetadata.bpm`. Producers other
 * than the parser (from_midi, the rlrr parser, hand-authored fakes) should
 * populate `Jot.tempoEvents` directly and leave `bpm` off all metadata.
 */
export type Metadata = {
  bpm?: number | BpmTransition;
  vol?: Volume | VolTransition;
  time?: TimeSignature;
  /** Maps each lane letter to an Instrument. Order is the rendered lane order. */
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
   * Jot-time (score-time, bar-1 downbeat = 0) anchor for the recorded
   * audio's lead-in. `songLeadIn` is the jot time at which the loaded audio
   * file begins (the audio time of the first drum onset, negated into jot
   * time). So with a 5.3s pre-drum intro, `songLeadIn` is
   * `-5.3`: bar 1 sits at jot 0, and `media = jot - songLeadIn` lines the
   * rendered drums up with the recorded ones. Optional, undefined / 0 mean
   * the audio starts at bar 1 (no pre-drum intro). The full set of derived
   * time anchors (incl. the view-only virtual lead-in) is the runtime
   * `Epochs` record; only this persisted alignment lives here.
   *
   * Pre-drum bars (audio that exists before the first drum hit) get
   * negative `bar.index` in the rendered jot. `leadBars` counts how many
   * such bars precede bar 1; carried here so consumers (mixer, debug
   * provenance, etc.) don't have to recount the leading rest bars
   * themselves.
   */
  songLeadIn?: number;
  leadBars?: number;
  /**
   * Grid density chosen by whichever producer built this jot, expressed
   * as 1/N-of-a-whole-note (so 48 = the 1/48 grid `from_midi` defaults
   * to, 12 slots per quarter). Advisory only: it does NOT change how the
   * DSL is interpreted (positions come from element weights), it tells
   * grid-aware consumers (note position readouts, the drum-offset slider,
   * sub-slot offset math) what slot resolution the producer was working
   * at. Absent → `DEFAULT_GRID_DIVISION` (see `src/grid.ts`).
   */
  gridDivision?: number;
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
 * specific span (`Note`, `Group`, `Bar`, `Layer`). Manually-authored Jot
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
  lane: string;
  /**
   * Signed sub-slot timing offset in milliseconds, relative to the note's
   * natural (slot-aligned) position. Lets a hit render and play between
   * grid slots, swing, ghost flams, push/pull feel; without snapping it
   * onto the grid. Set programmatically by producers (e.g. `from_midi`
   * when an onset lands far enough off any grid slot); there is no DSL
   * surface syntax for it yet. Absent → the note plays exactly on its slot.
   */
  offset?: number;
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
   * Set when this group originated from a `patternRef` (e.g. by the parser
   * or the reactive converter). The renderer uses it to draw outlines
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

// ---------- Bars, layers, patterns, jot ----------

/**
 * Mid-bar tempo marker anchored at an element index within a bar's
 * (post-parse, pre-expansion) element list. Emitted by the parser when
 * `{{bpm}}` appears between elements; consumed and stripped by the
 * tempo-hoist pass which turns it into a {@link TempoEvent} with the
 * anchor element's beat-within-bar position. Should not be present on a
 * parsed jot after `parse()` returns.
 */
export type BarTempoSource = {
  /** 0-based index into the bar's `elements` (the anchor element). */
  elementIndex: number;
  bpm: number | BpmTransition;
};

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
  /**
   * Transient parser output: mid-bar `{{bpm}}` markers anchored by
   * element index. Hoisted into `Jot.tempoEvents` and stripped post-parse.
   */
  tempoSources?: BarTempoSource[];
};

/**
 * One side of a global simultaneity (`||`). Multiple layers play in parallel
 * from a common start; the track length equals the longest layer. The DSL
 * surface syntax for joining layers is the `||` operator; nothing else in the
 * codebase should use the term "layer" for any other concept.
 *
 * `name` is purely a display hint (the renderer uses it instead of
 * "Layer 1/2/..." when present); it has no spec representation since DSL
 * source can't name a `||` side.
 */
export type Layer = {
  /** Optional display label, e.g. "Hands" / "Feet". Not parseable from DSL. */
  name?: string;
  /** Anacrusis/pickup elements before the first `|`. Not length-checked. */
  anacrusis?: Element[];
  bars: Bar[];
  /** Source range spanning the layer (from start to next `||` or EOF). */
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
  /** Layers laid out in parallel via `||`. A single-layer jot has length 1. */
  layers: Layer[];
  /**
   * Sticky tempo changes (mid-bar OK) sorted by (barIndex, beat). The
   * sole source of truth for tempo, including the song's initial tempo
   * (the event at the drums-enter downbeat); consumers walk this list
   * forward to compute the active bpm at any position. See
   * {@link TempoEvent}. The span before the first event defaults to 120
   * (see `tempo.initialBpm`).
   */
  tempoEvents?: TempoEvent[];
  /**
   * Per-bar performance drift in seconds, indexed by `layers[0].bars`
   * (lead-in bars = 0). How far each bar's *real* recorded downbeat sits
   * past where the clean uniform tempo grid puts it, the deviation the
   * tempo map smoothed away (`transcription.json`'s `barDrift`). The
   * displayed tempo stays uniform, but the waveform renderer + (eventually)
   * the playhead use `bar.startSec + drift` as the bar's true audio downbeat
   * to align bar lines to the recording. Lives at the jot level (not in
   * `globalMetadata`, which the DSL writer would dump into a `{{...}}` block)
   * and is dropped on DSL export; it's recording-specific, not authorable.
   * Undefined / all-zero for a metronomic recording or a hand-authored jot.
   */
  barDrift?: number[];
};

// ---------- Convenience helpers ----------

export const note = (lane: string, opts: Omit<Note, 'kind' | 'lane'> = {}): Note => ({
  kind: 'note',
  lane,
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
