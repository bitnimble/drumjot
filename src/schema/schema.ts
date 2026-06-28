/**
 * The Drumjot collaborative document schema.
 *
 * This is the canonical, editable, conflict-free representation a song; * deliberately FLAT and value-addressed, not the nested DSL weight-tree
 * (which stays an export-only serialization target). A note's musical
 * position is plain LWW register fields (`barId` + `beat`) and its lane is
 * a `lane` field, so every core edit, add, move along a track, move
 * track A→B, delete, is a register write or a keyed add/remove, and
 * rendering order is a deterministic sort rather than any CRDT sequence.
 *
 * The enums mirror `src/dsl/dsl.ts` exactly; the type-level tests in
 * `./test/schema.test.ts` assert `Infer<…>` equals the hand-written DSL
 * types so this can't silently drift from the domain.
 */
import { z } from 'zod';
import {
  idMap,
  type Infer,
  type Init,
  lazy,
  movableList,
  type ReactiveMap,
  record,
  type Snapshot,
  union,
  type UnionDescriptor,
} from './descriptors';
import { createReactiveDoc, type ReactiveDoc } from './reactive_doc';

// ---------- Leaf enums (mirror src/dsl/dsl.ts) ----------

// NB: accent/ghost are NOT modifiers in the schema. They're loudness, stored in
// `velocity`; the DSL `:a`/`:g` markers (still in the DSL `Modifier` type) are
// converted to a velocity in `from_dsl.ts`, and the accent ring / ghost glyph
// are derived from velocity at render time (see `bar_view.tsx`).
export const MODIFIER = z.enum([
  'c', // closed (hi-hat)
  'h', // half-open
  'o', // open
  'f', // foot / chick
  's', // splash
  'r', // rim shot
  'x', // cross-stick
  'z', // buzz / press
  'k', // choke
  'm', // mute
  'l', // let-ring
  'fl', // flam
  'dr', // drag
  'rf', // ruff
]);

export const STICKING = z.enum(['r', 'l', 'rf', 'lf']);
export const LIMB = z.enum(['lh', 'rh', 'lf', 'rf']);
export const INSTRUMENT_KIND = z.enum(['kick', 'snare', 'hihat', 'ride', 'crash', 'tom', 'custom']);

// ---------- Entities ----------

// ---------- Elements (the note | group | pattern tree) ----------

/**
 * Positional fields every element shares. Coordinates are RELATIVE to the
 * immediate container's space: top-level elements are positioned from
 * their bar's downbeat (and carry `barId`); a group's children are positioned
 * in the group's own internal space (no `barId`; the derivation converts to
 * global time via the group's duration scaling). A note's owning `||` layer is
 * NOT stored, it derives from its `trackId`'s placement in {@link
 * OrderingSchema}; only container elements carry a `layerId` routing fallback.
 */
const elementBase = {
  id: z.string(),
  /** Owning bar (top-level only; nested elements live in their group's space). */
  barId: z.string().optional(),
  /** Position in the immediate container's coordinate space. */
  beat: z.number(),
  /** Span in the immediate container's coordinate space. */
  duration: z.number(),
};

/** A single drum hit. Its home is a track (`trackId` → {@link TrackSchema},
 *  which carries the lane and, via `ordering`, the layer); loudness via the
 *  numeric `velocity`. `lane` is also kept on the note (and is the sole home for
 *  layer-agnostic pattern-body template notes, which carry no `trackId`); read
 *  a note's lane via `laneForNote`. A placed note carries NO `layerId`, its
 *  layer follows its track across layer moves with no per-note rewrite. */
export const NoteElementSchema = record({
  ...elementBase,
  kind: z.literal('note'),
  /** The owning track (placed notes). Absent on pattern-definition template
   *  notes, whose layer/track is resolved at each usage site. */
  trackId: z.string().optional(),
  lane: z.string(),
  /** Whole-array LWW register (rarely concurrent; merge the set atomically). */
  modifiers: z.array(MODIFIER),
  sticking: STICKING.optional(),
  /** Roll / buzz fill (DSL `~`). */
  roll: z.boolean().optional(),
  /** Signed sub-slot timing offset in milliseconds (DSL `Note.offset`). */
  offsetMs: z.number().optional(),
  /** MIDI velocity (0-127). The single source of loudness; authored dynamics
   *  (`pp`..`ff` in the DSL) are converted to a velocity at parse time. Absent
   *  = derive from modifiers (accent/ghost) at export, else the default. */
  velocity: z.number().optional(),
  /** Explicit MIDI note override; absent = derive from instrument/lane. */
  midiNote: z.number().optional(),
  /** Raw MIDI tick from a transcribed/imported source (provenance key). */
  midiTick: z.number().optional(),
});

/**
 * A container of child elements with its own `duration`. The children live
 * in the group's internal coordinate space; a **tuplet** is simply a group
 * whose children's natural span ≠ its `duration` (the ratio is the
 * scaling). Moving/stretching a group touches only its own `beat`/
 * `duration`, the children are untouched. Group-level `modifiers`/`roll`
 * apply to all descendants. Annotated `RecordDescriptor` so the
 * `lazy` self-reference type-checks.
 */
export const GroupElementSchema = record({
  ...elementBase,
  kind: z.literal('group'),
  /** Owning `||` layer (top-level only). A routing fallback used when the
   *  group has no placed-note descendant to derive the layer from; a group with
   *  notes follows its tracks' layer instead. */
  layerId: z.string().optional(),
  children: idMap(lazy((): UnionDescriptor => ElementSchema)),
  modifiers: z.array(MODIFIER).optional(),
  roll: z.boolean().optional(),
});

/** A leaf usage of a pattern definition, referenced by its internal id.
 *  The body lives once in `patterns`; rendering instantiates it scaled to
 *  this element's `duration`. (Substitutions deferred.) */
export const PatternElementSchema = record({
  ...elementBase,
  kind: z.literal('pattern'),
  /** Owning `||` layer (top-level only). A pattern usage's body is layer-
   *  agnostic (template notes carry no `trackId`), so its layer is stored here
   *  rather than derived. */
  layerId: z.string().optional(),
  patternId: z.string(),
});

/** An element: a note, a group (incl. tuplets), or a pattern usage. */
export const ElementSchema: UnionDescriptor = union({
  note: NoteElementSchema,
  group: GroupElementSchema,
  pattern: PatternElementSchema,
});

/**
 * A bar: time signature plus an optional per-bar tempo override. Ordered
 * by the `bars` movable list; `id` is stable and is what notes reference,
 * so inserting/moving a bar doesn't disturb notes in other bars. The
 * renderer's 1-based `index` is derived from list position, not stored.
 */
export const BarSchema = record({
  id: z.string(),
  /** Time-signature numerator (e.g. 7 in 7/8). */
  tsCount: z.number(),
  /** Time-signature denominator note value (4 = quarter, 8 = eighth, …). */
  tsUnit: z.number(),
  /** Per-bar tempo override in BPM; absent = inherit the running tempo. */
  tempoBpm: z.number().optional(),
  /** Anacrusis / pickup bar (DSL `layer.anacrusis`). Its length is sized to
   *  its content rather than the time signature; the derivation numbers it
   *  bar 0. Absent = a normal bar. */
  anacrusis: z.boolean().optional(),
});

/**
 * One `||` layer. Layers share the bar grid; they differ only in which
 * tracks (and thus notes) they own. `name` is a display hint ("Hands" /
 * "Feet"), not parseable from DSL. `color` tints the layer's background band
 * in the score (a `#rrggbb`); absent = transparent (the layer-1 default).
 */
export const LayerSchema = record({
  id: z.string(),
  name: z.string().optional(),
  color: z.string().optional(),
});

/**
 * A first-class track: one rendered row. An instrument track owns a `lane`
 * (the instrument) and is the home a {@link NoteElementSchema}'s `trackId`
 * points at; the same lane may appear on tracks in several layers (a snare in
 * layer 1 AND layer 2), but a single layer holds at most one track per lane.
 * Audio / lyrics tracks reference their session entity. A track's layer and
 * group are NOT stored here, they come from its placement in {@link
 * OrderingSchema} (reverse-lookup), so moving a track across layers never
 * re-homes its notes.
 */
export const TrackSchema = union({
  instrument: record({ id: z.string(), kind: z.literal('instrument'), lane: z.string() }),
  audio: record({ id: z.string(), kind: z.literal('audio'), audioId: z.string() }),
  lyrics: record({ id: z.string(), kind: z.literal('lyrics'), lyricsId: z.string() }),
});

/** A named cluster of tracks within a layer (e.g. a cymbal audio waveform +
 *  the crash & ride tracks). `name` is the heading shown in the score / panel;
 *  `color` an optional tint. Referenced by `groupId` from {@link
 *  OrderingSchema}; membership + order live there, not here. */
export const TrackGroupSchema = record({
  id: z.string(),
  name: z.string(),
  color: z.string().optional(),
});

/** One run within a layer: a named group (`groupId` set) or a loose
 *  (ungrouped) run (`groupId` null). Loose runs may repeat so loose tracks can
 *  sit both above and below a group. */
export const OrderSlotSchema = record({
  groupId: z.string().nullable(),
  tracks: movableList(record({ trackId: z.string() })),
});

/** One layer's arrangement: its groups + loose runs, in render order. */
export const OrderLayerSchema = record({
  layerId: z.string(),
  slots: movableList(OrderSlotSchema),
});

/**
 * The single source of truth for the score's row layout: the ordered layers
 * (top → bottom), each carrying its group/loose-run order and per-track order.
 * The Layers panel and the score's gutter both read + write this.
 * Reverse-lookups (`trackId → layerId`, `trackId → groupId`) derive a track's
 * placement. Modelled as a top-level movable list (not a wrapper record) so it
 * defaults to empty when unseeded; every real converter seeds it via
 * `buildDefaultOrdering`, and helpers tolerate an empty list.
 */
export const OrderingSchema = movableList(OrderLayerSchema);

/**
 * A sticky tempo change anchored at (`barId`, `beat`). Mirrors DSL
 * `TempoEvent` but anchors by stable bar id instead of array index. `bpm`
 * is a flat value or a transition ramp (`BpmTransition`), stored as one
 * LWW register.
 */
export const TempoEventSchema = record({
  id: z.string(),
  barId: z.string(),
  /** Beat-within-bar of the anchor (quarter notes from the downbeat). */
  beat: z.number(),
  bpm: z.union([
    z.number(),
    z.object({
      start: z.number().optional(),
      end: z.number(),
      /** Ramp length, in quarter-note beats (may be fractional). */
      duration: z.number(),
    }),
  ]),
});

/**
 * A reusable pattern definition, keyed in `patterns` by internal `id`
 * (referenced by a `pattern` element's `patternId`). `name` is the display
 * label; `body` is the reusable element tree in pattern-internal space.
 */
export const PatternDefSchema = record({
  id: z.string(),
  name: z.string(),
  body: idMap(lazy((): UnionDescriptor => ElementSchema)),
});

/**
 * Display + playback info for a lane lane, keyed in the `instruments`
 * idMap by the DSL lane letter. Mirrors `dsl.ts` `Instrument`.
 */
export const InstrumentSchema = record({
  kind: INSTRUMENT_KIND,
  name: z.string().optional(),
  limb: LIMB.optional(),
  /** Default MIDI note for the lane. */
  midiNote: z.number().optional(),
});

/**
 * The whole song. Global tempo/timeline live as top-level registers; the
 * collections are the editable entities. The bar grid is shared across
 * layers. Deferred to the editing-features phase: pattern definition
 * bodies (see {@link PatternSchema}).
 */
export const JotSchema = record({
  title: z.string(),
  /** Initial/global tempo in force before any per-bar override or event. */
  bpm: z.number(),
  /** Jot time (seconds, bar-1 = 0) at which the recorded audio begins; the
   *  lead-in alignment (<= 0). `media = jot - songLeadIn`. The derived runtime
   *  anchors live in the `Epochs` record. */
  songLeadIn: z.number().optional(),
  /** Number of pre-drum lead-in bars before bar 1. */
  leadBars: z.number().optional(),
  /** Producer grid density (1/N-of-a-whole-note); advisory. */
  gridDivision: z.number().optional(),
  /** Opaque, loader-supplied song metadata not otherwise modelled as a
   *  first-class field (artist, a global `vol`, free-text `comment`, the RLRR
   *  provenance sidecar, and any custom keys), stored verbatim as a JSON
   *  string. Seeded once at load from the DSL `globalMetadata` residual (the
   *  keys left after `bpm`/`time`/`instrumentMapping`/`songLeadIn`/`leadBars`/
   *  `gridDivision`/`title` are lifted to their own fields); read for the
   *  score header's artist/subtitle and re-emitted by the DSL exporter. Kept
   *  opaque because the RLRR sidecar is arbitrarily nested provenance the
   *  editor never structurally edits. */
  globalMetadataJson: z.string().optional(),
  /** `||` layers by id; a single-layer jot has one (or none → primary). */
  layers: idMap(LayerSchema),
  /** First-class tracks (instrument / audio / lyrics) by id; a note's home. */
  tracks: idMap(TrackSchema),
  /** Named track groups by id (membership/order live in `ordering`). */
  trackGroups: idMap(TrackGroupSchema),
  /** The row layout: ordered layers + per-layer group/track order. Empty until
   *  a converter seeds it via `buildDefaultOrdering`. */
  ordering: OrderingSchema,
  bars: movableList(BarSchema),
  /** The note | group | pattern tree, top-level entries keyed by id. */
  elements: idMap(ElementSchema),
  /** Lane letter → instrument display/playback info. */
  instruments: idMap(InstrumentSchema),
  /** Sticky tempo changes by id (sorted by bar order + beat at read time). */
  tempoEvents: idMap(TempoEventSchema),
  /** Pattern definitions by internal id. */
  patterns: idMap(PatternDefSchema),
});

// ---------- Consumer types ----------
// `ElementSchema` is a recursive union; its descriptor is annotated loosely
// (`UnionDescriptor`) to avoid the verbose explicit recursive type, so the
// precise element shapes are hand-written here for consumers to narrow on.

export type Modifier = z.infer<typeof MODIFIER>;
export type Sticking = z.infer<typeof STICKING>;
export type DrumInstrumentKind = z.infer<typeof INSTRUMENT_KIND>;

type ElementCommon = {
  id: string;
  barId?: string;
  beat: number;
  duration: number;
};
export type NoteElement = ElementCommon & {
  kind: 'note';
  /** Owning track (placed notes); absent on pattern-body template notes. The
   *  note's layer derives from this via `ordering`; notes store no `layerId`. */
  trackId?: string;
  lane: string;
  modifiers: Modifier[];
  sticking?: Sticking;
  roll?: boolean;
  offsetMs?: number;
  velocity?: number;
  midiNote?: number;
  midiTick?: number;
};
export type GroupElement = ElementCommon & {
  kind: 'group';
  /** Routing fallback layer; a group with placed notes follows their layer. */
  layerId?: string;
  children: ReactiveMap<Element>;
  modifiers?: Modifier[];
  roll?: boolean;
};
export type PatternElement = ElementCommon & { kind: 'pattern'; layerId?: string; patternId: string };
export type Element = NoteElement | GroupElement | PatternElement;
export type PatternDef = { id: string; name: string; body: ReactiveMap<Element> };

export type Bar = Infer<typeof BarSchema>;
export type Instrument = Infer<typeof InstrumentSchema>;
export type Layer = Infer<typeof LayerSchema>;
export type Track = Infer<typeof TrackSchema>;
export type InstrumentTrackEntity = Extract<Track, { kind: 'instrument' }>;
export type TrackGroup = Infer<typeof TrackGroupSchema>;
export type OrderSlot = Infer<typeof OrderSlotSchema>;
export type OrderLayer = Infer<typeof OrderLayerSchema>;
export type Ordering = Infer<typeof OrderingSchema>;
export type TempoEvent = Infer<typeof TempoEventSchema>;
/**
 * The live, mutable Jot: the deeply-observable MobX/Loro-backed model whose
 * collections are `ReactiveMap`/`ReactiveList`. Reads/writes are ordinary
 * property access; every write commits to the backing Loro doc. Snapshot it
 * to a plain {@link JotState} via `createMutableJot(...).snapshot()`.
 */
export type MutableJot = Infer<typeof JotSchema>;

/**
 * The immutable Jot: a plain JSON object with the same general schema as
 * {@link MutableJot} but plain JS collections (bars as an array,
 * notes/instruments as records keyed by id/lane) and no mutation surface.
 * What you serialize, diff, or seed a fresh {@link MutableJot} from.
 */
export type JotState = Snapshot<typeof JotSchema>;

/**
 * Create a mutable Jot document backed by Loro, optionally seeded from a
 * plain {@link JotState} (bars as an array, notes/instruments as records
 * keyed by id/lane). The returned `model` is the deeply-observable MobX
 * projection; reads/writes are ordinary property access. Call `.snapshot()`
 * on the result to read the current state back out as a plain {@link JotState}.
 */
export function createMutableJot(initial?: Init<typeof JotSchema>): ReactiveDoc<typeof JotSchema> {
  return createReactiveDoc(JotSchema, initial);
}

/**
 * Rebuild a live mutable Jot document from a plain {@link JotState} snapshot
 * (e.g. one read back out of a saved `.jot` file, or another
 * `createMutableJot(...).snapshot()`). The inverse of `.snapshot()`: round-
 * tripping `state → createMutableJotFromState(state).snapshot()` is lossless.
 *
 * A `Snapshot` is a fully-populated superset of the `Init` seed shape (same
 * plain-object projection, idMaps as records, movableLists as arrays), so the
 * cast through `unknown` is safe; it only exists because the two recursive
 * generic types don't structurally unify without blowing TS's depth limit
 * (the same reason `createReactiveDoc`'s body erases `Init<S>` to a plain
 * object). The seeding path (`populateRecord`) reads it as a plain object.
 */
export function createMutableJotFromState(state: JotState): ReactiveDoc<typeof JotSchema> {
  return createMutableJot(state as unknown as Init<typeof JotSchema>);
}
