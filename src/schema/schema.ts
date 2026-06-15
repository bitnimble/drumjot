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
  union,
  type UnionDescriptor,
} from './descriptors';
import { createReactiveDoc, type ReactiveDoc } from './reactive_doc';

// ---------- Leaf enums (mirror src/dsl/dsl.ts) ----------

export const MODIFIER = z.enum([
  'a', // accent
  'g', // ghost
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
export const VOLUME = z.enum(['pp', 'p', 'mp', 'mf', 'f', 'ff']);
export const INSTRUMENT_KIND = z.enum(['kick', 'snare', 'hihat', 'ride', 'crash', 'tom', 'custom']);

// ---------- Entities ----------

// ---------- Elements (the note | group | pattern tree) ----------

/**
 * Positional fields every element shares. Coordinates are RELATIVE to the
 * immediate container's space: top-level elements are positioned from
 * their bar's downbeat (and carry `barId`/`layerId`); a group's children
 * are positioned in the group's own internal space (no `barId`; the
 * derivation converts to global time via the group's duration scaling).
 */
const elementBase = {
  id: z.string(),
  /** Owning `||` layer (top-level only; nested elements inherit). */
  layerId: z.string().optional(),
  /** Owning bar (top-level only; nested elements live in their group's space). */
  barId: z.string().optional(),
  /** Position in the immediate container's coordinate space. */
  beat: z.number(),
  /** Span in the immediate container's coordinate space. */
  duration: z.number(),
};

/** A single drum hit. `lane` is the lane; dynamics via `vol` + modifiers. */
export const NoteElementSchema = record({
  ...elementBase,
  kind: z.literal('note'),
  lane: z.string(),
  /** Whole-array LWW register (rarely concurrent; merge the set atomically). */
  modifiers: z.array(MODIFIER),
  sticking: STICKING.optional(),
  /** Roll / buzz fill (DSL `~`). */
  roll: z.boolean().optional(),
  /** Signed sub-slot timing offset in milliseconds (DSL `Note.offset`). */
  offsetMs: z.number().optional(),
  /** Explicit MIDI velocity override; absent = derive from `vol`/modifiers. */
  velocity: z.number().optional(),
  /** Explicit MIDI note override; absent = derive from instrument/lane. */
  midiNote: z.number().optional(),
  /** Raw MIDI tick from a transcribed/imported source (provenance key). */
  midiTick: z.number().optional(),
  /** Symbolic dynamic (pp…ff). */
  vol: VOLUME.optional(),
});

/**
 * A container of child elements with its own `duration`. The children live
 * in the group's internal coordinate space; a **tuplet** is simply a group
 * whose children's natural span ≠ its `duration` (the ratio is the
 * scaling). Moving/stretching a group touches only its own `beat`/
 * `duration`, the children are untouched. Group-level `modifiers`/`roll`/
 * `vol` apply to all descendants. Annotated `RecordDescriptor` so the
 * `lazy` self-reference type-checks.
 */
export const GroupElementSchema = record({
  ...elementBase,
  kind: z.literal('group'),
  children: idMap(lazy((): UnionDescriptor => ElementSchema)),
  modifiers: z.array(MODIFIER).optional(),
  roll: z.boolean().optional(),
  vol: VOLUME.optional(),
});

/** A leaf usage of a pattern definition, referenced by its internal id.
 *  The body lives once in `patterns`; rendering instantiates it scaled to
 *  this element's `duration`. (Substitutions deferred.) */
export const PatternElementSchema = record({
  ...elementBase,
  kind: z.literal('pattern'),
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
 * notes they own (`note.layerId`). `name` is a display hint ("Hands" /
 * "Feet"), not parseable from DSL.
 */
export const LayerSchema = record({
  id: z.string(),
  name: z.string().optional(),
});

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
      /** Transition length, in bars. */
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
  /** `||` layers by id; a single-layer jot has one (or none → primary). */
  layers: idMap(LayerSchema),
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
export type Volume = z.infer<typeof VOLUME>;
export type DrumInstrumentKind = z.infer<typeof INSTRUMENT_KIND>;

type ElementCommon = {
  id: string;
  layerId?: string;
  barId?: string;
  beat: number;
  duration: number;
};
export type NoteElement = ElementCommon & {
  kind: 'note';
  lane: string;
  modifiers: Modifier[];
  sticking?: Sticking;
  roll?: boolean;
  offsetMs?: number;
  velocity?: number;
  midiNote?: number;
  midiTick?: number;
  vol?: Volume;
};
export type GroupElement = ElementCommon & {
  kind: 'group';
  children: ReactiveMap<Element>;
  modifiers?: Modifier[];
  roll?: boolean;
  vol?: Volume;
};
export type PatternElement = ElementCommon & { kind: 'pattern'; patternId: string };
export type Element = NoteElement | GroupElement | PatternElement;
export type PatternDef = { id: string; name: string; body: ReactiveMap<Element> };

export type Bar = Infer<typeof BarSchema>;
export type Instrument = Infer<typeof InstrumentSchema>;
export type Layer = Infer<typeof LayerSchema>;
export type TempoEvent = Infer<typeof TempoEventSchema>;
export type Jot = Infer<typeof JotSchema>;

/**
 * Create a reactive Jot document backed by Loro, optionally seeded from a
 * plain Jot object (bars as an array, notes/instruments as records keyed
 * by id/lane). The returned `model` is the deeply-observable MobX
 * projection; reads/writes are ordinary property access.
 */
export function createReactiveJot(initial?: Init<typeof JotSchema>): ReactiveDoc<typeof JotSchema> {
  return createReactiveDoc(JotSchema, initial);
}
