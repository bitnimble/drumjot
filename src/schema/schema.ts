/**
 * The Drumjot collaborative document schema.
 *
 * This is the canonical, editable, conflict-free representation a song; * deliberately FLAT and value-addressed, not the nested DSL weight-tree
 * (which stays an export-only serialization target). A note's musical
 * position is plain LWW register fields (`barId` + `beat`) and its lane is
 * a `pitch` field, so every core edit, add, move along a track, move
 * track A→B, delete, is a register write or a keyed add/remove, and
 * rendering order is a deterministic sort rather than any CRDT sequence.
 *
 * The enums mirror `src/dsl/dsl.ts` exactly; the type-level tests in
 * `./test/schema.test.ts` assert `Infer<…>` equals the hand-written DSL
 * types so this can't silently drift from the domain.
 */
import { z } from 'zod';
import { idMap, movableList, record, type Infer, type Init } from './descriptors';
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
export const INSTRUMENT_KIND = z.enum(['kick', 'snare', 'hihat', 'ride', 'crash', 'tom', 'custom']);

// ---------- Entities ----------

/**
 * A single drum hit. `barId` + `beat` place it (quarter-note beats from
 * the owning bar's downbeat); `pitch` is the DSL lane letter. All are LWW
 * registers, so a drag-to-move is one `beat` write and a crash→ride
 * correction is one `pitch` write. `id` is the stable identity selection,
 * undo, React keys and provenance all key off (it equals the `notes`
 * idMap key, carried on the record for convenience + export).
 */
export const NoteSchema = record({
  id: z.string(),
  barId: z.string(),
  beat: z.number(),
  pitch: z.string(),
  /** Onset duration in quarter-note beats. */
  duration: z.number(),
  /** Whole-array LWW register (rarely concurrent; merge the set atomically). */
  modifiers: z.array(MODIFIER),
  sticking: STICKING.optional(),
  /** Roll / buzz fill (DSL `~`). Absent = false. */
  roll: z.boolean().optional(),
  /** Signed sub-slot timing offset in milliseconds (DSL `Note.offset`). */
  offsetMs: z.number().optional(),
  /** Explicit MIDI velocity override (0–127); absent = derive from dynamics. */
  velocity: z.number().optional(),
  /** Explicit MIDI note override; absent = derive from instrument/pitch. */
  midiNote: z.number().optional(),
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
});

/**
 * Display + playback info for a pitch lane, keyed in the `instruments`
 * idMap by the DSL pitch letter. Mirrors `dsl.ts` `Instrument`.
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
 * three collections are the editable entities. Deferred (added when their
 * editing flows land): patterns, multiple `||` voices, and tempo
 * transitions (`BpmTransition`).
 */
export const JotSchema = record({
  title: z.string(),
  /** Initial/global tempo in force before any per-bar override. */
  bpm: z.number(),
  /** Audio time (seconds) of the first drum onset; the lead-in offset. */
  drumsT0Sec: z.number().optional(),
  /** Number of pre-drum lead-in bars before bar 1. */
  leadBars: z.number().optional(),
  /** Producer grid density (1/N-of-a-whole-note); advisory. */
  gridDivision: z.number().optional(),
  bars: movableList(BarSchema),
  notes: idMap(NoteSchema),
  /** Pitch letter → instrument display/playback info. */
  instruments: idMap(InstrumentSchema),
});

export type Note = Infer<typeof NoteSchema>;
export type Bar = Infer<typeof BarSchema>;
export type Instrument = Infer<typeof InstrumentSchema>;
export type Jot = Infer<typeof JotSchema>;

/**
 * Create a reactive Jot document backed by Loro, optionally seeded from a
 * plain Jot object (bars as an array, notes/instruments as records keyed
 * by id/pitch). The returned `model` is the deeply-observable MobX
 * projection; reads/writes are ordinary property access.
 */
export function createReactiveJot(initial?: Init<typeof JotSchema>): ReactiveDoc<typeof JotSchema> {
  return createReactiveDoc(JotSchema, initial);
}
