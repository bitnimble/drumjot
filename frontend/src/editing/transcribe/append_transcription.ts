/**
 * Insert a transcription's drum score into the *current* loaded jot, in
 * place, as a single undoable edit.
 *
 * Unlike a wholesale debug-bundle load (which replaces the document and drops
 * the session's audio tracks / lyrics / mixer), this mutates the live reactive
 * document so its identity, and therefore the loaded audio tracks, lyrics, and
 * undo history, survive. Two shapes:
 *
 *  - **replace**: the current jot has no notes (a blank jot, or one the user
 *    cleared). The transcription becomes the whole score, its bars, tempo, and
 *    drum notes seeded straight in.
 *  - **append**: the current jot already has notes. The transcription's notes
 *    land as a brand-new `||` layer at the bottom; the global tempo (bpm +
 *    tempo events) and bar grid are replaced by the transcription's; existing
 *    notes are kept on the same bar indices (so they stay aligned when the new
 *    grid matches, and "break" the user can fix/delete when it doesn't). The
 *    grid is extended to the longer of the two so no existing note is dropped.
 *
 * The whole edit runs inside one {@link transact} so it commits once = one
 * undo step.
 */
import type { LoroDoc } from 'loro-crdt';
import type { Jot } from 'src/schema/dsl/dsl';
import { dslToInit } from 'src/schema/dsl/from_dsl';
import type { MutableJot } from 'src/schema/schema';
import { transact } from 'src/schema/reactive_doc';

// Plain projection of the transcription's seed state (what `dslToInit`
// produces: idMaps as records, lists as arrays). Hand-written for ergonomic
// reads; structurally identical to the schema's `Init` / `JotState`.
type PlainBar = { id: string; tsCount: number; tsUnit: number; tempoBpm?: number; anacrusis?: boolean };
type PlainTrack =
  | { id: string; kind: 'instrument'; lane: string }
  | { id: string; kind: 'audio'; audioId: string }
  | { id: string; kind: 'lyrics'; lyricsId: string };
type PlainNote = {
  kind: 'note';
  id: string;
  barId?: string;
  beat: number;
  duration: number;
  trackId?: string;
  lane: string;
  modifiers: string[];
  sticking?: string;
  roll?: boolean;
  offsetMs?: number;
  velocity?: number;
  midiNote?: number;
  midiTick?: number;
};
type PlainGroup = {
  kind: 'group';
  id: string;
  barId?: string;
  layerId?: string;
  beat: number;
  duration: number;
  children: Record<string, PlainElement>;
  modifiers?: string[];
  roll?: boolean;
};
type PlainPattern = {
  kind: 'pattern';
  id: string;
  barId?: string;
  layerId?: string;
  beat: number;
  duration: number;
  patternId: string;
};
type PlainElement = PlainNote | PlainGroup | PlainPattern;
type PlainOrderLayer = {
  layerId: string;
  slots: { groupId: string | null; tracks: { trackId: string }[] }[];
};
type PlainInstrument = { kind: string; name?: string; limb?: string; midiNote?: number };
type PlainTempoEvent = { id: string; barId: string; beat: number; bpm: number | object };
type PlainPatternDef = { id: string; name: string; body: Record<string, PlainElement> };
type PlainJot = {
  title: string;
  songLeadIn?: number;
  leadBars?: number;
  gridDivision?: number;
  barDriftJson?: string;
  layers: Record<string, { id: string; name?: string; color?: string }>;
  tracks: Record<string, PlainTrack>;
  trackGroups: Record<string, { id: string; name: string; color?: string }>;
  ordering: PlainOrderLayer[];
  bars: PlainBar[];
  elements: Record<string, PlainElement>;
  instruments: Record<string, PlainInstrument>;
  tempoEvents: Record<string, PlainTempoEvent>;
  patterns: Record<string, PlainPatternDef>;
};

/** What the merge did, so the caller can decide whether to warn the user that
 *  pre-existing content was changed. */
export type AppendTranscriptionResult = {
  /** `replace` when the jot was empty (nothing to preserve), else `append`. */
  mode: 'append' | 'replace';
  /** Whether the jot had notes before the merge (always true for `append`). */
  hadNotes: boolean;
  /** Number of pre-existing tempo events that were dropped + replaced. */
  replacedTempoCount: number;
};

/** Mint unique ids that can't collide with the live document's existing ids
 *  (or a previous merge's). Deterministic when a `prefix` is supplied (tests);
 *  random otherwise. */
function makeMinter(prefix?: string): (kind: string) => string {
  const salt = prefix ?? `mx${Math.random().toString(36).slice(2, 8)}`;
  let n = 0;
  return (kind: string) => `${salt}_${kind}${n++}`;
}

function omitUndef<T extends Record<string, unknown>>(o: T): T {
  const out = {} as Record<string, unknown>;
  for (const k of Object.keys(o)) if (o[k] !== undefined) out[k] = o[k];
  return out as T;
}

/** Loose write surface over the live model's collections. The schema's
 *  `Infer` types are exact, but the merge writes plain seed objects (the same
 *  shape `dslToInit` emits), so a structural view keeps the body readable. */
type IdMapW = {
  size: number;
  has(id: string): boolean;
  keys(): IterableIterator<string>;
  set(id: string, value: Record<string, unknown>): void;
  setAll(entries: Iterable<[string, Record<string, unknown>]>): void;
  delete(...ids: string[]): void;
};
type ListW = {
  length: number;
  at(i: number): Record<string, unknown> | undefined;
  push(value: Record<string, unknown>): void;
  delete(index: number): void;
};
type ModelW = {
  title: string;
  songLeadIn?: number;
  leadBars?: number;
  gridDivision?: number;
  barDriftJson?: string;
  layers: IdMapW;
  tracks: IdMapW;
  trackGroups: IdMapW;
  ordering: ListW;
  bars: ListW;
  elements: IdMapW;
  instruments: IdMapW;
  tempoEvents: IdMapW;
  patterns: IdMapW;
};

/**
 * Insert `transcribed` (a freshly converted transcription jot) into the live
 * `model`. `doc` is the model's backing Loro doc, used to batch the whole edit
 * into one commit. `layerName` labels the inserted layer (e.g. the source
 * audio track's name) so A/B passes stay distinguishable.
 */
export function appendTranscription(
  doc: LoroDoc,
  model: MutableJot,
  transcribed: Jot,
  opts: { layerName?: string; idPrefix?: string } = {}
): AppendTranscriptionResult {
  const tx = dslToInit(transcribed) as unknown as PlainJot;
  const m = model as unknown as ModelW;
  const mint = makeMinter(opts.idPrefix);

  const hadNotes = m.elements.size > 0;
  const replacedTempoCount = m.tempoEvents.size;
  const mode: 'append' | 'replace' = hadNotes ? 'append' : 'replace';

  transact(doc, () => {
    if (mode === 'replace') {
      replaceContent(m, tx, opts.layerName);
    } else {
      appendLayer(m, tx, mint, opts.layerName);
    }
  });

  return { mode, hadNotes, replacedTempoCount };
}

/** Empty-jot path: clear every collection and seed the transcription verbatim
 *  (its own ids are safe, the model is now empty). The jot's title is kept. */
function replaceContent(m: ModelW, tx: PlainJot, layerName: string | undefined): void {
  for (const map of [
    m.elements,
    m.tempoEvents,
    m.ordering,
    m.bars,
    m.layers,
    m.tracks,
    m.trackGroups,
    m.instruments,
    m.patterns,
  ] as Array<IdMapW | ListW>) {
    clearCollection(map);
  }

  for (const bar of tx.bars) m.bars.push(omitUndef({ ...bar }));
  // Name the (single) transcription layer if a label was given.
  const layers = { ...tx.layers };
  if (layerName !== undefined) {
    const only = Object.keys(layers);
    if (only.length === 1) layers[only[0]] = { ...layers[only[0]], name: layerName };
  }
  m.layers.setAll(entriesOf(layers));
  m.tracks.setAll(entriesOf(tx.tracks));
  m.trackGroups.setAll(entriesOf(tx.trackGroups));
  m.instruments.setAll(entriesOf(tx.instruments));
  m.patterns.setAll(entriesOf(tx.patterns));
  m.elements.setAll(entriesOf(tx.elements));
  m.tempoEvents.setAll(entriesOf(tx.tempoEvents));
  for (const ol of tx.ordering) m.ordering.push(ol as unknown as Record<string, unknown>);

  // The initial tempo rides in `tx.tempoEvents` (the event at the first source
  // bar) now, copied above; no separate `bpm` register.
  setOptionalNumber(m, 'songLeadIn', tx.songLeadIn);
  setOptionalNumber(m, 'leadBars', tx.leadBars);
  setOptionalNumber(m, 'gridDivision', tx.gridDivision);
  // Replacing wholesale: carry the transcription's per-bar drift (its bar grid
  // becomes the document's). The overlay path doesn't (the merged grid would
  // misalign the drift index), so it leaves any existing drift untouched.
  m.barDriftJson = tx.barDriftJson;
}

/** Non-empty path: keep existing notes, overlay the transcription as a new
 *  bottom layer, replace the global tempo + bar grid (extended to the longer
 *  of the two so no existing note loses its bar). */
function appendLayer(
  m: ModelW,
  tx: PlainJot,
  mint: (kind: string) => string,
  layerName: string | undefined
): void {
  // --- bars: reuse existing ids for the overlapping prefix, append the rest.
  const txBarIndex = new Map<string, number>();
  tx.bars.forEach((b, i) => txBarIndex.set(b.id, i));
  const barIdByTxId = new Map<string, string>();
  const existing = m.bars.length;
  for (let i = 0; i < tx.bars.length; i++) {
    const src = tx.bars[i];
    if (i < existing) {
      const bar = m.bars.at(i) as { id: string; tsCount: number; tsUnit: number; tempoBpm?: number; anacrusis?: boolean };
      bar.tsCount = src.tsCount;
      bar.tsUnit = src.tsUnit;
      bar.tempoBpm = src.tempoBpm;
      bar.anacrusis = src.anacrusis;
      barIdByTxId.set(src.id, bar.id);
    } else {
      const id = mint('b');
      m.bars.push(omitUndef({ id, tsCount: src.tsCount, tsUnit: src.tsUnit, tempoBpm: src.tempoBpm, anacrusis: src.anacrusis }));
      barIdByTxId.set(src.id, id);
    }
  }

  // --- tempo: drop all existing, add the transcription's (re-anchored).
  m.tempoEvents.delete(...[...m.tempoEvents.keys()]);
  const tempoEntries: Array<[string, Record<string, unknown>]> = [];
  for (const ev of Object.values(tx.tempoEvents)) {
    const barId = barIdByTxId.get(ev.barId);
    if (barId === undefined) continue;
    const id = mint('t');
    tempoEntries.push([id, { id, barId, beat: ev.beat, bpm: ev.bpm }]);
  }
  if (tempoEntries.length > 0) m.tempoEvents.setAll(tempoEntries);

  // --- global registers from the transcription.
  // The initial tempo rides in `tx.tempoEvents` (the event at the first source
  // bar) now, copied above; no separate `bpm` register.
  setOptionalNumber(m, 'songLeadIn', tx.songLeadIn);
  setOptionalNumber(m, 'leadBars', tx.leadBars);
  setOptionalNumber(m, 'gridDivision', tx.gridDivision);

  // --- instruments: union (keep existing lane mappings, add new lanes).
  for (const [lane, inst] of Object.entries(tx.instruments)) {
    if (!m.instruments.has(lane)) m.instruments.set(lane, omitUndef({ ...inst }));
  }

  // --- patterns: copy with fresh ids (two-pass so body pattern-refs remap).
  const patternIdByTxId = new Map<string, string>();
  for (const id of Object.keys(tx.patterns)) patternIdByTxId.set(id, mint('p'));

  // --- layers + tracks for the transcription.
  const layerIdByTxId = new Map<string, string>();
  for (const [txId, layer] of Object.entries(tx.layers)) {
    const id = mint('l');
    layerIdByTxId.set(txId, id);
    const name = layerName ?? layer.name;
    m.layers.set(id, omitUndef({ id, name, color: layer.color }));
  }
  const trackIdByTxId = new Map<string, string>();
  for (const [txId, track] of Object.entries(tx.tracks)) {
    const id = mint('tr');
    trackIdByTxId.set(txId, id);
    // Transcriptions only emit instrument tracks; copy any other kinds verbatim.
    if (track.kind === 'instrument') {
      m.tracks.set(id, { id, kind: 'instrument', lane: track.lane });
    } else {
      m.tracks.set(id, omitUndef({ ...track, id }));
    }
  }

  const remap = {
    barId: (txBarId: string | undefined) => (txBarId !== undefined ? barIdByTxId.get(txBarId) : undefined),
    trackId: (txTrackId: string | undefined) => (txTrackId !== undefined ? trackIdByTxId.get(txTrackId) : undefined),
    layerId: (txLayerId: string | undefined) => (txLayerId !== undefined ? layerIdByTxId.get(txLayerId) : undefined),
    patternId: (txPatternId: string) => patternIdByTxId.get(txPatternId) ?? txPatternId,
    mint,
  };

  // Pattern bodies (after the id maps + remap exist).
  for (const [txId, def] of Object.entries(tx.patterns)) {
    const id = patternIdByTxId.get(txId)!;
    const body: Record<string, Record<string, unknown>> = {};
    for (const el of Object.values(def.body)) {
      const [newId, value] = remapElement(el, remap);
      body[newId] = value;
    }
    m.patterns.set(id, { id, name: def.name, body });
  }

  // --- elements: remap every top-level element of the transcription.
  const elemEntries: Array<[string, Record<string, unknown>]> = [];
  for (const el of Object.values(tx.elements)) {
    elemEntries.push(remapElement(el, remap));
  }
  if (elemEntries.length > 0) m.elements.setAll(elemEntries);

  // --- ordering: append one bottom OrderLayer per transcription layer.
  for (const ol of tx.ordering) {
    const layerId = layerIdByTxId.get(ol.layerId);
    if (layerId === undefined) continue;
    const slots = ol.slots.map((slot) => ({
      groupId: slot.groupId, // transcriptions have no track groups
      tracks: slot.tracks
        .map((t) => ({ trackId: trackIdByTxId.get(t.trackId) }))
        .filter((t): t is { trackId: string } => t.trackId !== undefined),
    }));
    m.ordering.push({ layerId, slots });
  }
}

type Remap = {
  barId: (id: string | undefined) => string | undefined;
  trackId: (id: string | undefined) => string | undefined;
  layerId: (id: string | undefined) => string | undefined;
  patternId: (id: string) => string;
  mint: (kind: string) => string;
};

/** Deep-copy one transcription element into the live document's id space,
 *  re-anchoring bar / track / layer / pattern references. */
function remapElement(el: PlainElement, remap: Remap): [string, Record<string, unknown>] {
  const id = remap.mint('e');
  if (el.kind === 'note') {
    return [
      id,
      omitUndef({
        kind: 'note',
        id,
        barId: remap.barId(el.barId),
        beat: el.beat,
        duration: el.duration,
        trackId: remap.trackId(el.trackId),
        lane: el.lane,
        modifiers: [...el.modifiers],
        sticking: el.sticking,
        roll: el.roll,
        offsetMs: el.offsetMs,
        velocity: el.velocity,
        midiNote: el.midiNote,
        midiTick: el.midiTick,
      }),
    ];
  }
  if (el.kind === 'group') {
    const children: Record<string, Record<string, unknown>> = {};
    for (const child of Object.values(el.children)) {
      const [childId, value] = remapElement(child, remap);
      children[childId] = value;
    }
    return [
      id,
      omitUndef({
        kind: 'group',
        id,
        barId: remap.barId(el.barId),
        layerId: remap.layerId(el.layerId),
        beat: el.beat,
        duration: el.duration,
        children,
        modifiers: el.modifiers ? [...el.modifiers] : undefined,
        roll: el.roll,
      }),
    ];
  }
  return [
    id,
    omitUndef({
      kind: 'pattern',
      id,
      barId: remap.barId(el.barId),
      layerId: remap.layerId(el.layerId),
      beat: el.beat,
      duration: el.duration,
      patternId: remap.patternId(el.patternId),
    }),
  ];
}

/** Empty a collection. Counts/keys are captured ONCE up front: inside a
 *  `transact` the observable surfaces don't reflect the deferred deletes
 *  (the cache only updates on commit), so re-reading `.length` mid-loop would
 *  see a stale value and over-delete. The underlying Loro container does apply
 *  each delete immediately, so deleting index 0 `n` times drains the list. */
function clearCollection(c: IdMapW | ListW): void {
  if ('keys' in c) {
    const ids = [...c.keys()];
    if (ids.length > 0) c.delete(...ids);
  } else {
    const n = c.length;
    for (let i = 0; i < n; i++) c.delete(0);
  }
}

function entriesOf(rec: Record<string, unknown>): Array<[string, Record<string, unknown>]> {
  return Object.entries(rec) as Array<[string, Record<string, unknown>]>;
}

/** Set an optional numeric register, clearing it when the source is absent. */
function setOptionalNumber(m: ModelW, key: 'songLeadIn' | 'leadBars' | 'gridDivision', value: number | undefined): void {
  m[key] = value;
}
