import { comparer, computed, makeObservable } from 'mobx';
import { computedFn } from 'mobx-utils';
import { isDyadic } from 'src/schema/dsl/element_metrics';
import type { Element, Jot, PatternDef } from 'src/schema/schema';

/**
 * Beat-addressed score structure derived from the reactive Jot's element
 * tree, the grouping/indexing layer that used to live in `JotStructure`.
 *
 * The reactive model is a hierarchy of `note | group | pattern` elements
 * with coordinates RELATIVE to their container; this store walks it,
 * converting to bar-absolute beats (a group of `duration` D whose children
 * span an internal length L scales them by D/L, that's the tuplet ratio),
 * and flattens it into per-bar, per-lane tracks plus tuplet + pattern
 * spans.
 *
 * **Granularity (load-bearing for editor perf).** The derivation is NOT one
 * monolithic computed: it's a tree of computeds keyed by a stable **bar-cell
 * key** (one `(bar, layer)` cell), each gated by structural equality.
 * Adding/editing a note re-flattens ONLY its own bar cell; every other bar's
 * `contentsFor` keeps its cached value, and a `trackFor(cell, lane)` for an
 * untouched lane does not even notify its observers. So an edit reaches React
 * only for the one bar+lane it changed; everything else stops at the MobX
 * layer. The full-structure `layers` getter (for export / playback / tempo /
 * density / lane-list consumers) is composed from those cached pieces, so it
 * stays correct while paying only for the bar that changed.
 *
 * Pixels, palette colours, tempo and the drum-offset transform are NOT
 * here, they live in their own domain stores and read this structure.
 */

const EPS = 1e-9;
/** Separates barId from layerId in a bar-cell key. Bar and layer ids are
 *  converter-generated slugs (`b0`, `v0`, `primary`, …) that never contain a
 *  colon, so this can't produce an ambiguous key. */
const KEY_SEP = ':';
const EMPTY_IDS: readonly string[] = Object.freeze([]);
/** Pass to a `computedFn` so unchanged content (by value) doesn't notify
 *  observers, even when the upstream bar cell recomputed. */
const STRUCTURAL = { equals: comparer.structural } as const;

export type StructNote = {
  id: string;
  lane: string;
  /** Quarter-note beats from the owning bar's downbeat (absolute). */
  beat: number;
  duration: number;
  modifiers: readonly string[];
  sticking?: string;
  roll: boolean;
  /** Onset lands on the binary (dyadic) grid. `isDyadic(beat)`. */
  straight: boolean;
  /** Playback fields (carried so the events / MIDI / RLRR derivations can
   *  resolve the note's MIDI/dynamics without the structure store knowing
   *  about it). */
  offsetMs?: number;
  velocity?: number;
  midiNote?: number;
  midiTick?: number;
  vol?: string;
};

export type StructTrack = { lane: string; notes: StructNote[] };

export type StructPatternSpan = {
  name: string;
  startBeat: number;
  endBeat: number;
  lanes: ReadonlySet<string>;
  colorIndex: number;
};

/** A group whose children don't fill its own duration, drawn as a bracket
 *  with `count` above it. */
export type StructTupletSpan = {
  count: number;
  startBeat: number;
  endBeat: number;
};

export type StructBar = {
  id: string;
  index: number;
  beats: number;
  tsCount: number;
  tsUnit: number;
  anacrusis: boolean;
  tracks: Record<string, StructTrack>;
  patternSpans: StructPatternSpan[];
  tupletSpans: StructTupletSpan[];
};

export type StructLayer = {
  id: string;
  name?: string;
  bars: StructBar[];
  lanes: string[];
};

/** A bar's geometry (no note content), the stable spine the render path and
 *  every layout/length consumer read. Independent of note edits except the
 *  `beats` of an anacrusis bar, which is sized from its content. */
export type StructBarGeometry = {
  id: string;
  index: number;
  beats: number;
  tsCount: number;
  tsUnit: number;
  anacrusis: boolean;
};

export const PRIMARY_LAYER = 'primary';

/** Stable id for the view-only "virtual" lead-in bar (one with no
 *  corresponding entry in `jot.bars`; it exists only in the rendered view
 *  structure, never in export/playback). Distinct from any real bar id so
 *  consumers can recognise it. */
export const LEAD_IN_BAR_ID = '__leadin__';

/** Project bars to the minimal shape {@link buildBarTempos} needs, flagging
 *  the view-only virtual lead-in bar as `synthetic` so tempo-event anchoring
 *  (indexed against the source bars) skips it. */
export function toTempoBars(
  bars: readonly StructBar[]
): { beats: number; synthetic?: boolean }[] {
  return bars.map((b) => ({ beats: b.beats, synthetic: b.id === LEAD_IN_BAR_ID }));
}

/** The flattened contents of one bar cell: per-lane tracks + span chrome. */
type BarContents = {
  tracks: Record<string, StructTrack>;
  tupletSpans: StructTupletSpan[];
  patternSpans: StructPatternSpan[];
  /** Natural content length (latest onset end), used to size anacrusis bars. */
  contentBeats: number;
};

const EMPTY_CONTENTS: BarContents = {
  tracks: {},
  tupletSpans: [],
  patternSpans: [],
  contentBeats: 0,
};

export class StructureStore {
  constructor(private readonly getJot: () => Jot | undefined) {
    makeObservable(this, {
      layerOrder: computed,
      barOrder: computed,
      membership: computed,
      patternColors: computed.struct,
      layers: computed,
    });
  }

  /** True when the jot declares no explicit `||` layers (a single primary
   *  layer whose notes carry no `layerId`). */
  private get singleLayer(): boolean {
    const jot = this.getJot();
    return !jot || jot.layers.size === 0;
  }

  /** Composite bar-cell key for a `(bar, layer)`. Single-layer jots key by bar
   *  alone, since their elements carry no `layerId`. Public so the presenter
   *  can address the same cells for its lane-scoped render path. */
  keyFor(barId: string, layerId: string): string {
    return this.singleLayer ? barId : `${barId}${KEY_SEP}${layerId}`;
  }

  /** Ordered layers; a jot with no declared layers gets one synthetic primary.
   *  idMap order isn't authoring order, so a numeric id-sort restores `v0,
   *  v1, …` (layer 0 must be the first `||` layer). */
  get layerOrder(): { id: string; name?: string }[] {
    const jot = this.getJot();
    if (!jot) return [];
    const declared = [...jot.layers.values()].sort((a, b) =>
      (a as { id: string }).id.localeCompare((b as { id: string }).id, undefined, { numeric: true })
    );
    return declared.length === 0
      ? [{ id: PRIMARY_LAYER }]
      : declared.map((l) => ({ id: (l as { id: string }).id, name: (l as { name?: string }).name }));
  }

  /** Bar order + 1-based render index (lead-in negative, anacrusis 0). Pure
   *  geometry from `jot.bars` + `leadBars`, independent of note content, so a
   *  note edit never perturbs it. */
  get barOrder(): { id: string; index: number; tsCount: number; tsUnit: number; anacrusis: boolean }[] {
    const jot = this.getJot();
    if (!jot) return [];
    const leadBars = jot.leadBars ?? 0;
    const out: { id: string; index: number; tsCount: number; tsUnit: number; anacrusis: boolean }[] = [];
    let gridPos = 0;
    for (const bar of jot.bars) {
      const anacrusis = bar.anacrusis === true;
      const index = anacrusis ? 0 : gridPos < leadBars ? gridPos - leadBars : gridPos - leadBars + 1;
      if (!anacrusis) gridPos++;
      out.push({ id: bar.id, index, tsCount: bar.tsCount, tsUnit: bar.tsUnit, anacrusis });
    }
    return out;
  }

  /** Top-level element ids grouped by `(bar, layer)` bar cell. Reads only the
   *  routing fields (`id` / `barId` / `layerId`), NOT note content, so it
   *  re-runs only on add / remove / re-home, never on a lane/beat/modifier
   *  edit. The single O(elements) pass per structural edit. */
  get membership(): Map<string, string[]> {
    const jot = this.getJot();
    const map = new Map<string, string[]>();
    if (!jot) return map;
    const single = this.singleLayer;
    for (const value of jot.elements.values()) {
      const el = value as Element;
      const barId = el.barId;
      if (barId === undefined) continue;
      const key = single ? barId : `${barId}${KEY_SEP}${el.layerId ?? ''}`;
      let arr = map.get(key);
      if (!arr) {
        arr = [];
        map.set(key, arr);
      }
      arr.push(el.id);
    }
    for (const arr of map.values()) arr.sort();
    return map;
  }

  /** Element ids in one bar cell, structurally gated: an add/remove in a
   *  DIFFERENT cell leaves this array (and `contentsFor`, `trackFor`, … off
   *  it) untouched at the MobX layer even though `membership` rebuilt. */
  barCellIds = computedFn(
    (key: string): readonly string[] => this.membership.get(key) ?? EMPTY_IDS,
    STRUCTURAL
  );

  /** Global pattern-name → colour slot, assigned in first-usage order across
   *  the whole jot (bar then layer order). `computed.struct` so a non-pattern
   *  edit never perturbs it, and thus never invalidates the pattern bars whose
   *  `contentsFor` reads it via the lazy `colorOf` callback. */
  get patternColors(): Record<string, number> {
    const jot = this.getJot();
    const out: Record<string, number> = {};
    if (!jot) return out;
    let next = 0;
    const assign = (name: string) => {
      if (!(name in out)) out[name] = next++;
    };
    for (const layer of this.layerOrder) {
      for (const bar of this.barOrder) {
        for (const id of this.barCellIds(this.keyFor(bar.id, layer.id))) {
          const el = jot.elements.get(id) as Element | undefined;
          if (el) collectPatternNames(el, jot, assign);
        }
      }
    }
    return out;
  }

  /** Flattened contents of one `(bar, layer)` cell. Re-runs only when that
   *  cell's membership or a member element's content changes (or, for a bar
   *  that contains a pattern, when the global pattern-colour map shifts).
   *  Structurally gated so a no-op rewrite doesn't propagate. */
  contentsFor = computedFn((key: string): BarContents => {
    const jot = this.getJot();
    const ids = this.barCellIds(key);
    if (!jot || ids.length === 0) return EMPTY_CONTENTS;
    // Read the global colour map lazily: only a pattern element triggers the
    // dependency, so plain-note bars never couple to it.
    const colorOf = (name: string) => this.patternColors[name] ?? 0;
    const notes: StructNote[] = [];
    const tuplets: StructTupletSpan[] = [];
    const patternSpans: StructPatternSpan[] = [];
    let contentBeats = 0;
    for (const id of ids) {
      const el = jot.elements.get(id) as Element | undefined;
      if (!el) continue;
      // Top-level coordinates are already bar-relative.
      flattenInto(el, el.beat, el.duration, jot, colorOf, { notes, tuplets, patternSpans });
      contentBeats = Math.max(contentBeats, el.beat + el.duration);
    }
    const tracks: Record<string, StructTrack> = {};
    for (const note of notes) {
      let track = tracks[note.lane];
      if (!track) {
        track = { lane: note.lane, notes: [] };
        tracks[note.lane] = track;
      }
      track.notes.push(note);
    }
    for (const lane of Object.keys(tracks)) {
      tracks[lane].notes.sort((a, b) => a.beat - b.beat);
    }
    return { tracks, tupletSpans: tuplets, patternSpans, contentBeats };
  }, STRUCTURAL);

  /** One lane's track within a bar cell, structurally gated. Adding a note to a
   *  sibling lane re-runs this (its cell changed) but, finding the same notes,
   *  does NOT notify, so that lane's row never re-renders. */
  trackFor = computedFn(
    (key: string, lane: string): StructTrack => this.contentsFor(key).tracks[lane] ?? { lane, notes: [] },
    STRUCTURAL
  );

  /** A lane's notes in one bar, UNIONED across every `||` layer that places a
   *  note on that lane. The mixer renders one row per lane (not per
   *  layer+lane), so a lane that lives in a non-first layer (e.g. the kick in a
   *  hands/feet split) must still surface in its row. Single-layer jots collapse
   *  to a single `trackFor`, so the common case is unchanged and stays granular.
   *  Structurally gated by beat; a sibling-lane or sibling-layer edit that
   *  leaves these notes alone doesn't notify. */
  mergedTrackFor = computedFn((barId: string, lane: string): StructTrack => {
    const layers = this.layerOrder;
    if (layers.length <= 1) {
      return this.trackFor(this.keyFor(barId, layers[0]?.id ?? ''), lane);
    }
    const notes: StructNote[] = [];
    for (const layer of layers) {
      notes.push(...this.trackFor(this.keyFor(barId, layer.id), lane).notes);
    }
    notes.sort((a, b) => a.beat - b.beat);
    return { lane, notes };
  }, STRUCTURAL);

  /** The id of the layer that owns `lane`, the first `||` layer (in
   *  {@link layerOrder}) that carries a note on it, for placing inserted /
   *  moved notes so they land in the row the user clicked. `undefined` for a
   *  single-layer jot (its notes carry no `layerId`) or a brand-new lane no
   *  layer has yet; callers fall back accordingly. */
  ownerLayerFor = computedFn((lane: string): string | undefined => {
    if (this.singleLayer) return undefined;
    for (const layer of this.layerOrder) {
      if (this.lanesForLayer(layer.id).includes(lane)) return layer.id;
    }
    return undefined;
  });

  /** A bar cell's lane-spanning chrome (pattern + tuplet brackets),
   *  structurally gated (a plain-note edit leaves it untouched). */
  spansFor = computedFn((key: string): { patternSpans: StructPatternSpan[]; tupletSpans: StructTupletSpan[] } => {
    const c = this.contentsFor(key);
    return { patternSpans: c.patternSpans, tupletSpans: c.tupletSpans };
  }, STRUCTURAL);

  /** A bar's bracket chrome UNIONED across every `||` layer, so a tuplet /
   *  pattern authored in a non-first layer still draws its bracket (mirrors
   *  {@link mergedTrackFor} for notes). Single-layer jots collapse to one
   *  `spansFor`, leaving the common case untouched. */
  mergedSpansFor = computedFn((barId: string): { patternSpans: StructPatternSpan[]; tupletSpans: StructTupletSpan[] } => {
    const layers = this.layerOrder;
    if (layers.length <= 1) {
      return this.spansFor(this.keyFor(barId, layers[0]?.id ?? ''));
    }
    const patternSpans: StructPatternSpan[] = [];
    const tupletSpans: StructTupletSpan[] = [];
    for (const layer of layers) {
      const s = this.spansFor(this.keyFor(barId, layer.id));
      patternSpans.push(...s.patternSpans);
      tupletSpans.push(...s.tupletSpans);
    }
    return { patternSpans, tupletSpans };
  }, STRUCTURAL);

  /** Bar geometry for one layer: index + time-signature beats (an anacrusis
   *  bar's beats is content-sized, the only note dependency here).
   *  Structurally gated, so it's stable across note edits to non-anacrusis
   *  bars, the spine the render path + layout consumers can read cheaply. */
  geometryFor = computedFn((layerId: string): StructBarGeometry[] => {
    return this.barOrder.map((b) => ({
      id: b.id,
      index: b.index,
      beats: b.anacrusis
        ? this.contentsFor(this.keyFor(b.id, layerId)).contentBeats
        : (b.tsCount * 4) / b.tsUnit,
      tsCount: b.tsCount,
      tsUnit: b.tsUnit,
      anacrusis: b.anacrusis,
    }));
  }, STRUCTURAL);

  /** Ordered lane list for a layer (lanes that carry at least one note),
   *  structurally gated: adding a note to an existing lane doesn't change it,
   *  so the mixer's row list never churns on an in-lane edit. */
  lanesForLayer = computedFn((layerId: string): string[] => {
    const jot = this.getJot();
    if (!jot) return [];
    const seen: string[] = [];
    for (const b of this.barOrder) {
      const tracks = this.contentsFor(this.keyFor(b.id, layerId)).tracks;
      for (const lane in tracks) if (!seen.includes(lane)) seen.push(lane);
    }
    return orderLanes(seen, [...jot.instruments.keys()]);
  }, STRUCTURAL);

  /** The full musical structure, composed from the cached per-cell pieces.
   *  Recomputes on any edit (it reads every cell) but pays only for the bar
   *  that changed; the rest return cached `contentsFor`. For export / playback
   *  / tempo / density / lane-list consumers; the render path reads the
   *  granular getters above instead. */
  get layers(): StructLayer[] {
    const jot = this.getJot();
    if (!jot) return [];
    return this.layerOrder.map((layer) => {
      const bars: StructBar[] = this.geometryFor(layer.id).map((geo) => {
        const c = this.contentsFor(this.keyFor(geo.id, layer.id));
        return {
          id: geo.id,
          index: geo.index,
          beats: geo.beats,
          tsCount: geo.tsCount,
          tsUnit: geo.tsUnit,
          anacrusis: geo.anacrusis,
          tracks: c.tracks,
          patternSpans: c.patternSpans,
          tupletSpans: c.tupletSpans,
        };
      });
      return { id: layer.id, name: layer.name, bars, lanes: this.lanesForLayer(layer.id) };
    });
  }
}

/**
 * Recursively flatten an element positioned at bar-absolute `[absBeat,
 * absBeat+absDur)` into notes (with absolute beats), tuplet spans (groups
 * that don't fill their duration) and pattern spans (pattern usages).
 */
function flattenInto(
  el: Element,
  absBeat: number,
  absDur: number,
  jot: Jot,
  colorOf: (name: string) => number,
  out: { notes: StructNote[]; tuplets: StructTupletSpan[]; patternSpans: StructPatternSpan[] }
): void {
  if (el.kind === 'note') {
    out.notes.push({
      id: el.id,
      lane: el.lane,
      beat: absBeat,
      duration: absDur,
      modifiers: el.modifiers,
      sticking: el.sticking,
      roll: el.roll === true,
      straight: isDyadic(absBeat),
      offsetMs: el.offsetMs,
      velocity: el.velocity,
      midiNote: el.midiNote,
      midiTick: el.midiTick,
      vol: el.vol,
    });
    return;
  }

  if (el.kind === 'group') {
    const children = [...el.children.values()] as Element[];
    const internalLen = naturalSpan(children);
    if (internalLen > EPS && Math.abs(internalLen - el.duration) > EPS) {
      out.tuplets.push({ count: children.length, startBeat: absBeat, endBeat: absBeat + absDur });
    }
    const scale = internalLen > EPS ? absDur / internalLen : 1;
    for (const child of children) {
      flattenInto(child, absBeat + child.beat * scale, child.duration * scale, jot, colorOf, out);
    }
    return;
  }

  // pattern: instantiate the referenced definition, scaled to this usage.
  const def = jot.patterns.get(el.patternId) as PatternDef | undefined;
  if (!def) return;
  const lanes = new Set<string>();
  const body = [...def.body.values()] as Element[];
  collectLanes(body, jot, lanes);
  out.patternSpans.push({
    name: def.name,
    startBeat: absBeat,
    endBeat: absBeat + absDur,
    lanes,
    colorIndex: colorOf(def.name),
  });
  const internalLen = naturalSpan(body);
  const scale = internalLen > EPS ? absDur / internalLen : 1;
  for (const child of body) {
    flattenInto(child, absBeat + child.beat * scale, child.duration * scale, jot, colorOf, out);
  }
}

/** Natural internal length of a container's children (latest onset end). */
function naturalSpan(children: readonly Element[]): number {
  let end = 0;
  for (const child of children) end = Math.max(end, child.beat + child.duration);
  return end;
}

/** Lanes a subtree plays, for a pattern span's lane set. */
function collectLanes(elements: readonly Element[], jot: Jot, into: Set<string>): void {
  for (const el of elements) {
    if (el.kind === 'note') into.add(el.lane);
    else if (el.kind === 'group') collectLanes([...el.children.values()] as Element[], jot, into);
    else {
      const def = jot.patterns.get(el.patternId) as PatternDef | undefined;
      if (def) collectLanes([...def.body.values()] as Element[], jot, into);
    }
  }
}

/** Pattern names used by a subtree, in encounter order (for colour slots). */
function collectPatternNames(el: Element, jot: Jot, assign: (name: string) => void): void {
  if (el.kind === 'note') return;
  if (el.kind === 'group') {
    for (const child of el.children.values()) collectPatternNames(child as Element, jot, assign);
    return;
  }
  const def = jot.patterns.get(el.patternId) as PatternDef | undefined;
  if (!def) return;
  assign(def.name);
  for (const child of def.body.values()) collectPatternNames(child as Element, jot, assign);
}

/** Order the lanes seen in a layer by the instrument-mapping order, then any
 *  stragglers (lanes with no mapping entry) in first-seen order. */
function orderLanes(seen: readonly string[], mappedOrder: readonly string[]): string[] {
  const out: string[] = [];
  for (const p of mappedOrder) {
    if (seen.includes(p) && !out.includes(p)) out.push(p);
  }
  for (const p of seen) {
    if (!out.includes(p)) out.push(p);
  }
  return out;
}
