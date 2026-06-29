/**
 * Reactive `MutableJot` -> DSL `Jot` converter: the inverse of `dslToInit`
 * (`from_dsl.ts`). Used by the DSL exporter (`writeDsl`) so a `.jot` text
 * export reflects the CURRENT reactive document (every edit), not the
 * originally-loaded source. The reactive model is the single runtime source of
 * truth; this rebuilds a DSL save-file view of it on demand.
 *
 * The forward conversion is intentionally lossy in ways this inverse must
 * reconstruct rather than recover verbatim:
 *   - rests are dropped (they only consume sibling space) -> rebuilt from the
 *     gaps between consecutive onsets;
 *   - a simultaneity becomes coincident elements -> regrouped by shared onset;
 *   - `:a` / `:g` accent/ghost markers and `{vol}` become a `velocity` ->
 *     mapped back to the closest authoring form;
 *   - element weights become absolute bar-space `duration`s -> recovered as
 *     ratios (normalised against the smallest sibling span).
 *
 * The result is therefore a CANONICAL DSL rendering of the content (it does not
 * reproduce the author's exact text), chosen so that re-parsing it yields a
 * structurally identical reactive document and re-exporting is a fixpoint.
 */
import type {
  Bar as DslBar,
  Element as DslElement,
  Group as DslGroup,
  Instrument as DslInstrument,
  Jot as DslJot,
  Layer as DslLayer,
  Metadata,
  Modifier,
  Note as DslNote,
  Pattern as DslPattern,
  TempoEvent as DslTempoEvent,
  TimeSignature,
  Volume,
} from 'src/schema/dsl/dsl';
import type { Element, GroupElement, MutableJot, NoteElement } from 'src/schema/schema';
import { laneForNote, layerIdOfTrack } from 'src/schema/ordering';
import { ACCENT_VELOCITY, GHOST_VELOCITY, VOLUME_TO_VELOCITY } from 'src/dynamics/dynamics';

const EPS = 1e-6;

/** Inverse of {@link VOLUME_TO_VELOCITY}: recover a `pp`..`ff` marker from an
 *  exact stored velocity (injective map, so the inverse is unambiguous). */
const VELOCITY_TO_VOLUME = new Map<number, Volume>(
  (Object.entries(VOLUME_TO_VELOCITY) as [Volume, number][]).map(([vol, vel]) => [vel, vol])
);

export function mutableToDsl(jot: MutableJot): DslJot {
  const patternNameById = new Map<string, string>();
  for (const p of jot.patterns.values()) patternNameById.set(p.id, p.name);

  // Bars: stable order from the movable list. The (at most one) anacrusis bar
  // is pulled out so it can sit on its layer's pickup line; the rest are the
  // shared real-bar grid every layer iterates.
  const allBars = [...jot.bars];
  const anacrusisBar = allBars.find((b) => b.anacrusis === true);
  const realBars = allBars.filter((b) => b.anacrusis !== true);

  // Group every top-level element by `${layerId} ${barId}` so each
  // (layer, bar) cell can be reconstructed independently.
  const cells = new Map<string, Element[]>();
  const cellKey = (layerId: string, barId: string) => `${layerId} ${barId}`;
  const firstLayerId = orderedLayerIds(jot)[0] ?? 'primary';
  for (const value of jot.elements.values()) {
    const el = value as Element;
    if (el.barId === undefined) continue;
    const layerId = layerOfTopLevel(jot, el) || firstLayerId;
    const key = cellKey(layerId, el.barId);
    let arr = cells.get(key);
    if (!arr) cells.set(key, (arr = []));
    arr.push(el);
  }

  const ctx: Ctx = { jot, patternNameById };

  const layers: DslLayer[] = orderedLayerIds(jot).map((layerId) => {
    const meta = jot.layers.get(layerId) as { name?: string } | undefined;
    const dslBars: DslBar[] = realBars.map((bar) => {
      const beats = barBeats(bar.tsCount, bar.tsUnit);
      const elements = reconstructSequence(cells.get(cellKey(layerId, bar.id)) ?? [], beats, true, ctx);
      return { elements, metadata: { time: { count: bar.tsCount, unit: bar.tsUnit } } };
    });
    const layer: DslLayer = { bars: dslBars };
    if (meta?.name !== undefined) layer.name = meta.name;
    if (anacrusisBar) {
      const beats = barBeats(anacrusisBar.tsCount, anacrusisBar.tsUnit);
      const anac = reconstructSequence(cells.get(cellKey(layerId, anacrusisBar.id)) ?? [], beats, false, ctx);
      if (anac.length > 0) layer.anacrusis = anac;
    }
    return layer;
  });

  const patterns = buildPatterns(jot, ctx);
  const tempoEvents = buildTempoEvents(jot, realBars);
  const globalMetadata = buildGlobalMetadata(jot, realBars);

  const out: DslJot = { title: jot.title, globalMetadata, layers };
  if (Object.keys(patterns).length > 0) out.patterns = patterns;
  if (tempoEvents.length > 0) out.tempoEvents = tempoEvents;
  return out;
}

type Ctx = { jot: MutableJot; patternNameById: Map<string, string> };

// ---------- Sequence reconstruction (rests, simuls, weights) ----------

/**
 * Rebuild a DSL element sequence from a set of sibling reactive elements that
 * tile `[0, totalSpan)`. Elements sharing an onset become a simultaneity; the
 * gaps between onsets become rests (`fillTrailing` adds a trailing rest so a
 * top-level bar stays full, but a group/pattern body, whose natural span is
 * its content, does not). Weights are the per-slot spans normalised against
 * the smallest slot, so the ratios re-parse to the same beats.
 */
function reconstructSequence(
  elements: Element[],
  totalSpan: number,
  fillTrailing: boolean,
  ctx: Ctx
): DslElement[] {
  if (elements.length === 0) {
    return fillTrailing && totalSpan > EPS ? [{ kind: 'rest' }] : [];
  }
  // Bucket by onset beat -> slots, each carrying its coincident elements.
  const byBeat = new Map<number, Element[]>();
  const beats: number[] = [];
  for (const el of elements) {
    const key = roundBeat(el.beat);
    let arr = byBeat.get(key);
    if (!arr) {
      byBeat.set(key, (arr = []));
      beats.push(key);
    }
    arr.push(el);
  }
  beats.sort((a, b) => a - b);

  // Build raw items (slots + interior/leading/trailing rest gaps) with spans.
  type Item = { kind: 'slot'; els: Element[]; span: number } | { kind: 'rest'; span: number };
  const items: Item[] = [];
  let cursor = 0;
  for (const beat of beats) {
    if (beat > cursor + EPS) items.push({ kind: 'rest', span: beat - cursor });
    const els = byBeat.get(beat)!;
    const span = Math.max(...els.map((e) => e.duration));
    items.push({ kind: 'slot', els, span });
    cursor = beat + span;
  }
  if (fillTrailing && totalSpan - cursor > EPS) items.push({ kind: 'rest', span: totalSpan - cursor });

  // Normalise spans to weights against the smallest positive span.
  const positives = items.map((i) => i.span).filter((s) => s > EPS);
  const minSpan = positives.length > 0 ? Math.min(...positives) : 0;
  const weightOf = (span: number): number | undefined => {
    if (!(minSpan > EPS)) return undefined;
    const w = cleanNumber(span / minSpan);
    return w === 1 ? undefined : w;
  };

  return items.map((item) => {
    if (item.kind === 'rest') {
      const w = weightOf(item.span);
      return w === undefined ? { kind: 'rest' } : { kind: 'rest', weight: w };
    }
    const weight = weightOf(item.span);
    if (item.els.length === 1) {
      const el = convertElement(item.els[0], ctx);
      if (weight !== undefined) (el as { weight?: number }).weight = weight;
      return el;
    }
    // Coincident, co-extensive elements -> a simultaneity. Members carry no
    // weight of their own (they all share the simul's onset + span).
    const simul: DslElement = { kind: 'simul', elements: item.els.map((e) => convertElement(e, ctx)) };
    if (weight !== undefined) simul.weight = weight;
    return simul;
  });
}

// ---------- Element conversion ----------

function convertElement(el: Element, ctx: Ctx): DslElement {
  if (el.kind === 'note') return noteToDsl(el, ctx);
  if (el.kind === 'group') return groupToDsl(el, ctx);
  // pattern usage
  const name = ctx.patternNameById.get(el.patternId) ?? el.patternId;
  return { kind: 'patternRef', name };
}

function noteToDsl(note: NoteElement, ctx: Ctx): DslNote {
  const out: DslNote = { kind: 'note', lane: laneForNote(ctx.jot, note) };
  const modifiers: Modifier[] = [...(note.modifiers as Modifier[])];
  const metadata: Metadata = {};
  applyVelocity(note.velocity, modifiers, metadata);
  if (note.midiNote !== undefined) setMidi(metadata, 'note', note.midiNote);
  if (note.midiTick !== undefined) setMidi(metadata, 'tick', note.midiTick);
  if (modifiers.length > 0) out.modifiers = modifiers;
  if (note.sticking) out.sticking = note.sticking;
  if (note.roll) out.roll = true;
  if (note.offsetMs !== undefined) out.offset = note.offsetMs;
  if (Object.keys(metadata).length > 0) out.metadata = metadata;
  return out;
}

function groupToDsl(group: GroupElement, ctx: Ctx): DslGroup {
  const children = [...group.children.values()] as Element[];
  const internalLen = Math.max(0, ...children.map((c) => c.beat + c.duration));
  const out: DslGroup = { kind: 'group', elements: reconstructSequence(children, internalLen, false, ctx) };
  if (group.modifiers && group.modifiers.length > 0) out.modifiers = [...(group.modifiers as Modifier[])];
  if (group.roll) out.roll = true;
  return out;
}

/** Map a stored velocity back to the nearest authoring form: the `:a` / `:g`
 *  accent/ghost markers, a `{vol}` marker, or (for an arbitrary value) a
 *  `{midi:{velocity}}` sidecar. A note's natural (unset) velocity stays bare. */
function applyVelocity(velocity: number | undefined, modifiers: Modifier[], metadata: Metadata): void {
  if (velocity === undefined) return;
  if (velocity === ACCENT_VELOCITY) {
    modifiers.push('a');
    return;
  }
  if (velocity === GHOST_VELOCITY) {
    modifiers.push('g');
    return;
  }
  const vol = VELOCITY_TO_VOLUME.get(velocity);
  if (vol) {
    metadata.vol = vol;
    return;
  }
  setMidi(metadata, 'velocity', velocity);
}

function setMidi(metadata: Metadata, key: 'note' | 'velocity' | 'tick', value: number): void {
  const midi = (metadata.midi as Record<string, number> | undefined) ?? {};
  midi[key] = value;
  (metadata as { midi?: unknown }).midi = midi;
}

// ---------- Patterns, tempo, metadata ----------

function buildPatterns(jot: MutableJot, ctx: Ctx): Record<string, DslPattern> {
  const out: Record<string, DslPattern> = {};
  for (const p of jot.patterns.values()) {
    const body = [...p.body.values()] as Element[];
    const internalLen = Math.max(0, ...body.map((c) => c.beat + c.duration));
    out[p.name] = { name: p.name, elements: reconstructSequence(body, internalLen, false, ctx) };
  }
  return out;
}

function buildTempoEvents(jot: MutableJot, realBars: { id: string }[]): DslTempoEvent[] {
  const barIndexById = new Map<string, number>();
  realBars.forEach((b, i) => barIndexById.set(b.id, i));
  const events: DslTempoEvent[] = [];
  for (const ev of jot.tempoEvents.values()) {
    const barIndex = barIndexById.get(ev.barId);
    if (barIndex === undefined) continue;
    events.push({ barIndex, beat: ev.beat, bpm: ev.bpm });
  }
  events.sort((a, b) => (a.barIndex !== b.barIndex ? a.barIndex - b.barIndex : a.beat - b.beat));
  return events;
}

function buildGlobalMetadata(
  jot: MutableJot,
  realBars: { tsCount: number; tsUnit: number }[]
): Metadata {
  const residual = parseResidual(jot.globalMetadataJson);
  // Tempo (incl. the initial) exports via `tempoEvents`, not a metadata `bpm`;
  // `barDrift` is recording-specific and intentionally dropped on DSL export.
  const meta: Metadata = { ...residual };
  const time: TimeSignature | undefined = realBars[0]
    ? { count: realBars[0].tsCount, unit: realBars[0].tsUnit }
    : undefined;
  if (time) meta.time = time;
  const instrumentMapping = buildInstrumentMapping(jot);
  if (Object.keys(instrumentMapping).length > 0) meta.instrumentMapping = instrumentMapping;
  if (jot.songLeadIn !== undefined) meta.songLeadIn = jot.songLeadIn;
  if (jot.leadBars !== undefined) meta.leadBars = jot.leadBars;
  if (jot.gridDivision !== undefined) meta.gridDivision = jot.gridDivision;
  return meta;
}

function buildInstrumentMapping(jot: MutableJot): Record<string, DslInstrument> {
  const out: Record<string, DslInstrument> = {};
  for (const [lane, inst] of jot.instruments) {
    const i = inst as { kind: DslInstrument['kind']; name?: string; limb?: DslInstrument['limb']; midiNote?: number };
    const dsl: DslInstrument = { kind: i.kind };
    if (i.name !== undefined) dsl.name = i.name;
    if (i.limb !== undefined) dsl.limb = i.limb;
    if (i.midiNote !== undefined) dsl.midi = { note: i.midiNote };
    out[lane] = dsl;
  }
  return out;
}

function parseResidual(json: string | undefined): Metadata {
  if (!json) return {};
  try {
    const parsed = JSON.parse(json);
    return parsed && typeof parsed === 'object' ? (parsed as Metadata) : {};
  } catch {
    return {};
  }
}

// ---------- Layer derivation (mirrors StructureStore.membership) ----------

function orderedLayerIds(jot: MutableJot): string[] {
  const ids = [...jot.layers.values()].map((l) => (l as { id: string }).id);
  ids.sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  return ids;
}

/** The layer a top-level element renders in: a note via its track's placement
 *  in `ordering`, a group via its first descendant note's track, else the
 *  stored `layerId` fallback. Empty string when unresolved. */
function layerOfTopLevel(jot: MutableJot, el: Element): string {
  const tid = firstDescendantTrackId(el);
  if (tid !== undefined) return layerIdOfTrack(jot, tid) ?? '';
  return el.kind !== 'note' ? (el.layerId ?? '') : '';
}

function firstDescendantTrackId(el: Element): string | undefined {
  if (el.kind === 'note') return el.trackId;
  if (el.kind === 'group') {
    for (const child of el.children.values()) {
      const t = firstDescendantTrackId(child as Element);
      if (t !== undefined) return t;
    }
  }
  return undefined;
}

// ---------- Numeric helpers ----------

function barBeats(tsCount: number, tsUnit: number): number {
  return (tsCount * 4) / tsUnit;
}

function roundBeat(beat: number): number {
  return Math.round(beat * 1e6) / 1e6;
}

/** Snap a weight ratio to a clean value: integers within fp tolerance collapse
 *  to the integer; otherwise round to 6 dp so the fixpoint is stable. */
function cleanNumber(n: number): number {
  const r = Math.round(n);
  if (Math.abs(n - r) < 1e-4) return r;
  return Math.round(n * 1e6) / 1e6;
}
