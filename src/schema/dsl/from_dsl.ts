/**
 * DSL `Jot` → reactive Jot converter. The loaders (`parse`, `fromMidi`,
 * `rlrr_to_jot`, fakes) all emit a DSL `Jot`, a nested weight-tree; this
 * maps it onto the reactive `JotSchema` element tree so a single reactive
 * document backs the whole app.
 *
 * Coordinates: top-level elements are distributed across their bar's beats
 * (bar-relative); a group's children keep their *natural* weights as the
 * group's internal coordinate space, and the group's bar-space `duration`
 * vs that internal span is the tuplet scaling the structure store applies.
 * Rests are dropped (they only consume sibling space); a simultaneity
 * becomes coincident note elements.
 */
import type { Element as DslElement, Jot as DslJot, TimeSignature } from 'src/schema/dsl/dsl';
import { elementWeight, sumWeights } from 'src/schema/dsl/element_metrics';
import { initialBpm } from 'src/schema/dsl/tempo';
import { ACCENT_VELOCITY, GHOST_VELOCITY, VOLUME_TO_VELOCITY } from 'src/dynamics/dynamics';
import type { Init } from '../descriptors';
import { createMutableJot, JotSchema } from '../schema';
import { TrackBuilder } from '../ordering';
import { compareLanesByDefaultMixerOrder } from 'src/instruments/mixer_order';

type Obj = Record<string, unknown>;

/** Drop the `:a`/`:g` loudness markers (not schema modifiers); `undefined` when
 *  nothing's left, matching the group schema's optional `modifiers`. */
function groupModifiers(mods: readonly string[] | undefined): string[] | undefined {
  const kept = (mods ?? []).filter((m) => m !== 'a' && m !== 'g');
  return kept.length > 0 ? kept : undefined;
}

function barBeats(time: TimeSignature): number {
  return (time.count * 4) / time.unit;
}

function omitUndef(o: Obj): Obj {
  const out: Obj = {};
  for (const k of Object.keys(o)) if (o[k] !== undefined) out[k] = o[k];
  return out;
}

type Ctx = {
  ids: { n: number };
  layerId?: string;
  barId?: string;
  patternIdByName: Record<string, string>;
  /** Allocates the note's track per `(layerId, lane)`. Absent in pattern-body
   *  context (template notes are layer-agnostic, so carry no `trackId`). */
  tracks?: TrackBuilder;
};

function genId(prefix: string, ctx: Ctx): string {
  return `${prefix}${ctx.ids.n++}`;
}

/** The MIDI sidecar a producer stashed on a note (`note`, `velocity`,
 *  `tick`), reached via the untyped `Metadata` index signature. */
function midiMeta(el: DslElement): { note?: number; velocity?: number; tick?: number } | undefined {
  return el.kind === 'note'
    ? (el.metadata?.midi as { note?: number; velocity?: number; tick?: number } | undefined)
    : undefined;
}

/** Convert one DSL element at bar/group-space `[beat, beat+duration)` into
 *  `[id, init]` pairs (a simultaneity yields several). */
function convertElement(
  el: DslElement,
  beat: number,
  duration: number,
  ctx: Ctx,
  topLevel: boolean
): Array<[string, Obj]> {
  // Containers (group / pattern) carry `barId` + `layerId` as a routing
  // fallback; a note carries only `barId`, its layer is derived from its
  // `trackId`'s placement in `ordering` (so moving a track across layers never
  // re-homes the note).
  const anchor = topLevel ? omitUndef({ barId: ctx.barId, layerId: ctx.layerId }) : {};
  switch (el.kind) {
    case 'rest':
      return [];
    case 'note': {
      const id = genId('e', ctx);
      // Placed notes (those in a layer) get a track; pattern-body template
      // notes (no layer context) keep only `lane`.
      const trackId =
        ctx.layerId !== undefined && ctx.tracks ? ctx.tracks.track(ctx.layerId, el.lane) : undefined;
      // Accent/ghost aren't schema modifiers: they're loudness. An explicit MIDI
      // velocity wins, else `:a`/`:g` map to a representative velocity, else a
      // `pp`..`ff` marker is converted. The `:a`/`:g` markers are dropped from
      // the stored modifiers (the accent ring / ghost glyph derive from
      // velocity at render time).
      const dslMods = el.modifiers ?? [];
      const accentGhostVel = dslMods.includes('a')
        ? ACCENT_VELOCITY
        : dslMods.includes('g')
          ? GHOST_VELOCITY
          : undefined;
      return [
        [
          id,
          omitUndef({
            kind: 'note',
            id,
            ...(topLevel ? omitUndef({ barId: ctx.barId }) : {}),
            beat,
            duration,
            trackId,
            lane: el.lane,
            modifiers: dslMods.filter((m) => m !== 'a' && m !== 'g'),
            sticking: el.sticking,
            roll: el.roll ? true : undefined,
            offsetMs: el.offset,
            midiNote: midiMeta(el)?.note,
            velocity:
              midiMeta(el)?.velocity ??
              accentGhostVel ??
              (typeof el.metadata?.vol === 'string'
                ? VOLUME_TO_VELOCITY[el.metadata.vol]
                : undefined),
            midiTick: midiMeta(el)?.tick,
          }),
        ],
      ];
    }
    case 'simul':
      // Inner elements share this onset and span.
      return el.elements.flatMap((inner) => convertElement(inner, beat, duration, ctx, topLevel));
    case 'group': {
      const id = genId('e', ctx);
      return [
        [
          id,
          omitUndef({
            kind: 'group',
            id,
            ...anchor,
            beat,
            duration,
            children: convertChildren(el.elements, ctx),
            // Drop the accent/ghost loudness markers (not schema modifiers).
            modifiers: groupModifiers(el.modifiers),
            roll: el.roll ? true : undefined,
          }),
        ],
      ];
    }
    case 'patternRef': {
      const id = genId('e', ctx);
      return [
        [
          id,
          omitUndef({
            kind: 'pattern',
            id,
            ...anchor,
            beat,
            duration,
            patternId: ctx.patternIdByName[el.name],
          }),
        ],
      ];
    }
  }
}

/** A group/pattern body's children, keyed by id, in natural-weight space. */
function convertChildren(els: DslElement[], ctx: Ctx): Obj {
  const out: Obj = {};
  let cursor = 0;
  for (const el of els) {
    const w = elementWeight(el);
    for (const [id, init] of convertElement(el, cursor, w, ctx, false)) out[id] = init;
    cursor += w;
  }
  return out;
}

/** A bar's top-level elements, distributed across `totalBeats`. */
function convertBarElements(els: DslElement[], totalBeats: number, ctx: Ctx): Obj {
  const out: Obj = {};
  const weights = els.map(elementWeight);
  const total = weights.reduce((a, b) => a + b, 0) || 1;
  let cursor = 0;
  for (let i = 0; i < els.length; i++) {
    const span = (weights[i] / total) * totalBeats;
    for (const [id, init] of convertElement(els[i], cursor, span, ctx, true)) out[id] = init;
    cursor += span;
  }
  return out;
}

/** Build the plain-object initial state for {@link createMutableJot}. */
export function dslToInit(jot: DslJot): Init<typeof JotSchema> {
  const ids = { n: 0 };
  const gm = jot.globalMetadata;
  const globalTime = gm.time ?? { count: 4, unit: 4 };

  // Patterns: assign ids first (so refs resolve), then convert bodies.
  const patternIdByName: Record<string, string> = {};
  for (const name of Object.keys(jot.patterns ?? {})) patternIdByName[name] = `p${ids.n++}`;
  const patterns: Obj = {};
  for (const [name, pattern] of Object.entries(jot.patterns ?? {})) {
    const id = patternIdByName[name];
    patterns[id] = {
      id,
      name,
      body: convertChildren(pattern.elements, { ids, patternIdByName }),
    };
  }

  // Layers (always at least one explicit layer).
  const layers: Obj = {};
  const layerIds: string[] = [];
  const dslLayers = jot.layers.length > 0 ? jot.layers : [{ bars: [] }];
  dslLayers.forEach((v, i) => {
    const id = `v${i}`;
    layerIds[i] = id;
    layers[id] = omitUndef({ id, name: v.name });
  });

  // Bar grid from layer 0 (shared); prepend an anacrusis bar if present.
  const grid = dslLayers[0];
  const realBars = grid.bars;
  const realBarIds: string[] = [];
  const bars: Obj[] = [];
  let anacrusisBarId: string | undefined;
  if (grid.anacrusis && grid.anacrusis.length > 0) {
    anacrusisBarId = genId('b', { ids, patternIdByName });
    bars.push({ id: anacrusisBarId, tsCount: globalTime.count, tsUnit: globalTime.unit, anacrusis: true });
  }
  let activeTime = globalTime;
  for (let i = 0; i < realBars.length; i++) {
    const t = realBars[i].metadata?.time ?? activeTime;
    activeTime = t;
    const id = `b${ids.n++}`;
    realBarIds[i] = id;
    bars.push({ id, tsCount: t.count, tsUnit: t.unit });
  }

  // Elements: every layer's bars (+ anacrusis) flattened into one map. The
  // track builder mints one instrument track per (layer, lane) as notes are
  // converted, and yields the default ordering afterwards.
  const trackBuilder = new TrackBuilder();
  const elements: Obj = {};
  dslLayers.forEach((v, vi) => {
    const ctx = (barId: string): Ctx => ({
      ids,
      layerId: layerIds[vi],
      barId,
      patternIdByName,
      tracks: trackBuilder,
    });
    if (anacrusisBarId && v.anacrusis && v.anacrusis.length > 0) {
      Object.assign(
        elements,
        convertBarElements(v.anacrusis, sumWeights(v.anacrusis), ctx(anacrusisBarId))
      );
    }
    v.bars.forEach((b, bi) => {
      const barId = realBarIds[bi];
      if (barId === undefined) return;
      const t = b.metadata?.time ?? globalTime;
      Object.assign(elements, convertBarElements(b.elements, barBeats(t), ctx(barId)));
    });
  });

  // Tempo events, anchored by bar id.
  const tempoEvents: Obj = {};
  for (const ev of jot.tempoEvents ?? []) {
    const barId = realBarIds[ev.barIndex];
    if (barId === undefined) continue;
    const id = `t${ids.n++}`;
    tempoEvents[id] = { id, barId, beat: ev.beat, bpm: ev.bpm };
  }

  // Instruments by lane letter.
  const instruments: Obj = {};
  for (const [lane, inst] of Object.entries(gm.instrumentMapping ?? {})) {
    instruments[lane] = omitUndef({
      kind: inst.kind,
      name: inst.name,
      limb: inst.limb,
      midiNote: inst.midi?.note,
    });
  }

  return omitUndef({
    title: jot.title,
    bpm: initialBpm(jot),
    songLeadIn: gm.songLeadIn,
    leadBars: gm.leadBars,
    gridDivision: gm.gridDivision,
    layers,
    tracks: trackBuilder.tracks,
    trackGroups: {},
    ordering: trackBuilder.ordering(layerIds, (a, b) =>
      compareLanesByDefaultMixerOrder(a, b, (lane) => gm.instrumentMapping?.[lane])
    ),
    bars,
    elements,
    instruments,
    tempoEvents,
    patterns,
  }) as Init<typeof JotSchema>;
}

/** Convert a DSL `Jot` into a live mutable Jot document. */
export function dslToMutable(jot: DslJot) {
  return createMutableJot(dslToInit(jot));
}
