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
import type { Element as DslElement, Jot as DslJot, TimeSignature } from 'src/dsl/dsl';
import { elementWeight, sumWeights } from 'src/jot/pattern_expansion';
import { initialBpm } from 'src/tempo/tempo';
import type { Init } from './descriptors';
import { createReactiveJot, JotSchema } from './schema';

type Obj = Record<string, unknown>;

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
  voiceId?: string;
  barId?: string;
  patternIdByName: Record<string, string>;
};

function genId(prefix: string, ctx: Ctx): string {
  return `${prefix}${ctx.ids.n++}`;
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
  const anchor = topLevel ? omitUndef({ barId: ctx.barId, voiceId: ctx.voiceId }) : {};
  switch (el.kind) {
    case 'rest':
      return [];
    case 'note': {
      const id = genId('e', ctx);
      return [
        [
          id,
          omitUndef({
            kind: 'note',
            id,
            ...anchor,
            beat,
            duration,
            pitch: el.pitch,
            modifiers: el.modifiers ?? [],
            sticking: el.sticking,
            roll: el.roll ? true : undefined,
            offsetMs: el.offset,
            midiNote: (el.metadata?.midi as { note?: number } | undefined)?.note,
            vol: typeof el.metadata?.vol === 'string' ? el.metadata.vol : undefined,
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
            modifiers: el.modifiers && el.modifiers.length > 0 ? el.modifiers : undefined,
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

/** Build the plain-object initial state for {@link createReactiveJot}. */
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

  // Voices (always at least one explicit voice).
  const voices: Obj = {};
  const voiceIds: string[] = [];
  const dslVoices = jot.voices.length > 0 ? jot.voices : [{ bars: [] }];
  dslVoices.forEach((v, i) => {
    const id = `v${i}`;
    voiceIds[i] = id;
    voices[id] = omitUndef({ id, name: v.name });
  });

  // Bar grid from voice 0 (shared); prepend an anacrusis bar if present.
  const grid = dslVoices[0];
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

  // Elements: every voice's bars (+ anacrusis) flattened into one map.
  const elements: Obj = {};
  dslVoices.forEach((v, vi) => {
    const ctx = (barId: string): Ctx => ({ ids, voiceId: voiceIds[vi], barId, patternIdByName });
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

  // Instruments by pitch letter.
  const instruments: Obj = {};
  for (const [pitch, inst] of Object.entries(gm.instrumentMapping ?? {})) {
    instruments[pitch] = omitUndef({
      kind: inst.kind,
      name: inst.name,
      limb: inst.limb,
      midiNote: inst.midi?.note,
    });
  }

  return omitUndef({
    title: jot.title,
    bpm: initialBpm(jot),
    drumsT0Sec: gm.drumsT0Sec,
    signalT0Sec: gm.signalT0Sec,
    leadBars: gm.leadBars,
    gridDivision: gm.gridDivision,
    voices,
    bars,
    elements,
    instruments,
    tempoEvents,
    patterns,
  }) as Init<typeof JotSchema>;
}

/** Convert a DSL `Jot` into a live reactive Jot document. */
export function dslToReactive(jot: DslJot) {
  return createReactiveJot(dslToInit(jot));
}
