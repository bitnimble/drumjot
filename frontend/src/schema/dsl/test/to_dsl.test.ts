import { describe, expect, it } from 'bun:test';
import { parse } from 'src/schema/dsl/parser/parser';
import { writeDsl } from 'src/schema/dsl/writer';
import { dslToMutable } from 'src/schema/dsl/from_dsl';
import { mutableToDsl } from 'src/schema/dsl/to_dsl';
import { buildStructural } from 'src/editing/jot_editor_store';
import type { Jot } from 'src/schema/dsl/dsl';
import type { StructLayer } from 'src/editing/structure/structure_store';

/**
 * `mutableToDsl` is the reactive->DSL exporter. The contract isn't a verbatim
 * text round-trip (the forward conversion is lossy: rests, simuls, accent
 * markers and weights are all reconstructed canonically), it's that re-parsing
 * the exported DSL yields a structurally IDENTICAL document, and that a second
 * export is a fixpoint. This mirrors the browser-level `round_trip.e2e.ts` but
 * runs as a fast pure unit test.
 */

const r = (x: number) => Math.round(x * 1e4) / 1e4;

/** Normalise the structure-store layers to an id-free, comparison-stable shape
 *  (the same fields `round_trip.e2e.ts` snapshots in the browser). */
function normalize(layers: readonly StructLayer[]) {
  return layers.map((v) => ({
    lanes: v.lanes,
    bars: v.bars.map((bar) => ({
      index: bar.index,
      beats: r(bar.beats),
      tracks: Object.fromEntries(
        Object.keys(bar.tracks)
          .sort()
          .map((p) => [
            p,
            bar.tracks[p].notes.map((n) => ({
              beat: r(n.beat),
              dur: r(n.duration),
              roll: !!n.roll,
              mods: [...(n.modifiers ?? [])].sort(),
              stick: n.sticking ?? null,
              velocity: n.velocity ?? null,
            })),
          ])
      ),
      patternSpans: bar.patternSpans.map((s) => ({ name: s.name, start: r(s.startBeat), end: r(s.endBeat) })),
      tupletSpans: bar.tupletSpans.map((s) => ({ count: s.count, start: r(s.startBeat), end: r(s.endBeat) })),
    })),
  }));
}

function structureOf(dsl: Jot) {
  return normalize(buildStructural(dsl).musicalLayers);
}

/** Export the reactive doc built from `src` to DSL, then re-parse it. */
function exportReimport(src: string): { dsl: string; reparsed: Jot } {
  const reactive = dslToMutable(parse(src)).model;
  const dsl = writeDsl(mutableToDsl(reactive));
  return { dsl, reparsed: parse(dsl) };
}

// The rich round-trip song: pattern def + usage, a weighted tuplet of
// simultaneities, accent + flam modifiers with L/R sticking, an explicit rest,
// an open roll, a 3:4 polyrhythm, and two layers.
const SONG = `{{ title: "Round Trip", bpm: 120, time: "4/4",
  instrumentMapping: { k:{name:"Kick"}, s:{name:"Snare"}, h:{name:"HiHat"}, c:{name:"Crash"}, a:{name:"TomA"}, b:{name:"TomB"} } }}
[Groove=(k.s.kks.)]
| [Groove] (k+s k+s k+s)_4 |
| s:fl@l k@r s@r:a . k@r s@l k@r k@r |
||
| h:c h:c h:c h:c |
| (a a a)_4 + (b b b b)_4 |
`;

describe('mutableToDsl', () => {
  it('round-trips the rich song to an identical structure', () => {
    const before = structureOf(parse(SONG));
    const { reparsed } = exportReimport(SONG);
    const after = structureOf(reparsed);
    expect(after).toEqual(before);
  });

  it('is a text fixpoint (re-export equals export)', () => {
    const first = exportReimport(SONG);
    const second = writeDsl(mutableToDsl(dslToMutable(first.reparsed).model));
    expect(second).toBe(first.dsl);
  });

  it('preserves the title in the exported DSL', () => {
    const { dsl } = exportReimport(SONG);
    expect(dsl).toContain('title: "Round Trip"');
  });

  it('reflects a reactive tempo-event edit in the export', () => {
    const reactive = dslToMutable(parse('{{ bpm: 120 }}\n| k s k s |\n| k s k s |')).model;
    const barId = [...reactive.bars][1].id;
    reactive.tempoEvents.set('t1', { id: 't1', barId, beat: 0, bpm: 160 });
    const dsl = writeDsl(mutableToDsl(reactive));
    expect(dsl).toContain('bpm: 160');
    // And it re-parses to a tempo event on bar index 1.
    const reparsed = parse(dsl);
    expect(reparsed.tempoEvents?.some((e) => e.barIndex === 1 && e.bpm === 160)).toBe(true);
  });
});
