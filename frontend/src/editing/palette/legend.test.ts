/**
 * Characterization / parity contract for `PaletteStore.legend`.
 *
 * `legend` is the score-header colour/name legend. Its LANE ORDER has always
 * been the structure walk's per-bar track-record insertion order, which is a
 * DIFFERENT order from the jot-wide lane list (`PaletteStore.jotLanes`, itself
 * `structure.layerOrder` + `lanesForLayer`, which reorders by instrument-mapping
 * order). This locks the exact current output so a perf refactor of `legend`'s
 * body would be provably behaviour-preserving, and records that a jotLanes-based
 * rebuild agrees on the lane SET but DIVERGES in ORDER, so such a rebuild is NOT
 * behaviour-preserving and the granular refactor was not shipped (see below).
 */
import { describe, expect, it } from 'bun:test';
import { parse } from 'src/schema/dsl/parser/parser';
import { JotEditorStore } from 'src/editing/jot_editor_store';
import type { PaletteStore } from 'src/editing/palette/palette_store';

function load(src: string): PaletteStore {
  const store = new JotEditorStore();
  store.loadSource(parse(src));
  return store.palette!;
}

const laneOf = (entry: readonly [string, unknown]) => entry[0];

// The exact divergent jot from `score/test/legend.e2e.ts`: authored `h s k`,
// but the structure walk emits the legend in `s, h, k` order.
const DIVERGENT = `{{ bpm: 120, time: "4/4", title: "Legend Order",
  instrumentMapping: { h: { name: "HiHat" }, s: { name: "Snare" }, k: { name: "Kick" } } }}
(h s k h s k)
`;

// A mapped lane (`x`) with NO notes: reveals whether legend includes empty
// mapped lanes. It walks note-bearing tracks only, so `x` must be absent.
const EMPTY_MAPPED = `{{ bpm: 120, time: "4/4",
  instrumentMapping: { k: { name: "Kick" }, s: { name: "Snare" }, x: { name: "Cowbell" } } }}
| k s . . |
`;

describe('PaletteStore.legend (parity contract)', () => {
  it('emits [lane, {color, name}] in the structure-walk order, with mapping names', () => {
    const palette = load(DIVERGENT);
    // Exact current output: order s, h, k (NOT authoring/jotLanes order), each
    // with its instrument-mapping name and a real palette colour.
    expect([...palette.legend]).toEqual([
      ['s', { color: '#5BA8E8', name: 'Snare' }],
      ['h', { color: '#7BC74D', name: 'HiHat' }],
      ['k', { color: '#FF8C55', name: 'Kick' }],
    ]);
  });

  it('excludes a mapped lane that has no notes', () => {
    const palette = load(EMPTY_MAPPED);
    const lanes = palette.legend.map(laneOf);
    expect(lanes).toEqual(['k', 's']);
    expect(lanes).not.toContain('x');
  });

  it('legend and a raw jotLanes rebuild agree on the lane SET but DIVERGE in ORDER', () => {
    const palette = load(DIVERGENT);
    const legendLanes = palette.legend.map(laneOf);
    const jotLanes = palette.jotLanes;

    // Same set of note-bearing lanes...
    expect([...legendLanes].sort()).toEqual([...jotLanes].sort());
    // ...but a different ORDER: `legend` follows the structure walk's per-bar
    // track-record insertion order, while `jotLanes` reorders by the
    // instrument-mapping order (`lanesForLayer` -> `orderLanes`). This is the
    // load-bearing finding: a RAW jotLanes rebuild is NOT order-preserving for
    // `legend`; the refactor must reorder to the structure-walk order to match.
    expect(legendLanes).toEqual(['s', 'h', 'k']);
    expect(jotLanes).toEqual(['k', 's', 'h']);
    expect(legendLanes).not.toEqual(jotLanes);
  });

  it('legend agrees with a jotLanes rebuild when the two orders coincide', () => {
    // When authoring order matches instrument-mapping order, the two agree
    // fully, so a jotLanes rebuild would be a valid (order-preserving) source
    // here. Only the divergent case above breaks parity.
    const palette = load(
      `{{ time: "4/4", instrumentMapping: { k: { name: "Kick" }, s: { name: "Snare" } } }}
| k s . . |`
    );
    expect(palette.legend.map(laneOf)).toEqual(palette.jotLanes);
    expect(palette.jotLanes).toEqual(['k', 's']);
  });
});
