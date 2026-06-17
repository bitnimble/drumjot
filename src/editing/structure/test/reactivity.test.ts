import { describe, expect, it } from 'bun:test';
import { autorun, runInAction } from 'mobx';
import { createMutableJot, type MutableJot } from 'src/schema/schema';
import type { Jot } from 'src/schema/dsl/dsl';
import { StructureStore, type StructBar } from 'src/editing/structure/structure_store';
import { StructuralPresenter } from 'src/editing/structure/structural_presenter';
import { PaletteStore } from 'src/editing/palette/palette_store';
import { LayoutStore } from 'src/editing/viewport/layout_store';
import { ViewConfig } from 'src/editing/viewport/view_config';

/**
 * The editor-perf contract: adding a note must reach the reactive graph for
 * AT MOST the one bar+lane it touches. Every other row's `barsForLane` keeps
 * its cached value (its observer never re-fires => no React re-render / no
 * reconciliation), unchanged bars within the touched row keep their object
 * identity, and the touched bar's other lanes are left alone. These are the
 * guarantees the granular `StructureStore` derivation exists to provide; this
 * test fails loudly if a future change reintroduces the monolithic
 * "recompute everything on any edit" behaviour.
 */

/** Wire the render-facing peers around a mutable reactive model, mirroring
 *  `buildJotPeers` but exposing the model so the test can edit it. */
function peers(model: MutableJot): StructuralPresenter {
  const viewConfig = new ViewConfig();
  const structureStore = new StructureStore(() => model);
  const palette = new PaletteStore(structureStore, () => viewConfig.palette, () => model);
  const layoutStore = new LayoutStore(
    structureStore,
    () => viewConfig.barWidth as number,
    () => viewConfig.barNotePaddingBeats
  );
  const source = {
    globalMetadata: { bpm: 120, instrumentMapping: { k: { kind: 'kick' }, s: { kind: 'snare' } } },
  } as unknown as Jot;
  return new StructuralPresenter(structureStore, palette, layoutStore, source, viewConfig);
}

function model(): MutableJot {
  return createMutableJot({
    title: '',
    bpm: 120,
    bars: [
      { id: 'b1', tsCount: 4, tsUnit: 4 },
      { id: 'b2', tsCount: 4, tsUnit: 4 },
    ],
    elements: {
      k1: { kind: 'note', id: 'k1', barId: 'b1', beat: 0, duration: 1, lane: 'k', modifiers: [] },
      s1: { kind: 'note', id: 's1', barId: 'b1', beat: 1, duration: 1, lane: 's', modifiers: [] },
      k2: { kind: 'note', id: 'k2', barId: 'b2', beat: 0, duration: 1, lane: 'k', modifiers: [] },
    },
    instruments: { k: { kind: 'kick' }, s: { kind: 'snare' } },
  }).model;
}

const addNote = (m: MutableJot, id: string, barId: string, beat: number, lane: string) =>
  runInAction(() =>
    m.elements.set(id, { kind: 'note', id, barId, beat, duration: 1, lane, modifiers: [] })
  );

const findBar = (bars: readonly StructBar[], id: string) => bars.find((b) => b.id === id)!;

describe('granular reactivity', () => {
  it('adding a note re-renders only the target bar+lane, nothing else', () => {
    const m = model();
    const s = peers(m);

    // One "observer" (autorun) per lane row, like each InstrumentTrackView.
    const fires = { k: 0, s: 0 };
    const dk = autorun(() => {
      fires.k++;
      void s.barsForLane('k').bars;
    });
    const ds = autorun(() => {
      fires.s++;
      void s.barsForLane('s').bars;
    });

    const before = { ...fires };
    const kBarsBefore = s.barsForLane('k').bars;
    const sRowBefore = s.barsForLane('s'); // whole cached row object for lane s

    addNote(m, 'k1b', 'b1', 2, 'k'); // bar b1, lane k

    // Capture while the autoruns are still alive (a `computedFn` is only
    // memoised while observed, so reading it after disposal would recompute).
    const kBarsAfter = s.barsForLane('k').bars;
    const sRowAfter = s.barsForLane('s');
    dk();
    ds();

    // The touched lane's row re-rendered exactly once more...
    expect(fires.k).toBe(before.k + 1);
    // ...and the OTHER lane's row did NOT re-render at all (stopped at MobX).
    expect(fires.s).toBe(before.s);
    // ...nor did it recompute: its whole row object identity is unchanged.
    expect(sRowAfter).toBe(sRowBefore);

    // Within the touched row: the untouched bar (b2) keeps its identity; only
    // the touched bar (b1) is a fresh object.
    expect(findBar(kBarsAfter, 'b2')).toBe(findBar(kBarsBefore, 'b2'));
    expect(findBar(kBarsAfter, 'b1')).not.toBe(findBar(kBarsBefore, 'b1'));

    // The new note actually landed in b1 / lane k.
    expect(findBar(kBarsAfter, 'b1').tracks['k'].notes.map((n) => n.beat)).toEqual([0, 2]);
  });

  it('a note edit does not recompute the lane list (mixer row order stable)', () => {
    const m = model();
    const s = peers(m);
    let lanesFires = 0;
    const d = autorun(() => {
      lanesFires++;
      void s.lanes;
    });
    const before = lanesFires;
    addNote(m, 'k1c', 'b1', 3, 'k'); // existing lane k
    d();
    expect(lanesFires).toBe(before);
  });
});
