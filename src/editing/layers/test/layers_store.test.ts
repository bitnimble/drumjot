import { describe, expect, it } from 'bun:test';
import { autorun, runInAction } from 'mobx';
import { LayersStore } from 'src/editing/layers/layers_store';
import { JotEditorStore } from 'src/editing/jot_editor_store';
import { PICKER_PALETTE } from 'src/editing/tracks/tracks';
import { dslToReactive } from 'src/schema/dsl/from_dsl';
import { parse } from 'src/schema/dsl/parser/parser';
import type { Jot } from 'src/schema/schema';

function storeFrom(src: string): { jot: Jot; store: LayersStore } {
  const jot = dslToReactive(parse(src)).model;
  return { jot, store: new LayersStore(() => jot) };
}

describe('LayersStore read-model', () => {
  it('layout reacts to a jot reload (a freshly-added second layer appears)', () => {
    // Regression: the store reads `jotEditorStore.jot` (the swapped-on-reload
    // reactive doc). When that read isn't reactive to the swap, an observed
    // layout stays pinned to the previous song, so reloading single -> two
    // layer never surfaces the second layer's band (only v0 renders).
    const editor = new JotEditorStore();
    const layers = new LayersStore(() => editor.jot);
    runInAction(() =>
      editor.loadSource(parse('{{ time: "4/4", instrumentMapping: { h:{name:"HiHat"} } }} | h h h h |'))
    );
    let layerIds: string[] = [];
    const dispose = autorun(() => {
      layerIds = layers.layout.map((l) => l.id);
    });
    expect(layerIds).toEqual(['v0']);
    runInAction(() =>
      editor.loadSource(
        parse(
          '{{ time: "4/4", instrumentMapping: { h:{name:"HiHat"}, k:{name:"Kick"} } }} | h h h h | || | k . . . |'
        )
      )
    );
    expect(layerIds).toEqual(['v0', 'v1']);
    dispose();
  });

  it('projects ordering into layer → loose-slot → track views', () => {
    const { store } = storeFrom('| h h h h | || | k . s . |');
    const layout = store.layout;

    expect(layout.map((l) => l.id)).toEqual(['v0', 'v1']);
    for (const layer of layout) {
      expect(layer.slots).toHaveLength(1);
      expect(layer.slots[0].kind).toBe('loose');
    }
    // v0 → [h]; v1 → [k, s] (instrument tracks carry their lane).
    const lanesOf = (i: number) =>
      layout[i].slots.flatMap((s) =>
        s.tracks.map((t) => (t.kind === 'instrument' ? t.lane : t.kind))
      );
    expect(lanesOf(0)).toEqual(['h']);
    expect(new Set(lanesOf(1))).toEqual(new Set(['k', 's']));
  });

  it('reverse-looks-up a track to its layer and loose (null) group', () => {
    const { store } = storeFrom('| h h h h | || | k . s . |');
    const v0Track = store.layout[0].slots[0].tracks[0].id;
    expect(store.layerIdOfTrack(v0Track)).toBe('v0');
    expect(store.groupIdOfTrack(v0Track)).toBeNull();
    expect(store.layerIdOfTrack('missing')).toBeUndefined();
  });

  it('defaults layer 1 to transparent and later layers to a palette rotation', () => {
    const { store } = storeFrom('| h h h h | || | k . s . |');
    const [l0, l1] = store.layout;
    expect(l0.color).toBeUndefined();
    expect(l1.color).toBe(PICKER_PALETTE[0]);
  });

  it('is empty with no jot', () => {
    const store = new LayersStore(() => undefined);
    expect(store.layout).toEqual([]);
    expect(store.layerIdOfTrack('x')).toBeUndefined();
  });
});
