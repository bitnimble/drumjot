import { describe, expect, it } from 'bun:test';
import { createBlankJot } from 'src/editing/new_jot';
import { initialBpm } from 'src/schema/dsl/tempo';
import { JotEditorStore } from 'src/editing/jot_editor_store';
import { LayersStore } from 'src/editing/layers/layers_store';

describe('createBlankJot', () => {
  it('declares the lighter stock kit, top-to-bottom, with no toms', () => {
    const jot = createBlankJot();
    const mapping = jot.globalMetadata.instrumentMapping!;
    expect(Object.keys(mapping)).toEqual(['c', 'd', 'h', 's', 'k']);
    expect(Object.values(mapping).map((i) => i.kind)).toEqual([
      'crash',
      'ride',
      'hihat',
      'snare',
      'kick',
    ]);
  });

  it('defaults to 120 bpm, 4/4, a title, and one empty bar', () => {
    const jot = createBlankJot();
    expect(jot.title).toBe('New Jot');
    // No `bpm` field; the initial tempo defaults to 120 via `tempoEvents`.
    expect(initialBpm(jot)).toBe(120);
    expect(jot.globalMetadata.time).toEqual({ count: 4, unit: 4 });
    expect(jot.layers).toHaveLength(1);
    expect(jot.layers[0].bars).toHaveLength(1);
    expect(jot.layers[0].bars[0].elements).toEqual([]);
  });

  it('carries no notes, patterns, tempo events, or audio', () => {
    const jot = createBlankJot();
    expect(jot.patterns).toBeUndefined();
    expect(jot.tempoEvents).toBeUndefined();
    // No element on any layer carries a note.
    expect(jot.layers.flatMap((l) => l.bars.flatMap((b) => b.elements))).toEqual([]);
  });

  it('takes a custom title', () => {
    expect(createBlankJot('My groove').title).toBe('My groove');
  });

  it('renders its declared lanes as empty rows once loaded', () => {
    const store = new JotEditorStore();
    store.loadSource(createBlankJot());
    // One empty 4/4 bar exists...
    expect(store.structural!.viewGeometry.some((b) => b.index === 1)).toBe(true);
    // ...no lane carries a note (the structural row list, driven by notes, is empty)...
    expect(store.structural!.lanes).toEqual([]);
    // ...yet the rendered rows (tracks/ordering model) cover the full kit, in
    // the canonical top-of-kit-first order.
    const layers = new LayersStore(() => store.jot);
    const renderedLanes = layers.layout.flatMap((l) =>
      l.slots.flatMap((s) => s.tracks.filter((t) => t.kind === 'instrument').map((t) => t.lane))
    );
    expect(renderedLanes).toEqual(['c', 'd', 'h', 's', 'k']);
  });
});
