import { describe, expect, it } from 'bun:test';
import { JotEditorStore } from 'src/editing/jot_editor_store';
import { parse } from 'src/schema/dsl/parser/parser';

// Per-bar drift travels jot.barDrift -> CRDT barDriftJson (seeded in from_dsl)
// -> StructuralPresenter.barDrift, the reactive replacement for the old frozen
// `source.barDrift` read. The waveform stretch + playback DriftMap consume it.
describe('barDrift ingestion through the reactive model', () => {
  it('carries jot.barDrift to structural.barDrift', () => {
    const store = new JotEditorStore();
    const jot = parse('{{ time: "4/4" }}\n| k s k s |\n| k s k s |');
    jot.barDrift = [0, 0.03];
    store.loadSource(jot);
    expect([...store.structural!.barDrift]).toEqual([0, 0.03]);
  });

  it('is empty for a jot with no drift', () => {
    const store = new JotEditorStore();
    store.loadSource(parse('| k s k s |'));
    expect([...store.structural!.barDrift]).toEqual([]);
  });
});
