import { describe, expect, it } from 'bun:test';
import { parse } from 'src/schema/dsl/parser/parser';
import { JotEditorStore } from 'src/editing/jot_editor_store';
import { TempoEditPresenter } from 'src/editing/playback/tempo_edit_presenter';

function load(dsl: string): { store: JotEditorStore; tempo: TempoEditPresenter } {
  const store = new JotEditorStore();
  store.loadSource(parse(dsl));
  return { store, tempo: new TempoEditPresenter(store) };
}

const TWO_BARS_WITH_CHANGE = `{{ bpm: 120 }}
| k s k s |
{{ bpm: 140 }}
| k s k s |`;

describe('TempoEditPresenter', () => {
  it('exposes the initial pill plus one marker per flat tempo event', () => {
    const { tempo } = load(TWO_BARS_WITH_CHANGE);
    const markers = tempo.bpmMarkers;
    expect(markers.map((m) => ({ bpm: m.bpm, kind: m.source.kind }))).toEqual([
      { bpm: 120, kind: 'initial' },
      { bpm: 140, kind: 'event' },
    ]);
    // The initial pill anchors at the very start; events sit later.
    expect(markers[0].globalBeat).toBe(0);
    expect(markers[1].globalBeat).toBeGreaterThan(0);
  });

  it('edits the initial tempo in place via the initial pill', () => {
    const { store, tempo } = load(TWO_BARS_WITH_CHANGE);
    tempo.commitMarker({ kind: 'initial' }, '160');
    expect(store.jot!.bpm).toBe(160);
    expect(tempo.bpmMarkers[0].bpm).toBe(160);
  });

  it('mutates an existing event in place (stable id, not delete+add)', () => {
    const { store, tempo } = load(TWO_BARS_WITH_CHANGE);
    const event = tempo.bpmMarkers.find((m) => m.source.kind === 'event')!;
    const id = (event.source as { id: string }).id;
    tempo.commitMarker(event.source, '155');
    expect(store.jot!.tempoEvents.get(id)?.bpm).toBe(155);
    expect(store.jot!.tempoEvents.size).toBe(1);
  });

  it('clamps edits to [20, 400] integers', () => {
    const { store, tempo } = load(TWO_BARS_WITH_CHANGE);
    tempo.commitMarker({ kind: 'initial' }, '5000');
    expect(store.jot!.bpm).toBe(400);
    tempo.commitMarker({ kind: 'initial' }, '1');
    expect(store.jot!.bpm).toBe(20);
  });

  it('deletes an event when its pill is cleared, but never the initial pill', () => {
    const { store, tempo } = load(TWO_BARS_WITH_CHANGE);
    const event = tempo.bpmMarkers.find((m) => m.source.kind === 'event')!;
    tempo.commitMarker(event.source, '');
    expect(store.jot!.tempoEvents.size).toBe(0);
    expect(tempo.bpmMarkers.map((m) => m.source.kind)).toEqual(['initial']);

    // Clearing the initial pill is a no-op (the base tempo can't be removed).
    tempo.commitMarker({ kind: 'initial' }, '');
    expect(store.jot!.bpm).toBe(120);
  });

  it('canDelete is true for events, false for the initial pill', () => {
    const { tempo } = load(TWO_BARS_WITH_CHANGE);
    expect(tempo.canDelete({ kind: 'initial' })).toBe(false);
    expect(tempo.canDelete({ kind: 'event', id: 'x' })).toBe(true);
  });

  it('creates a tempo change at a clicked x, seeded with the tempo in force', () => {
    const { store, tempo } = load(`{{ bpm: 120 }}\n| k s k s |\n| k s k s |`);
    const ppb = store.structural!.pxPerBeat;
    // Click on the second real bar's downbeat (global beat = lead-in + 4).
    const event = tempo.bpmMarkers; // [initial]
    expect(event).toHaveLength(1);
    const leadInBeats = store.structural!.layers[0]!.bars[0]!.beats;
    const id = tempo.createTempoChangeAtX((leadInBeats + 4) * ppb);
    expect(id).toBeDefined();
    const created = store.jot!.tempoEvents.get(id!);
    expect(created?.bpm).toBe(120); // the tempo in force there
    expect(created?.beat).toBe(0);
    expect(tempo.bpmMarkers).toHaveLength(2);
  });

  it('returns the existing event id instead of duplicating at the same anchor', () => {
    const { store, tempo } = load(TWO_BARS_WITH_CHANGE);
    const ppb = store.structural!.pxPerBeat;
    const leadInBeats = store.structural!.layers[0]!.bars[0]!.beats;
    const before = store.jot!.tempoEvents.size;
    // Bar index 1 downbeat already has the 140 event.
    const id = tempo.createTempoChangeAtX((leadInBeats + 4) * ppb);
    expect(store.jot!.tempoEvents.size).toBe(before);
    expect(store.jot!.tempoEvents.get(id!)?.bpm).toBe(140);
  });
});
