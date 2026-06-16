/**
 * Unit tests for EditingPresenter: mode switching and committing an
 * insert-mode placeholder as a real note in the reactive document. The
 * inserted note must reflow into the structural layers.
 */
import { describe, expect, it } from 'bun:test';
import { parse } from 'src/schema/dsl/parser/parser';
import { JotEditorStore } from 'src/editing/jot_editor_store';
import { EditingStore } from 'src/editing/editing_store';
import { EditingPresenter } from 'src/editing/editing_presenter';
import { LayersPresenter } from 'src/editing/layers/layers_presenter';
import { SettingsStore } from 'src/settings/settings_store';
import { SelectionStore } from 'src/editing/selection/selection';
import { SelectionPresenter, orderedNotes } from 'src/editing/selection/selection_presenter';
import { layerIdOfTrack } from 'src/schema/ordering';

// One kick on each downbeat of two 4/4 bars.
const SRC =
  '{{ bpm: 120, time: "4/4", instrumentMapping: { k:{name:"Kick"} } }} | k . . . | k . . . |';

function setup() {
  const store = new JotEditorStore();
  store.loadSource(parse(SRC));
  const editingStore = new EditingStore();
  const settings = new SettingsStore();
  const selection = new SelectionStore();
  const selectionPresenter = new SelectionPresenter(selection, () =>
    orderedNotes(store.structural?.musicalLayers ?? [])
  );
  const layersPresenter = new LayersPresenter(() => store.jot);
  const presenter = new EditingPresenter(
    editingStore,
    store,
    settings,
    selection,
    selectionPresenter,
    layersPresenter
  );
  return { store, editingStore, settings, selection, selectionPresenter, presenter };
}

describe('EditingPresenter', () => {
  it('defaults to select mode', () => {
    const { editingStore } = setup();
    expect(editingStore.mode).toBe('select');
  });

  it('switching out of insert mode clears the placeholder', () => {
    const { editingStore, presenter } = setup();
    presenter.setMode('insert');
    presenter.movePlaceholder({ lane: 'k', barId: 'x', beat: 1, absBeat: 1, barBeats: 4 });
    expect(editingStore.placeholder).toBeDefined();
    presenter.setMode('select');
    expect(editingStore.placeholder).toBeUndefined();
  });

  it('insertNote is a no-op without a placeholder', () => {
    const { store, presenter } = setup();
    const before = store.structural!.musicalLayers[0].bars[0].tracks['k'].notes.length;
    presenter.insertNote();
    const after = store.structural!.musicalLayers[0].bars[0].tracks['k'].notes.length;
    expect(after).toBe(before);
  });

  it('commits the placeholder as a note at its exact bar + beat', () => {
    const { store, presenter } = setup();
    const bar0 = store.structural!.musicalLayers[0].bars[0];
    presenter.setMode('insert');
    presenter.movePlaceholder({ lane: 'k', barId: bar0.id, beat: 2, absBeat: 2, barBeats: 4 });
    presenter.insertNote();

    const notes = store.structural!.musicalLayers[0].bars[0].tracks['k'].notes;
    // The original downbeat kick plus the newly inserted one at beat 2.
    expect(notes.map((n) => n.beat).sort((a, b) => a - b)).toEqual([0, 2]);
  });

  it('snaps the placeholder to the grid when snapping is enabled', () => {
    const { editingStore, settings, presenter } = setup();
    // Default grid has main beats + 16ths; nearest 16th to 1.1 is 1.0.
    settings.gridLines = { ...settings.gridLines, subBeat16: true };
    presenter.setSnapping(true);
    presenter.setMode('insert');
    presenter.movePlaceholder({ lane: 'k', barId: 'x', beat: 1.1, absBeat: 1.1, barBeats: 4 });
    expect(editingStore.placeholder!.beat).toBeCloseTo(1.0, 9);
    expect(editingStore.placeholder!.absBeat).toBeCloseTo(1.0, 9);
  });

  it('leaves the placeholder unsnapped when snapping is disabled', () => {
    const { editingStore, presenter } = setup();
    presenter.setSnapping(false); // snapping is on by default
    presenter.setMode('insert');
    presenter.movePlaceholder({ lane: 'k', barId: 'x', beat: 1.1, absBeat: 1.1, barBeats: 4 });
    expect(editingStore.placeholder!.beat).toBeCloseTo(1.1, 9);
  });

  const kicks = (store: ReturnType<typeof setup>['store']) =>
    store.structural!.musicalLayers[0].bars;

  it('deleteSelection removes every selected note and clears the selection', () => {
    const { store, selection, selectionPresenter, presenter } = setup();
    const k0 = kicks(store)[0].tracks['k'].notes[0];
    selectionPresenter.replace(k0);
    presenter.deleteSelection();
    expect(kicks(store)[0].tracks['k']?.notes.length ?? 0).toBe(0);
    expect(selection.selectedNotes.size).toBe(0);
  });

  it('moveSelection shifts a note within its bar', () => {
    const { store, selectionPresenter, presenter } = setup();
    const k0 = kicks(store)[0].tracks['k'].notes[0];
    selectionPresenter.replace(k0);
    presenter.moveSelection(k0, 1);
    expect(kicks(store)[0].tracks['k'].notes.map((n) => n.beat)).toEqual([1]);
  });

  it('moveSelection re-homes a note across a bar boundary', () => {
    const { store, selectionPresenter, presenter } = setup();
    const k0 = kicks(store)[0].tracks['k'].notes[0];
    selectionPresenter.replace(k0);
    presenter.moveSelection(k0, 4); // bar0 beat0 -> bar1 beat0
    expect(kicks(store)[0].tracks['k']?.notes.length ?? 0).toBe(0);
    expect(kicks(store)[1].tracks['k'].notes.map((n) => n.beat).sort()).toEqual([0, 0]);
  });

  it('moveSelection preserves relative spacing across the group', () => {
    const { store, selection, selectionPresenter, presenter } = setup();
    const bars = kicks(store);
    const k0 = bars[0].tracks['k'].notes[0];
    const k1 = bars[1].tracks['k'].notes[0];
    selection.state = { type: 'notes', notes: new Set([k0, k1]) };
    presenter.moveSelection(k0, 0.5);
    expect(kicks(store)[0].tracks['k'].notes.map((n) => n.beat)).toEqual([0.5]);
    expect(kicks(store)[1].tracks['k'].notes.map((n) => n.beat)).toEqual([0.5]);
  });

  it('moveSelection snaps the anchor to the grid when snapping is on', () => {
    const { store, selectionPresenter, presenter } = setup();
    const k0 = kicks(store)[0].tracks['k'].notes[0];
    selectionPresenter.replace(k0);
    presenter.setSnapping(true); // default grid: main beats + 16ths
    presenter.moveSelection(k0, 0.3); // 0.3 -> nearest 16th 0.25
    expect(kicks(store)[0].tracks['k'].notes[0].beat).toBeCloseTo(0.25, 9);
  });

  it('snapDeltaFn snaps the live drag delta to the grid (anchor-relative)', () => {
    const { store, presenter } = setup();
    const k0 = kicks(store)[0].tracks['k'].notes[0]; // anchor at abs beat 0
    presenter.setSnapping(true); // default grid: main beats + 16ths
    const snap = presenter.snapDeltaFn(k0);
    // A raw 0.3-beat drag snaps the anchor's target (0.3) to the nearest 16th.
    expect(snap(0.3)).toBeCloseTo(0.25, 9);
    expect(snap(0.1)).toBeCloseTo(0, 9);
  });

  it('snapDeltaFn is identity when snapping is off', () => {
    const { store, presenter } = setup();
    const k0 = kicks(store)[0].tracks['k'].notes[0];
    presenter.setSnapping(false);
    expect(presenter.snapDeltaFn(k0)(0.3)).toBeCloseTo(0.3, 9);
  });

  it('moveSelection remaps lanes via laneMap (cross-lane move)', () => {
    const { store, selectionPresenter, presenter } = setup();
    const k0 = kicks(store)[0].tracks['k'].notes[0];
    selectionPresenter.replace(k0);
    presenter.moveSelection(k0, 0, () => 's');
    expect(kicks(store)[0].tracks['k']?.notes.length ?? 0).toBe(0);
    expect(kicks(store)[0].tracks['s'].notes.map((n) => n.beat)).toEqual([0]);
  });
});

// Hands (hi-hat) on layer 0, feet (kick) on layer 1, joined by `||`.
const MULTI_SRC =
  '{{ bpm: 120, time: "4/4", instrumentMapping: { h:{name:"HiHat"}, k:{name:"Kick"} } }} ' +
  '| h . . . | || | k . . . |';

function setupMulti() {
  const store = new JotEditorStore();
  store.loadSource(parse(MULTI_SRC));
  const editingStore = new EditingStore();
  const settings = new SettingsStore();
  const selection = new SelectionStore();
  const selectionPresenter = new SelectionPresenter(selection, () =>
    orderedNotes(store.structural?.musicalLayers ?? [])
  );
  const layersPresenter = new LayersPresenter(() => store.jot);
  const presenter = new EditingPresenter(
    editingStore,
    store,
    settings,
    selection,
    selectionPresenter,
    layersPresenter
  );
  return { store, presenter, selection, selectionPresenter };
}

describe('EditingPresenter, multi-layer (hands/feet split)', () => {
  // A placed note stores no `layerId`; its layer derives from its `trackId`'s
  // placement in `ordering`. So assert on the derived layer, not a stored field.
  const noteLayerIds = (store: JotEditorStore, lane: string) =>
    [...store.jot!.elements.values()]
      .filter((e) => (e as { kind: string; lane?: string }).kind === 'note' && (e as { lane?: string }).lane === lane)
      .map((e) => {
        const trackId = (e as { trackId?: string }).trackId;
        return trackId !== undefined ? layerIdOfTrack(store.jot!, trackId) : undefined;
      });

  it('renders a lane that lives in the non-first layer (the kick row)', () => {
    const { store } = setupMulti();
    const kBar0 = store.structural!.barsForLane('k').bars.find((b) => b.tracks['k']?.notes.length);
    expect(kBar0?.tracks['k'].notes.map((n) => n.beat)).toEqual([0]);
  });

  it('ownerLayerFor returns each lane its own layer', () => {
    const { store } = setupMulti();
    const hLayer = store.structural!.ownerLayerFor('h');
    const kLayer = store.structural!.ownerLayerFor('k');
    expect(hLayer).toBeDefined();
    expect(kLayer).toBeDefined();
    expect(hLayer).not.toBe(kLayer);
  });

  it('inserting into the kick lane tags the note with the kick layer', () => {
    const { store, presenter } = setupMulti();
    const kLayer = store.structural!.ownerLayerFor('k');
    const bar0 = store.structural!.musicalLayers[0].bars[0];
    presenter.setMode('insert');
    presenter.movePlaceholder({ lane: 'k', barId: bar0.id, beat: 2, absBeat: 2, barBeats: 4 });
    presenter.insertNote();
    // Both kicks (original + inserted) carry the kick layer, none leak to hands.
    expect(noteLayerIds(store, 'k')).toEqual([kLayer, kLayer]);
  });

  it('surfaces a tuplet bracket authored in a non-first layer', () => {
    const store = new JotEditorStore();
    store.loadSource(
      parse(
        '{{ time: "4/4", instrumentMapping: { h:{name:"HiHat"}, s:{name:"Snare"}, k:{name:"Kick"} } }} ' +
          '| h h h h | || | (s k s) . . . |'
      )
    );
    // The triplet lives in layer 1 (feet); its bracket must still surface in
    // the bar's lane-spanning chrome (bar-level, so any lane's bars expose it).
    const bars = store.structural!.barsForLane('s').bars;
    const withTuplet = bars.find((b) => b.tupletSpans.length > 0);
    expect(withTuplet?.tupletSpans[0].count).toBe(3);
  });

  it('keeps a single-lane tuplet per lane (each draws above its own row)', () => {
    // A snare triplet and a kick triplet over the same beats (kick on its own
    // layer): both stay, each tagged with its single lane, so each draws above
    // its own row rather than coinciding.
    const store = new JotEditorStore();
    store.loadSource(
      parse(
        '{{ time: "4/4", instrumentMapping: { s:{name:"Snare"}, k:{name:"Kick"} } }} ' +
          '| (s s s) . . . | || | (k k k) . . . |'
      )
    );
    const bracketBar = store
      .structural!.barsForLane('s')
      .bars.find((b) => b.tupletSpans.length > 0)!;
    const laneSets = bracketBar.tupletSpans.map((t) => [...t.lanes].join(',')).sort();
    expect(laneSets).toEqual(['k', 's']);
    expect(bracketBar.tupletSpans.every((t) => t.lanes.size === 1)).toBe(true);
  });

  it('de-duplicates two identical same-lane tuplets across layers', () => {
    // Both layers carry the snare triplet: one bracket, not two stacked copies.
    const store = new JotEditorStore();
    store.loadSource(
      parse(
        '{{ time: "4/4", instrumentMapping: { s:{name:"Snare"} } }} ' +
          '| (s s s) . . . | || | (s s s) . . . |'
      )
    );
    const bracketBar = store
      .structural!.barsForLane('s')
      .bars.find((b) => b.tupletSpans.length > 0)!;
    expect(bracketBar.tupletSpans.length).toBe(1);
  });

  it('a multi-lane tuplet carries all its lanes', () => {
    const store = new JotEditorStore();
    store.loadSource(
      parse(
        '{{ time: "4/4", instrumentMapping: { s:{name:"Snare"}, k:{name:"Kick"} } }} ' +
          '| (s k s) . . . |'
      )
    );
    const bracketBar = store
      .structural!.barsForLane('s')
      .bars.find((b) => b.tupletSpans.length > 0)!;
    expect([...bracketBar.tupletSpans[0].lanes].sort()).toEqual(['k', 's']);
  });

  it('moving a hi-hat onto the kick lane re-homes it to the kick layer', () => {
    const { store, selectionPresenter, presenter } = setupMulti();
    const hBar = store.structural!.barsForLane('h').bars.find((b) => b.tracks['h']?.notes.length)!;
    const h0 = hBar.tracks['h'].notes[0];
    selectionPresenter.replace(h0);
    presenter.moveSelection(h0, 0, () => 'k');
    // The moved note now plays on the kick lane and carries the kick layer.
    expect(noteLayerIds(store, 'k')).toEqual([
      store.structural!.ownerLayerFor('k'),
      store.structural!.ownerLayerFor('k'),
    ]);
  });
});
