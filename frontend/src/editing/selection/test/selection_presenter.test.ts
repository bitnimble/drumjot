import { describe, expect, it } from 'bun:test';
import type { StructNote } from 'src/editing/structure/structure_store';
import { SelectionStore } from 'src/editing/selection/selection';
import { SelectionPresenter } from 'src/editing/selection/selection_presenter';

/** Minimal StructNote stand-in; only identity matters to the selection logic. */
function note(id: string): StructNote {
  return { id, lane: 'h', beat: 0, duration: 0.25, modifiers: [], roll: false, straight: true };
}

function setup(notes: StructNote[]) {
  const store = new SelectionStore();
  const presenter = new SelectionPresenter(store, () => notes);
  return { store, presenter };
}

const ids = (store: SelectionStore) => [...store.selectedNotes].map((n) => n.id).sort();

describe('SelectionPresenter, Explorer semantics', () => {
  it('replace selects exactly one note and sets the pivot', () => {
    const ns = [note('a'), note('b'), note('c')];
    const { store, presenter } = setup(ns);
    presenter.replace(ns[1]);
    expect(ids(store)).toEqual(['b']);
    expect(store.anchor).toBe(ns[1]);
  });

  it('ctrl-toggle adds then removes an individual note', () => {
    const ns = [note('a'), note('b'), note('c')];
    const { store, presenter } = setup(ns);
    presenter.replace(ns[0]);
    presenter.toggle(ns[2]);
    expect(ids(store)).toEqual(['a', 'c']);
    presenter.toggle(ns[2]);
    expect(ids(store)).toEqual(['a']);
  });

  it('shift-extend selects the contiguous run from the pivot', () => {
    const ns = [note('a'), note('b'), note('c'), note('d')];
    const { store, presenter } = setup(ns);
    presenter.replace(ns[0]);
    presenter.extendTo(ns[2]);
    expect(ids(store)).toEqual(['a', 'b', 'c']);
  });

  it('re-shift-extend recomputes the range from the same pivot', () => {
    const ns = [note('a'), note('b'), note('c'), note('d')];
    const { store, presenter } = setup(ns);
    presenter.replace(ns[0]);
    presenter.extendTo(ns[3]); // a..d
    presenter.extendTo(ns[1]); // recompute a..b
    expect(ids(store)).toEqual(['a', 'b']);
  });

  it('shift-extend works backwards from the pivot', () => {
    const ns = [note('a'), note('b'), note('c'), note('d')];
    const { store, presenter } = setup(ns);
    presenter.replace(ns[2]);
    presenter.extendTo(ns[0]);
    expect(ids(store)).toEqual(['a', 'b', 'c']);
  });

  it('mixed: shift-range, ctrl-deselect middle, shift further down', () => {
    const ns = [note('a'), note('b'), note('c'), note('d'), note('e')];
    const { store, presenter } = setup(ns);
    presenter.replace(ns[0]);
    presenter.extendTo(ns[2]); // a,b,c
    presenter.toggle(ns[1]); // ctrl-deselect b -> a,c ; pivot now b ; base = {a,c}
    presenter.extendTo(ns[4]); // base ∪ range(b..e) = a,c ∪ b,c,d,e
    expect(ids(store)).toEqual(['a', 'b', 'c', 'd', 'e']);
  });

  it('extendTo with no pivot falls back to replace', () => {
    const ns = [note('a'), note('b')];
    const { store, presenter } = setup(ns);
    presenter.extendTo(ns[1]);
    expect(ids(store)).toEqual(['b']);
  });

  it('clear empties the selection', () => {
    const ns = [note('a')];
    const { store, presenter } = setup(ns);
    presenter.replace(ns[0]);
    presenter.clear();
    expect(ids(store)).toEqual([]);
    expect(store.state).toBeUndefined();
  });
});
