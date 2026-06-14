import { describe, expect, it } from 'bun:test';
import { autorun, runInAction } from 'mobx';
import { z } from 'zod';
import { idMap, movableList, record } from 'src/schema/descriptors';
import { createReactiveDoc } from 'src/schema/reactive_doc';

const Note = record({ pitch: z.string(), beat: z.number() });

describe('register fields', () => {
  it('reads the hydrated default and reflects a write', () => {
    const Song = record({ title: z.string() });
    const { model } = createReactiveDoc(Song, { title: 'untitled' });
    expect(model.title).toBe('untitled');
    runInAction(() => {
      model.title = 'breakbeat';
    });
    expect(model.title).toBe('breakbeat');
  });

  it('is observable: a reaction fires when a register changes', () => {
    const Song = record({ title: z.string() });
    const { model } = createReactiveDoc(Song, { title: 'a' });
    const seen: string[] = [];
    const dispose = autorun(() => seen.push(model.title));
    runInAction(() => {
      model.title = 'b';
    });
    dispose();
    expect(seen).toEqual(['a', 'b']);
  });

  it('writes through to the backing Loro doc', () => {
    const Song = record({ title: z.string() });
    const { model, doc } = createReactiveDoc(Song, { title: 'a' });
    runInAction(() => {
      model.title = 'c';
    });
    expect(doc.getMap('root').get('title')).toBe('c');
  });
});

describe('idMap', () => {
  it('materializes a keyed child and exposes its fields', () => {
    const Song = record({ notes: idMap(Note) });
    const { model } = createReactiveDoc(Song);
    runInAction(() => {
      model.notes.set('n1', { pitch: 'h', beat: 0 });
    });
    expect(model.notes.size).toBe(1);
    expect(model.notes.get('n1')!.pitch).toBe('h');
    expect(model.notes.get('n1')!.beat).toBe(0);
  });

  it('child field writes are observable', () => {
    const Song = record({ notes: idMap(Note) });
    const { model } = createReactiveDoc(Song);
    runInAction(() => {
      model.notes.set('n1', { pitch: 'h', beat: 0 });
    });
    const note = model.notes.get('n1')!;
    const seen: string[] = [];
    const dispose = autorun(() => seen.push(note.pitch));
    runInAction(() => {
      note.pitch = 'rd';
    });
    dispose();
    expect(seen).toEqual(['h', 'rd']);
  });

  it('membership is observable and delete removes the child', () => {
    const Song = record({ notes: idMap(Note) });
    const { model } = createReactiveDoc(Song);
    const sizes: number[] = [];
    const dispose = autorun(() => sizes.push(model.notes.size));
    runInAction(() => {
      model.notes.set('n1', { pitch: 'h', beat: 0 });
    });
    runInAction(() => {
      model.notes.delete('n1');
    });
    dispose();
    expect(sizes).toEqual([0, 1, 0]);
    expect(model.notes.get('n1')).toBeUndefined();
  });
});

describe('movableList', () => {
  const Bars = record({ bars: movableList(record({ n: z.number() })) });

  it('push appends; at/length read; membership is observable', () => {
    const { model } = createReactiveDoc(Bars);
    const lengths: number[] = [];
    const dispose = autorun(() => lengths.push(model.bars.length));
    runInAction(() => {
      model.bars.push({ n: 1 });
    });
    runInAction(() => {
      model.bars.push({ n: 2 });
    });
    dispose();
    expect(lengths).toEqual([0, 1, 2]);
    expect(model.bars.at(0)!.n).toBe(1);
    expect(model.bars.at(1)!.n).toBe(2);
  });

  it('move reorders and preserves child identity', () => {
    const { model } = createReactiveDoc(Bars);
    runInAction(() => {
      model.bars.push({ n: 1 });
      model.bars.push({ n: 2 });
      model.bars.push({ n: 3 });
    });
    const first = model.bars.at(0)!;
    runInAction(() => {
      model.bars.move(0, 2);
    });
    expect([...model.bars].map((b) => b.n)).toEqual([2, 3, 1]);
    expect(model.bars.at(2)).toBe(first);
  });

  it('delete removes an item', () => {
    const { model } = createReactiveDoc(Bars);
    runInAction(() => {
      model.bars.push({ n: 1 });
      model.bars.push({ n: 2 });
    });
    runInAction(() => {
      model.bars.delete(0);
    });
    expect([...model.bars].map((b) => b.n)).toEqual([2]);
  });

  it('child field writes are observable', () => {
    const { model } = createReactiveDoc(Bars);
    runInAction(() => {
      model.bars.push({ n: 1 });
    });
    const bar = model.bars.at(0)!;
    const seen: number[] = [];
    const dispose = autorun(() => seen.push(bar.n));
    runInAction(() => {
      bar.n = 9;
    });
    dispose();
    expect(seen).toEqual([1, 9]);
  });
});

describe('deep initialization', () => {
  it('initializes nested records and idMap entries from a plain object', () => {
    const Doc = record({
      meta: record({ title: z.string() }),
      notes: idMap(Note),
    });
    const { model } = createReactiveDoc(Doc, {
      meta: { title: 'song' },
      notes: { n1: { pitch: 'h', beat: 0 }, n2: { pitch: 'k', beat: 1 } },
    });
    expect(model.meta.title).toBe('song');
    expect(model.notes.size).toBe(2);
    expect(model.notes.get('n2')!.beat).toBe(1);
  });
});

describe('convergence', () => {
  it('a remote add materializes in the local model after import', () => {
    const Song = record({ notes: idMap(Note) });
    const a = createReactiveDoc(Song);
    const b = createReactiveDoc(Song);
    runInAction(() => {
      a.model.notes.set('n1', { pitch: 'h', beat: 0 });
    });
    b.doc.import(a.doc.export({ mode: 'update' }));
    expect(b.model.notes.size).toBe(1);
    expect(b.model.notes.get('n1')!.pitch).toBe('h');
  });

  it('concurrent adds in different docs both survive the merge', () => {
    const Song = record({ notes: idMap(Note) });
    const a = createReactiveDoc(Song);
    const b = createReactiveDoc(Song);
    runInAction(() => {
      a.model.notes.set('na', { pitch: 'h', beat: 0 });
    });
    runInAction(() => {
      b.model.notes.set('nb', { pitch: 'k', beat: 1 });
    });
    a.doc.import(b.doc.export({ mode: 'update' }));
    b.doc.import(a.doc.export({ mode: 'update' }));
    expect(a.model.notes.size).toBe(2);
    expect(b.model.notes.size).toBe(2);
    expect(a.model.notes.get('nb')!.pitch).toBe('k');
    expect(b.model.notes.get('na')!.pitch).toBe('h');
  });
});

describe('robustness / API surface', () => {
  it('rejects a movableList whose entry type is not a record', () => {
    const Bad = record({ bars: movableList(idMap(Note)) });
    expect(() => createReactiveDoc(Bad)).toThrow(/movableList/i);
  });

  it('seeds a register field whose value is an array (not misread as a container)', () => {
    const WithMods = record({ mods: z.array(z.string()) });
    const { model } = createReactiveDoc(WithMods, { mods: ['fl', 'gh'] });
    expect(model.mods).toEqual(['fl', 'gh']);
  });

  it('set replaces the entry, dropping fields absent from the new value', () => {
    const Stickable = record({ pitch: z.string(), sticking: z.string().optional() });
    const Song = record({ notes: idMap(Stickable) });
    const { model } = createReactiveDoc(Song);
    runInAction(() => {
      model.notes.set('n1', { pitch: 'h', sticking: 'r' });
    });
    expect(model.notes.get('n1')!.sticking).toBe('r');
    runInAction(() => {
      model.notes.set('n1', { pitch: 'h' });
    });
    expect(model.notes.get('n1')!.sticking).toBeUndefined();
  });

  it('rejects an idMap whose entry type is not a record', () => {
    const Bad = record({ x: idMap(idMap(Note)) });
    expect(() => createReactiveDoc(Bad)).toThrow(/record/i);
  });

  it('dispose() unregisters every container', () => {
    const Doc = record({ meta: record({ x: z.number() }), notes: idMap(Note) });
    const rd = createReactiveDoc(Doc);
    expect(rd.containerCount()).toBeGreaterThan(0);
    rd.dispose();
    expect(rd.containerCount()).toBe(0);
  });

  it('deleting an idMap entry unregisters its container (no registry leak)', () => {
    const Doc = record({ notes: idMap(Note) });
    const rd = createReactiveDoc(Doc);
    const base = rd.containerCount();
    runInAction(() => {
      rd.model.notes.set('n1', { pitch: 'h', beat: 0 });
    });
    expect(rd.containerCount()).toBe(base + 1);
    runInAction(() => {
      rd.model.notes.delete('n1');
    });
    expect(rd.containerCount()).toBe(base);
  });

  it('supports Map-style iteration over an idMap', () => {
    const Song = record({ notes: idMap(Note) });
    const { model } = createReactiveDoc(Song);
    runInAction(() => {
      model.notes.set('n1', { pitch: 'h', beat: 0 });
      model.notes.set('n2', { pitch: 'k', beat: 1 });
    });
    expect(
      [...model.notes.values()].map((n) => n.pitch).sort()
    ).toEqual(['h', 'k']);
    expect([...model.notes.keys()].sort()).toEqual(['n1', 'n2']);
    expect([...model.notes].map(([id]) => id).sort()).toEqual(['n1', 'n2']);
    const byForEach: string[] = [];
    model.notes.forEach((_n, id) => byForEach.push(id));
    expect(byForEach.sort()).toEqual(['n1', 'n2']);
  });
});
