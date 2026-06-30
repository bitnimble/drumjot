import { describe, expect, it } from 'bun:test';
import { autorun, runInAction } from 'mobx';
import { z } from 'zod';
import { derived, record } from 'src/schema/descriptors';
import { createReactiveDoc } from 'src/schema/reactive_doc';
import { createDerivedRegistry, declareDerived, slot } from 'src/schema/derived_registry';

describe('derived fields through a reactive doc', () => {
  const Decl = declareDerived({ doubled: slot<number>() });
  const Doc = record({ n: z.number(), doubled: derived(Decl.doubled) });

  it('reads through to the registered impl and reacts to stored changes', () => {
    const reg = createDerivedRegistry(Decl);
    const { model } = createReactiveDoc(Doc, { n: 3 }, reg);
    reg.doubled.define(() => model.n * 2);
    const seen: number[] = [];
    const dispose = autorun(() => seen.push(model.doubled));
    expect(seen).toEqual([6]);
    runInAction(() => {
      model.n = 5;
    });
    expect(seen.at(-1)).toBe(10);
    dispose();
  });

  it('throws reading a derived field when no registry was provided', () => {
    const { model } = createReactiveDoc(Doc, { n: 1 });
    expect(() => model.doubled).toThrow(/no derived registry/);
  });

  it('throws reading a derived field whose impl was never installed', () => {
    const reg = createDerivedRegistry(Decl);
    const { model } = createReactiveDoc(Doc, { n: 1 }, reg);
    expect(() => model.doubled).toThrow(/before it was defined/);
  });

  it('one slot referenced from two tree positions resolves to one impl', () => {
    const D = declareDerived({ shared: slot<number>() });
    const Nested = record({
      a: derived(D.shared),
      inner: record({ b: derived(D.shared) }),
    });
    const reg = createDerivedRegistry(D);
    const { model } = createReactiveDoc(Nested, {}, reg);
    reg.shared.define(() => 7);
    expect(model.a).toBe(7);
    expect(model.inner.b).toBe(7);
  });

  it('derived fields are excluded from snapshots', () => {
    const reg = createDerivedRegistry(Decl);
    const doc = createReactiveDoc(Doc, { n: 4 }, reg);
    reg.doubled.define(() => 999);
    const snap = doc.snapshot() as Record<string, unknown>;
    expect(snap.n).toBe(4);
    expect('doubled' in snap).toBe(false);
  });
});
