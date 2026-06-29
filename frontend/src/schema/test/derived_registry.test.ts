import { describe, expect, it } from 'bun:test';
import { autorun, observable, runInAction } from 'mobx';
import {
  createDerivedRegistry,
  declareDerived,
  fnSlot,
  RESOLVE,
  slot,
} from 'src/schema/derived_registry';

describe('DerivedRegistry', () => {
  const Decl = declareDerived({
    answer: slot<number>(),
    scaled: fnSlot<number, number>(),
  });

  it('resolves a defined value', () => {
    const r = createDerivedRegistry(Decl);
    r.answer.define(() => 42);
    expect(r[RESOLVE](Decl.answer).read()).toBe(42);
  });

  it('throws reading before define', () => {
    const r = createDerivedRegistry(Decl);
    expect(() => r[RESOLVE](Decl.answer).read()).toThrow(/before it was defined/);
  });

  it('throws on double define', () => {
    const r = createDerivedRegistry(Decl);
    r.answer.define(() => 1);
    expect(() => r.answer.define(() => 2)).toThrow(/already defined/);
  });

  it('throws resolving a slot from another declaration', () => {
    const r = createDerivedRegistry(Decl);
    const foreign = declareDerived({ x: slot<number>() });
    expect(() => r[RESOLVE](foreign.x)).toThrow(/not part of this registry/);
  });

  it('fnSlot memoizes per argument', () => {
    const r = createDerivedRegistry(Decl);
    let calls = 0;
    r.scaled.define((n) => {
      calls++;
      return n * 2;
    });
    const read = r[RESOLVE](Decl.scaled);
    let a = 0;
    let b = 0;
    // computedFn caches per-arg only while observed in a reactive context.
    const dispose = autorun(() => {
      a = read.read(2) as number;
      b = read.read(2) as number;
    });
    expect([a, b]).toEqual([4, 4]);
    expect(calls).toBe(1);
    dispose();
  });

  it('a reactive read recovers on late define and tracks upstream', () => {
    const r = createDerivedRegistry(Decl);
    const src = observable.box(10);
    const seen: Array<number | string> = [];
    const dispose = autorun(() => {
      try {
        seen.push(r[RESOLVE](Decl.answer).read() as number);
      } catch {
        seen.push('threw');
      }
    });
    expect(seen).toEqual(['threw']);
    r.answer.define(() => src.get() * 2);
    expect(seen.at(-1)).toBe(20);
    runInAction(() => src.set(11));
    expect(seen.at(-1)).toBe(22);
    dispose();
  });
});
