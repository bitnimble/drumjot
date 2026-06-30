/**
 * The per-document registry of derived-field implementations.
 *
 * Derived fields are declared first-class in a schema `record` via
 * `derived(slot)` (see {@link DerivedDescriptor}); the *implementation* of each
 * is installed at runtime by the owning feature presenter through this registry.
 * The registry is the standalone mediator that decouples producers from
 * consumers: a presenter calls `registry.<field>.define(() => …)`, the reactive
 * model resolves a read of `jot.<field>` back to that implementation by the
 * slot's identity, and consumers never import the producing presenter.
 *
 * Lifecycle is per-document: a fresh registry is created alongside each loaded
 * jot and injected into both the model build and the presenters. Construction
 * order is irrelevant (registration stores a closure; reads are lazy), and an
 * async-loaded module can install late. The only failure path is a read of a
 * field whose implementation was never installed, which throws at the point of
 * read; a double `define` throws immediately.
 */
import { computed, type IComputedValue, observable, runInAction } from 'mobx';
import { computedFn } from 'mobx-utils';
import type { AnySlot, DerivedFnSlot, DerivedSlot } from './descriptors';

/** Symbol-keyed resolver on the registry surface, so a field literally named
 *  `resolve` can never collide with it. The reactive model uses this to map a
 *  descriptor's slot to its live implementation. */
export const RESOLVE = Symbol('derivedResolve');

/** The read side the reactive model uses, erased of the slot's value type. */
export interface DerivedReadSlot {
  read(arg?: unknown): unknown;
}

/** The minimal capability the reactive-doc build path needs: map a slot to its
 *  live implementation. The full registry surface is a superset of this. */
export interface DerivedResolver {
  [RESOLVE](slot: AnySlot): DerivedReadSlot;
}

/** Public install surface for a nullary derived field. */
export interface ValueSlotApi<T> {
  define(impl: () => T): void;
}

/** Public install surface for a keyed (function-valued) derived field. */
export interface FnSlotApi<A, T> {
  define(impl: (arg: A) => T): void;
}

class ValueSlot<T> implements ValueSlotApi<T> {
  private readonly impl = observable.box<(() => T) | undefined>(undefined, { deep: false });
  private memo: IComputedValue<T> | undefined;

  constructor(private readonly name: string) {}

  define(impl: () => T): void {
    if (this.impl.get() != null) {
      throw new Error(`derived field '${this.name}' is already defined`);
    }
    runInAction(() => this.impl.set(impl));
  }

  // Reads the observable `impl` box BEFORE throwing so a consumer that read
  // too early (and threw) re-runs once the implementation is installed.
  read(): T {
    const impl = this.impl.get();
    if (impl == null) {
      throw new Error(`derived field '${this.name}' was read before it was defined`);
    }
    if (this.memo == null) this.memo = computed(() => this.impl.get()!());
    return this.memo.get();
  }
}

class FnSlot<A, T> implements FnSlotApi<A, T> {
  private readonly impl = observable.box<((arg: A) => T) | undefined>(undefined, { deep: false });
  private memo: ((arg: A) => T) | undefined;

  constructor(private readonly name: string) {}

  define(impl: (arg: A) => T): void {
    if (this.impl.get() != null) {
      throw new Error(`derived field '${this.name}' is already defined`);
    }
    runInAction(() => this.impl.set(impl));
  }

  read(arg: A): T {
    const impl = this.impl.get();
    if (impl == null) {
      throw new Error(`derived field '${this.name}' was read before it was defined`);
    }
    if (this.memo == null) this.memo = computedFn((a: A) => this.impl.get()!(a));
    return this.memo(arg);
  }
}

/** Mint a nullary derived-field slot. Its type `T` is the single source of
 *  truth for the model surface and `define`. */
export function slot<T>(): DerivedSlot<T> {
  return { derivedKind: 'value', name: '<unnamed>' };
}

/** Mint a keyed (function-valued) derived-field slot. */
export function fnSlot<A, T>(): DerivedFnSlot<A, T> {
  return { derivedKind: 'fn', name: '<unnamed>' };
}

/** Stamp each slot with its declared field name (for error messages) and return
 *  the declaration. The declaration is the static set of slot identities the
 *  schema references and `createDerivedRegistry` instantiates. */
export function declareDerived<D extends Record<string, AnySlot>>(decl: D): D {
  for (const key of Object.keys(decl)) decl[key].name = key;
  return decl;
}

type SlotApi<S> =
  S extends DerivedFnSlot<infer A, infer T> ? FnSlotApi<A, T>
  : S extends DerivedSlot<infer T> ? ValueSlotApi<T>
  : never;

/** The per-document registry instance: a typed `define` surface per declared
 *  field, plus the symbol-keyed resolver the reactive model uses. */
export type DerivedRegistry<D extends Record<string, AnySlot>> = {
  [K in keyof D]: SlotApi<D[K]>;
} & DerivedResolver;

/**
 * Instantiate a per-document registry from a static declaration. Each declared
 * slot gets a fresh live implementation slot (its own observable + lazy
 * memoised computed), so a reload's presenters install into a clean registry
 * with no double-define clash and no stale closure over the previous document.
 */
export function createDerivedRegistry<D extends Record<string, AnySlot>>(decl: D): DerivedRegistry<D> {
  const bySlot = new Map<AnySlot, DerivedReadSlot>();
  const surface: Record<string, unknown> & { [RESOLVE]?: DerivedResolver[typeof RESOLVE] } = {};
  for (const key of Object.keys(decl)) {
    const declared = decl[key];
    const live = declared.derivedKind === 'fn' ? new FnSlot(key) : new ValueSlot(key);
    surface[key] = live;
    // The one cast in the mechanism: erase the slot's value type for the
    // type-agnostic resolver the model reads through. The public `define`
    // surface stays fully typed via `SlotApi`.
    bySlot.set(declared, live as unknown as DerivedReadSlot);
  }
  surface[RESOLVE] = (slot: AnySlot): DerivedReadSlot => {
    const live = bySlot.get(slot);
    if (live == null) {
      throw new Error(`derived: slot '${slot.name}' is not part of this registry`);
    }
    return live;
  };
  return surface as DerivedRegistry<D>;
}
