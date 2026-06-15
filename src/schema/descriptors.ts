import type { z, ZodType } from 'zod';

/**
 * CRDT-aware schema descriptors. A schema is defined once with these
 * factories and drives three things: the extracted TypeScript type
 * ({@link Infer}), the observable MobX node tree, and the backing Loro
 * doc. Every descriptor carries a {@link KIND} brand so the engine, and
 * `record()`'s normalization, can tell a descriptor apart from a bare
 * Zod leaf schema.
 *
 * Governing rule: inside a `record`, a bare Zod type is one LWW register
 * (the whole value merges atomically); you reach for a container factory
 * (`idMap` / `movableList` / `list` / `text` / `counter`) only when you
 * want sub-element collaboration.
 */
export const KIND = Symbol('crdtKind');

// These are `interface`s, not type aliases: the descriptor shapes are
// mutually recursive (a record holds descriptors; an idMap's value is a
// descriptor), and only interfaces tolerate that lazy self-reference
// without TS2456. `Descriptor` stays a union alias over them.

/** A bare LWW-register leaf, holding a Zod schema for inference + defaults. */
export interface RegDescriptor<T extends ZodType = ZodType> {
  [KIND]: true;
  kind: 'reg';
  schema: T;
}

export interface RecordDescriptor<
  F extends Record<string, ZodType | Descriptor> = Record<string, ZodType | Descriptor>,
> {
  [KIND]: true;
  kind: 'record';
  /** Normalized fields; every bare Zod value has been wrapped in `reg`. */
  fields: { [K in keyof F]: F[K] extends Descriptor ? F[K] : RegDescriptor<Extract<F[K], ZodType>> };
}

export interface IdMapDescriptor<V extends Descriptor = Descriptor> {
  [KIND]: true;
  kind: 'idMap';
  value: V;
}

export interface MovableListDescriptor<V extends Descriptor = Descriptor> {
  [KIND]: true;
  kind: 'movableList';
  value: V;
}

/**
 * A discriminated union of record variants, selected at runtime by the
 * `discriminant` field (default `'kind'`). Each variant must carry that
 * field as a literal matching its key. The active variant is fixed at
 * creation, changing a value's kind means deleting and re-adding it.
 */
export interface UnionDescriptor<
  V extends Record<string, RecordDescriptor> = Record<string, RecordDescriptor>,
> {
  [KIND]: true;
  kind: 'union';
  discriminant: string;
  variants: V;
}

/**
 * A descriptor resolved on demand, for recursive schemas (a value that
 * contains values of its own type). The `resolve` thunk defers the
 * self-reference past the cyclic `const` initialization; the recursive
 * type itself needs an explicit annotation to break the cycle (like Zod's
 * `z.lazy`).
 */
export interface LazyDescriptor<D extends Descriptor = Descriptor> {
  [KIND]: true;
  kind: 'lazy';
  resolve: () => D;
}

export type Descriptor =
  | RegDescriptor
  | RecordDescriptor
  | IdMapDescriptor
  | MovableListDescriptor
  | UnionDescriptor
  | LazyDescriptor;

export function isDescriptor(v: unknown): v is Descriptor {
  return typeof v === 'object' && v !== null && KIND in v;
}

/** Internal: wrap a bare Zod leaf as an LWW register. Never written by
 *  hand, `record()` injects it for any field that isn't a descriptor. */
export function reg<T extends ZodType>(schema: T): RegDescriptor<T> {
  return { [KIND]: true, kind: 'reg', schema };
}

export function record<F extends Record<string, ZodType | Descriptor>>(
  shape: F
): RecordDescriptor<F> {
  const fields = {} as Record<string, Descriptor>;
  for (const key of Object.keys(shape)) {
    const value = shape[key];
    fields[key] = isDescriptor(value) ? value : reg(value);
  }
  return { [KIND]: true, kind: 'record', fields } as RecordDescriptor<F>;
}

export function idMap<V extends Descriptor>(value: V): IdMapDescriptor<V> {
  return { [KIND]: true, kind: 'idMap', value };
}

export function union<V extends Record<string, RecordDescriptor>>(
  variants: V,
  discriminant: string = 'kind'
): UnionDescriptor<V> {
  return { [KIND]: true, kind: 'union', discriminant, variants };
}

/** A recursive reference. Annotate the recursive schema's descriptor type
 *  explicitly (`const X: XDesc = union(...)`) so the `() => X` thunk's
 *  return type breaks the cyclic-initializer error. */
export function lazy<D extends Descriptor = Descriptor>(resolve: () => D): LazyDescriptor<D> {
  return { [KIND]: true, kind: 'lazy', resolve };
}

// ---------- Surface types ----------

/**
 * Read/write surface an `idMap` field projects to. Deliberately a narrow,
 * honest interface rather than the full `Map`, every member here is
 * actually implemented by the engine, so the types can't promise an
 * iteration method that throws at runtime. Iteration reflects live
 * membership (MobX-observable).
 */
export interface ReactiveMap<T> {
  readonly size: number;
  has(id: string): boolean;
  get(id: string): T | undefined;
  /** Create or replace the keyed child from a plain value object. */
  set(id: string, value: T): void;
  delete(id: string): void;
  keys(): IterableIterator<string>;
  values(): IterableIterator<T>;
  entries(): IterableIterator<[string, T]>;
  forEach(cb: (value: T, id: string, map: ReactiveMap<T>) => void): void;
  [Symbol.iterator](): IterableIterator<[string, T]>;
}

/**
 * Read/write surface a `movableList` field projects to. Ordered; `move`
 * is cycle-safe under concurrency. Like {@link ReactiveMap}, a narrow
 * interface whose every member the engine actually implements. Iteration
 * and `length` reflect live order (MobX-observable).
 */
export interface ReactiveList<T> {
  readonly length: number;
  at(index: number): T | undefined;
  /** Append a new entry built from a plain value object. */
  push(value: T): void;
  insert(index: number, value: T): void;
  delete(index: number): void;
  move(from: number, to: number): void;
  forEach(cb: (value: T, index: number, list: ReactiveList<T>) => void): void;
  [Symbol.iterator](): IterableIterator<T>;
}

// ---------- Type extraction ----------

/** Infer the plain TypeScript shape a descriptor models. `idMap` → a
 *  keyed `Map`, `movableList` → an array, `record` → an object, `reg` →
 *  the leaf Zod type. A bare Zod field is treated as its inferred type
 *  (it gets normalized to `reg` at runtime). */
// Recursion fuel: a precise schema terminates well before the cap (via the
// `ReactiveMap`/`ReactiveList` interface boundary), but the *broad*
// `Descriptor`, which TS expands in generic contexts now that `union`/
// `lazy` are members, needs a floor or it instantiates forever.
type Depths = [never, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

export type Infer<D, N extends number = 15> =
  N extends 0 ? unknown
  : D extends RegDescriptor<infer T> ? z.infer<T>
  : D extends RecordDescriptor<infer F> ? InferRecord<F, N>
  : D extends IdMapDescriptor<infer V> ? ReactiveMap<Infer<V, Depths[N]>>
  : D extends MovableListDescriptor<infer V> ? ReactiveList<Infer<V, Depths[N]>>
  : D extends UnionDescriptor<infer V> ? Infer<V[keyof V], Depths[N]>
  : D extends LazyDescriptor<infer R> ? Infer<R, Depths[N]>
  : never;

type InferField<X, N extends number> =
  X extends Descriptor ? Infer<X, N> : X extends ZodType ? z.infer<X> : never;

/** Project a record's fields, making any field whose value admits
 *  `undefined` (an optional Zod leaf) an optional *key*, matching Zod's
 *  own `infer`, rather than a required key of `T | undefined`. */
type InferRecord<F, N extends number> = Prettify<
  MakeUndefinedOptional<{ [K in keyof F]: InferField<F[K], Depths[N]> }>
>;

type MakeUndefinedOptional<T> =
  { [K in keyof T as undefined extends T[K] ? K : never]?: Exclude<T[K], undefined> } & {
    [K in keyof T as undefined extends T[K] ? never : K]: T[K];
  };

type Prettify<T> = { [K in keyof T]: T[K] } & {};

/**
 * Plain-data shape for deep-initializing a doc. Like {@link Infer} but the
 * live collection surfaces are plain JS: an `idMap` is a record keyed by id
 * and a `movableList` is an array, i.e. you pass the *data*, not the
 * `ReactiveMap`/`ReactiveList` you'd read back. Records and leaves are the
 * same as `Infer`.
 */
export type Init<D, N extends number = 15> =
  N extends 0 ? unknown
  : D extends RegDescriptor<infer T> ? z.infer<T>
  : D extends RecordDescriptor<infer F> ? InitRecord<F, N>
  : D extends IdMapDescriptor<infer V> ? InitMap<Init<V, Depths[N]>>
  : D extends MovableListDescriptor<infer V> ? Init<V, Depths[N]>[]
  : D extends UnionDescriptor<infer V> ? Init<V[keyof V], Depths[N]>
  : D extends LazyDescriptor<infer R> ? Init<R, Depths[N]>
  : never;

/** Interface (not `Record<…>`) so the value type defers expansion, keeps
 *  the recursive `Init` from eagerly instantiating to infinity on the
 *  generic descriptor, the same way `ReactiveMap` does for `Infer`. */
interface InitMap<T> {
  [id: string]: T;
}

type InitField<X, N extends number> =
  X extends Descriptor ? Init<X, N> : X extends ZodType ? z.infer<X> : never;

// Every field is optional for initialization: you seed the data you have
// (a missing scalar is just unset; a missing collection starts empty).
type InitRecord<F, N extends number> = Prettify<{ [K in keyof F]?: InitField<F[K], Depths[N]> }>;

export function movableList<V extends Descriptor>(value: V): MovableListDescriptor<V> {
  return { [KIND]: true, kind: 'movableList', value };
}
