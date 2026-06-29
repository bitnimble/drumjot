# Schema-declared derived state via an external registry

## Problem

The mutable jot (a schema-driven MobX façade over Loro, `src/schema/`) declares
only **stored** fields. All derived state lives one layer up as MobX `computed`
getters on per-domain stores/presenters. As the product grows, presenters
increasingly read derived state that logically belongs to another domain, by
importing that domain's presenter directly:

- `PlaybackStore.epochs` reads `TempoPresenter.timeline`
- playback `events.ts` reads `StructuralPresenter.musicalLayers` / `tempoSource`
- `WaveformChunker`, `TempoEditPresenter` read `StructuralPresenter.tempoSource` / `pxPerBeat`

This cross-wiring couples consumers to producer presenters and grows
combinatorially.

## Goal

Let derived/computed state be **declared first-class in the schema tree**, with
the *implementation* still supplied by the owning feature presenter. A presenter
"extends" the mutable document with computed properties that any other presenter
can read off the model; `jot.tempoTimeline`; without importing or depending on
the producing presenter. Reading a derived field whose implementation was never
installed crashes at the point of read.

### Non-goals

- Not a wholesale migration of all store/presenter computeds. Only state
  consumed **across** domains is promoted (opt-in). Intra-domain computeds stay
  local on their store/presenter.
- Derived values are **not** Loro-backed and never enter the CRDT container path.
- No runtime validation of derived values (the Zod/type is a type carrier only).

## Design

### Slots are the single source of truth for type + identity

A static declaration, co-located with the schema. Each slot is declared once
with its type; `slot<T>()` / `fnSlot<A, T>()` mint a **unique identity object**,
so two slots never collide even with identical field names anywhere in the tree.

```ts
// src/schema/derived_fields.ts
export const Derived = declareDerived({
  tempoTimeline: slot<JotTimeline>(),
  dominantBpm:   slot<number | undefined>(),
  barsForLane:   fnSlot<string, LaneBars>(),
});
```

The registry namespace is **flat**; one entry per slot; and decoupled from
where the slot is referenced in the schema *tree*. That is what lets the same
field name appear in multiple subtrees (each pointing at a distinct slot), or one
slot be referenced from several tree positions sharing a single implementation.

### A single `derived(...)` descriptor takes the slot directly

The schema references the slot directly, at any depth in the tree. The slot
already encodes its own arity, so one `derived(...)` factory covers both nullary
and keyed slots; there is no separate `derivedFn`.

```ts
// src/schema/schema.ts
JotSchema = record({
  bpm: z.number(),
  bars: movableList(BarSchema),
  // …stored fields…
  tempoTimeline: derived(Derived.tempoTimeline),   // nullary
  dominantBpm:   derived(Derived.dominantBpm),
  barsForLane:   derived(Derived.barsForLane),      // keyed
});
```

`derived(slot)` returns a `DerivedDescriptor` carrying the slot reference (for
runtime resolution) and, through the slot's type parameter, the value type (for
the type mapper). At runtime the descriptor reads the slot's arity marker to
build the correct getter shape.

### Type flow (no `.z`, no `unknown` at the public surface)

The model-type mapper (`InferModel<typeof JotSchema>` → `MutableJot`) recognizes
a `DerivedDescriptor` and unwraps the underlying slot via conditional types:

- `DerivedDescriptor` wrapping `DerivedSlot<T>`     → readonly `T` property
- `DerivedDescriptor` wrapping `DerivedFnSlot<A, T>` → `(arg: A) => T` method

Single-sourced: `slot<JotTimeline>()` → `derived(slot)` → `MutableJot` property
typed `JotTimeline`. The only cast in the entire mechanism is inside the
registry's private storage map; every public API surface is fully typed.

### Two surfaces: static `Derived`, per-document `registry`

The schema is built once at module load, but implementations are
per-loaded-document. So the declaration and the live instance are separate
objects sharing slot identity + type:

- **`Derived`** (static, module scope), types + identities. The schema
  references `Derived.X`. Must be static.
- **`registry = createDerivedRegistry(Derived)`** (per-document instance), same
  keys, each now a live slot with `define` (install) and `get` (read). Created in
  the composition root, injected into both the model build and the presenters.
  Per-document so a jot reload gets fresh implementations: no double-define clash,
  no stale closure over a disposed document, no cross-document key collision.

### Install + read (typed)

```ts
interface DerivedSlot<T> {
  define(impl: () => T): void;   // duplicate define throws
  get(): T;                      // unregistered throws; lazily wraps impl in computed();
                                 // observes membership so a late define re-triggers a read that threw
}
interface DerivedFnSlot<A, T> {
  define(impl: (arg: A) => T): void;
  get(arg: A): T;                // backed by computedFn for per-argument memoization
}
```

Presenter side, fully type-checked against the slot's `T`:

```ts
registry.tempoTimeline.define(() => buildTimeline(this.structural));   // () => JotTimeline ✓
registry.barsForLane.define((lane) => …);                             // (lane: string) => LaneBars ✓
```

`get` is internal. Consumers read off the model, `jot.tempoTimeline`,
`jot.barsForLane('snare')`, and the model getter calls `slot.get()` underneath.

### Model integration

In `reactive_doc.ts`, when building a record node, a `derived` field does **not**
create a Loro container. Instead it defines a getter that resolves through the
per-document registry by the descriptor's slot identity:

```ts
// nullary
Object.defineProperty(model, key, { get: () => registry.resolve(slot).get(), enumerable: true });
// keyed: getter returns the memoized function
Object.defineProperty(model, key, { get: () => (arg) => registry.resolve(slot).get(arg), enumerable: true });
```

The registry is threaded into the same composition root that constructs the
`ReactiveDoc` and runs `buildJotPeers` (`src/editing/jot_editor_store.ts`).

### Reactivity

Each live slot holds an `observable.box<impl | undefined>` plus a lazily-built
`computed` (or `computedFn`). `get` reads the box (registering the observation)
before throwing, so a consumer that read before `define`, and threw, re-runs
once the impl is installed. This is also what makes the lazy-async-module case
safe: a module that installs *and* consumes its own derived fields arrives as a
unit, so the consumer never observes the unregistered state in practice; if it
does, it self-heals on define rather than staying broken.

### Error semantics

- **Unregistered read → throw** at the point of read. The only failure path.
- **Double `define` → throw** immediately (a slot has exactly one owner).
- **No eager seal.** Construction order is not guaranteed (presenters may
  construct before the document; unit tests vary; async modules install late), so
  there is no post-construction assertion that every slot is filled. Lazy
  crash-on-read is the sole backstop and fires only for a genuinely-unregistered
  read.

### Layer rules

- **Presenters call `define`; stores never touch the registry.** The
  single-writer rule stays intact, installing an implementation is orchestration,
  which lives on presenters. The impl closure may delegate to a store computed or
  inline the derivation in the `define(() => …)` body.
- **Opt-in promotion.** Only cross-domain derived state is declared in `Derived`
  and installed. The `Derived` declaration thus documents exactly the
  cross-cutting contract of the document.
- **Pull-down.** A store computed that exists *only* to feed a now-promoted
  derived field moves out of the store into the owning presenter (inlined into the
  `define` body when small), shrinking stores back toward pure data.

## First adopter: TempoPresenter

`TempoPresenter` becomes a registrar. In its constructor it calls `define` for
the cross-domain fields it owns:

- `tempoTimeline`  ← `buildTimeline(this.structural)` (currently `TempoPresenter.timeline`)
- `dominantBpm` / `dominantTime` ← `pickDominantBpmAndTime(...)`
- `tempoSource`    ← currently on `StructuralPresenter`, consumed by playback /
  tempo-edit / waveform; pulled down into the tempo domain's `define` body

Consumers (`PlaybackStore.epochs`, playback `events.ts`, `WaveformChunker`,
`TempoEditPresenter`) stop importing `TempoPresenter` / `StructuralPresenter` for
these and read `jot.tempoTimeline` etc. `TempoPresenter` is still instantiated
for its registration side-effect, but no consumer needs a typed handle to it; that is the decoupling win.

`musicalLayers` is the strongest follow-on candidate; migrate tempo cleanly
first, then promote others incrementally.

## Files

- `src/schema/derived_fields.ts` (new), `slot` / `fnSlot` / `declareDerived`,
  the `DerivedSlot` / `DerivedFnSlot` types, the static `Derived` declaration.
- `src/schema/derived_registry.ts` (new), `createDerivedRegistry`, the live
  per-document slot instances (`define` / `get`, observable box + lazy
  computed/computedFn), duplicate + unregistered guards.
- `src/schema/descriptors.ts`, `derived(slot)` descriptor factory + the
  `DerivedDescriptor` kind; `record()` normalization passes it through unchanged.
- `src/schema/reactive_doc.ts`, recognize `DerivedDescriptor` in the build path:
  skip container creation, define a registry-backed getter; thread the registry
  reference in.
- `src/schema/` model-type mapper, conditional-type unwrap of
  `DerivedSlot` / `DerivedFnSlot` into property / method types on `MutableJot`.
- `src/schema/schema.ts`, declare the initial cross-domain derived fields.
- `src/editing/jot_editor_store.ts`, create the registry at the composition
  root, inject into the `ReactiveDoc` build and `buildJotPeers`.
- `src/editing/playback/tempo_presenter.ts`, install the tempo-domain derived
  fields via `define`.
- `src/editing/structure/structural_presenter.ts`, move `tempoSource` down into
  the tempo `define`; remove the store computed if nothing else consumes it.
- Consumers above, switch reads to the model getters.

## Testing

- **`DerivedRegistry` unit tests:** read-after-define and read-before-define
  (reactivity via a MobX `autorun` that re-runs once `define` lands),
  duplicate-define throws, unregistered read throws, `fnSlot` per-argument
  memoization.
- **Schema/model test:** a `derived` field reads through to its installed impl
  and reacts to upstream observable changes; an unregistered read throws; a slot
  referenced from two tree positions resolves to one impl.
- **Regression:** existing tempo/playback unit + e2e suites cover the migration.
  Run `bun run typecheck` / `bun run test`, and `bun run e2e` (any `src/**`
  change).
