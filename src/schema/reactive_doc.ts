import { LoroDoc, LoroMap, LoroMovableList, type ContainerID } from 'loro-crdt';
import { observable, runInAction } from 'mobx';
import type {
  Descriptor,
  IdMapDescriptor,
  Infer,
  Init,
  MovableListDescriptor,
  ReactiveList,
  ReactiveMap,
  RecordDescriptor,
} from './descriptors';

/** Root container name holding the top-level record. */
export const ROOT = 'root';

export type ReactiveDoc<S extends RecordDescriptor> = {
  /** The deeply-observable MobX projection. Reads/writes are ordinary
   *  property access; writes go Loro-first then commit, and the commit's
   *  synchronous event updates the cache. */
  model: Infer<S>;
  /** The backing Loro doc, the source of truth for merge/persistence.
   *  App code shouldn't touch it; it's the synchroniser. */
  doc: LoroDoc;
  /** Stop listening to Loro events and unregister every node. */
  dispose: () => void;
  /** Number of live containers currently registered. Diagnostic, lets
   *  leak tests assert teardown returns the registry to baseline. */
  containerCount: () => number;
};

/**
 * A node knows how to fold one Loro container diff into its observable
 * surface. Routing is by container id (stable), never by event `path`
 * (a shifting list index). The registry maps every live container id to
 * its node, and child nodes register/unregister themselves as keyed
 * entries come and go.
 */
interface Node {
  /** Fold a map container's diff into the cache (record / idMap nodes). */
  applyMapDiff?(updated: Record<string, unknown>): void;
  /** Reconcile a list container's order/membership (movableList nodes).
   *  Driven off the diff event but reads current state to preserve child
   *  identity across moves. */
  applyListDiff?(): void;
}

type Registry = Map<ContainerID, Node>;

// ---------- Record node ----------

/**
 * Materialized record: register fields are cached scalars (fast plain-JS
 * reads, no WASM crossing); container fields are stable child surfaces
 * built once. The cache is written only by the event handler, so local
 * writes and remote imports share one sync path.
 */
function buildRecordNode(
  desc: RecordDescriptor,
  lmap: LoroMap,
  doc: LoroDoc,
  registry: Registry
): { surface: Record<string, unknown>; dispose: () => void } {
  const cache = observable.map<string, unknown>({}, { deep: false });
  const childSurfaces: Record<string, unknown> = {};
  // Teardown for every container this record owns, so disposing the node
  // (or deleting it as an idMap entry) unregisters its whole subtree, not
  // just its own container id.
  const childDisposers: Array<() => void> = [];
  // The precise mapped `fields` type is for `Infer`; the generic engine
  // iterates them as a plain descriptor map.
  const fields = desc.fields as Record<string, Descriptor>;

  for (const key of Object.keys(fields)) {
    const field = fields[key];
    if (field.kind === 'reg') continue;
    // Container-typed field: ensure its child container exists, build its
    // node now so the surface is stable for the node's lifetime.
    // `ensureMergeable*` (not `getOrCreateContainer`) for schema-fixed
    // keys: two peers that independently build the same skeleton must
    // address the SAME logical container, else a merge mints competing
    // container ids and one peer's subtree is silently dropped.
    if (field.kind === 'record') {
      const child = lmap.ensureMergeableMap(key);
      const built = buildRecordNode(field, child, doc, registry);
      childSurfaces[key] = built.surface;
      childDisposers.push(built.dispose);
    } else if (field.kind === 'idMap') {
      const child = lmap.ensureMergeableMap(key);
      const built = buildIdMapNode(field, child, doc, registry);
      childSurfaces[key] = built.surface;
      childDisposers.push(built.dispose);
    } else if (field.kind === 'movableList') {
      const child = lmap.ensureMergeableMovableList(key);
      const built = buildMovableListNode(field, child, doc, registry);
      childSurfaces[key] = built.surface;
      childDisposers.push(built.dispose);
    } else {
      // Fail fast rather than leave the surface getter returning undefined:
      // an unhandled container kind in a public schema would otherwise be
      // silent data loss. (text/counter land in later cycles.) Unreachable
      // for today's kinds, so `field` narrows to `never`, cast to read it.
      throw new Error(`reactive_doc: container kind '${(field as Descriptor).kind}' is not yet implemented (field '${key}')`);
    }
  }

  const node: Node = {
    applyMapDiff(updated) {
      for (const k of Object.keys(updated)) {
        // Only register fields are cached as scalars; container-field keys
        // carry child snapshots the child node handles itself. Read the
        // authoritative value back so a removed key (e.g. an optional
        // field cleared by a replacing `set`) clears the cache rather than
        // caching a delete sentinel.
        if (fields[k]?.kind !== 'reg') continue;
        const v = lmap.get(k);
        if (v === undefined) cache.delete(k);
        else cache.set(k, v);
      }
    },
  };
  registry.set(lmap.id, node);

  const surface: Record<string, unknown> = {};
  for (const key of Object.keys(fields)) {
    const field = fields[key];
    if (field.kind === 'reg') {
      Object.defineProperty(surface, key, {
        enumerable: true,
        get: () => cache.get(key),
        set: (value: unknown) => {
          lmap.set(key, value as never);
          doc.commit();
        },
      });
    } else {
      Object.defineProperty(surface, key, {
        enumerable: true,
        get: () => childSurfaces[key],
      });
    }
  }

  const dispose = () => {
    registry.delete(lmap.id);
    for (const d of childDisposers) d();
  };
  return { surface, dispose };
}

// ---------- idMap node ----------

/**
 * A keyed collection of child records (e.g. notes by id). Entries are
 * themselves child containers; the observable `entries` map holds their
 * surfaces and is reconciled against the Loro map on every diff.
 */
function buildIdMapNode(
  desc: IdMapDescriptor,
  lmap: LoroMap,
  doc: LoroDoc,
  registry: Registry
): { surface: ObservableIdMap; dispose: () => void } {
  // Entries are record containers; nothing else is buildable as one. Fail
  // fast on an unsupported entry kind rather than mis-cast it below.
  const entryDesc = desc.value;
  if (entryDesc.kind !== 'record') {
    throw new Error(`reactive_doc: idMap value must be a record (got '${entryDesc.kind}')`);
  }

  const entries = observable.map<string, Record<string, unknown>>({}, { deep: false });
  // Per-entry teardown, disposes the entry's whole subtree (its container
  // plus any nested containers the record owns), so a delete leaves no
  // orphaned nodes in the registry.
  const entryDisposers = new Map<string, () => void>();

  const node: Node = {
    applyMapDiff(updated) {
      for (const id of Object.keys(updated)) {
        const raw = lmap.get(id);
        const present = raw !== undefined;
        if (present && !entries.has(id)) {
          const child = raw as LoroMap;
          const built = buildRecordNode(entryDesc, child, doc, registry);
          entryDisposers.set(id, built.dispose);
          entries.set(id, built.surface);
        } else if (!present && entries.has(id)) {
          entryDisposers.get(id)?.();
          entryDisposers.delete(id);
          entries.delete(id);
        }
      }
    },
  };
  registry.set(lmap.id, node);

  const dispose = () => {
    registry.delete(lmap.id);
    for (const d of entryDisposers.values()) d();
    entryDisposers.clear();
  };
  return { surface: new ObservableIdMap(entries, lmap, doc), dispose };
}

/**
 * Map-like surface over an `idMap` container. Reads come from the
 * observable `entries`; writes create/replace/delete the child container
 * and commit, and the resulting event reconciles `entries`.
 */
class ObservableIdMap implements ReactiveMap<Record<string, unknown>> {
  constructor(
    private readonly store: ReturnType<typeof observable.map<string, Record<string, unknown>>>,
    private readonly lmap: LoroMap,
    private readonly doc: LoroDoc
  ) {}

  get size(): number {
    return this.store.size;
  }

  has(id: string): boolean {
    return this.store.has(id);
  }

  get(id: string): Record<string, unknown> | undefined {
    return this.store.get(id);
  }

  /** Create (or replace) the keyed child from a plain value object. Flat
   *  register children only for now. */
  set(id: string, value: Record<string, unknown>): void {
    // Mergeable so two peers concurrently creating the same id (rare, but
    // possible) converge on one container rather than competing ones.
    const child = this.lmap.ensureMergeableMap(id);
    // Replace semantics (Map.set contract): drop any existing keys absent
    // from the new value, then write the new keys. The container id is
    // kept, so concurrent edits to surviving keys aren't clobbered.
    // Materialise the key list first, deleting while iterating a live
    // WASM-backed iterator could skip or revisit entries.
    for (const k of [...child.keys()]) {
      if (!(k in value)) child.delete(k);
    }
    for (const k of Object.keys(value)) child.set(k, value[k] as never);
    this.doc.commit();
  }

  delete(id: string): void {
    this.lmap.delete(id);
    this.doc.commit();
  }

  keys(): IterableIterator<string> {
    return this.store.keys();
  }

  values(): IterableIterator<Record<string, unknown>> {
    return this.store.values();
  }

  entries(): IterableIterator<[string, Record<string, unknown>]> {
    return this.store.entries();
  }

  forEach(cb: (value: Record<string, unknown>, id: string, map: ObservableIdMap) => void): void {
    this.store.forEach((value, id) => cb(value, id, this));
  }

  [Symbol.iterator](): IterableIterator<[string, Record<string, unknown>]> {
    return this.store.entries();
  }
}

// ---------- movableList node ----------

/**
 * An ordered list of child records (e.g. bars). On any list event the node
 * reconciles the observable array against the list's current contents,
 * keyed by container id, so a `move` (which Loro reports as delete+insert)
 * reuses the existing child surface and preserves its identity rather than
 * tearing it down and rebuilding.
 */
function buildMovableListNode(
  desc: MovableListDescriptor,
  llist: LoroMovableList,
  doc: LoroDoc,
  registry: Registry
): { surface: ObservableList; dispose: () => void } {
  const entryDesc = desc.value;
  if (entryDesc.kind !== 'record') {
    throw new Error(`reactive_doc: movableList value must be a record (got '${entryDesc.kind}')`);
  }

  const items = observable.array<Record<string, unknown>>([], { deep: false });
  const childrenByCid = new Map<ContainerID, { surface: Record<string, unknown>; dispose: () => void }>();

  const reconcile = () => {
    const len = llist.length;
    const next: Record<string, unknown>[] = new Array(len);
    const seen = new Set<ContainerID>();
    for (let i = 0; i < len; i++) {
      const child = llist.get(i) as LoroMap;
      const cid = child.id;
      seen.add(cid);
      let entry = childrenByCid.get(cid);
      if (!entry) {
        entry = buildRecordNode(entryDesc, child, doc, registry);
        childrenByCid.set(cid, entry);
      }
      next[i] = entry.surface;
    }
    for (const [cid, entry] of childrenByCid) {
      if (!seen.has(cid)) {
        entry.dispose();
        childrenByCid.delete(cid);
      }
    }
    items.replace(next);
  };

  const node: Node = { applyListDiff: reconcile };
  registry.set(llist.id, node);

  const dispose = () => {
    registry.delete(llist.id);
    for (const entry of childrenByCid.values()) entry.dispose();
    childrenByCid.clear();
  };
  return { surface: new ObservableList(items, llist, doc, entryDesc), dispose };
}

/**
 * Array-like surface over a `movableList` container. Reads come from the
 * observable `items`; writes create/move/delete child containers and
 * commit, and the resulting event reconciles `items`.
 */
class ObservableList implements ReactiveList<Record<string, unknown>> {
  constructor(
    private readonly items: ReturnType<typeof observable.array<Record<string, unknown>>>,
    private readonly llist: LoroMovableList,
    private readonly doc: LoroDoc,
    private readonly entryDesc: RecordDescriptor
  ) {}

  get length(): number {
    return this.items.length;
  }

  at(index: number): Record<string, unknown> | undefined {
    return this.items[index];
  }

  push(value: Record<string, unknown>): void {
    this.insert(this.llist.length, value);
  }

  insert(index: number, value: Record<string, unknown>): void {
    const child = this.llist.insertContainer(index, new LoroMap());
    populateRecord(this.entryDesc, child, value);
    this.doc.commit();
  }

  delete(index: number): void {
    this.llist.delete(index, 1);
    this.doc.commit();
  }

  move(from: number, to: number): void {
    this.llist.move(from, to);
    this.doc.commit();
  }

  forEach(cb: (value: Record<string, unknown>, index: number, list: ObservableList) => void): void {
    this.items.forEach((value, index) => cb(value, index, this));
  }

  [Symbol.iterator](): IterableIterator<Record<string, unknown>> {
    return this.items[Symbol.iterator]();
  }
}

// ---------- Deep population (initial state) ----------

/** Write a plain record value into a Loro map: registers via `set`,
 *  container fields recursively. Used for deep initialization and for
 *  creating collection entries. */
function populateRecord(
  desc: RecordDescriptor,
  lmap: LoroMap,
  value: Record<string, unknown>
): void {
  const fields = desc.fields as Record<string, Descriptor>;
  for (const key of Object.keys(value)) {
    const field = fields[key];
    if (!field) continue;
    const v = value[key];
    if (field.kind === 'reg') {
      lmap.set(key, v as never);
    } else if (field.kind === 'record') {
      populateRecord(field, lmap.ensureMergeableMap(key), v as Record<string, unknown>);
    } else if (field.kind === 'idMap') {
      populateIdMap(field, lmap.ensureMergeableMap(key), v as Record<string, Record<string, unknown>>);
    } else if (field.kind === 'movableList') {
      populateList(field, lmap.ensureMergeableMovableList(key), v as Record<string, unknown>[]);
    } else {
      throw new Error(`reactive_doc: cannot initialize container kind '${(field as Descriptor).kind}' (field '${key}')`);
    }
  }
}

function populateList(
  desc: MovableListDescriptor,
  llist: LoroMovableList,
  value: Record<string, unknown>[]
): void {
  const entryDesc = desc.value;
  if (entryDesc.kind !== 'record') {
    throw new Error(`reactive_doc: movableList value must be a record (got '${entryDesc.kind}')`);
  }
  for (let i = 0; i < value.length; i++) {
    populateRecord(entryDesc, llist.insertContainer(i, new LoroMap()), value[i]);
  }
}

function populateIdMap(
  desc: IdMapDescriptor,
  lmap: LoroMap,
  value: Record<string, Record<string, unknown>>
): void {
  const entryDesc = desc.value;
  if (entryDesc.kind !== 'record') {
    throw new Error(`reactive_doc: idMap value must be a record (got '${entryDesc.kind}')`);
  }
  for (const id of Object.keys(value)) {
    populateRecord(entryDesc, lmap.ensureMergeableMap(id), value[id]);
  }
}

// ---------- Factory ----------

export function createReactiveDoc<S extends RecordDescriptor>(
  schema: S,
  // Deep initial state as a plain object, `idMap`s as records keyed by id,
  // `movableList`s as arrays. Populated through Loro so the cache hydrates
  // via the same event path as every later write.
  initial?: Init<S>
): ReactiveDoc<S> {
  const doc = new LoroDoc();
  const rootMap = doc.getMap(ROOT);
  const registry: Registry = new Map();

  const unsubscribe = doc.subscribe((batch) => {
    runInAction(() => {
      for (const event of batch.events) {
        const node = registry.get(event.target);
        if (!node) continue;
        if (event.diff.type === 'map') {
          node.applyMapDiff?.(event.diff.updated as Record<string, unknown>);
        } else if (event.diff.type === 'list') {
          node.applyListDiff?.();
        }
      }
    });
  });

  // Build the node tree (creating child containers + registering nodes),
  // then commit once to flush those creations through the event path.
  const root = buildRecordNode(schema, rootMap, doc, registry);
  const surface = root.surface;
  doc.commit();

  // Deep-populate the initial plain object through Loro; the commit's
  // events then hydrate every node's cache and materialize collections.
  if (initial) {
    populateRecord(schema, rootMap, initial as Record<string, unknown>);
    doc.commit();
  }

  const dispose = () => {
    unsubscribe();
    root.dispose();
  };
  return { model: surface as Infer<S>, doc, dispose, containerCount: () => registry.size };
}
