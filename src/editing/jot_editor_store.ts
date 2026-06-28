import { makeAutoObservable, observable } from 'mobx';
import type { LoroDoc } from 'loro-crdt';
import { Jot } from 'src/schema/dsl/dsl';
import type { MutableJot, JotSchema, JotState } from 'src/schema/schema';
import { createMutableJotFromState } from 'src/schema/schema';
import { createJotDerivedRegistry, type JotDerivedRegistry } from 'src/schema/derived_fields';
import type { ReactiveDoc } from 'src/schema/reactive_doc';
import { ExampleJot } from 'src/fakes/fakes';
import { dslToMutable } from 'src/schema/dsl/from_dsl';
import { ViewConfig } from 'src/editing/viewport/view_config';
import { StructureStore } from 'src/editing/structure/structure_store';
import { StructuralPresenter } from 'src/editing/structure/structural_presenter';
import { PaletteStore } from 'src/editing/palette/palette_store';
import { LayoutStore } from 'src/editing/viewport/layout_store';
import { TempoPresenter } from 'src/editing/playback/tempo_presenter';

/**
 * The loaded song and the chrome around loading it. The song is held as its
 * peer domains, {@link structural} (structure / layout / drum-offset),
 * {@link palette} (colours / legend), {@link tempo} (bar tempos / timeline).
 * Every other store reads the peer it needs off this one (viewport reads
 * `structural.pxPerBeat`, the mixer reads `palette` + lanes, playback reads
 * `tempo`); those stores take a reference to this one. The peers (plus the
 * backing {@link jot}) are always set / cleared together by the presenter (a
 * single atomic load), so any one being defined implies all are. The DSL is a
 * load/export-only format: it's converted into the reactive {@link jot} once at
 * load and never read again at runtime (export goes the other way, via
 * `mutableToDsl`).
 *
 * Observables + computeds, plus the {@link loadSource} peer-constructor: it
 * builds the mutable document and the structure / palette / tempo views over
 * it (the one place that does, the sanctioned exception to "stores hold no
 * logic"). All higher-level load orchestration (parse, convert, replace-song
 * coordination) still lives on the presenter, which is the only caller of
 * `loadSource`. Document edits go through {@link jot} (`store.jot.elements.set`).
 */
export class JotEditorStore {
  /** Mutable Loro-backed document for the loaded song, or `undefined` before
   *  the first load. The source of truth the peers below are views over;
   *  edits go through {@link jot}. Held as a plain (non-observable) handle,
   *  load reactivity flows through the peer fields, set together with it. */
  private mutableDoc: ReactiveDoc<typeof JotSchema> | undefined;

  /** Structure / layout / pixels / drum-offset for the loaded song. */
  structural: StructuralPresenter | undefined;
  /** Per-lane colours + legend for the loaded song. */
  palette: PaletteStore | undefined;
  /** Per-bar tempo segments, dominant bpm/time, and the audio timeline. */
  tempo: TempoPresenter | undefined;

  /**
   * Shared layout config threaded into every loaded song's peers, so the
   * zoom slider mutates a single config object and the layout reflows
   * reactively (ViewConfig is MobX-observable; `StructuralPresenter.pxPerBeat`
   * reads `barWidth`).
   */
  viewConfig: ViewConfig = new ViewConfig();

  /** Built-in example jots offered in the toolbar / empty state. */
  examples: readonly ExampleJot[] = [];

  /** Id of the example currently loaded, or `undefined` when the loaded
   *  jot didn't come from the example picker. */
  currentExampleId: string | undefined = undefined;

  /**
   * In-flight file-load counter. Each top-level loader (jot / midi / paradb
   * map / debug bundle / audio track) brackets its work so the modal
   * overlay surfaces. Nested calls (e.g. the debug bundle loading its
   * per-stem audio tracks) bump the count too but keep the outer label so
   * the overlay reads as one operation.
   */
  loadingCount: number = 0;
  loadingLabel: string | undefined = undefined;

  /**
   * Whether the loaded song has unsaved edits: `false` right after a load or a
   * save, `true` once the document is edited. Drives the "discard changes?"
   * prompt before a wholesale replace (e.g. File → New jot). Written only by
   * {@link JotEditorPresenter} (the load/save orchestrator + the per-doc edit
   * subscription); `false` with nothing loaded.
   */
  dirty: boolean = false;

  constructor() {
    // `mutableDoc` is observed only by REFERENCE (`observable.ref`): deep-
    // observing a Loro doc would be wrong (its contents are already reactive
    // through the schema façade), but the *swap* on reload must notify, so a
    // consumer that reads the mutable jot directly (e.g. `LayersStore.layout`
    // off `jot.ordering`) re-derives when a new song loads rather than staying
    // pinned to the previous doc. `jot` stays a plain getter that reads it.
    makeAutoObservable<this, 'mutableDoc'>(this, {
      mutableDoc: observable.ref,
      jot: false,
      loroDoc: false,
    });
  }

  get isLoading(): boolean {
    return this.loadingCount > 0;
  }

  /** The backing Loro document for the loaded song, or `undefined` before the
   *  first load. The source of truth for undo/redo: `HistoryPresenter` builds
   *  its `UndoManager` from this and reattaches when the reference swaps on
   *  reload. App edits still go through {@link jot}; this is the synchroniser
   *  handle, not an edit surface. Reading it tracks the `observable.ref`
   *  {@link mutableDoc}, so a reaction over it re-fires when a new song loads. */
  get loroDoc(): LoroDoc | undefined {
    return this.mutableDoc?.doc;
  }

  /** The mutable document model for the loaded song, or `undefined` before
   *  the first load. Edits go straight through it, e.g.
   *  `store.jot?.elements.set(id, note)`; the commit reflows `structural`. */
  get jot(): MutableJot | undefined {
    return this.mutableDoc?.model;
  }

  /** Plain {@link JotState} snapshot of the loaded song's edited document,
   *  or `undefined` before the first load. What the mutable `.jot` save
   *  format serialises (the lossless superset, post-edit); the inverse of
   *  {@link loadState}. */
  snapshot(): JotState | undefined {
    return this.mutableDoc?.snapshot();
  }

  /**
   * Build (or clear) the loaded song's mutable document and its peer views,
   * installing them atomically. The sole writer of `mutableDoc` / `source` /
   * `structural` / `palette` / `tempo`; the presenter calls it from inside its
   * per-load `runInAction`. Pass `undefined` to clear the loaded song (empty
   * state). Disposes the previous document so its Loro subscription is torn
   * down (leak-test safety).
   */
  loadSource(source: Jot | undefined): void {
    this.mutableDoc?.dispose();
    if (!source) {
      this.mutableDoc = undefined;
      this.structural = undefined;
      this.palette = undefined;
      this.tempo = undefined;
      return;
    }
    const registry = createJotDerivedRegistry();
    this.installPeers(dslToMutable(source, registry), registry);
  }

  /**
   * Install a song from a saved {@link JotState} snapshot (a mutable `.jot`
   * file) rather than from DSL: the document is seeded directly from
   * `document` (preserving the edits the snapshot captured, which a DSL
   * round-trip would lose). The whole song, including the `globalMetadata` the
   * peers read, lives in the snapshot, so no separate DSL source is needed. The
   * same single-writer / atomic-swap contract as {@link loadSource}; the
   * presenter is the only caller and wraps it in its per-load `runInAction`.
   */
  loadState(document: JotState): void {
    this.mutableDoc?.dispose();
    const registry = createJotDerivedRegistry();
    this.installPeers(createMutableJotFromState(document, registry), registry);
  }

  /** Set the loaded-song peer fields together (the atomic swap both {@link
   *  loadSource} and {@link loadState} share). Sole writer of `mutableDoc` /
   *  `structural` / `palette` / `tempo`. */
  private installPeers(doc: ReactiveDoc<typeof JotSchema>, registry: JotDerivedRegistry): void {
    const peers = buildJotPeers(doc, registry, this.viewConfig);
    this.mutableDoc = peers.doc;
    this.structural = peers.structural;
    this.palette = peers.palette;
    this.tempo = peers.tempo;
  }
}

/**
 * Build the structure / palette / tempo views over an already-constructed
 * mutable document `doc` (seeded from DSL via `dslToMutable`, or from a saved
 * snapshot via `createMutableJotFromState`). Every peer reads the reactive
 * document directly (incl. `globalMetadata`, now lifted into the schema); the
 * `StructureStore` + `LayoutStore` are captured by the peers' closures and
 * intentionally not surfaced; `doc` is passed back through so the owner can
 * hold it for edits + disposal. Pass the shared {@link ViewConfig} so the zoom
 * slider (which mutates `viewConfig.barWidth`) drives this song's `pxPerBeat` /
 * layout.
 */
function buildJotPeers(
  doc: ReactiveDoc<typeof JotSchema>,
  registry: JotDerivedRegistry,
  viewConfig: ViewConfig
) {
  const mutable = doc.model;
  const structureStore = new StructureStore(() => mutable);
  const palette = new PaletteStore(
    structureStore,
    () => viewConfig.palette,
    () => mutable
  );
  const layoutStore = new LayoutStore(
    structureStore,
    () => viewConfig.barWidth as number,
    () => viewConfig.barNotePaddingBeats
  );
  const structural = new StructuralPresenter(
    structureStore,
    palette,
    layoutStore,
    () => mutable,
    viewConfig
  );
  const tempo = new TempoPresenter(structural, registry);
  return { doc, structural, palette, tempo };
}

/**
 * Store-free structure builder for one-shot consumers that only need the
 * derived structure of a plain `Jot` (MIDI / RLRR export, unit tests,
 * Storybook), with no document lifecycle. Returns just the
 * {@link StructuralPresenter}; the underlying mutable doc is transient.
 */
export function buildStructural(
  source: Jot,
  viewConfig: ViewConfig = new ViewConfig()
): StructuralPresenter {
  const registry = createJotDerivedRegistry();
  return buildJotPeers(dslToMutable(source, registry), registry, viewConfig).structural;
}
