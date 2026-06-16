import { makeAutoObservable, observable } from 'mobx';
import { Jot } from 'src/schema/dsl/dsl';
import type { Jot as ReactiveJot, JotSchema } from 'src/schema/schema';
import type { ReactiveDoc } from 'src/schema/reactive_doc';
import { ExampleJot } from 'src/fakes/fakes';
import { dslToReactive } from 'src/schema/dsl/from_dsl';
import { ViewConfig } from 'src/editing/viewport/view_config';
import { StructureStore } from 'src/editing/structure/structure_store';
import { StructuralPresenter } from 'src/editing/structure/structural_presenter';
import { PaletteStore } from 'src/editing/palette/palette_store';
import { LayoutStore } from 'src/editing/viewport/layout_store';
import { TempoPresenter } from 'src/editing/playback/tempo_presenter';

/**
 * The loaded song and the chrome around loading it. The song is held as its
 * peer domains, {@link source} (raw DSL: title / globalMetadata),
 * {@link structural} (structure / layout / drum-offset), {@link palette}
 * (colours / legend), {@link tempo} (bar tempos / timeline). Every other
 * store reads the peer it needs off this one (viewport reads
 * `structural.pxPerBeat`, the mixer reads `palette` + lanes, playback reads
 * `tempo`); those stores take a reference to this one. The four fields are
 * always set / cleared together by the presenter (a single atomic load), so
 * any one being defined implies all are.
 *
 * Observables + computeds, plus the {@link loadSource} peer-constructor: it
 * builds the reactive document and the structure / palette / tempo views over
 * it (the one place that does, the sanctioned exception to "stores hold no
 * logic"). All higher-level load orchestration (parse, convert, replace-song
 * coordination) still lives on the presenter, which is the only caller of
 * `loadSource`. Document edits go through {@link jot} (`store.jot.elements.set`).
 */
export class JotEditorStore {
  /** Reactive Loro-backed document for the loaded song, or `undefined` before
   *  the first load. The source of truth the peers below are views over;
   *  edits go through {@link jot}. Held as a plain (non-observable) handle,
   *  load reactivity flows through the peer fields, set together with it. */
  private reactiveDoc: ReactiveDoc<typeof JotSchema> | undefined;

  /** Raw source of the loaded song (title / globalMetadata read off this),
   *  or `undefined` before the first load (the empty-state welcome screen
   *  renders then). */
  source: Jot | undefined;
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

  constructor() {
    // `reactiveDoc` is observed only by REFERENCE (`observable.ref`): deep-
    // observing a Loro doc would be wrong (its contents are already reactive
    // through the schema façade), but the *swap* on reload must notify, so a
    // consumer that reads the reactive jot directly (e.g. `LayersStore.layout`
    // off `jot.ordering`) re-derives when a new song loads rather than staying
    // pinned to the previous doc. `jot` stays a plain getter that reads it.
    makeAutoObservable<this, 'reactiveDoc'>(this, { reactiveDoc: observable.ref, jot: false });
  }

  get isLoading(): boolean {
    return this.loadingCount > 0;
  }

  /** The reactive document model for the loaded song, or `undefined` before
   *  the first load. Edits go straight through it, e.g.
   *  `store.jot?.elements.set(id, note)`; the commit reflows `structural`. */
  get jot(): ReactiveJot | undefined {
    return this.reactiveDoc?.model;
  }

  /**
   * Build (or clear) the loaded song's reactive document and its peer views,
   * installing them atomically. The sole writer of `reactiveDoc` / `source` /
   * `structural` / `palette` / `tempo`; the presenter calls it from inside its
   * per-load `runInAction`. Pass `undefined` to clear the loaded song (empty
   * state). Disposes the previous document so its Loro subscription is torn
   * down (leak-test safety).
   */
  loadSource(source: Jot | undefined): void {
    this.reactiveDoc?.dispose();
    if (!source) {
      this.reactiveDoc = undefined;
      this.source = undefined;
      this.structural = undefined;
      this.palette = undefined;
      this.tempo = undefined;
      return;
    }
    const peers = buildJotPeers(source, this.viewConfig);
    this.reactiveDoc = peers.doc;
    this.source = source;
    this.structural = peers.structural;
    this.palette = peers.palette;
    this.tempo = peers.tempo;
  }
}

/**
 * Build the reactive document for `source` plus the structure / palette /
 * tempo views over it. The `StructureStore` + `LayoutStore` are captured by the
 * peers' closures and intentionally not surfaced; the `doc` is returned so the
 * owner can hold it for edits + disposal. Pass the shared {@link ViewConfig} so
 * the zoom slider (which mutates `viewConfig.barWidth`) drives this song's
 * `pxPerBeat` / layout; standalone callers can omit it for a fresh default.
 */
function buildJotPeers(source: Jot, viewConfig: ViewConfig) {
  const doc = dslToReactive(source);
  const reactive = doc.model;
  const structureStore = new StructureStore(() => reactive);
  const palette = new PaletteStore(
    structureStore,
    () => viewConfig.palette,
    () => reactive
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
    source,
    viewConfig
  );
  const tempo = new TempoPresenter(structural);
  return { doc, structural, palette, tempo };
}

/**
 * Store-free structure builder for one-shot consumers that only need the
 * derived structure of a plain `Jot` (MIDI / RLRR export, unit tests,
 * Storybook), with no document lifecycle. Returns just the
 * {@link StructuralPresenter}; the underlying reactive doc is transient.
 */
export function buildStructural(
  source: Jot,
  viewConfig: ViewConfig = new ViewConfig()
): StructuralPresenter {
  return buildJotPeers(source, viewConfig).structural;
}
