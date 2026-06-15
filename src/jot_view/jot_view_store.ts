import { makeAutoObservable } from 'mobx';
import { Jot } from 'src/schema/dsl/dsl';
import { ExampleJot } from 'src/fakes/fakes';
import { dslToReactive } from 'src/schema/dsl/from_dsl';
import { ViewConfig } from 'src/jot_view/viewport/view_config';
import { StructureStore } from 'src/jot_view/structure/structure_store';
import { StructuralPresenter } from 'src/jot_view/structure/structural_presenter';
import { PaletteStore } from 'src/jot_view/palette/palette_store';
import { LayoutStore } from 'src/jot_view/viewport/layout_store';
import { TempoPresenter } from 'src/jot_view/playback/tempo_presenter';

/**
 * The loaded song and the chrome around loading it. The song is held as its
 * peer domains, {@link source} (raw DSL: title / globalMetadata),
 * {@link structural} (structure / layout / drum-offset), {@link palette}
 * (colours / legend), {@link tempo} (bar tempos / timeline). Every other
 * store reads the peer it needs off this one (viewport reads
 * `structural.pxPerBeat`, the mixer reads `palette` + pitches, playback reads
 * `tempo`); those stores take a reference to this one. The four fields are
 * always set / cleared together by the presenter (a single atomic load), so
 * any one being defined implies all are.
 *
 * Pure data: observables + one computed. All loading orchestration (parse,
 * convert, build the peers, replace-song coordination) lives on the presenter;
 * it is the only thing that writes these fields.
 */
export class JotViewStore {
  /** Raw source of the loaded song (title / globalMetadata read off this),
   *  or `undefined` before the first load (the empty-state welcome screen
   *  renders then). */
  source: Jot | undefined;
  /** Structure / layout / pixels / drum-offset for the loaded song. */
  structural: StructuralPresenter | undefined;
  /** Per-pitch colours + legend for the loaded song. */
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
    makeAutoObservable(this);
  }

  get isLoading(): boolean {
    return this.loadingCount > 0;
  }
}

/** The peer domains for one loaded song (the value the presenter splays into
 *  {@link JotViewStore}'s `structural` / `palette` / `tempo` fields). */
export interface JotModel {
  structural: StructuralPresenter;
  palette: PaletteStore;
  tempo: TempoPresenter;
}

/**
 * Build the peer domains for `source`: convert the DSL into the reactive
 * document and construct the structure / palette / tempo views over it. The
 * shared `reactive` doc + `StructureStore` + `LayoutStore` are captured by the
 * peers' closures and intentionally not surfaced. Pass the shared
 * {@link ViewConfig} so the zoom slider (which mutates `viewConfig.barWidth`)
 * drives this song's `pxPerBeat` / layout; standalone callers (MIDI / RLRR
 * export, unit tests) can omit it and get a fresh default config.
 */
export function buildJotModel(source: Jot, viewConfig: ViewConfig = new ViewConfig()): JotModel {
  const reactive = dslToReactive(source).model;
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
  return { structural, palette, tempo };
}
