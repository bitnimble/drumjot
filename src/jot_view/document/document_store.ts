import { makeAutoObservable } from 'mobx';
import { ExampleJot } from 'src/fakes/fakes';
import { RenderedJot } from 'src/jot/resolved_jot';
import { ViewConfig } from 'src/jot/view_config';

/**
 * The loaded song and the chrome around loading it. The `currentJot` is
 * the shared spine every other store reads from (viewport reads its
 * `pxPerBeat`, the mixer reads its pitches, the playback transport plays
 * it); those stores take a reference to this one.
 *
 * Pure data: observables + one computed. All loading orchestration
 * (parse, convert, replace-song coordination) lives on the presenter; it
 * is the only thing that writes these fields.
 */
export class DocumentStore {
  /** The currently-loaded, laid-out jot, or `undefined` before the first
   *  load (the empty-state welcome screen renders in that case). */
  currentJot: RenderedJot | undefined;

  /**
   * Shared layout config threaded into every new `RenderedJot`, so the
   * zoom slider mutates a single config object and the layout reflows
   * reactively (ViewConfig is MobX-observable; RenderedJot's `layoutJot`
   * is a computedFn that reads `barWidth`).
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
