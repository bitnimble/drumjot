import { makeAutoObservable } from 'mobx';
import { GridLineSettings, SettingsStore } from './stores/settings_store';

/**
 * Dependencies the presenter orchestrates over. Every store is a plain
 * data container; the presenter is the single place that mutates them.
 *
 * This grows one entry per extracted store as the `JotViewStore` carve-up
 * proceeds. It is a TEMPORARY catch-all for all orchestration that used
 * to (incorrectly) live on `JotViewStore`; once the carve-up is complete
 * the methods here get split into per-feature presenters, each owning the
 * subset of stores its feature touches.
 */
export type JotViewerPresenterDeps = {
  settings: SettingsStore;
};

/**
 * Catch-all presenter for the jot viewer. Holds the actions, reactions,
 * and orchestration that mutate the data-only stores; React components
 * bind its methods to UI callbacks and read store state for rendering.
 *
 * The split exists so business logic can be unit-tested with mocked
 * stores (e.g. `presenter.setUniformWaveforms(true)` and assert the
 * mocked `SettingsStore` was updated) without standing up React or the
 * full store graph.
 *
 * Methods are grouped by the feature/domain they'll eventually move to;
 * each group is fronted by a `// --- <domain> ---` banner.
 */
export class JotViewerPresenter {
  // Store dependencies. `makeAutoObservable` is told to leave these
  // non-observable (they're already-observable stores; the presenter
  // only holds references, it doesn't own their reactivity).
  readonly settings: SettingsStore;

  constructor(deps: JotViewerPresenterDeps) {
    this.settings = deps.settings;
    makeAutoObservable(this, { settings: false }, { autoBind: true });
  }

  // --- settings ---

  toggleGridLine(key: keyof GridLineSettings) {
    this.settings.gridLines = {
      ...this.settings.gridLines,
      [key]: !this.settings.gridLines[key],
    };
  }

  setUniformWaveforms(on: boolean) {
    this.settings.uniformWaveforms = on;
  }
}
