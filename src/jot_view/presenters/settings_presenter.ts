import { makeAutoObservable } from 'mobx';
import { GridLineSettings, SettingsStore } from '../stores/settings_store';

/**
 * Mutations over {@link SettingsStore}, the display-toggle settings the
 * View dropdown drives. One of the per-domain presenters the legacy
 * `JotViewerPresenter` was split into; stores stay data-only, presenters
 * are the sole writers.
 */
export class SettingsPresenter {
  readonly settings: SettingsStore;

  constructor(settings: SettingsStore) {
    this.settings = settings;
    makeAutoObservable(this, { settings: false });
  }

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
