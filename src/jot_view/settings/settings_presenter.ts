import { makeAutoObservable } from 'mobx';
import { GridLineSettings, SettingsStore } from './settings_store';

/**
 * Mutations over {@link SettingsStore}, the display-toggle settings the
 * View dropdown drives. One of the per-domain presenters the legacy
 * catch-all `JotViewStore` was split into; stores stay data-only,
 * presenters are the sole writers.
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

  /**
   * Switch the grid to the 48ths overlay used for transcribed bundles.
   * The transcribe pipeline routinely emits triplet subdivisions; 48ths
   * is the LCM of 16ths + triplets so it visualises both. Called by the
   * bundle loader (DocumentPresenter) to override the store-wide 16ths
   * default for that load specifically.
   */
  useTranscribeGridLines() {
    this.settings.gridLines = {
      mainBeat: true,
      subBeat16: false,
      subBeatQuarterTriplet: false,
      subBeatTriplet: false,
      subBeat48: true,
    };
  }
}
