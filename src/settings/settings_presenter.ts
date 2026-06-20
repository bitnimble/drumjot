import { makeAutoObservable } from 'mobx';
import { GridLineSettings, SettingsStore } from './settings_store';

/**
 * Mutations over {@link SettingsStore}, the display-toggle settings the
 * View dropdown drives. One of the per-domain presenters the legacy
 * catch-all `JotEditorStore` was split into; stores stay data-only,
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

  setMergeLayers(on: boolean) {
    this.settings.mergeLayers = on;
  }

  /**
   * Re-apply the display settings carried in a loaded `.jot` save file's
   * editor metadata. Called after {@link SettingsStore.reset} + the new
   * song is installed, so a saved grid-line overlay / waveform / merge
   * choice travels with the song. A partial snapshot leaves the reset
   * defaults in place for any absent field.
   */
  applySettings(settings: SettingsState): void {
    if (settings.gridLines) this.settings.gridLines = { ...settings.gridLines };
    if (settings.uniformWaveforms !== undefined) {
      this.settings.uniformWaveforms = settings.uniformWaveforms;
    }
    if (settings.mergeLayers !== undefined) this.settings.mergeLayers = settings.mergeLayers;
  }

  /**
   * Switch the grid to the 48ths overlay used for transcribed bundles.
   * The transcribe pipeline routinely emits triplet subdivisions; 48ths
   * is the LCM of 16ths + triplets so it visualises both. Called by the
   * bundle loader (JotEditorPresenter) to override the store-wide 16ths
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

/**
 * Serialisable display-settings snapshot for the `.jot` save format's editor
 * metadata. Every field optional so the loader tolerates files written by an
 * older app version (missing fields keep their reset defaults).
 */
export type SettingsState = {
  gridLines?: GridLineSettings;
  uniformWaveforms?: boolean;
  mergeLayers?: boolean;
};
