import { makeAutoObservable } from 'mobx';
import type { Resettable } from 'src/editing/session_reset';

/**
 * Toggleable grid lines drawn behind the notes in every bar. `mainBeat`
 * is on by default to match the score's classic look; the sub-beat
 * variants are off by default; they're practise aids the user can flip
 * on from the View menu when they want a denser reference grid.
 *
 * The four sub-beat families are orthogonal:
 *   - 16ths               (4 per beat, duple)
 *   - quarter triplets    (1.5 per beat = 3 per 2 beats, duple-pair triplet)
 *   - 8th triplets        (3 per beat)
 *   - 48ths               (12 per beat; LCM of 16ths + 8th-triplets)
 * Each can be toggled independently.
 */
export type GridLineSettings = {
  mainBeat: boolean;
  subBeat16: boolean;
  subBeatQuarterTriplet: boolean;
  subBeatTriplet: boolean;
  subBeat48: boolean;
};

/** Fresh-load defaults for {@link SettingsStore.gridLines}: main beats +
 *  16ths on, the rest off. Shared by the field initialiser and {@link
 *  SettingsStore.reset} so a song load returns to exactly the boot state. */
export const DEFAULT_GRID_LINES: GridLineSettings = {
  mainBeat: true,
  subBeat16: true,
  subBeatQuarterTriplet: false,
  subBeatTriplet: false,
  subBeat48: false,
};

/**
 * Per-song editor display settings: the grid-line overlay, waveform
 * normalisation, and the visually-merge-layers toggle. These are display
 * choices the user asked to travel with the song, so they {@link reset} on
 * each load and are then re-applied from a loaded save file's editor
 * metadata (a fresh DSL / MIDI load gets the {@link DEFAULT_GRID_LINES}
 * defaults). Kept as a single catch-all store rather than segmented per
 * setting type so the whole block serialises in one place.
 *
 * Pure data: only MobX observables (+ {@link reset}, the sanctioned
 * session-reset exception, see {@link Resettable}). Every other mutation
 * lives on the presenter (see `settings_presenter.ts`), so settings logic
 * can be unit-tested against a mocked store.
 */
export class SettingsStore implements Resettable {
  /**
   * Toggleable grid lines drawn behind notes in each bar. Default is
   * main beats + 16ths on for hand-authored / MIDI / example loads;
   * loading a debug bundle flips this to main beats + 48ths (the
   * presenter does so when applying a bundle) since transcribed scores
   * frequently land on triplet subdivisions the 16th grid alone can't
   * visualise. The View dropdown surfaces the toggles for manual override.
   */
  gridLines: GridLineSettings = { ...DEFAULT_GRID_LINES };

  /**
   * When true, each audio-track waveform is rendered with a per-track
   * normalisation factor so the median non-silent peak lands near the
   * top of the row regardless of the source recording's amplitude.
   * Silence still renders as silence; only the visual gain changes.
   * Default on so quiet recordings stay readable; toggle off via the
   * View dropdown to see accurate (un-normalised) signal levels.
   */
  uniformWaveforms: boolean = true;

  /**
   * When true, the audio-track waveform rows draw the same bar lines and
   * beat grid the score shows above them (which sub-beat families are on
   * follows {@link gridLines}, so a vertical line traces cleanly from the
   * score down through every waveform). Purely a reference overlay, no
   * data change. Default on; toggle off via the View dropdown for a clean
   * waveform.
   */
  waveformGridLines: boolean = true;

  /**
   * View-only "Visually merge layers". When true, the score collapses all
   * tracks of the same lane (across every `||` layer) into a single row and
   * drops the layer bands, showing the old flat per-lane view. Purely a
   * rendering choice, no data change: each note keeps its own layer, so edits
   * route per-note and a click-to-add lands on the firstmost layer carrying
   * the lane. Off by default (the score is layer-first).
   */
  mergeLayers: boolean = false;

  constructor() {
    makeAutoObservable(this);
  }

  /** Session reset: return every display setting to its fresh-load default.
   *  A loaded save file's `editor.settings` is applied afterwards (by
   *  `SettingsPresenter.applySettings`); a DSL / MIDI load keeps these
   *  defaults. */
  reset(): void {
    this.gridLines = { ...DEFAULT_GRID_LINES };
    this.uniformWaveforms = true;
    this.waveformGridLines = true;
    this.mergeLayers = false;
  }
}
