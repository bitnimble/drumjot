import { makeAutoObservable } from 'mobx';

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

/**
 * Persistent (eventually) user settings, display toggles and similar
 * preferences that live independently of any one loaded song. Kept as a
 * single catch-all store rather than segmented per setting type so the
 * whole thing can be serialised to localStorage in one place later.
 *
 * Pure data: only MobX observables (+ computeds). Every mutation lives on
 * the presenter (see `presenters/settings_presenter.ts`), so settings
 * logic can be unit-tested against a mocked store.
 */
export class SettingsStore {
  /**
   * Toggleable grid lines drawn behind notes in each bar. Default is
   * main beats + 16ths on for hand-authored / MIDI / example loads;
   * loading a debug bundle flips this to main beats + 48ths (the
   * presenter does so when applying a bundle) since transcribed scores
   * frequently land on triplet subdivisions the 16th grid alone can't
   * visualise. The View dropdown surfaces the toggles for manual override.
   */
  gridLines: GridLineSettings = {
    mainBeat: true,
    subBeat16: true,
    subBeatQuarterTriplet: false,
    subBeatTriplet: false,
    subBeat48: false,
  };

  /**
   * When true, each audio-track waveform is rendered with a per-track
   * normalisation factor so the median non-silent peak lands near the
   * top of the row regardless of the source recording's amplitude.
   * Silence still renders as silence; only the visual gain changes.
   * Default on so quiet recordings stay readable; toggle off via the
   * View dropdown to see accurate (un-normalised) signal levels.
   */
  uniformWaveforms: boolean = true;

  constructor() {
    makeAutoObservable(this);
  }
}
