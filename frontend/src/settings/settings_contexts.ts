import React from 'react';
import { GridLineSettings } from './settings_store';
import type { SettingsStore } from './settings_store';
import type { SettingsPresenter } from './settings_presenter';

/**
 * The display-settings store + its presenter, provided at the JotEditor level
 * so the View dropdown reads/toggles them off context instead of the app shell
 * threading a dozen `on*`/flag props through the Toolbar. `null` outside the
 * View (e.g. a Toolbar-less render); consumers no-op in that case.
 */
export const SettingsStoreContext = React.createContext<SettingsStore | null>(null);
export const SettingsPresenterContext = React.createContext<SettingsPresenter | null>(null);

/**
 * Grid-line toggles surfaced through the View dropdown. Threaded as
 * context so the deep `BarView` can read each setting without every
 * intermediate (`MixerView` → `InstrumentTrackView` → `BarView`) carrying a
 * prop. Defaults match the store's initial state so a BarView rendered
 * outside the View (e.g. unit tests, future embedded usage) still has the
 * classic look.
 */
export const GridLineSettingsContext = React.createContext<GridLineSettings>({
  mainBeat: true,
  subBeat16: false,
  subBeatQuarterTriplet: false,
  subBeatTriplet: false,
  subBeat48: false,
});
