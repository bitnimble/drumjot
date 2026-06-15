import React from 'react';
import { GridLineSettings } from './settings_store';

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
