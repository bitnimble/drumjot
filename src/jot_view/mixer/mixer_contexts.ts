import React from 'react';
import { MixerStore } from './mixer_store';

/**
 * Routes the {@link MixerStore} to deep consumers that read mixer state
 * (today: `MixerView`'s row order, the per-row audio-split status, the
 * per-instrument colour view-models). `null` outside the view.
 */
export const MixerStoreContext = React.createContext<MixerStore | null>(null);

/**
 * Toolbar toggle: render audio-track waveforms with per-track
 * normalisation so the median non-silent peak fills most of the row,
 * regardless of source amplitude. Defaults to `false` so a canvas
 * rendered outside the View still shows the accurate signal level.
 * Read by `AudioTrackWaveformCanvas` (the mixer's audio-track row is
 * what renders the waveforms).
 */
export const UniformWaveformsContext = React.createContext<boolean>(false);
