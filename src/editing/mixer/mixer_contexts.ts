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

/**
 * Toolbar toggle: draw bar lines + the score's beat grid over each
 * audio-track waveform row (the grid families shown follow the score's
 * `--grid-display-*` vars, so the waveform mirrors the score above it).
 * Defaults to `true` (enabled by default, per the View menu). Read by
 * `AudioTrackView` to mount its `WaveformGridOverlay`.
 */
export const WaveformGridLinesContext = React.createContext<boolean>(true);

/**
 * View toggle "Visually merge layers": when true the score collapses tracks of
 * the same lane across layers into one row (no layer bands). Read by
 * `MixerView` to pick the row source. Defaults to `false` (layer-first).
 */
export const MergeLayersContext = React.createContext<boolean>(false);
