import React from 'react';
import { MixerStore } from './mixer_store';

/**
 * Routes the {@link MixerStore} to deep consumers that read mixer state
 * (today: `MixerView`'s row order, the per-row audio-split status, the
 * per-instrument colour view-models). `null` outside the view.
 */
export const MixerStoreContext = React.createContext<MixerStore | null>(null);
