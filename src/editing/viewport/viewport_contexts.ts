import React from 'react';
import { ViewportStore } from './viewport_store';

/**
 * Routes the {@link ViewportStore} to deep consumers that read scroll /
 * zoom / visible-range state (today: score `WindowedTicks` / `PopoverPortal`
 * and `WindowedLyricLines`, the mixer rows' windowing, the toolbar zoom
 * slider). `null` outside the view.
 */
export const ViewportStoreContext = React.createContext<ViewportStore | null>(null);
