import React from 'react';
import { LyricsAlignStore } from './lyrics_align_store';
import { LyricsPresenter } from './lyrics_presenter';

/**
 * Routes the {@link LyricsAlignStore} to deep consumers that read lyrics
 * align state (today: `LyricsTrackView`'s per-row align spinner). `null`
 * outside the view.
 */
export const LyricsAlignStoreContext = React.createContext<LyricsAlignStore | null>(null);

/**
 * Routes the {@link LyricsPresenter} to deep consumers that mutate
 * lyrics state (today: `LyricsTrackView` removing its own row via
 * `removeLyricsTrack`). `null` outside the view; consumers no-op when
 * absent.
 */
export const LyricsPresenterContext = React.createContext<LyricsPresenter | null>(null);
