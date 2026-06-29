import React from 'react';
import type { PlaybackStore } from './playback_store';
import type { PlaybackPresenter } from './playback_presenter';

/**
 * The transport store + presenter, provided once at the JotEditor level so
 * non-transport surfaces can read/drive playback state without prop
 * plumbing. Today the Edit menu's Alignment section reads the live
 * drum-beat / audio offsets off {@link PlaybackStoreContext} and nudges
 * them through {@link PlaybackPresenterContext}. `null` outside the View
 * (tests / standalone renders), where consumers no-op.
 */
export const PlaybackStoreContext = React.createContext<PlaybackStore | null>(null);
export const PlaybackPresenterContext = React.createContext<PlaybackPresenter | null>(null);

/**
 * Whether the score auto-scrolls to keep the playhead centred during
 * playback, and the toggle that flips it. Read by two distant
 * consumers: `PlayheadAutoScroller` (skips the per-frame `scrollLeft`
 * write when `follow` is false) and the `FollowToggle` button stacked
 * above the playhead label. Threading through `JotEditor →
 * TimelineHeader → Playhead → PlayheadLabel` for one boolean + one
 * handler is more noise than it's worth, hence the context. Defaults
 * to `{ follow: true, toggle: noop }` so a Playhead rendered outside
 * the View still behaves like today's always-follow build.
 */
export type FollowPlayheadContextValue = {
  follow: boolean;
  toggle: () => void;
};

export const FollowPlayheadContext = React.createContext<FollowPlayheadContextValue>({
  follow: true,
  toggle: () => {},
});
