import React from 'react';

/**
 * Whether the score auto-scrolls to keep the playhead centred during
 * playback, and the toggle that flips it. Read by two distant
 * consumers: `PlayheadAutoScroller` (skips the per-frame `scrollLeft`
 * write when `follow` is false) and the `FollowToggle` button stacked
 * above the playhead label. Threading through `JotView →
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
