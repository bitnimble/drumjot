/**
 * Re-export barrel. The former monolithic `JotViewStore` is gone: its
 * data split into the per-concern data stores under `./stores/*`, and its
 * orchestration/actions into the per-domain presenters under
 * `./presenters/*`. These re-exports keep existing `from './store'`
 * import paths working for the shared types + helpers that several files
 * still pull from here.
 */
export type { GridLineSettings } from './stores/settings_store';
export type { TranscribeStatus, TranscribeOptions } from './stores/transcribe_store';
export type { LyricsAlignStatus } from './stores/lyrics_align_store';
export type { AudioTrackSplitStatus } from './stores/mixer_store';
export { collectJotPitches, clampVolume, VOLUME_STEP } from './stores/mixer_store';
export {
  BASE_BAR_WIDTH,
  MIN_ZOOM,
  MAX_ZOOM,
  DEFAULT_GUTTER_WIDTH,
  MIN_GUTTER_WIDTH,
  MAX_GUTTER_WIDTH,
  snapToDevicePx,
} from './stores/viewport_store';
export { trackKeyEq, type TrackKey } from 'src/tracks';
