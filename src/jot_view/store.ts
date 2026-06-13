/**
 * Re-export barrel. The former monolithic `JotViewStore` is gone: its
 * data split into the per-concern data stores under `./stores/*`, and its
 * orchestration/actions into the per-domain presenters under
 * `./presenters/*`. These re-exports keep existing `from './store'`
 * import paths working for the shared types + helpers that several files
 * still pull from here.
 */
export type { GridLineSettings } from './settings/settings_store';
export type { TranscribeStatus, TranscribeOptions } from './transcribe/transcribe_store';
export type { LyricsAlignStatus } from './lyrics/lyrics_align_store';
export type { AudioTrackSplitStatus } from './mixer/mixer_store';
export { collectJotPitches, clampVolume, VOLUME_STEP } from './mixer/mixer_store';
export {
  BASE_BAR_WIDTH,
  MIN_ZOOM,
  MAX_ZOOM,
  DEFAULT_GUTTER_WIDTH,
  MIN_GUTTER_WIDTH,
  MAX_GUTTER_WIDTH,
  snapToDevicePx,
} from './viewport/viewport_store';
export { trackKeyEq, type TrackKey } from 'src/tracks';
