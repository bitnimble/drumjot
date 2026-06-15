/**
 * Globally-important time anchors for the loaded song, in JOT-TIME seconds
 * (the score's coordinate system, where the bar-1 downbeat = 0). Replaces the
 * old scattered `drumsT0Sec` arithmetic with one named record.
 *
 * Ordering (all lead-in anchors are <= 0):
 *
 *     fullLeadIn  <=  songLeadIn  <=  drums (0)
 *
 * `songLeadIn` is the live audio alignment (tunable by the drum-offset
 * control); `fullLeadIn` comes from the rendered structure (and so includes
 * the view-only virtual lead-in bar). The audio <-> jot mapping is
 * `media = jot - songLeadIn` (see {@link jotToMedia}).
 */
export type Epochs = {
  /** Bar-1 downbeat / first drum onset. The jot-time origin (always 0). */
  drums: number;
  /** Jot time at which the recorded audio begins (audio buffer t=0). The
   *  song's audio lead-in; the negative of the old `drumsT0Sec`. <= drums. */
  songLeadIn: number;
  /** Jot time of the rendered left edge (the virtual lead-in bar): the
   *  leftmost seekable / visible point on the timeline. <= songLeadIn. */
  fullLeadIn: number;
};

/** Build an {@link Epochs} from the live audio alignment (`songLeadIn`, <= 0)
 *  and the rendered left edge (`fullLeadIn`, <= 0). `drums` is always 0. */
export function makeEpochs(songLeadIn: number, fullLeadIn: number): Epochs {
  return { drums: 0, songLeadIn, fullLeadIn };
}

/** Audio (media) time for a jot time: `media = jot - songLeadIn`. */
export function jotToMedia(epochs: Epochs, jotSec: number): number {
  return jotSec - epochs.songLeadIn;
}

/** Jot time for an audio (media) time: the inverse of {@link jotToMedia}. */
export function mediaToJot(epochs: Epochs, mediaSec: number): number {
  return mediaSec + epochs.songLeadIn;
}
