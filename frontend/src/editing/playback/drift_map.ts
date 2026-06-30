/**
 * Per-bar piecewise jot-time ↔ media-time (recorded-audio-time) mapping.
 *
 * The flat identity is `media = jot - songLeadIn`: the score's uniform tempo
 * grid lines the synth up with the recording. But a human recording *drifts*; * a bar the drummer leant on is a few ms longer than the grid says. We display
 * a uniform grid anyway (that's the point of a transcription) and absorb the
 * deviation into a per-bar {@link BarTiming.driftSec} channel, so a given
 * musical position maps to a slightly shifted spot in the recording.
 *
 * This map makes that shift first-class for playback: each bar's uniform jot
 * span `[startSec, startSec+durationSec)` maps onto its REAL recorded span
 * `[startSec+drift, nextStartSec+nextDrift) − songLeadIn`. So:
 *
 *  - the playhead (media→jot) tracks where the recording actually is,
 *  - synth notes (jot→media) schedule against the recording, not the grid,
 *  - the audio buffer seek (jot→media) lands on the true sample, and
 *  - it matches the waveform renderer's stretch exactly (same per-bar span),
 *    so bar lines, waveform, audio, and notes all agree.
 *
 * Crucially, when every `driftSec` is 0 (a metronomic recording, the common
 * case) the map collapses to the flat `jot - songLeadIn` and is byte-identical
 * to the old arithmetic, drift is a no-op until a wandering recording needs
 * it. Notes carry no residual drift offset of their own: the grid (this map)
 * owns the drift, the note keeps only its sub-grid micro-timing.
 */
import { BarTiming } from './timeline';

export type DriftMap = {
  /** False when every bar's drift is 0; callers may skip drift handling. */
  readonly hasDrift: boolean;
  /** Recorded-audio (media) time for a jot time: drift-aware `jot - songLeadIn`. */
  jotToMedia(jotSec: number): number;
  /** Jot time for a recorded-audio (media) time: the inverse of {@link jotToMedia}. */
  mediaToJot(mediaSec: number): number;
};

/**
 * Build a {@link DriftMap} from the timeline's per-bar jot spans + drift and
 * the live audio alignment (`songLeadInSec`, <= 0). Bars are contiguous in jot
 * time and (drift being tiny vs. a bar) monotonic in media time, so a forward
 * scan locates the containing bar; positions outside the bar range extrapolate
 * linearly off the nearest edge bar (the lead-in scrub / past-the-end tail).
 */
export function buildDriftMap(bars: readonly BarTiming[], songLeadInSec: number): DriftMap {
  const n = bars.length;
  if (n === 0) {
    return {
      hasDrift: false,
      jotToMedia: (j) => j - songLeadInSec,
      mediaToJot: (m) => m + songLeadInSec,
    };
  }

  // Per-bar jot + media spans. mediaStart[i] is the bar's REAL audio downbeat;
  // mediaEnd[i] is the next bar's real downbeat (contiguous), so the bar's
  // media duration absorbs the drift delta to the next bar.
  const jotStart = new Float64Array(n);
  const jotEnd = new Float64Array(n);
  const mediaStart = new Float64Array(n);
  const mediaEnd = new Float64Array(n);
  let hasDrift = false;
  for (let i = 0; i < n; i++) {
    const b = bars[i];
    const drift = b.driftSec ?? 0;
    const nextDrift = i + 1 < n ? bars[i + 1].driftSec ?? 0 : drift;
    if (drift !== 0 || nextDrift !== 0) hasDrift = true;
    jotStart[i] = b.startSec;
    jotEnd[i] = b.startSec + b.durationSec;
    mediaStart[i] = b.startSec + drift - songLeadInSec;
    mediaEnd[i] = b.startSec + b.durationSec + nextDrift - songLeadInSec;
  }

  // No drift anywhere → the flat mapping, exactly as before (and O(1)).
  if (!hasDrift) {
    return {
      hasDrift: false,
      jotToMedia: (j) => j - songLeadInSec,
      mediaToJot: (m) => m + songLeadInSec,
    };
  }

  const jotToMedia = (jotSec: number): number => {
    let i = 0;
    while (i < n - 1 && jotSec >= jotEnd[i]) i++;
    const jSpan = jotEnd[i] - jotStart[i];
    if (jSpan <= 0) return mediaStart[i];
    const frac = (jotSec - jotStart[i]) / jSpan;
    return mediaStart[i] + frac * (mediaEnd[i] - mediaStart[i]);
  };

  const mediaToJot = (mediaSec: number): number => {
    let i = 0;
    while (i < n - 1 && mediaSec >= mediaEnd[i]) i++;
    const mSpan = mediaEnd[i] - mediaStart[i];
    if (mSpan <= 0) return jotStart[i];
    const frac = (mediaSec - mediaStart[i]) / mSpan;
    return jotStart[i] + frac * (jotEnd[i] - jotStart[i]);
  };

  return { hasDrift: true, jotToMedia, mediaToJot };
}
