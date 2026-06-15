/**
 * Pure waveform peak compute primitives, shared between the main
 * thread and the {@link Worker} that powers
 * {@link waveform_worker_client}. No DOM / Web Audio / React deps so
 * the same code can execute in either context.
 *
 * Each function returns a flat `Float32Array` of length `2 * widthPx`
 * (interleaved `[min, max]` per pixel column) so callers can fillRect
 * into Canvas without re-allocating per row.
 */

/**
 * One column-strip on the score: where the bar's left edge sits in CSS
 * pixels (`x`), its CSS width, and the absolute jot-time range the
 * bar covers. Used by {@link computeWaveformPeaksFromChannels} to map
 * each pixel column back to the buffer-sample range it represents.
 *
 * Flattened off the structural layers on the main thread so the worker
 * doesn't need to know about React / MobX. See
 * `audio_tracks.ts::buildBarSlices`.
 */
export type BarSlice = {
  x: number;
  width: number;
  startSec: number;
  durationSec: number;
};

/**
 * Per-channel PCM + the rate metadata the peak loops need. Channel
 * arrays are *copies* of the original `AudioBuffer.getChannelData(ch)`
 * (the worker holds its own copy so the main-thread `AudioBuffer`
 * stays untouched and usable by the BufferSource playback path).
 */
export type ChannelData = {
  channels: Float32Array[];
  sampleRate: number;
  length: number;
};

/**
 * Magnitude below which a sample / pixel column is treated as silence
 * for the uniform-waveform median. Background noise on a clean mix
 * typically sits around -60 to -40 dBFS (~0.001–0.01); excluding it
 * stops the median from collapsing toward the floor on tracks that
 * are mostly gaps between hits. Mirrors the constant of the same
 * purpose in `mixer.tsx`'s old per-bitmap normaliser.
 */
const SILENCE_FLOOR = 0.05;

/**
 * Target amplitude (in normalised [-1, 1] sample units) that the
 * median non-silent magnitude is scaled to in "uniform amplitude"
 * mode. 0.25 means the median sample lands at 25 % of the row's
 * half-height; i.e. the median peak-to-peak covers ~50 % of the
 * row, leaving headroom so transients sit inside the lane instead
 * of clipping at the top/bottom edges. Tweak this single value to
 * change how full uniform waveforms render.
 */
const UNIFORM_WAVEFORM_TARGET = 0.3;

/**
 * Per-track amplitude scale for "uniform amplitude" mode. Computed
 * once on track registration against the decoded PCM (NOT against a
 * particular bitmap) so every chunk of a tiled waveform normalises
 * against the SAME number; no visible amplitude seams between
 * neighbouring chunks of the same track, no zoom dependency.
 *
 * Method: stride through the channel data, fold to mono inline, take
 * the median magnitude above {@link SILENCE_FLOOR}, return
 * {@link UNIFORM_WAVEFORM_TARGET} / median so the median sample lands
 * at the target fraction of the row's half height. Returns `1` when
 * the track is entirely silent (nothing to normalise) or has too few
 * samples to take a median.
 *
 * Stride keeps the cost a fixed ~10 k samples regardless of track
 * length; typically ~1 ms on warm engines, called once per track.
 */
export function computeTrackAmpScale(data: ChannelData): number {
  const { channels, length } = data;
  if (length === 0 || channels.length === 0) return 1;
  const stride = Math.max(1, Math.floor(length / 10000));
  const mags: number[] = [];
  if (channels.length === 1) {
    const d = channels[0];
    for (let s = 0; s < length; s += stride) {
      const v = Math.abs(d[s]);
      if (v > SILENCE_FLOOR) mags.push(v);
    }
  } else if (channels.length === 2) {
    const c0 = channels[0];
    const c1 = channels[1];
    for (let s = 0; s < length; s += stride) {
      const v = Math.abs((c0[s] + c1[s]) * 0.5);
      if (v > SILENCE_FLOOR) mags.push(v);
    }
  } else {
    const numChannels = channels.length;
    const channelScale = 1 / numChannels;
    for (let s = 0; s < length; s += stride) {
      let v = 0;
      for (let ch = 0; ch < numChannels; ch++) v += channels[ch][s];
      v = Math.abs(v * channelScale);
      if (v > SILENCE_FLOOR) mags.push(v);
    }
  }
  if (mags.length === 0) return 1;
  mags.sort((a, b) => a - b);
  const median = mags[Math.floor(mags.length / 2)];
  if (median <= 0) return 1;
  return Math.max(0.25, Math.min(25, UNIFORM_WAVEFORM_TARGET / median));
}

/**
 * Copy each channel out of an `AudioBuffer` so the result can be
 * shipped to a worker (or held independently of the original buffer).
 * The copy is intentional: the AudioBuffer stays exclusively owned by
 * the main thread for sample-accurate BufferSource playback.
 */
export function extractChannels(buffer: AudioBuffer): ChannelData {
  const numChannels = buffer.numberOfChannels;
  const channels: Float32Array[] = new Array(numChannels);
  for (let ch = 0; ch < numChannels; ch++) {
    const src = buffer.getChannelData(ch);
    // `slice()` produces a brand-new ArrayBuffer-backed Float32Array,
    // which is what we want when transferring to a worker.
    channels[ch] = src.slice();
  }
  return { channels, sampleRate: buffer.sampleRate, length: buffer.length };
}

/**
 * Bar-by-bar peak extraction; the canvas-mixer waveform. Mirrors the
 * legacy `computeWaveformPeaks` semantics: each pixel column inside a
 * bar's pixel range maps to the bar's audio-time slice (= jot-time +
 * `drumsT0Sec`), and the [min, max] envelope of the channels collapsed
 * to mono goes into `peaks[2*p, 2*p+1]`. Pixels outside any bar stay
 * at 0/0 (the array is zero-initialised by `Float32Array`).
 */
export function computeWaveformPeaksFromChannels(
  data: ChannelData,
  bars: BarSlice[],
  totalWidthPx: number,
  drumsT0Sec: number
): Float32Array {
  const peaks = new Float32Array(totalWidthPx * 2);
  if (totalWidthPx <= 0 || bars.length === 0) return peaks;
  const { channels, sampleRate, length } = data;
  const numChannels = channels.length;
  const channelScale = numChannels > 0 ? 1 / numChannels : 1;
  for (const bar of bars) {
    const x0 = bar.x;
    const w = bar.width;
    const pxStart = Math.max(0, Math.floor(x0));
    const pxEnd = Math.min(totalWidthPx, Math.ceil(x0 + w));
    for (let p = pxStart; p < pxEnd; p++) {
      const frac0 = (p - x0) / w;
      const frac1 = (p + 1 - x0) / w;
      const tJot0 = bar.startSec + frac0 * bar.durationSec;
      const tJot1 = bar.startSec + frac1 * bar.durationSec;
      const tAudio0 = tJot0 + drumsT0Sec;
      const tAudio1 = tJot1 + drumsT0Sec;
      const s0 = Math.max(0, Math.floor(tAudio0 * sampleRate));
      const s1 = Math.min(length, Math.ceil(tAudio1 * sampleRate));
      writePixelPeak(channels, numChannels, channelScale, s0, s1, peaks, p * 2);
    }
  }
  return peaks;
}

/**
 * Arbitrary audio-time window peak extraction. Used by the timing-viz
 * snippet next to each note's debug overlay; `startSec` /
 * `durationSec` are in the buffer's own time frame (seconds from
 * t=0). Out-of-buffer pixels write 0/0 so silent edges render flat
 * instead of throwing.
 */
export function computeWindowPeaksFromChannels(
  data: ChannelData,
  startSec: number,
  durationSec: number,
  widthPx: number
): Float32Array {
  const peaks = new Float32Array(widthPx * 2);
  if (widthPx <= 0 || durationSec <= 0) return peaks;
  const { channels, sampleRate, length } = data;
  const numChannels = channels.length;
  const channelScale = numChannels > 0 ? 1 / numChannels : 1;
  const secPerPx = durationSec / widthPx;
  for (let p = 0; p < widthPx; p++) {
    const t0 = startSec + p * secPerPx;
    const t1 = startSec + (p + 1) * secPerPx;
    const s0 = Math.max(0, Math.floor(t0 * sampleRate));
    const s1 = Math.min(length, Math.ceil(t1 * sampleRate));
    writePixelPeak(channels, numChannels, channelScale, s0, s1, peaks, p * 2);
  }
  return peaks;
}

/**
 * Scan one pixel column's worth of samples across every channel, fold
 * them to mono on the fly, and write the [min, max] envelope into
 * `peaks[pIdx]` / `peaks[pIdx + 1]`. Mono / stereo get specialised
 * inner loops (the common cases, saves a tight inner-loop branch and
 * a multiplication per sample); >2 channels fall through to a generic
 * sum. Empty ranges write zeros so silent regions render flat.
 */
function writePixelPeak(
  channels: Float32Array[],
  numChannels: number,
  channelScale: number,
  s0: number,
  s1: number,
  peaks: Float32Array,
  pIdx: number
): void {
  if (s1 <= s0) {
    peaks[pIdx] = 0;
    peaks[pIdx + 1] = 0;
    return;
  }
  let mn = Infinity;
  let mx = -Infinity;
  if (numChannels === 1) {
    const data = channels[0];
    for (let s = s0; s < s1; s++) {
      const v = data[s];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
  } else if (numChannels === 2) {
    const c0 = channels[0];
    const c1 = channels[1];
    for (let s = s0; s < s1; s++) {
      const v = (c0[s] + c1[s]) * 0.5;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
  } else {
    for (let s = s0; s < s1; s++) {
      let v = 0;
      for (let ch = 0; ch < numChannels; ch++) v += channels[ch][s];
      v *= channelScale;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
  }
  if (mn === Infinity) {
    peaks[pIdx] = 0;
    peaks[pIdx + 1] = 0;
  } else {
    peaks[pIdx] = mn;
    peaks[pIdx + 1] = mx;
  }
}
