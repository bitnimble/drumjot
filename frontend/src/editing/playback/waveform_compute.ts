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
  /**
   * Performance drift (seconds) at this bar's downbeat and the next bar's,
   * from `globalMetadata.barDrift`. The bar's uniform pixel `width` maps onto
   * its REAL recorded span, `[startSec + driftSec, nextStartSec +
   * nextDriftSec)`, so transients render under the bar lines without any
   * raster resampling. Both 0 (no stretch) for a metronomic recording.
   */
  driftSec?: number;
  nextDriftSec?: number;
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
 * Multi-resolution min/max summary of a track's mono-folded PCM. `levels[0]`
 * holds the [min, max] of every {@link PYRAMID_BASE}-sample block; each higher
 * level halves the resolution (pairwise-reduced from the one below). A pixel
 * covering S samples reads ~1-2 blocks from the level whose `blockSize <= S`,
 * so a render is O(pixels) regardless of zoom instead of O(samples) -- the raw
 * PCM is scanned exactly ONCE (here, to build this) rather than on every paint.
 *
 * Mono-folded because the waveform displays the channel-averaged envelope, so
 * one pyramid serves any channel count and is a fraction of the raw PCM size
 * (the worker keeps only this and frees the channels after building it).
 */
type PyramidLevel = { min: Float32Array; max: Float32Array; blockSize: number };

export type WaveformPyramid = {
  levels: PyramidLevel[];
  length: number;
};

/** A track's render-ready peaks: the {@link WaveformPyramid} + the rate/length
 *  the per-pixel sample mapping needs. Replaces holding the raw channels. */
export type TrackPeaks = {
  pyramid: WaveformPyramid;
  sampleRate: number;
  length: number;
};

/**
 * Level-0 block size (samples). 16 keeps the summary accurate well past the
 * app's max zoom (which never shows fewer than ~tens of samples per pixel) and
 * the pyramid at ~1/8 the size of the mono PCM. Powers of two above it form
 * the higher levels.
 */
const PYRAMID_BASE = 16;

/** Build a track's {@link TrackPeaks} from decoded channels. One O(samples)
 *  pass (folds to mono + the level-0 blocks), then cheap pairwise reductions. */
export function buildTrackPeaks(data: ChannelData): TrackPeaks {
  return { pyramid: buildPyramid(data), sampleRate: data.sampleRate, length: data.length };
}

function buildPyramid(data: ChannelData): WaveformPyramid {
  const { channels, length } = data;
  const numChannels = channels.length;
  const n0 = Math.max(1, Math.ceil(length / PYRAMID_BASE));
  const min0 = new Float32Array(n0);
  const max0 = new Float32Array(n0);
  for (let b = 0; b < n0; b++) {
    const s0 = b * PYRAMID_BASE;
    const s1 = Math.min(length, s0 + PYRAMID_BASE);
    let mn = Infinity;
    let mx = -Infinity;
    if (numChannels === 1) {
      const d = channels[0];
      for (let s = s0; s < s1; s++) {
        const v = d[s];
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
    } else if (numChannels > 2) {
      const inv = 1 / numChannels;
      for (let s = s0; s < s1; s++) {
        let v = 0;
        for (let ch = 0; ch < numChannels; ch++) v += channels[ch][s];
        v *= inv;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    }
    min0[b] = mn === Infinity ? 0 : mn;
    max0[b] = mx === -Infinity ? 0 : mx;
  }
  const levels = [{ min: min0, max: max0, blockSize: PYRAMID_BASE }];
  let cur = levels[0];
  while (cur.min.length > 1) {
    const pn = cur.min.length;
    const nn = Math.ceil(pn / 2);
    const min = new Float32Array(nn);
    const max = new Float32Array(nn);
    for (let i = 0; i < nn; i++) {
      const a = 2 * i;
      let lo = cur.min[a];
      let hi = cur.max[a];
      if (a + 1 < pn) {
        if (cur.min[a + 1] < lo) lo = cur.min[a + 1];
        if (cur.max[a + 1] > hi) hi = cur.max[a + 1];
      }
      min[i] = lo;
      max[i] = hi;
    }
    cur = { min, max, blockSize: cur.blockSize * 2 };
    levels.push(cur);
  }
  return { levels, length };
}

/**
 * Pick the pyramid level whose blocks are no larger than `spp` samples, so a
 * pixel spanning `spp` samples reads ~1-2 blocks. Hoisted out of the per-pixel
 * loop -- the samples-per-pixel rate is ~uniform within a bar / window -- so the
 * `log2` runs once per bar, not once per pixel.
 */
function chooseLevel(pyramid: WaveformPyramid, spp: number): PyramidLevel {
  let level = 0;
  if (spp > PYRAMID_BASE) {
    level = Math.floor(Math.log2(spp / PYRAMID_BASE));
    if (level >= pyramid.levels.length) level = pyramid.levels.length - 1;
  }
  return pyramid.levels[level];
}

/**
 * Min/max of the mono envelope over `[s0, s1)`, written into `out[outIdx]` /
 * `out[outIdx + 1]`, reading whole blocks from `lvl` (chosen by
 * {@link chooseLevel}). ~1-2 reads per pixel; the block alignment over-reads by
 * at most one block (sub-pixel at these zooms, an exact-vs-approximate tradeoff
 * that's invisible on real audio).
 */
function queryLevel(
  lvl: PyramidLevel,
  lastBlock: number,
  s0: number,
  s1: number,
  out: Float32Array,
  outIdx: number
): void {
  if (s1 <= s0) {
    out[outIdx] = 0;
    out[outIdx + 1] = 0;
    return;
  }
  const { min, max, blockSize } = lvl;
  let b0 = Math.floor(s0 / blockSize);
  let b1 = Math.floor((s1 - 1) / blockSize);
  if (b0 < 0) b0 = 0;
  if (b1 > lastBlock) b1 = lastBlock;
  let mn = Infinity;
  let mx = -Infinity;
  for (let b = b0; b <= b1; b++) {
    if (min[b] < mn) mn = min[b];
    if (max[b] > mx) mx = max[b];
  }
  if (mn === Infinity) {
    out[outIdx] = 0;
    out[outIdx + 1] = 0;
  } else {
    out[outIdx] = mn;
    out[outIdx + 1] = mx;
  }
}

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
 * bar's pixel range maps to the bar's audio-time slice (= jot-time -
 * `songLeadIn`), and the [min, max] envelope of the channels collapsed
 * to mono goes into `peaks[2*p, 2*p+1]`. Pixels outside any bar stay
 * at 0/0 (the array is zero-initialised by `Float32Array`).
 */
export function computeWaveformPeaks(
  data: TrackPeaks,
  bars: BarSlice[],
  totalWidthPx: number,
  songLeadInSec: number
): Float32Array {
  const peaks = new Float32Array(totalWidthPx * 2);
  if (totalWidthPx <= 0 || bars.length === 0) return peaks;
  const { pyramid, sampleRate, length } = data;
  for (const bar of bars) {
    const x0 = bar.x;
    const w = bar.width;
    // The bar's uniform pixel width maps onto its REAL recorded span, not
    // its uniform jot span: shift the start by this bar's drift and stretch
    // the duration by the drift delta to the next bar. So a bar the drummer
    // held renders slightly compressed and the transient lands on the bar
    // line, a known scale baked into the per-pixel sample range here (no
    // bitmap resampling). Both drifts 0 ⇒ identical to the plain mapping.
    const drift = bar.driftSec ?? 0;
    const nextDrift = bar.nextDriftSec ?? drift;
    const audioStart = bar.startSec + drift - songLeadInSec;
    const audioDur = bar.durationSec + (nextDrift - drift);
    // Samples per pixel are ~uniform across the bar, so pick the pyramid level
    // once here rather than per pixel.
    const lvl = chooseLevel(pyramid, w > 0 ? Math.max(1, (audioDur * sampleRate) / w) : 1);
    const lastBlock = lvl.min.length - 1;
    const pxStart = Math.max(0, Math.floor(x0));
    const pxEnd = Math.min(totalWidthPx, Math.ceil(x0 + w));
    for (let p = pxStart; p < pxEnd; p++) {
      const frac0 = (p - x0) / w;
      const frac1 = (p + 1 - x0) / w;
      const tAudio0 = audioStart + frac0 * audioDur;
      const tAudio1 = audioStart + frac1 * audioDur;
      const s0 = Math.max(0, Math.floor(tAudio0 * sampleRate));
      const s1 = Math.min(length, Math.ceil(tAudio1 * sampleRate));
      queryLevel(lvl, lastBlock, s0, s1, peaks, p * 2);
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
export function computeWindowPeaks(
  data: TrackPeaks,
  startSec: number,
  durationSec: number,
  widthPx: number
): Float32Array {
  const peaks = new Float32Array(widthPx * 2);
  if (widthPx <= 0 || durationSec <= 0) return peaks;
  const { pyramid, sampleRate, length } = data;
  const secPerPx = durationSec / widthPx;
  // Uniform samples per pixel across the window: pick the level once.
  const lvl = chooseLevel(pyramid, Math.max(1, secPerPx * sampleRate));
  const lastBlock = lvl.min.length - 1;
  for (let p = 0; p < widthPx; p++) {
    const t0 = startSec + p * secPerPx;
    const t1 = startSec + (p + 1) * secPerPx;
    const s0 = Math.max(0, Math.floor(t0 * sampleRate));
    const s1 = Math.min(length, Math.ceil(t1 * sampleRate));
    queryLevel(lvl, lastBlock, s0, s1, peaks, p * 2);
  }
  return peaks;
}
