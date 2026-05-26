/**
 * Main-thread client for the waveform peaks worker.
 *
 * The worker (see {@link ./waveform_worker}) keeps a copy of each
 * loaded track's PCM so peaks can be recomputed (per-zoom redraw,
 * per-onset timing-viz) without blocking the main thread during heavy
 * UI work like score relayout.
 *
 * Lifecycle:
 *  - The worker is spawned lazily on first use (singleton).
 *  - {@link registerTrack} copies the PCM out of the live `AudioBuffer`
 *    once and ships it across; subsequent peak requests are
 *    just-the-bars + width, no buffer payload.
 *  - {@link dropTrack} frees the worker-side copy when the track is
 *    cleared.
 *
 * If the worker can't be constructed (very old runtime, test env
 * without `Worker`), the client falls back to running the same pure
 * compute functions on the main thread. The async API stays the same
 * so callers don't branch.
 */
import { makeAutoObservable, runInAction } from 'mobx';
import { AudioTrackId } from './audio_tracks';
import {
  BarSlice,
  ChannelData,
  computeTrackAmpScale,
  computeWaveformPeaksFromChannels,
  computeWindowPeaksFromChannels,
  extractChannels,
} from './waveform_compute';
import type {
  WaveformWorkerRequest,
  WaveformWorkerResponse,
} from './waveform_worker';

export type { BarSlice } from './waveform_compute';

class WaveformWorkerClient {
  private worker: Worker | undefined;
  // On the fallback (no Worker) path we keep the channel copies here
  // and execute compute synchronously so callers still see the same
  // Promise<Float32Array> shape.
  private fallback: Map<AudioTrackId, ChannelData> = new Map();
  /**
   * Per-track uniform-waveform amplitude scale. Computed once on
   * {@link registerTrack} (cheap subsampled scan, ~1 ms) so every
   * chunk normalises against the same value; no amplitude seams
   * between neighbouring chunks of the same track. Observable so a
   * canvas / waveform consumer re-renders the moment registration
   * publishes the real scale (before registration, {@link getAmpScale}
   * returns 1 as a passthrough).
   */
  ampScales: Map<AudioTrackId, number> = new Map();
  private nextReqId = 1;
  private pending: Map<
    number,
    { resolve: (peaks: Float32Array) => void; reject: (err: Error) => void }
  > = new Map();

  constructor() {
    makeAutoObservable(this);
  }

  /**
   * Lazily construct (or return) the Worker. We don't build it at
   * import time because `Worker` doesn't exist in unit-test
   * environments (bun's `bun test` runs a pure JS host) and we'd
   * crash the module before any test could opt out via
   * {@link forceFallback}.
   */
  private ensureWorker(): Worker | undefined {
    if (this.worker) return this.worker;
    if (typeof Worker === 'undefined') return undefined;
    try {
      const w = new Worker(new URL('./waveform_worker.ts', import.meta.url), {
        type: 'module',
      });
      w.onmessage = (e: MessageEvent<WaveformWorkerResponse>) => this.onMessage(e.data);
      w.onerror = (err) => {
        console.error('[waveform-worker] uncaught:', err);
      };
      this.worker = w;
      return w;
    } catch (err) {
      console.warn(
        '[waveform-worker] could not spawn (running compute on main thread):',
        err,
      );
      return undefined;
    }
  }

  /**
   * Hand the worker a copy of the decoded PCM. Sent once per track on
   * load. Channel arrays are transferred (the worker takes
   * ownership), which is safe because {@link extractChannels} just
   * produced fresh copies; the original `AudioBuffer` stays
   * untouched and usable by the BufferSource playback path.
   */
  registerTrack(id: AudioTrackId, buffer: AudioBuffer): void {
    const data = extractChannels(buffer);
    // Per-track amplitude scale: computed BEFORE the channel data is
    // (potentially) transferred to the worker so we still have local
    // access to the Float32Arrays. ~1 ms even on long tracks (the
    // function strides through ~10 k samples).
    const scale = computeTrackAmpScale(data);
    runInAction(() => {
      this.ampScales.set(id, scale);
    });
    const worker = this.ensureWorker();
    if (!worker) {
      this.fallback.set(id, data);
      return;
    }
    const transfer: Transferable[] = data.channels.map((c) => c.buffer);
    worker.postMessage(
      {
        kind: 'register',
        id,
        channels: data.channels,
        sampleRate: data.sampleRate,
        length: data.length,
      } satisfies WaveformWorkerRequest,
      transfer,
    );
  }

  /**
   * Synchronous lookup for the per-track uniform-amplitude scale.
   * Returns `1` (= passthrough) before the track is registered, so
   * callers don't have to wait on a Promise.
   */
  getAmpScale(id: AudioTrackId): number {
    return this.ampScales.get(id) ?? 1;
  }

  /**
   * Drop the worker-side PCM copy for `id`. Sent when the track is
   * cleared so the worker doesn't accumulate dead buffers across a
   * session.
   */
  dropTrack(id: AudioTrackId): void {
    runInAction(() => {
      this.ampScales.delete(id);
    });
    if (this.fallback.delete(id)) return;
    const worker = this.ensureWorker();
    if (!worker) return;
    worker.postMessage({ kind: 'drop', id } satisfies WaveformWorkerRequest);
  }

  /**
   * Peaks for an arbitrary contiguous region of the score, sized to
   * `widthPx` pixels (chunk-local). `bars` is the pre-shifted bar
   * layout; each bar's `x` is in chunk-local pixel coordinates
   * (negative when the bar extends to the left of the chunk; past
   * `widthPx` when it extends to the right). Bars outside the chunk
   * naturally drop out via the existing pixel-range clamp inside the
   * compute fn, so callers don't need to filter.
   *
   * Used by the tiled mixer waveform: one call per visible chunk.
   * Each chunk picks its own `widthPx` (= `chunkBeats *
   * chunkRenderedPxPerBeat`), so chunks at high zoom get sharper
   * bitmaps independently of any global canvas-dimension cap.
   */
  computePeaks(
    id: AudioTrackId,
    bars: BarSlice[],
    widthPx: number,
    drumsT0Sec: number,
  ): Promise<Float32Array> {
    return this.request({
      kind: 'peaks',
      id,
      bars,
      totalWidthPx: widthPx,
      drumsT0Sec,
    });
  }

  /**
   * Arbitrary buffer-time window peaks for `id`, for the per-note
   * timing-viz snippet.
   */
  computeWindow(
    id: AudioTrackId,
    startSec: number,
    durationSec: number,
    widthPx: number,
  ): Promise<Float32Array> {
    return this.request({
      kind: 'window',
      id,
      startSec,
      durationSec,
      widthPx,
    });
  }

  private request(
    base:
      | {
          kind: 'peaks';
          id: AudioTrackId;
          bars: BarSlice[];
          totalWidthPx: number;
          drumsT0Sec: number;
        }
      | {
          kind: 'window';
          id: AudioTrackId;
          startSec: number;
          durationSec: number;
          widthPx: number;
        },
  ): Promise<Float32Array> {
    const reqId = this.nextReqId++;
    const worker = this.ensureWorker();
    if (!worker) {
      // Synchronous fallback (no Worker available). Still returns a
      // Promise so the caller's shape is identical.
      try {
        const data = this.fallback.get(base.id);
        if (!data) {
          return Promise.reject(new Error(`unregistered track ${base.id}`));
        }
        const peaks =
          base.kind === 'peaks'
            ? computeWaveformPeaksFromChannels(
                data,
                base.bars,
                base.totalWidthPx,
                base.drumsT0Sec,
              )
            : computeWindowPeaksFromChannels(
                data,
                base.startSec,
                base.durationSec,
                base.widthPx,
              );
        return Promise.resolve(peaks);
      } catch (err) {
        return Promise.reject(err instanceof Error ? err : new Error(String(err)));
      }
    }
    return new Promise<Float32Array>((resolve, reject) => {
      this.pending.set(reqId, { resolve, reject });
      worker.postMessage({ ...base, reqId } satisfies WaveformWorkerRequest);
    });
  }

  private onMessage(msg: WaveformWorkerResponse): void {
    const pending = this.pending.get(msg.reqId);
    if (!pending) return;
    this.pending.delete(msg.reqId);
    if (msg.kind === 'result') pending.resolve(msg.peaks);
    else pending.reject(new Error(msg.message));
  }
}

export const waveformWorker = new WaveformWorkerClient();
