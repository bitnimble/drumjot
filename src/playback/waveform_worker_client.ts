/**
 * Main-thread client for the waveform worker.
 *
 * The worker (see {@link ./waveform_worker}) keeps a copy of each
 * loaded track's PCM and serves two coexisting APIs:
 *
 *  - **Peaks-returning** ({@link computePeaks} / {@link computeWindow},
 *    used by the minimap and the per-note timing-viz snippet): the
 *    worker computes `[min, max]` per pixel column and ships the
 *    `Float32Array` back as a transferable.
 *  - **OffscreenCanvas** ({@link attachChunk} / {@link renderChunk} /
 *    {@link releaseChunk}, used by the mixer's per-chunk waveform
 *    tiles): the tile's `<canvas>` is transferred to the worker once
 *    on mount, after which the worker computes peaks AND paints
 *    directly into it. No bytes cross back to the main thread on
 *    redraw, so a sustained zoom gesture costs ~0 main-thread ms
 *    regardless of how many tiles are visible.
 *
 * Lifecycle:
 *  - The worker is spawned lazily on first use (singleton).
 *  - {@link registerTrack} copies the PCM out of the live `AudioBuffer`
 *    once and ships it across; subsequent peak / render requests are
 *    just-the-bars + width, no buffer payload.
 *  - {@link dropTrack} frees the worker-side copy when the track is
 *    cleared.
 *
 * Workers are assumed available (every supported browser ships them;
 * see AGENTS.md §5.11). Construction failure throws; a test that
 * exercises these APIs without mocking will fail loudly rather than
 * silently degrading to a main-thread fallback that masks the missing
 * coverage.
 */
import { makeAutoObservable, runInAction } from 'mobx';
import { AudioTrackId } from './audio_tracks';
import {
  BarSlice,
  computeTrackAmpScale,
  extractChannels,
} from './waveform_compute';
import type {
  WaveformWorkerRequest,
  WaveformWorkerResponse,
} from './waveform_worker';

export type { BarSlice } from './waveform_compute';

class WaveformWorkerClient {
  private worker: Worker | undefined;
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
   * Lazily construct (or return) the Worker. Deferred to first use so
   * module load stays side-effect-free for test runners that never
   * exercise the worker code path.
   */
  private ensureWorker(): Worker {
    if (this.worker) return this.worker;
    const w = new Worker(new URL('./waveform_worker.ts', import.meta.url), {
      type: 'module',
    });
    w.onmessage = (e: MessageEvent<WaveformWorkerResponse>) => this.onMessage(e.data);
    w.onerror = (err) => {
      console.error('[waveform-worker] uncaught:', err);
    };
    this.worker = w;
    return w;
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
    const transfer: Transferable[] = data.channels.map((c) => c.buffer);
    this.ensureWorker().postMessage(
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
    this.ensureWorker().postMessage({ kind: 'drop', id } satisfies WaveformWorkerRequest);
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

  /**
   * Hand the worker control of a tile's `<canvas>` so it can paint
   * into it directly without the peak bytes ever reaching the main
   * thread. Call once per tile on mount, after
   * `HTMLCanvasElement.transferControlToOffscreen()` produces the
   * `OffscreenCanvas`. `chunkKey` must be globally unique across
   * tracks; the convention is `${trackId}:${chunk.key}`.
   */
  attachChunk(chunkKey: string, canvas: OffscreenCanvas, trackId: AudioTrackId): void {
    this.ensureWorker().postMessage(
      { kind: 'attachChunk', chunkKey, trackId, canvas } satisfies WaveformWorkerRequest,
      [canvas],
    );
  }

  /**
   * Trigger a (re)paint of an already-attached tile. Fire-and-forget;
   * no Promise to await. The worker recomputes peaks against the
   * stored PCM and paints into the tile's `OffscreenCanvas` directly.
   * Cheap to call rapidly (callers should still rAF-coalesce sustained
   * gestures so the queue doesn't pile up faster than the worker can
   * drain; see `mixer.tsx`).
   */
  renderChunk(
    chunkKey: string,
    bars: BarSlice[],
    widthPx: number,
    height: number,
    backingW: number,
    backingH: number,
    drumsT0Sec: number,
    pitchColor: string,
    ampScale: number,
  ): void {
    this.ensureWorker().postMessage({
      kind: 'renderChunk',
      chunkKey,
      bars,
      widthPx,
      height,
      backingW,
      backingH,
      drumsT0Sec,
      pitchColor,
      ampScale,
    } satisfies WaveformWorkerRequest);
  }

  /**
   * Drop the worker-side slot for a tile. Called on tile unmount so
   * the worker doesn't accumulate dead `OffscreenCanvas` references.
   */
  releaseChunk(chunkKey: string): void {
    this.ensureWorker().postMessage({ kind: 'releaseChunk', chunkKey } satisfies WaveformWorkerRequest);
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
