/**
 * Main-thread client for the waveform workers.
 *
 * Architecture: one Worker per audio track. Spawned on
 * {@link registerTrack}, terminated on {@link dropTrack}. The worker
 * module itself (see {@link ./waveform_worker}) is unchanged and still
 * keyed internally by track id; each instance just happens to own the
 * one track it was spawned for, so cross-track peak compute and
 * OffscreenCanvas tile painting overlap up to hardwareConcurrency
 * cores. For a typical mixer with N tracks, a sustained zoom gesture
 * fans out across N worker threads instead of serializing on one.
 *
 * Two coexisting modes (each scoped to a single track):
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
 * Chunk routing: callers only know the `chunkKey` (string), not the
 * trackId, when calling {@link renderChunk} / {@link releaseChunk}.
 * {@link attachChunk} records the chunkKey-to-trackId mapping so
 * later chunk calls find the right worker. {@link dropTrack} also
 * sweeps this map so a chunk message arriving after its owner was
 * dropped is a safe no-op.
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

/**
 * Per-track Worker wrapper. Owns the Worker, the request-id counter,
 * and the pending-promise map for `peaks`/`window` requests. Each
 * handle stores PCM for exactly one track; the worker module's
 * internal `buffers` Map still exists but only ever holds the single
 * track this handle was spawned for.
 */
class TrackWorkerHandle {
  readonly worker: Worker;
  private nextReqId = 1;
  private pending: Map<
    number,
    { resolve: (peaks: Float32Array) => void; reject: (err: Error) => void }
  > = new Map();

  constructor() {
    this.worker = new Worker(new URL('./waveform_worker.ts', import.meta.url), {
      type: 'module',
    });
    this.worker.onmessage = (e: MessageEvent<WaveformWorkerResponse>) => {
      const msg = e.data;
      const pending = this.pending.get(msg.reqId);
      if (!pending) return;
      this.pending.delete(msg.reqId);
      if (msg.kind === 'result') pending.resolve(msg.peaks);
      else pending.reject(new Error(msg.message));
    };
    this.worker.onerror = (err) => {
      console.error('[waveform-worker] uncaught:', err);
    };
  }

  post(msg: WaveformWorkerRequest, transfer: Transferable[] = []): void {
    this.worker.postMessage(msg, transfer);
  }

  request(
    base:
      | {
          kind: 'peaks';
          id: AudioTrackId;
          bars: BarSlice[];
          totalWidthPx: number;
          songLeadInSec: number;
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
    return new Promise<Float32Array>((resolve, reject) => {
      this.pending.set(reqId, { resolve, reject });
      this.worker.postMessage({ ...base, reqId } satisfies WaveformWorkerRequest);
    });
  }

  terminate(): void {
    this.worker.terminate();
    // Reject any in-flight peak/window promises so callers don't hang
    // on a worker that no longer exists.
    for (const { reject } of this.pending.values()) {
      reject(new Error('audio track was dropped'));
    }
    this.pending.clear();
  }
}

class WaveformWorkerClient {
  private workers: Map<AudioTrackId, TrackWorkerHandle> = new Map();
  /**
   * chunkKey to owning trackId so {@link renderChunk} and
   * {@link releaseChunk} can route to the right worker without their
   * callers having to thread trackId through. Populated by
   * {@link attachChunk}, pruned by {@link releaseChunk} and
   * {@link dropTrack}.
   */
  private chunkOwner: Map<string, AudioTrackId> = new Map();
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

  constructor() {
    makeAutoObservable(this);
  }

  /**
   * Spawn a dedicated worker for the track and hand it a copy of the
   * decoded PCM. Sent once per track on load. Channel arrays are
   * transferred (the worker takes ownership), which is safe because
   * {@link extractChannels} just produced fresh copies; the original
   * `AudioBuffer` stays untouched and usable by the BufferSource
   * playback path.
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
    // Defensive: ids are monotonic so re-registration shouldn't
    // happen, but if it ever does, tear down the prior worker first
    // so we don't leak it.
    this.workers.get(id)?.terminate();
    const handle = new TrackWorkerHandle();
    this.workers.set(id, handle);
    const transfer: Transferable[] = data.channels.map((c) => c.buffer);
    handle.post(
      {
        kind: 'register',
        id,
        channels: data.channels,
        sampleRate: data.sampleRate,
        length: data.length,
      },
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
   * Drop a track: terminate its worker (also rejects any in-flight
   * peak/window promises) and forget every chunkKey owned by it.
   * Sent when the track is cleared. Chunk tiles still mounted will
   * fire {@link releaseChunk} on unmount; with the routing entries
   * gone, those calls are safe no-ops.
   */
  dropTrack(id: AudioTrackId): void {
    runInAction(() => {
      this.ampScales.delete(id);
    });
    const handle = this.workers.get(id);
    if (handle) {
      handle.terminate();
      this.workers.delete(id);
    }
    for (const [chunkKey, owner] of this.chunkOwner) {
      if (owner === id) this.chunkOwner.delete(chunkKey);
    }
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
    songLeadInSec: number,
  ): Promise<Float32Array> {
    const handle = this.workers.get(id);
    if (!handle) return Promise.reject(new Error(`unregistered track ${id}`));
    return handle.request({
      kind: 'peaks',
      id,
      bars,
      totalWidthPx: widthPx,
      songLeadInSec,
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
    const handle = this.workers.get(id);
    if (!handle) return Promise.reject(new Error(`unregistered track ${id}`));
    return handle.request({
      kind: 'window',
      id,
      startSec,
      durationSec,
      widthPx,
    });
  }

  /**
   * Hand the track's worker control of a tile's `<canvas>` so it can
   * paint into it directly without the peak bytes ever reaching the
   * main thread. Call once per tile on mount, after
   * `HTMLCanvasElement.transferControlToOffscreen()` produces the
   * `OffscreenCanvas`. `chunkKey` must be globally unique across
   * tracks; the convention is `${trackId}:${chunk.key}`.
   *
   * Records the chunkKey-to-trackId mapping so later renderChunk /
   * releaseChunk calls (which carry only the chunkKey) can find the
   * right worker. If the track was already dropped between the React
   * mount and this effect, the call is a safe no-op.
   */
  attachChunk(chunkKey: string, canvas: OffscreenCanvas, trackId: AudioTrackId): void {
    const handle = this.workers.get(trackId);
    if (!handle) return;
    this.chunkOwner.set(chunkKey, trackId);
    handle.post(
      { kind: 'attachChunk', chunkKey, trackId, canvas },
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
    songLeadInSec: number,
    laneColor: string,
    ampScale: number,
  ): void {
    const owner = this.chunkOwner.get(chunkKey);
    if (owner === undefined) return;
    const handle = this.workers.get(owner);
    if (!handle) return;
    handle.post({
      kind: 'renderChunk',
      chunkKey,
      bars,
      widthPx,
      height,
      backingW,
      backingH,
      songLeadInSec,
      laneColor,
      ampScale,
    });
  }

  /**
   * Drop the worker-side slot for a tile. Called on tile unmount so
   * the worker doesn't accumulate dead `OffscreenCanvas` references.
   */
  releaseChunk(chunkKey: string): void {
    const owner = this.chunkOwner.get(chunkKey);
    if (owner === undefined) return;
    this.chunkOwner.delete(chunkKey);
    const handle = this.workers.get(owner);
    if (!handle) return;
    handle.post({ kind: 'releaseChunk', chunkKey });
  }
}

export const waveformWorker = new WaveformWorkerClient();
