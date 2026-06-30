/// <reference lib="webworker" />
/**
 * Worker that owns a copy of each loaded audio track's PCM (one
 * `Float32Array` per channel) and computes / paints waveform tiles
 * off the main thread. Owned by {@link waveform_worker_client.ts};
 * clients never talk to it directly.
 *
 * Two coexisting modes:
 *
 *  - **Peaks-returning mode** (legacy, still used by the minimap and
 *    the per-note timing-viz snippet): main thread asks for a
 *    `Float32Array` of `[min, max]` per pixel column and renders it
 *    itself. `peaks` / `window` requests.
 *  - **OffscreenCanvas mode** (the mixer's per-chunk waveform tiles):
 *    main thread `transferControlToOffscreen()`s the tile's canvas to
 *    the worker once at mount; thereafter the worker computes peaks
 *    AND paints directly into the OffscreenCanvas, no bytes crossing
 *    back to the main thread. Drawing happens entirely off-main, so
 *    a sustained wheel-zoom gesture costs ~0 main-thread ms per
 *    frame regardless of how many tiles are visible.
 *
 * Protocol (see {@link WaveformWorkerRequest} /
 * {@link WaveformWorkerResponse}):
 *
 *  - `register`: stash the PCM for a track id. Sent once on load.
 *  - `drop`: free the PCM for a track id. Sent on track clear.
 *  - `peaks`: compute peaks against stored PCM and reply with a
 *             transferable `Float32Array`.
 *  - `window`: same for the per-note timing-viz snippet.
 *  - `attachChunk`: stash the transferred `OffscreenCanvas` for a
 *                   chunk under its globally-unique key. Sent once
 *                   per tile, on mount.
 *  - `renderChunk`: compute peaks + paint them into the chunk's
 *                   `OffscreenCanvas`. Fire-and-forget; no reply.
 *  - `releaseChunk`: drop the worker-side `OffscreenCanvas` slot.
 *                    Sent on tile unmount.
 *
 * All peaks-mode responses carry the originating `reqId` so the
 * client can match them to the right pending Promise.
 */
import {
  BarSlice,
  buildTrackPeaks,
  computeWaveformPeaks,
  computeWindowPeaks,
  TrackPeaks,
} from './waveform_compute';
import { paintWaveform } from './waveform_paint';

export type WaveformWorkerRequest =
  | {
      kind: 'register';
      id: string;
      channels: Float32Array[];
      sampleRate: number;
      length: number;
    }
  | { kind: 'drop'; id: string }
  | {
      kind: 'peaks';
      reqId: number;
      id: string;
      bars: BarSlice[];
      totalWidthPx: number;
      songLeadInSec: number;
    }
  | {
      kind: 'window';
      reqId: number;
      id: string;
      startSec: number;
      durationSec: number;
      widthPx: number;
    }
  | {
      kind: 'attachChunk';
      chunkKey: string;
      trackId: string;
      canvas: OffscreenCanvas;
    }
  | {
      kind: 'renderChunk';
      chunkKey: string;
      bars: BarSlice[];
      widthPx: number;
      height: number;
      backingW: number;
      backingH: number;
      songLeadInSec: number;
      laneColor: string;
      ampScale: number;
    }
  | { kind: 'releaseChunk'; chunkKey: string };

export type WaveformWorkerResponse =
  | { kind: 'result'; reqId: number; peaks: Float32Array }
  | { kind: 'error'; reqId: number; message: string };

const buffers = new Map<string, TrackPeaks>();

/**
 * Per-tile peak cache: the last computed peaks plus a signature of the inputs
 * that drive them (tile width, drum/audio offset, bar geometry). A repaint that
 * only changes colour / amplitude scale reuses these instead of re-querying the
 * pyramid. Pruned on `releaseChunk`.
 */
const peakCache = new Map<string, { sig: string; peaks: Float32Array }>();

/**
 * Per-tile slot: the `OffscreenCanvas` transferred from main thread
 * plus the audio track id that owns its PCM. Looked up by
 * `chunkKey = `${trackId}:${chunk.key}`` so two tracks with
 * coincidentally-equal `chunk.key` values don't collide.
 */
const attachedChunks = new Map<string, { canvas: OffscreenCanvas; trackId: string }>();

const ctx = self as unknown as DedicatedWorkerGlobalScope;
ctx.onmessage = (e: MessageEvent<WaveformWorkerRequest>) => {
  const msg = e.data;
  switch (msg.kind) {
    case 'register': {
      // The channel arrays arrive as transferables (the client copies off the
      // live `AudioBuffer` before postMessaging). Build the min/max pyramid
      // from them once, then let the raw PCM go: the pyramid is all the render
      // path needs and a fraction of the size. The channels are unreferenced
      // after this call and get GC'd.
      buffers.set(
        msg.id,
        buildTrackPeaks({
          channels: msg.channels,
          sampleRate: msg.sampleRate,
          length: msg.length,
        }),
      );
      // Rebuilding a track's pyramid (e.g. replacing the backing audio in
      // place) invalidates any peaks cached against the old audio. The cache
      // key (`chunkKey`) is per-track and survives a same-id re-register, so
      // clear it here rather than relying on the client tearing the worker
      // down first.
      peakCache.clear();
      return;
    }
    case 'drop': {
      buffers.delete(msg.id);
      peakCache.clear();
      return;
    }
    case 'peaks': {
      const data = buffers.get(msg.id);
      if (!data) {
        reply({ kind: 'error', reqId: msg.reqId, message: `unregistered track ${msg.id}` });
        return;
      }
      const peaks = computeWaveformPeaks(
        data,
        msg.bars,
        msg.totalWidthPx,
        msg.songLeadInSec,
      );
      reply({ kind: 'result', reqId: msg.reqId, peaks }, [peaks.buffer]);
      return;
    }
    case 'window': {
      const data = buffers.get(msg.id);
      if (!data) {
        reply({ kind: 'error', reqId: msg.reqId, message: `unregistered track ${msg.id}` });
        return;
      }
      const peaks = computeWindowPeaks(
        data,
        msg.startSec,
        msg.durationSec,
        msg.widthPx,
      );
      reply({ kind: 'result', reqId: msg.reqId, peaks }, [peaks.buffer]);
      return;
    }
    case 'attachChunk': {
      attachedChunks.set(msg.chunkKey, { canvas: msg.canvas, trackId: msg.trackId });
      return;
    }
    case 'renderChunk': {
      const slot = attachedChunks.get(msg.chunkKey);
      if (!slot) return; // late render after release; safe to drop
      const data = buffers.get(slot.trackId);
      if (!data) return; // track unregistered ahead of release; nothing to draw
      renderChunkInto(slot.canvas, data, msg);
      return;
    }
    case 'releaseChunk': {
      attachedChunks.delete(msg.chunkKey);
      peakCache.delete(msg.chunkKey);
      return;
    }
  }
};

/**
 * Paint a single tile. Mirrors the legacy main-thread render path
 * verbatim (peaks compute, vertical-line `fillRect` per column) so
 * the visual output is identical; the only difference is the
 * rendering context is an `OffscreenCanvas` owned by this worker
 * instead of an `HTMLCanvasElement` on the main thread.
 *
 * Re-assigning `canvas.width` / `canvas.height` resets the canvas
 * state (transform, clip, fillStyle, …), which serves as our
 * "implicit clear"; we still `clearRect` after the transform to be
 * explicit and to handle the no-dimension-change case.
 */
function renderChunkInto(
  canvas: OffscreenCanvas,
  data: TrackPeaks,
  msg: Extract<WaveformWorkerRequest, { kind: 'renderChunk' }>,
): void {
  const { chunkKey, bars, widthPx, height, backingW, backingH, songLeadInSec, laneColor, ampScale } =
    msg;
  if (widthPx <= 0 || height <= 0) return;
  // Reuse the last peaks for this tile when only paint params (colour /
  // amplitude scale) changed; recompute only when width / offset / bar geometry
  // change (see {@link peakCache}). Hash EVERY bar's fields, not just the
  // endpoints: a tempo/drift edit to a bar in the middle of the chunk must
  // invalidate too (chunks usually hold 1-2 bars, so this stays cheap).
  let sig = `${widthPx}|${songLeadInSec}`;
  for (const b of bars) {
    sig += `|${b.x},${b.width},${b.startSec},${b.durationSec},${b.driftSec ?? 0},${b.nextDriftSec ?? 0}`;
  }
  let peaks: Float32Array;
  const cached = peakCache.get(chunkKey);
  if (cached && cached.sig === sig) {
    peaks = cached.peaks;
  } else {
    peaks = computeWaveformPeaks(data, bars, widthPx, songLeadInSec);
    peakCache.set(chunkKey, { sig, peaks });
  }
  canvas.width = backingW;
  canvas.height = backingH;
  const ctx2d = canvas.getContext('2d');
  if (!ctx2d) return;
  paintWaveform(ctx2d, peaks, widthPx, height, backingW, backingH, laneColor, ampScale);
}

function reply(msg: WaveformWorkerResponse, transfer: Transferable[] = []): void {
  ctx.postMessage(msg, transfer);
}
