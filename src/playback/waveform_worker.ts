/// <reference lib="webworker" />
/**
 * Worker that owns a copy of each loaded audio track's PCM (one
 * `Float32Array` per channel) and computes waveform peaks off the
 * main thread. Owned by {@link waveform_worker_client.ts}; clients
 * never talk to it directly.
 *
 * Protocol (see {@link WaveformWorkerRequest} / {@link
 * WaveformWorkerResponse}):
 *
 *  - `register`: stash the PCM for a track id. Sent once on load.
 *  - `drop`: free the PCM for a track id. Sent on track clear.
 *  - `peaks`: compute mixer-waveform peaks against stored PCM and
 *             reply with a transferable `Float32Array`.
 *  - `window`: same for the per-note timing-viz snippet.
 *
 * All responses carry the originating `reqId` so the client can match
 * them to the right pending Promise.
 */
import {
  BarSlice,
  ChannelData,
  computeWaveformPeaksFromChannels,
  computeWindowPeaksFromChannels,
} from './waveform_compute';

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
      drumsT0Sec: number;
    }
  | {
      kind: 'window';
      reqId: number;
      id: string;
      startSec: number;
      durationSec: number;
      widthPx: number;
    };

export type WaveformWorkerResponse =
  | { kind: 'result'; reqId: number; peaks: Float32Array }
  | { kind: 'error'; reqId: number; message: string };

const buffers = new Map<string, ChannelData>();

const ctx = self as unknown as DedicatedWorkerGlobalScope;
ctx.onmessage = (e: MessageEvent<WaveformWorkerRequest>) => {
  const msg = e.data;
  switch (msg.kind) {
    case 'register': {
      // The channel arrays arrive as transferables (the client copies
      // off the live `AudioBuffer` before postMessaging), so the
      // worker takes sole ownership of these buffers.
      buffers.set(msg.id, {
        channels: msg.channels,
        sampleRate: msg.sampleRate,
        length: msg.length,
      });
      return;
    }
    case 'drop': {
      buffers.delete(msg.id);
      return;
    }
    case 'peaks': {
      const data = buffers.get(msg.id);
      if (!data) {
        reply({ kind: 'error', reqId: msg.reqId, message: `unregistered track ${msg.id}` });
        return;
      }
      const peaks = computeWaveformPeaksFromChannels(
        data,
        msg.bars,
        msg.totalWidthPx,
        msg.drumsT0Sec,
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
      const peaks = computeWindowPeaksFromChannels(
        data,
        msg.startSec,
        msg.durationSec,
        msg.widthPx,
      );
      reply({ kind: 'result', reqId: msg.reqId, peaks }, [peaks.buffer]);
      return;
    }
  }
};

function reply(msg: WaveformWorkerResponse, transfer: Transferable[] = []): void {
  ctx.postMessage(msg, transfer);
}
