import { readFile } from '@tauri-apps/plugin-fs';
import { convertFileSrc } from '@tauri-apps/api/core';
import {
  PROTOCOL_VERSION,
  type ClientMessage,
  type ResultRef,
} from 'src/net/control_protocol';
import { type CapabilityPresenter } from './capability_presenter';
import { type DesktopBridge } from './desktop_bridge';

export type TranscribeResult = {
  /** Predicted-onsets MIDI bytes (feed to `src/midi/from_midi.ts`). */
  midi: Uint8Array;
  /** Separated audio (stems + drumless) as loadable URLs for Web Audio. */
  audioUrls: string[];
};

/** Resolve a backend result reference to bytes. Local path → fs read; remote
 *  url → fetch; tiny inline → base64. */
export async function resolveBytes(ref: ResultRef): Promise<Uint8Array> {
  if (ref.kind === 'path') {
    return readFile(ref.path);
  }
  if (ref.kind === 'url') {
    return new Uint8Array(await (await fetch(ref.url)).arrayBuffer());
  }
  const binary = atob(ref.bytesB64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

/** A loadable URL for an audio artifact: local paths go through Tauri's asset
 *  protocol (range-request streaming for Web Audio), remote urls pass through. */
export function resolveMediaUrl(ref: ResultRef): string {
  if (ref.kind === 'path') {
    return convertFileSrc(ref.path);
  }
  if (ref.kind === 'url') {
    return ref.url;
  }
  return `data:application/octet-stream;base64,${ref.bytesB64}`;
}

/**
 * Drives a transcription over the sidecar (desktop): ensures the transcription
 * capability is installed, runs the job through the Rust broker, and returns the
 * MIDI bytes + stem URLs. The bytes-resolver is injectable so this is
 * unit-testable without a Tauri runtime.
 */
export class DesktopTranscriber {
  constructor(
    private readonly bridge: DesktopBridge,
    private readonly capabilities: CapabilityPresenter,
    private readonly toBytes: (ref: ResultRef) => Promise<Uint8Array> = resolveBytes,
    private readonly toMediaUrl: (ref: ResultRef) => string = resolveMediaUrl,
  ) {}

  async transcribe(
    audioPath: string,
    opts: {
      params?: Record<string, unknown>;
      onProgress?: (stage: string, frac: number) => void;
    } = {},
  ): Promise<TranscribeResult> {
    if (!this.capabilities.store.isReady('transcription')) {
      await this.capabilities.install('transcription');
      if (!this.capabilities.store.isReady('transcription')) {
        throw new Error('transcription capability is not available');
      }
    }

    const request: ClientMessage = {
      v: PROTOCOL_VERSION,
      type: 'request',
      id: crypto.randomUUID(),
      op: 'transcribe',
      args: { audio: { kind: 'path', path: audioPath }, params: opts.params ?? {} },
    };

    let midiRef: ResultRef | undefined;
    const audioRefs: ResultRef[] = [];
    let failure: string | undefined;

    await this.bridge.runJob(request, (msg) => {
      if (msg.type === 'progress') {
        opts.onProgress?.(msg.stage, msg.frac);
      } else if (msg.type === 'result') {
        for (const artifact of msg.artifacts) {
          if (artifact.role === 'midi') {
            midiRef = artifact.ref;
          } else {
            audioRefs.push(artifact.ref);
          }
        }
      } else if (msg.type === 'error') {
        failure = `${msg.code}: ${msg.message}`;
      }
    });

    if (failure != null) {
      throw new Error(`transcribe failed (${failure})`);
    }
    if (midiRef == null) {
      throw new Error('transcribe produced no MIDI');
    }
    return {
      midi: await this.toBytes(midiRef),
      audioUrls: audioRefs.map((ref) => this.toMediaUrl(ref)),
    };
  }
}
