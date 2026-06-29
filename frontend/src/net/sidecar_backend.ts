import { convertFileSrc } from '@tauri-apps/api/core';
import {
  PROTOCOL_VERSION,
  type Artifact,
  type ClientMessage,
  type Op,
  type ResultRef,
} from './control_protocol';
import { TauriBridge, type DesktopBridge } from 'src/desktop/desktop_bridge';
import { type AudioInput, type BackendClient, type RunOptions, type RunResult } from './backend';

/**
 * Backend over the Tauri sidecar (desktop): a job per op through the Rust
 * broker (`run_job`), control-protocol frames in both directions. Blob inputs
 * are staged to a temp file the sidecar reads off disk. Web-safe to import, the
 * plugin-fs / path APIs are loaded lazily inside the methods so this never
 * touches a Tauri-only module in the browser boot path.
 *
 * Does NOT gate capabilities: callers gate at the UI entry point (the menu
 * action), so by the time a job runs the venv has the deps. See
 * CapabilityPresenter.requestCapability.
 */
export class SidecarBackendClient implements BackendClient {
  constructor(private readonly bridge: DesktopBridge = new TauriBridge()) {}

  async run(
    op: Op,
    audio: AudioInput,
    params: Record<string, unknown>,
    opts: RunOptions = {},
  ): Promise<RunResult> {
    const path = audio.kind === 'path' ? audio.path : await stageTempInput(audio.blob, audio.filename);
    const request: ClientMessage = {
      v: PROTOCOL_VERSION,
      type: 'request',
      id: crypto.randomUUID(),
      op,
      args: { audio: { kind: 'path', path }, params },
    };

    const artifacts: Artifact[] = [];
    let data: unknown;
    let failure: string | undefined;

    const onAbort = (): void => void this.bridge.cancelJob(request.id);
    opts.signal?.addEventListener('abort', onAbort, { once: true });
    try {
      await this.bridge.runJob(request, (msg) => {
        if (msg.type === 'progress') {
          opts.onProgress?.({ stage: msg.stage, frac: msg.frac, message: msg.message });
        } else if (msg.type === 'result') {
          artifacts.push(...msg.artifacts);
          data = msg.data;
        } else if (msg.type === 'error') {
          failure = `${msg.code}: ${msg.message}`;
        }
      });
    } finally {
      opts.signal?.removeEventListener('abort', onAbort);
    }

    if (opts.signal?.aborted) {
      throw new DOMException('Aborted', 'AbortError');
    }
    if (failure != null) {
      throw new Error(`${op} failed (${failure})`);
    }
    return { artifacts, data };
  }

  async resolveBytes(ref: ResultRef): Promise<Uint8Array> {
    if (ref.kind === 'path') {
      const { readFile } = await import('@tauri-apps/plugin-fs');
      return readFile(ref.path);
    }
    if (ref.kind === 'url') {
      return new Uint8Array(await (await fetch(ref.url)).arrayBuffer());
    }
    return base64ToBytes(ref.bytesB64);
  }

  /** Local paths go through Tauri's asset protocol (range-request streaming for
   *  Web Audio); remote urls pass through; inline becomes a data URI. */
  resolveMediaUrl(ref: ResultRef): string {
    if (ref.kind === 'path') {
      return convertFileSrc(ref.path);
    }
    if (ref.kind === 'url') {
      return ref.url;
    }
    return `data:application/octet-stream;base64,${ref.bytesB64}`;
  }
}

/** Stage in-memory audio to a temp file the sidecar process can read off disk.
 *  Returns the absolute path. The `$TEMP/drumjot` scope is granted in
 *  src-tauri/capabilities/default.json. */
async function stageTempInput(blob: Blob, filename: string): Promise<string> {
  const { BaseDirectory, mkdir, writeFile } = await import('@tauri-apps/plugin-fs');
  const { join, tempDir } = await import('@tauri-apps/api/path');
  const safe = filename.replace(/[/\\]/g, '_');
  const rel = `drumjot/input-${crypto.randomUUID()}-${safe}`;
  await mkdir('drumjot', { baseDir: BaseDirectory.Temp, recursive: true });
  await writeFile(rel, new Uint8Array(await blob.arrayBuffer()), { baseDir: BaseDirectory.Temp });
  return join(await tempDir(), rel);
}

function base64ToBytes(b64: string): Uint8Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}
