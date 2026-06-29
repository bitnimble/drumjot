import { convertFileSrc } from '@tauri-apps/api/core';
import {
  PROTOCOL_VERSION,
  type Artifact,
  type ClientMessage,
  type ResultRef,
} from './control_protocol';
import { TauriBridge, type DesktopBridge } from 'src/desktop/desktop_bridge';
import {
  type AudioInput,
  type BackendClient,
  type RunOptions,
  type RunRequest,
  type RunResult,
} from './backend';

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

  async run(request: RunRequest, audio: AudioInput, opts: RunOptions = {}): Promise<RunResult> {
    // Already aborted before we started: skip staging + dispatching entirely.
    if (opts.signal?.aborted) {
      throw new DOMException('Aborted', 'AbortError');
    }
    // Stage blob input to a temp file we own + delete; a path input we don't.
    let staged: string | undefined;
    let path: string;
    if (audio.kind === 'path') {
      path = audio.path;
    } else {
      staged = await stageTempInput(audio.blob, audio.filename);
      path = staged;
    }
    const message: ClientMessage = {
      v: PROTOCOL_VERSION,
      type: 'request',
      id: crypto.randomUUID(),
      op: request.op,
      args: { audio: { kind: 'path', path }, params: request.params },
    };

    const artifacts: Artifact[] = [];
    let data: unknown;
    let failure: string | undefined;

    const onAbort = (): void => void this.bridge.cancelJob(message.id);
    opts.signal?.addEventListener('abort', onAbort, { once: true });
    try {
      // An abort that fired during staging won't trigger the listener (the event
      // already passed), so the sidecar would never get cancelled, bail before
      // dispatching. The finally still removes the listener + cleans the temp.
      if (opts.signal?.aborted) {
        throw new DOMException('Aborted', 'AbortError');
      }
      await this.bridge.runJob(message, (msg) => {
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
      // The sidecar has finished reading the input by the time runJob resolves;
      // drop the staged temp file (best-effort) so they don't accumulate.
      if (staged != null) void removeTempInput(staged);
    }

    if (opts.signal?.aborted) {
      throw new DOMException('Aborted', 'AbortError');
    }
    if (failure != null) {
      throw new Error(`${request.op} failed (${failure})`);
    }
    return { artifacts, data };
  }

  async resolveBytes(ref: ResultRef): Promise<Uint8Array> {
    if (ref.kind === 'path') {
      // Read through the asset protocol (runtime-scoped to the outputs dir in
      // lib.rs setup) rather than plugin-fs, so it works wherever data_root puts
      // outputs (portable <exe>/data, or app-local-data when installed) without
      // depending on a static fs-capability path.
      const res = await fetch(convertFileSrc(ref.path));
      if (!res.ok) throw new Error(`Asset read failed (${res.status})`);
      return new Uint8Array(await res.arrayBuffer());
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

/** Best-effort delete of a staged temp input (needs `fs:allow-remove` for
 *  `$TEMP/drumjot`); failures are swallowed so a denied scope just leaks the
 *  file rather than breaking the job. */
async function removeTempInput(absPath: string): Promise<void> {
  try {
    const { remove } = await import('@tauri-apps/plugin-fs');
    await remove(absPath);
  } catch {
    // ignore, temp cleanup is non-critical; the OS reaps $TEMP eventually.
  }
}

function base64ToBytes(b64: string): Uint8Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}
