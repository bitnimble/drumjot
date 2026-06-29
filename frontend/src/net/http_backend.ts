import { backendFetch } from './backend_fetch';
import { type ResultRef } from './control_protocol';
import {
  type AlignLyricsParams,
  type AudioInput,
  type BackendClient,
  type RunOptions,
  type RunRequest,
  type RunResult,
  type TranscribeParams,
} from './backend';
import { STAGE_ORDER, stemUrl, transcriber, type TranscribeResponse } from 'src/editing/transcribe/transcriber';
import { alignLyricsForced, type AlignLyricsRequest } from 'src/lyrics/forced_align';

/**
 * Backend over HTTP to a remote / dev transcriber service (the web build's
 * transport). Wraps the existing streaming clients (transcriber.ts,
 * forced_align.ts) and normalises their results to the adapter shape. There is
 * no HTTP stem-separation endpoint, so `separate` is unsupported here.
 */
export class HttpBackendClient implements BackendClient {
  async run(request: RunRequest, audio: AudioInput, opts: RunOptions = {}): Promise<RunResult> {
    if (request.op === 'transcribe') return this.transcribe(toFile(audio), request.params, opts);
    if (request.op === 'alignLyrics') return this.alignLyrics(toFile(audio), request.params, opts);
    throw new Error('Stem separation is not available over HTTP (desktop only).');
  }

  private async transcribe(
    file: File,
    params: TranscribeParams,
    opts: RunOptions,
  ): Promise<RunResult> {
    const response = await transcriber.transcribe(file, {
      ...params,
      signal: opts.signal,
      onProgress: (event) => {
        // Stage `end` bookends are noise for the pill (they'd flash "end"
        // between stages); only forward stage starts + substage detail.
        if (event.kind === 'stage' && event.phase === 'end') return;
        const frac = STAGE_ORDER.indexOf(event.stage) / STAGE_ORDER.length;
        opts.onProgress?.({
          stage: event.stage,
          frac: frac >= 0 ? frac : 0.5,
          message: event.kind === 'substage' ? event.detail : undefined,
        });
      },
    });
    return { artifacts: artifactsFromResponse(response), data: response };
  }

  private async alignLyrics(
    file: File,
    params: AlignLyricsParams,
    opts: RunOptions,
  ): Promise<RunResult> {
    const req: AlignLyricsRequest = {
      kind: params.kind ?? 'mix',
      file,
      realign: { lines: params.lines, language: params.language },
    };
    const lines = await alignLyricsForced(req, {
      signal: opts.signal,
      onProgress: (event) =>
        opts.onProgress?.({ stage: event.kind, frac: event.kind === 'running' ? 0.5 : 0 }),
    });
    return { artifacts: [], data: { lines } };
  }

  async resolveBytes(ref: ResultRef): Promise<Uint8Array> {
    if (ref.kind === 'url') {
      const res = await backendFetch(ref.url);
      if (!res.ok) throw new Error(`Fetch failed (${res.status})`);
      return new Uint8Array(await res.arrayBuffer());
    }
    if (ref.kind === 'inline') {
      const binary = atob(ref.bytesB64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      return bytes;
    }
    throw new Error('HTTP backend cannot read a local path artifact');
  }

  resolveMediaUrl(ref: ResultRef): string {
    if (ref.kind === 'url') return ref.url;
    if (ref.kind === 'inline') return `data:application/octet-stream;base64,${ref.bytesB64}`;
    throw new Error('HTTP backend cannot serve a local path artifact');
  }
}

function toFile(audio: AudioInput): File {
  if (audio.kind === 'path') {
    throw new Error('HTTP backend cannot upload a local path; pass blob input');
  }
  return new File([audio.blob], audio.filename, { type: audio.blob.type });
}

/** Map a TranscribeResponse's stem/MIDI URLs to control-protocol artifacts. */
function artifactsFromResponse(response: TranscribeResponse): RunResult['artifacts'] {
  const artifacts: RunResult['artifacts'] = [];
  const midi = stemUrl(response.prediction_midi_url ?? null);
  if (midi != null) artifacts.push({ role: 'midi', ref: { kind: 'url', url: midi } });
  const drum = stemUrl(response.drum_stem_url ?? null);
  if (drum != null) artifacts.push({ role: 'stem', ref: { kind: 'url', url: drum } });
  const noDrums = stemUrl(response.no_drums_url ?? null);
  if (noDrums != null) artifacts.push({ role: 'audio', ref: { kind: 'url', url: noDrums } });
  return artifacts;
}
