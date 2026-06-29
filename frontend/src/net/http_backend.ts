import { backendFetch } from './backend_fetch';
import { type Op, type ResultRef } from './control_protocol';
import { type AudioInput, type BackendClient, type RunOptions, type RunResult } from './backend';
import {
  STAGE_ORDER,
  stemUrl,
  transcriber,
  type BeatInput,
  type LlmModel,
  type OnsetBackend,
  type TranscribeResponse,
} from 'src/editing/transcribe/transcriber';
import { alignLyricsForced, type AlignLyricsRequest } from 'src/lyrics/forced_align';
import { type LyricLine } from 'src/lyrics/lrc';

/**
 * Backend over HTTP to a remote / dev transcriber service (the web build's
 * transport). Wraps the existing streaming clients (transcriber.ts,
 * forced_align.ts) and normalises their results to the adapter shape. There is
 * no HTTP stem-separation endpoint, so `separate` is unsupported here.
 */
export class HttpBackendClient implements BackendClient {
  async run(
    op: Op,
    audio: AudioInput,
    params: Record<string, unknown>,
    opts: RunOptions = {},
  ): Promise<RunResult> {
    if (op === 'transcribe') return this.transcribe(toFile(audio), params, opts);
    if (op === 'alignLyrics') return this.alignLyrics(toFile(audio), params, opts);
    throw new Error('Stem separation is not available over HTTP (desktop only).');
  }

  private async transcribe(
    file: File,
    params: Record<string, unknown>,
    opts: RunOptions,
  ): Promise<RunResult> {
    const response = await transcriber.transcribe(file, {
      beatInput: params.beatInput as BeatInput | undefined,
      onsetBackend: params.onsetBackend as OnsetBackend | undefined,
      llmModel: params.llmModel as LlmModel | undefined,
      quantise: params.quantise as boolean | undefined,
      quantiseUseLlm: params.quantiseUseLlm as boolean | undefined,
      debug: params.debug as boolean | undefined,
      signal: opts.signal,
      onProgress: (event) => {
        const frac = STAGE_ORDER.indexOf(event.stage) / STAGE_ORDER.length;
        opts.onProgress?.({
          stage: event.stage,
          frac: frac >= 0 ? frac : 0.5,
          message: event.kind === 'substage' ? event.detail : event.phase,
        });
      },
    });
    return { artifacts: artifactsFromResponse(response), data: response };
  }

  private async alignLyrics(
    file: File,
    params: Record<string, unknown>,
    opts: RunOptions,
  ): Promise<RunResult> {
    const req: AlignLyricsRequest = {
      kind: (params.kind as 'mix' | 'vocals' | undefined) ?? 'mix',
      file,
      realign: {
        lines: (params.lines as Pick<LyricLine, 'startSec' | 'text'>[] | undefined) ?? [],
        language: params.language as string | undefined,
      },
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
