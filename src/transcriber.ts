/**
 * Client for the local / IaaS transcriber service.
 *
 * Requests always go to `/api` on the frontend's own origin. The Vite
 * dev / preview server (see vite.config.ts) proxies that prefix onward
 * to the actual transcriber URL configured via `TRANSCRIBER_URL` in the
 * server-side env. Keeping the browser bundle origin-relative avoids
 * cross-origin CORS when the frontend and the transcriber sit on
 * different hosts (e.g. drumjot.kumo.dev vs. a LAN GPU box).
 */
const TRANSCRIBER_BASE = '/api';

export type BarSummary = {
  bar: number;
  time_signature: string;
  tempo_bpm: number;
  feel: string;
  start_time: number;
};

export type TranscribeMetadata = {
  initial_tempo: number;
  initial_time_signature: [number, number];
  duration_seconds: number;
  stems_used: string[];
  bars: BarSummary[];
  has_tempo_changes: boolean;
  has_time_sig_changes: boolean;
};

export type TranscribeResponse = {
  metadata: TranscribeMetadata;
  /**
   * Onset candidates that fed the filter LLM, per pitch. Only populated
   * when the request set `include_candidates=true`. `bar` is 0-indexed;
   * `beat_in_bar` is a 1-indexed float (integer part = beat number,
   * fraction = position inside the beat). Both are -1 / -1.0 for onsets
   * outside the tracked beat range.
   */
  candidates?: Record<
    string,
    Array<{ time: number; strength: number; bar: number; beat_in_bar: number }>
  >;
  /**
   * Absolute path inside the transcriber container where intermediate
   * artifacts (drum stems, per-instrument stems, beats.json, onsets.json,
   * prediction.mid, note_provenance.json) were written. Null when debug
   * persistence is disabled. With the default docker-compose mount
   * (`./debug:/debug`), the host path is the same string with `/debug`
   * replaced by `./debug`.
   */
  debug_dir?: string | null;
  /**
   * URL path (with leading `/`) to the isolated drum mix as FLAC, or
   * null if the stem couldn't be produced. Compose against the
   * configured `TRANSCRIBER_BASE` to get a fetchable URL (e.g.
   * `${TRANSCRIBER_BASE}${drum_stem_url}` -> `/api/outputs/<id>/drum_stem.flac`
   * in dev). See `stemUrl()` below for the canonical helper.
   */
  drum_stem_url?: string | null;
  /**
   * URL path to the drumless (bass + other + vocals) mix as FLAC.
   * Same composition rules as `drum_stem_url`. Useful for backing-track
   * practice — pair with the rendered Jot to play along.
   */
  no_drums_url?: string | null;
  /**
   * URL path to the predicted onsets rendered as a MIDI file. The
   * frontend converts this to a Drumjot Jot via `src/midi/from_midi.ts`.
   */
  prediction_midi_url?: string | null;
  /**
   * URL path to the debug `.zip` bundle for this run — the predicted
   * MIDI, per-note provenance, MP3-encoded per-stem + drumless audio
   * tracks, and a JSON manifest with per-stage timings + the full
   * captured log stream. The web UI can load this zip back to
   * reconstitute the score + audio tracks + debug info offline (see
   * {@link loadDebugZip}). Null when the bundle couldn't be assembled
   * (e.g. nothing to bundle).
   */
  debug_zip_url?: string | null;
};

/** Pipeline stage names — kept in lockstep with `transcriber/app/pipeline/runner.py`'s
 *  `Stage` StrEnum. Used both as `resume_stage` form values and as the
 *  picker labels in the UI. */
export type TranscribeStage =
  | 'stems_all'
  | 'stems_per'
  | 'beats'
  | 'onsets'
  | 'filter'
  | 'quantise'
  | 'transcribe';

/** Audio fed into the beat tracker.
 *  See `transcriber/app/pipeline/runner.py::BeatInput`. */
export type BeatInput = 'full_mix' | 'drum_stem';

/**
 * Anthropic model used by the three classification stages
 * (`filter`; `hihat_split`; `cymbal_split`). The `quantise` stage's
 * Haiku is hard-coded server-side and ignored here.
 *
 * The wire value is sent verbatim as the `llm_model` form field; the
 * Python side passes it through to the Anthropic SDK, so any model id
 * the SDK accepts works in principle. The three exposed here are the
 * ones surfaced in the Transcribe dropdown — keep them in lockstep
 * with the UI's options.
 */
export type LlmModel =
  | 'claude-opus-4-7'
  | 'claude-sonnet-4-6'
  | 'claude-haiku-4-5-20251001';

/** Human-readable label for each {@link LlmModel} value. Used by the
 *  Transcribe-menu selector; kept here so the wire-format ↔ label
 *  mapping stays single-sourced. */
export const LLM_MODEL_LABELS: Record<LlmModel, string> = {
  'claude-opus-4-7': 'Opus 4.7',
  'claude-sonnet-4-6': 'Sonnet 4.6',
  'claude-haiku-4-5-20251001': 'Haiku 4.5',
};

/** Selector order, most-capable first. */
export const LLM_MODEL_ORDER: readonly LlmModel[] = [
  'claude-opus-4-7',
  'claude-sonnet-4-6',
  'claude-haiku-4-5-20251001',
];

/**
 * One entry in `GET /transcribe/list`. Mirrors the server-side
 * `TranscriptionSummary` (`transcriber/app/models.py`); see the docstring
 * there for the field semantics.
 */
export type TranscriptionSummary = {
  folder: string;
  original_filename: string | null;
  requested_at: string;
  last_run_at: string | null;
  last_resume_stage: string | null;
  resumable_stages: TranscribeStage[];
};

/**
 * One progress update streamed from /transcribe and /transcribe/resume.
 *
 * `stage` events bookend each pipeline stage with `phase: 'start'` /
 * `'end'`; `substage` events fire as in-stage milestones tick over
 * (e.g. per-instrument filter completions). The frontend uses these to
 * keep the toolbar pill live while the pipeline runs.
 */
export type TranscribeProgress =
  | {
      kind: 'stage';
      stage: TranscribeStage;
      phase: 'start' | 'end';
      elapsedSeconds?: number;
    }
  | {
      kind: 'substage';
      stage: TranscribeStage;
      detail: string;
    };

/**
 * Compose a stem URL path returned in a `TranscribeResponse` against the
 * configured transcriber base. Handles dev (`/api` -> Vite proxy) and
 * prod (absolute URL) uniformly.
 */
export function stemUrl(path: string | null | undefined): string | null {
  if (!path) return null;
  // Leading-slash paths join cleanly against any base (relative or absolute).
  const prefix = path.startsWith('/') ? '' : '/';
  return `${TRANSCRIBER_BASE}${prefix}${path}`;
}

export type TranscribeOptions = {
  includeCandidates?: boolean;
  /**
   * Which audio is fed into the beat tracker. `full_mix` (default) is
   * madmom's training distribution; `drum_stem` can help on tracks with
   * heavy non-drum syncopation. Server-side default lives in
   * `Settings.beat_input_default`.
   */
  beatInput?: BeatInput;
  /**
   * Anthropic model for the three classification stages (filter,
   * hihat_split, cymbal_split). Omitted = server falls back to
   * `Settings.llm_model`.
   */
  llmModel?: LlmModel;
  /**
   * Persist all intermediate audio + JSON artifacts to the transcriber's
   * debug directory. Required for the run to be resumable later (the
   * resume endpoint reads from `/debug/<folder>/`). See
   * transcriber/README.md for the layout.
   */
  debug?: boolean;
  /**
   * Fires once per `stage`/`substage` NDJSON event streamed from the
   * server. The store wires this to update `transcribeStatus` with the
   * live stage label so the toolbar pill reads e.g.
   * "Transcribing song.mp3 · stems_all…" then advances as the pipeline
   * progresses. Optional — omitting it loses progress visibility but
   * doesn't affect correctness.
   */
  onProgress?: (event: TranscribeProgress) => void;
  /** AbortSignal lets callers cancel slow requests (separation can take ~60s). */
  signal?: AbortSignal;
};

/**
 * Form-encoded body for `POST /transcribe/resume`. `resumeFolder` is the
 * basename of a folder under the configured debug base (typically a value
 * pulled from a {@link TranscriptionSummary}); `resumeStage` is one of
 * `STAGE_ORDER`. All other fields share their /transcribe semantics.
 */
export type ResumeOptions = {
  resumeFolder: string;
  resumeStage: TranscribeStage;
  includeCandidates?: boolean;
  beatInput?: BeatInput;
  /** Same semantics as {@link TranscribeOptions.llmModel}. */
  llmModel?: LlmModel;
  onProgress?: (event: TranscribeProgress) => void;
  signal?: AbortSignal;
};

export class TranscriberClient {
  constructor(private readonly baseUrl: string = TRANSCRIBER_BASE) {}

  /** Quick liveness probe; returns true if the backend is reachable. */
  async health(): Promise<{ ok: boolean; gpu: boolean; gpuName?: string }> {
    try {
      const res = await fetch(`${this.baseUrl}/health`);
      if (!res.ok) return { ok: false, gpu: false };
      const body = (await res.json()) as {
        status: string;
        gpu_available: boolean;
        gpu_name?: string;
      };
      return {
        ok: body.status === 'ok',
        gpu: !!body.gpu_available,
        gpuName: body.gpu_name ?? undefined,
      };
    } catch {
      return { ok: false, gpu: false };
    }
  }

  async transcribe(file: File, options: TranscribeOptions = {}): Promise<TranscribeResponse> {
    const form = new FormData();
    form.append('file', file);
    this.appendTranscribeFields(form, options);
    if (options.debug) {
      form.append('debug', 'true');
    }

    const res = await fetch(`${this.baseUrl}/transcribe`, {
      method: 'POST',
      body: form,
      signal: options.signal,
    });
    return this.readStream(res, options, 'Transcribe');
  }

  /**
   * Re-run the pipeline from `options.resumeStage` onward against a prior
   * /transcribe run's debug folder. The server expects the artifacts of
   * every stage strictly before `resumeStage` to be on disk under
   * `<DEBUG_DIR>/<resumeFolder>/` — surface a 400 with a stage-specific
   * message if anything is missing. Mirrors `/transcribe`'s response
   * shape so callers can reuse the same debug-zip-load path.
   */
  async resume(options: ResumeOptions): Promise<TranscribeResponse> {
    const form = new FormData();
    form.append('resume_folder', options.resumeFolder);
    form.append('resume_stage', options.resumeStage);
    this.appendTranscribeFields(form, options);

    const res = await fetch(`${this.baseUrl}/transcribe/resume`, {
      method: 'POST',
      body: form,
      signal: options.signal,
    });
    return this.readStream(res, options, 'Resume');
  }

  /**
   * Consume the NDJSON progress stream from /transcribe or /transcribe/resume.
   *
   * The server emits one JSON object per line:
   *   - `{type:"stage",   stage, phase, elapsed_seconds?}` — stage bookends
   *   - `{type:"substage",stage, detail}` — in-stage milestones
   *   - `{type:"result",  data: <TranscribeResponse>}` — terminal success
   *   - `{type:"error",   status_code, message}` — terminal failure
   *
   * The first three feed `options.onProgress`; the terminal event
   * resolves the returned promise (or throws). The stream may close
   * without a terminal event if the connection drops mid-pipeline —
   * we treat that as an error so the caller can show a sensible pill.
   */
  private async readStream(
    res: Response,
    options: TranscribeOptions | ResumeOptions,
    verb: string,
  ): Promise<TranscribeResponse> {
    if (!res.ok) {
      const detail = await safeReadError(res);
      throw new Error(`${verb} failed (${res.status}): ${detail}`);
    }
    const body = res.body;
    if (!body) {
      throw new Error(`${verb} returned no response body`);
    }
    const reader = body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let final: TranscribeResponse | null = null;
    let terminalError: { statusCode: number; message: string } | null = null;
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        // Split on newline; the last fragment may be a partial line so
        // keep it in `buffer` for the next chunk.
        let newlineIdx: number;
        while ((newlineIdx = buffer.indexOf('\n')) >= 0) {
          const line = buffer.slice(0, newlineIdx).trim();
          buffer = buffer.slice(newlineIdx + 1);
          if (!line) continue;
          const event = safeParseJson(line);
          if (!event) continue;
          this.dispatchEvent(event, options.onProgress, (r) => {
            final = r;
          }, (e) => {
            terminalError = e;
          });
          if (final || terminalError) break;
        }
        if (final || terminalError) break;
      }
      // Drain any remaining buffered line (servers SHOULD terminate with
      // a newline, but tolerate a missing trailing newline anyway).
      const tail = buffer.trim();
      if (tail && !final && !terminalError) {
        const event = safeParseJson(tail);
        if (event) {
          this.dispatchEvent(event, options.onProgress, (r) => {
            final = r;
          }, (e) => {
            terminalError = e;
          });
        }
      }
    } finally {
      // Reader cleanup is best-effort; aborts surface as a rejection
      // from `reader.read()` above and we just let the cancel chain run.
      try {
        await reader.cancel();
      } catch {
        // ignore — the stream is already done or aborted.
      }
    }
    if (terminalError) {
      const err = terminalError as { statusCode: number; message: string };
      throw new Error(`${verb} failed (${err.statusCode}): ${err.message}`);
    }
    if (!final) {
      throw new Error(
        `${verb} stream ended without a terminal result event`,
      );
    }
    return final;
  }

  private dispatchEvent(
    event: Record<string, unknown>,
    onProgress: TranscribeOptions['onProgress'],
    onResult: (r: TranscribeResponse) => void,
    onError: (e: { statusCode: number; message: string }) => void,
  ): void {
    const type = event.type;
    if (type === 'stage' && onProgress) {
      onProgress({
        kind: 'stage',
        stage: event.stage as TranscribeStage,
        phase: event.phase as 'start' | 'end',
        elapsedSeconds:
          typeof event.elapsed_seconds === 'number'
            ? event.elapsed_seconds
            : undefined,
      });
    } else if (type === 'substage' && onProgress) {
      onProgress({
        kind: 'substage',
        stage: event.stage as TranscribeStage,
        detail: String(event.detail ?? ''),
      });
    } else if (type === 'result') {
      onResult(event.data as TranscribeResponse);
    } else if (type === 'error') {
      onError({
        statusCode:
          typeof event.status_code === 'number' ? event.status_code : 500,
        message: String(event.message ?? 'unknown error'),
      });
    }
  }

  /** List recent /transcribe runs available on the server, most-recently
   *  -run first. Returns an empty array when the debug base is missing or
   *  empty. */
  async listTranscriptions(): Promise<TranscriptionSummary[]> {
    const res = await fetch(`${this.baseUrl}/transcribe/list`);
    if (!res.ok) {
      const detail = await safeReadError(res);
      throw new Error(`List transcriptions failed (${res.status}): ${detail}`);
    }
    return (await res.json()) as TranscriptionSummary[];
  }

  /** Shared form-field encoder for /transcribe + /transcribe/resume.
   *  Both endpoints accept the same toggles; only the file/resume
   *  identifier differs. */
  private appendTranscribeFields(
    form: FormData,
    options: TranscribeOptions | ResumeOptions,
  ): void {
    if (options.includeCandidates) {
      form.append('include_candidates', 'true');
    }
    if (options.beatInput !== undefined) {
      form.append('beat_input', options.beatInput);
    }
    if (options.llmModel !== undefined) {
      form.append('llm_model', options.llmModel);
    }
  }
}

/** Parse one NDJSON line without throwing — malformed events are logged
 *  and skipped so a single bad line can't kill the whole stream. */
function safeParseJson(line: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(line);
    return typeof parsed === 'object' && parsed !== null
      ? (parsed as Record<string, unknown>)
      : null;
  } catch (err) {
    // eslint-disable-next-line no-console
    console.warn('Skipping malformed NDJSON event:', line, err);
    return null;
  }
}

async function safeReadError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body === 'object' && body && 'detail' in body) {
      return String((body as { detail: unknown }).detail);
    }
    return JSON.stringify(body);
  } catch {
    try {
      return await res.text();
    } catch {
      return res.statusText;
    }
  }
}

/** Singleton instance for convenience. */
export const transcriber = new TranscriberClient();

/**
 * Derive a Jot title from an uploaded audio file's name. Strips the
 * file extension and trims whitespace; returns null when the input is
 * null/empty or yields an empty stem (so callers can skip applying it
 * rather than overwrite an LLM-emitted title with the empty string).
 */
export function titleFromFilename(filename: string | null | undefined): string | null {
  if (!filename) return null;
  // Strip a single trailing `.<ext>` (handles dotfiles like ".env" by
  // not matching, which leaves the stem untouched). Then trim.
  const stem = filename.replace(/\.[^./\\]+$/, '').trim();
  return stem || null;
}
