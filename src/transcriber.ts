/**
 * Client for the local / IaaS transcriber service.
 *
 * The service URL is resolved at module load time:
 *   - In dev, requests go through Vite's `/api` proxy (see vite.config.ts),
 *     which forwards them to `http://localhost:8001` (the docker-compose
 *     default for `transcriber/`). The browser bundle reads
 *     `VITE_TRANSCRIBER_URL` from `import.meta.env` and falls back to
 *     `/api` (i.e. the proxy path) when it isn't set.
 *   - In production, set `VITE_TRANSCRIBER_URL` at build time to a fully
 *     qualified base URL of a deployed transcriber instance. The same
 *     variable also feeds the dev-server proxy target, so dev and prod
 *     agree on the name.
 */
const TRANSCRIBER_BASE: string =
  (import.meta as unknown as { env?: Record<string, string> }).env
    ?.VITE_TRANSCRIBER_URL ?? '/api';

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

export type RefinementIteration = {
  level: 'lint' | 'macro' | 'structure' | 'onsets' | 'velocity';
  iteration: number;
  issues_detected: number;
  issues_sent_to_llm: number;
  score_before: number;
  score_after: number;
  accepted: boolean;
  note: string;
};

export type RefinementLog = {
  initial_score: number;
  final_score: number;
  elapsed_seconds: number;
  iterations: RefinementIteration[];
};

export type BestOfKLog = {
  samples: number;
  scores: number[];
  chosen_index: number;
};

export type TranscribeResponse = {
  jot_dsl: string;
  metadata: TranscribeMetadata;
  refinement?: RefinementLog | null;
  best_of_k?: BestOfKLog | null;
  /**
   * Onset candidates that fed the LLM, per pitch. Only populated when the
   * request set `include_candidates=true`. `bar` is 0-indexed; `beat_in_bar`
   * is a 1-indexed float (integer part = beat number, fraction = position
   * inside the beat). Both are -1 / -1.0 for onsets outside the tracked
   * beat range.
   */
  candidates?: Record<
    string,
    Array<{ time: number; strength: number; bar: number; beat_in_bar: number }>
  >;
  /**
   * Absolute path inside the transcriber container where intermediate
   * artifacts (drum stems, per-instrument stems, beats.json, initial.jot,
   * refinement.json, ...) were written. Null when debug persistence is
   * disabled. With the default docker-compose mount (`./debug:/debug`),
   * the host path is the same string with `/debug` replaced by `./debug`.
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
   * Set only by the `filter` transcribe path: URL path to the predicted
   * onsets rendered as a MIDI file. Either this OR `jot_dsl` carries
   * the score depending on `transcribe_mode`.
   */
  prediction_midi_url?: string | null;
  /**
   * URL path to the debug `.zip` bundle for this run — the score
   * (`final.jot`), MP3-encoded per-stem + drumless audio tracks, and a
   * JSON manifest with per-stage timings + the full captured log stream.
   * The web UI can load this zip back to reconstitute the score + audio
   * tracks + debug info offline (see {@link loadDebugZip}). Null when the
   * bundle couldn't be assembled (e.g. nothing to bundle).
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
  | 'transcribe'
  | 'refine';

/** Audio fed into the beat tracker.
 *  See `transcriber/app/pipeline/runner.py::BeatInput`. */
export type BeatInput = 'full_mix' | 'drum_stem';

/** Which transcribe pathway runs. `dsl` = LLM emits a Drumjot DSL line
 *  per instrument, recomposed + (optionally) F1-refined. `filter` = LLM
 *  only filters artifact onsets per instrument; kept onsets render
 *  straight to MIDI with their original times. */
export type TranscribeMode = 'dsl' | 'filter';

/** Per-stem onset detector backend. `librosa` = the legacy spectral-flux
 *  detector. `adtof` = ADTOF CRNN run per stem, with automatic per-stem
 *  fallback to `librosa` if ADTOF is unavailable/erroring. */
export type OnsetBackend = 'librosa' | 'adtof';

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
   * Run the F1-gated multi-level convergence loop (macro / structure /
   * onsets / velocity). Independent of `lint`. Filter mode ignores this
   * (the refine stage is skipped).
   */
  refine?: boolean;
  /**
   * Run the deterministic Jot linter (instrument-tier + performance-tier
   * checks) and ask the LLM to fix flagged regions surgically.
   * Independent of `refine` — you can enable either alone, both, or
   * neither. Filter mode ignores this.
   */
  lint?: boolean;
  /** Generate K candidate transcriptions and pick the highest-scoring one.
   *  Filter mode ignores this (single deterministic filter pass). */
  bestOfK?: number;
  /**
   * Which audio is fed into the beat tracker. `full_mix` (default) is
   * madmom's training distribution; `drum_stem` can help on tracks with
   * heavy non-drum syncopation. Server-side default lives in
   * `Settings.beat_input_default`.
   */
  beatInput?: BeatInput;
  /**
   * Which transcribe pathway runs. `dsl` (default) emits Drumjot DSL +
   * refinement; `filter` emits a predicted-onsets MIDI plus per-note
   * provenance and skips the refine stage entirely.
   */
  transcribeMode?: TranscribeMode;
  /**
   * Per-stem onset detector backend. `librosa` is the legacy
   * spectral-flux detector; `adtof` is the ADTOF CRNN run per stem with
   * a per-stem librosa fallback when ADTOF/its weights aren't available.
   */
  onsetBackend?: OnsetBackend;
  /**
   * Persist all intermediate audio + JSON artifacts to the transcriber's
   * debug directory. Required for the run to be resumable later (the
   * resume endpoint reads from `/debug/<folder>/`). See
   * transcriber/README.md for the layout.
   */
  debug?: boolean;
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
  refine?: boolean;
  lint?: boolean;
  bestOfK?: number;
  beatInput?: BeatInput;
  transcribeMode?: TranscribeMode;
  onsetBackend?: OnsetBackend;
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
    if (!res.ok) {
      const detail = await safeReadError(res);
      throw new Error(`Transcribe failed (${res.status}): ${detail}`);
    }
    return (await res.json()) as TranscribeResponse;
  }

  /**
   * Re-run the pipeline from `options.resumeStage` onward against a prior
   * /transcribe run's debug folder. The server expects the artifacts of
   * every stage strictly before `resumeStage` to be on disk under
   * `<DEBUG_DIR>/<resumeFolder>/` — surface a 400 with a stage-specific
   * message if anything is missing. Mirrors `/transcribe`'s response
   * shape so callers can reuse the same `parse(jot_dsl)` / debug-zip-load
   * path.
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
    if (!res.ok) {
      const detail = await safeReadError(res);
      throw new Error(`Resume failed (${res.status}): ${detail}`);
    }
    return (await res.json()) as TranscribeResponse;
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
    if (options.refine !== undefined) {
      form.append('refine', options.refine ? 'true' : 'false');
    }
    if (options.lint !== undefined) {
      form.append('lint', options.lint ? 'true' : 'false');
    }
    if (options.bestOfK !== undefined && options.bestOfK > 1) {
      form.append('best_of_k', String(options.bestOfK));
    }
    if (options.beatInput !== undefined) {
      form.append('beat_input', options.beatInput);
    }
    if (options.transcribeMode !== undefined) {
      form.append('transcribe_mode', options.transcribeMode);
    }
    if (options.onsetBackend !== undefined) {
      form.append('onset_backend', options.onsetBackend);
    }
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
 *
 * This used to live in `transcriber/app/pipeline/title.py` as a regex
 * pass over the DSL inside the Python service. It now lives here so
 * the canonical TS parser owns the DSL — the frontend sets
 * `jot.title` directly on the parsed Jot rather than mutating the
 * DSL text.
 */
export function titleFromFilename(filename: string | null | undefined): string | null {
  if (!filename) return null;
  // Strip a single trailing `.<ext>` (handles dotfiles like ".env" by
  // not matching, which leaves the stem untouched). Then trim.
  const stem = filename.replace(/\.[^./\\]+$/, '').trim();
  return stem || null;
}
