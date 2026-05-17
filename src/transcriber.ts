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
  level: 'macro' | 'structure' | 'onsets' | 'velocity';
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

export type SelfConsistencyLog = {
  samples: number;
  scores: number[];
  chosen_index: number;
};

export type TranscribeResponse = {
  jot_dsl: string;
  metadata: TranscribeMetadata;
  refinement?: RefinementLog | null;
  self_consistency?: SelfConsistencyLog | null;
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
};

export type TranscribeOptions = {
  includeCandidates?: boolean;
  /** Run the multi-level convergence loop after the initial transcription. */
  refine?: boolean;
  /** Generate K candidate transcriptions and pick the highest-scoring one. */
  selfConsistencySamples?: number;
  /**
   * Persist all intermediate audio + JSON artifacts to the transcriber's
   * debug directory. Useful for debugging stem separation, beat tracking,
   * or LLM output. See transcriber/README.md for the layout.
   */
  debug?: boolean;
  /** AbortSignal lets callers cancel slow requests (separation can take ~60s). */
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
    if (options.includeCandidates) {
      form.append('include_candidates', 'true');
    }
    if (options.refine !== undefined) {
      form.append('refine', options.refine ? 'true' : 'false');
    }
    if (options.selfConsistencySamples !== undefined && options.selfConsistencySamples > 1) {
      form.append('self_consistency_samples', String(options.selfConsistencySamples));
    }
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
