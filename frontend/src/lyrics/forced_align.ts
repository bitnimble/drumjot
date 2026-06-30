/**
 * Client for the Python backend's `/lyrics/align` endpoint. The endpoint
 * always runs in forced-alignment mode: the caller supplies the lyric
 * text + rough line timings (typically pulled straight from LRCLIB) and
 * the backend runs a CTC forced aligner (MMS-300m via
 * `ctc-forced-aligner`) to recompute per-word timings against an
 * uploaded audio source.
 *
 * The endpoint base is the shared {@link appSettingsStore.apiBase}
 * (`<transcriber origin>/api`), same as the transcribe client.
 */

import { backendFetch } from 'src/net/backend_fetch';
import { appSettingsStore } from 'src/settings/app_settings_presenter';
import type { LyricLine } from './lrc';

/**
 * Caller-provided lyric text + initial timings. The backend treats the
 * text as authoritative (no transcription pass) and only recomputes
 * word/line timings via the CTC forced aligner.
 */
export type AlignLyricsRealignInput = {
  lines: readonly Pick<LyricLine, 'startSec' | 'text'>[];
  /** Optional ISO-639-1 hint that pins the aligner's language head.
   *  Omitted = auto-detect from the lyric text + (fallback) the first
   *  30 s of audio. */
  language?: string;
};

export type AlignLyricsRequest = {
  /** `mix` runs the 2-stem vocals separator first; `vocals` skips
   *  separation and feeds the file straight to the aligner. */
  kind: 'mix' | 'vocals';
  file: File;
  realign: AlignLyricsRealignInput;
};

/** Non-terminal progress emitted while the alignment stream runs:
 *  `queued` while the request waits behind another in-flight GPU job
 *  (transcribe or another align), then `running` once it owns the GPU and
 *  alignment actually starts. */
export type AlignLyricsProgress = { kind: 'queued' | 'running' };

export type AlignLyricsOptions = {
  signal?: AbortSignal;
  /** Fires once per non-terminal stream envelope. Optional; omitting it
   *  just loses the queued/running wait-state feedback. */
  onProgress?: (event: AlignLyricsProgress) => void;
};

/**
 * POST to `/lyrics/align` and return the parsed lyric lines.
 *
 * The endpoint streams NDJSON: one envelope per line, `queued` (only
 * when the request waits behind another GPU job), `running` (alignment
 * started), then a terminal `result` carrying `{lines}` or an `error`.
 * Non-terminal envelopes drive `opts.onProgress` so the caller can show
 * a wait state; the `result` lines (shape `{lines: LyricLine[]}`, an
 * exact match for our in-memory type) go straight into
 * `lyricsStore.replace(id, ...)` via `JotEditorStore.alignLyricsForced`.
 *
 * Input-validation failures arrive as a real 4xx (before any stream
 * bytes) and surface here as the server's `detail` message, same as
 * before the endpoint streamed.
 */
export async function alignLyricsForced(
  req: AlignLyricsRequest,
  opts: AlignLyricsOptions = {},
): Promise<LyricLine[]> {
  const form = new FormData();
  if (req.kind === 'vocals') {
    form.set('vocals', req.file, req.file.name);
  } else {
    form.set('mix', req.file, req.file.name);
  }
  const payload = req.realign.lines.map((l) => ({
    startSec: l.startSec,
    text: l.text,
  }));
  form.set('lyrics', JSON.stringify(payload));
  if (req.realign.language !== undefined && req.realign.language.length > 0) {
    form.set('language', req.realign.language);
  }
  const res = await backendFetch(`${appSettingsStore.apiBase}/lyrics/align`, {
    method: 'POST',
    body: form,
    signal: opts.signal,
  });
  if (!res.ok) {
    let detail: string | undefined;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body?.detail === 'string') detail = body.detail;
    } catch {
      // Non-JSON body; fall through to the status-text fallback below.
    }
    throw new Error(detail ?? `lyrics/align failed (${res.status} ${res.statusText})`);
  }
  if (!res.body) {
    throw new Error('lyrics/align returned no response body');
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  let lines: LyricLine[] | null = null;
  let errorMessage: string | null = null;

  const handle = (event: Record<string, unknown>): void => {
    const type = event.type;
    if (type === 'queued' || type === 'running') {
      opts.onProgress?.({ kind: type });
    } else if (type === 'result') {
      const data = event.data as { lines?: LyricLine[] } | undefined;
      lines = Array.isArray(data?.lines) ? data.lines : [];
    } else if (type === 'error') {
      errorMessage = typeof event.message === 'string' ? event.message : 'lyrics/align failed';
    }
  };
  const settled = (): boolean => lines !== null || errorMessage !== null;

  try {
    while (!settled()) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // Split on newline; a trailing partial line stays buffered for the
      // next chunk.
      let newlineIdx: number;
      while ((newlineIdx = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, newlineIdx).trim();
        buffer = buffer.slice(newlineIdx + 1);
        if (line) {
          const event = parseNdjsonLine(line);
          if (event) handle(event);
        }
        if (settled()) break;
      }
    }
    // Tolerate a final envelope with no trailing newline.
    const tail = buffer.trim();
    if (!settled() && tail) {
      const event = parseNdjsonLine(tail);
      if (event) handle(event);
    }
  } finally {
    // Best-effort: aborts surface as a rejection from `reader.read()`
    // above; here we just let the cancel chain run.
    try {
      await reader.cancel();
    } catch {
      // already done or aborted
    }
  }

  if (errorMessage !== null) {
    throw new Error(errorMessage);
  }
  if (lines === null) {
    throw new Error('lyrics/align stream ended without a terminal result event');
  }
  return lines;
}

/** Parse one NDJSON line without throwing; a malformed envelope is
 *  logged and skipped so a single bad line can't kill the whole stream
 *  (mirrors the transcribe stream reader in `src/transcriber.ts`). */
function parseNdjsonLine(line: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(line);
    return typeof parsed === 'object' && parsed !== null
      ? (parsed as Record<string, unknown>)
      : null;
  } catch (err) {
    // eslint-disable-next-line no-console
    console.warn('Skipping malformed lyrics/align NDJSON event:', line, err);
    return null;
  }
}

/**
 * Filename heuristic for picking the vocals stem out of a paradb map's
 * `audioTracks` (or any other multi-track bundle). Matches common
 * names; `vocals`, `voice`, `vox`, `lead_vocal`, `singer`; case-
 * insensitively against the basename (with extension stripped).
 * Returns the file, or `undefined` when nothing matches; the caller
 * then falls back to running the vocals separator over the full mix.
 */
export function pickVocalsTrack(files: readonly File[]): File | undefined {
  const VOCAL_TOKENS = /\b(vocals?|voice|vox|sing(er|ing)?|lead[_-]?vocal)\b/i;
  for (const f of files) {
    const base = f.name.replace(/\.[^./\\]+$/, '');
    if (VOCAL_TOKENS.test(base)) return f;
  }
  return undefined;
}

/** Filename-based vocals heuristic on a track name (no `File` wrapper).
 *  Same regex as {@link pickVocalsTrack}; exported so the store can
 *  auto-pick from `AudioTrack`s without first hydrating their blobs. */
export function nameLooksLikeVocals(filename: string): boolean {
  const base = filename.replace(/\.[^./\\]+$/, '');
  return /\b(vocals?|voice|vox|sing(er|ing)?|lead[_-]?vocal)\b/i.test(base);
}
