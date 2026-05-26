/**
 * Client for the Python backend's `/lyrics/align` endpoint. The endpoint
 * always runs in forced-alignment mode: the caller supplies the lyric
 * text + rough line timings (typically pulled straight from LRCLIB) and
 * the backend uses whisperx's wav2vec2 aligner to recompute per-word
 * timings against an uploaded audio source.
 *
 * The endpoint base mirrors `src/transcriber.ts::TRANSCRIBER_BASE`:
 * `VITE_TRANSCRIBER_URL` in prod, `/api` in dev (proxied to the
 * transcriber service by Vite).
 */

import type { LyricLine } from './lrc';

const TRANSCRIBER_BASE: string =
  (import.meta as unknown as { env?: Record<string, string> }).env
    ?.VITE_TRANSCRIBER_URL ?? '/api';

/**
 * Caller-provided lyric text + initial timings. The backend treats the
 * text as authoritative (no Whisper transcription pass) and only
 * recomputes word/line timings via wav2vec2.
 */
export type AlignLyricsRealignInput = {
  lines: readonly Pick<LyricLine, 'startSec' | 'text'>[];
  /** Optional ISO-639-1 hint that pins the wav2vec2 aligner. Omitted =
   *  auto-detect from the lyric text + (fallback) the first 30 s of
   *  audio. */
  language?: string;
};

export type AlignLyricsRequest = {
  /** `mix` runs the 2-stem vocals separator first; `vocals` skips
   *  separation and feeds the file straight to wav2vec2. */
  kind: 'mix' | 'vocals';
  file: File;
  realign: AlignLyricsRealignInput;
};

export type AlignLyricsOptions = {
  signal?: AbortSignal;
};

/**
 * POST to `/lyrics/align` and return the parsed lyric lines. The server
 * response shape (`{lines: LyricLine[]}`) matches our in-memory type
 * exactly, so the response goes straight into `lyricsStore.load()`.
 */
export async function alignLyricsWhisper(
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
  const res = await fetch(`${TRANSCRIBER_BASE}/lyrics/align`, {
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
  const body = (await res.json()) as { lines?: LyricLine[] };
  return Array.isArray(body?.lines) ? body.lines : [];
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
