/**
 * LRCLIB search client. LRCLIB (https://lrclib.net) is a free, public,
 * no-API-key database of synced lyrics; CORS-friendly so we call it
 * directly from the browser. Today we only use the `/api/search`
 * endpoint; search returns the candidate list the modal needs for the
 * exact-match / multi-result logic; `/api/get` would return a single
 * best match but doesn't expose the alternates the modal lets the user
 * pick from.
 */

export type LrclibMatch = {
  id: number;
  trackName: string;
  artistName: string;
  albumName: string | null;
  duration: number | null;
  syncedLyrics: string | null;
  plainLyrics: string | null;
  instrumental: boolean;
};

const BASE_URL = 'https://lrclib.net/api';

export type SearchOptions = {
  trackName: string;
  artistName: string;
  signal?: AbortSignal;
};

/**
 * Search LRCLIB for `trackName` + `artistName`. Returns only entries
 * with non-empty `syncedLyrics` (the modal's exact-match / auto-load
 * logic only operates on synced lyrics; plain lyrics aren't useful
 * for the timeline-anchored row).
 *
 * Throws on transport / HTTP failures; the caller's `requestId` guard
 * is responsible for filtering stale responses against newer searches.
 */
export async function searchLrclib(opts: SearchOptions): Promise<LrclibMatch[]> {
  const params = new URLSearchParams();
  if (opts.trackName.trim().length > 0) params.set('track_name', opts.trackName.trim());
  if (opts.artistName.trim().length > 0) params.set('artist_name', opts.artistName.trim());
  // LRCLIB accepts an empty query (returns nothing); the modal guards
  // against firing when neither field has content, so by the time we
  // get here at least one of the two is set.
  const url = `${BASE_URL}/search?${params.toString()}`;
  // No custom request headers on purpose: any non-safelisted header
  // (LRCLIB's optional `Lrclib-Client` identifier) would force the
  // browser to send a CORS preflight before every GET, and LRCLIB
  // doesn't set `Access-Control-Max-Age` so the spec-default 5 s TTL
  // means the preflight refires for almost every search (~2 s tax).
  // Keeping only safelisted headers makes this a "simple" CORS request.
  const res = await fetch(url, {
    method: 'GET',
    signal: opts.signal,
  });
  if (!res.ok) {
    throw new Error(`LRCLIB search failed (${res.status} ${res.statusText})`);
  }
  const json = (await res.json()) as LrclibApiRow[];
  if (!Array.isArray(json)) return [];
  const out: LrclibMatch[] = [];
  for (const row of json) {
    if (typeof row?.syncedLyrics !== 'string' || row.syncedLyrics.length === 0) continue;
    out.push({
      id: typeof row.id === 'number' ? row.id : 0,
      trackName: row.trackName ?? '',
      artistName: row.artistName ?? '',
      albumName: row.albumName ?? null,
      duration: typeof row.duration === 'number' ? row.duration : null,
      syncedLyrics: row.syncedLyrics,
      plainLyrics: typeof row.plainLyrics === 'string' ? row.plainLyrics : null,
      instrumental: row.instrumental === true,
    });
  }
  return out;
}

type LrclibApiRow = {
  id?: number;
  trackName?: string;
  artistName?: string;
  albumName?: string | null;
  duration?: number | null;
  syncedLyrics?: string | null;
  plainLyrics?: string | null;
  instrumental?: boolean;
};

/** Case-insensitive trim-equality used by the modal's exact-match check.
 *  Doesn't normalise smart-quote vs straight-quote, NFC vs NFD, etc.; *  edge cases that don't pass this fall through to the multi-result
 *  picker, which is the correct UX for an ambiguous match. */
export function ciTrimEq(a: string, b: string): boolean {
  return a.trim().toLowerCase() === b.trim().toLowerCase();
}
