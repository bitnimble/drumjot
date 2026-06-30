/**
 * Loader for the per-kanji furigana split table (a trimmed JmdictFurigana
 * derivative; see `scripts/build-furigana-dict.ts`).
 *
 * The table answers one question for {@link FuriganaAnnotator}: given an
 * all-kanji compound and its reading, how does that reading divide across
 * the individual kanji (盲目的 + もうもくてき → 盲/もう 目/もく 的/てき)? It's
 * consulted only when the lyrics aligner split a compound across chips and
 * kuromoji's token-level reading can't otherwise be cut.
 *
 * The asset (`public/jmdict-furigana/furigana.txt.gz`) is one entry per
 * line, `surface|<utf16len>,<reading>;…`, where the length/reading pairs
 * tile the surface left to right. We fetch + gunzip it ourselves for the
 * same reason the kuromoji loader does (see `kuromoji_loader.ts`): a `.gz`
 * served with `Content-Encoding: gzip` (Vite/sirv) is transparently
 * inflated by the browser, while one served raw (Caddy, prod) is not, so
 * we gunzip only when the bytes actually start with the gzip magic.
 */
import type { RubySegment } from './furigana';

/** One reading of a surface and how it splits across the kanji. A surface
 *  with homographs (今日 = こんにち / こんじつ / …) carries several. */
export type FuriganaSplit = {
  /** Full hiragana reading (the concatenation of the segment readings),
   *  matched against kuromoji's token reading to pick the right homograph. */
  reading: string;
  /** One {@link RubySegment} per kanji (or per ruby range), tiling the
   *  surface. Every segment carries a reading; there are no bare runs (the
   *  source surface is all-kanji). */
  segs: RubySegment[];
};

/** Surface → its split readings. */
export type FuriganaSplitMap = Map<string, FuriganaSplit[]>;

/** gzip magic (RFC 1952): every member starts with 0x1f 0x8b. */
function looksGzipped(bytes: Uint8Array): boolean {
  return bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
}

/** Decompress `buf` when (and only when) it's actually gzip; an already-
 *  inflated body falls straight through. Mirrors `kuromoji_loader.ts`. */
async function maybeGunzip(buf: ArrayBuffer): Promise<ArrayBuffer> {
  if (!looksGzipped(new Uint8Array(buf))) return buf;
  try {
    const inflated = new Response(buf).body!.pipeThrough(
      new DecompressionStream('gzip'),
    );
    return await new Response(inflated).arrayBuffer();
  } catch {
    return buf;
  }
}

/** Parse one `surface|len,reading;len,reading;…` line into a map entry, or
 *  null when malformed (defensive; the asset is generated, so this should
 *  not fire). */
function parseLine(line: string): { surface: string; split: FuriganaSplit } | null {
  const bar = line.indexOf('|');
  if (bar <= 0) return null;
  const surface = line.slice(0, bar);
  const pairs = line.slice(bar + 1).split(';');
  const segs: RubySegment[] = [];
  let reading = '';
  let off = 0;
  for (const pair of pairs) {
    const comma = pair.indexOf(',');
    if (comma <= 0) return null;
    const len = Number(pair.slice(0, comma));
    const r = pair.slice(comma + 1);
    if (!Number.isInteger(len) || len <= 0 || !r) return null;
    const base = surface.slice(off, off + len);
    if (base.length !== len) return null; // ran past the surface
    segs.push({ base, reading: r });
    reading += r;
    off += len;
  }
  if (off !== surface.length || segs.length === 0) return null;
  return { surface, split: { reading, segs } };
}

/** Fetch + decode the split table from `url`. Rejects on a non-OK
 *  response; a malformed line is skipped rather than fatal. */
export async function loadFuriganaSplitMap(url: string): Promise<FuriganaSplitMap> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}: ${url}`);
  }
  const buf = await maybeGunzip(await res.arrayBuffer());
  const text = new TextDecoder('utf-8').decode(buf);
  const map: FuriganaSplitMap = new Map();
  for (const line of text.split('\n')) {
    if (!line) continue;
    const parsed = parseLine(line);
    if (!parsed) continue;
    const existing = map.get(parsed.surface);
    if (existing) existing.push(parsed.split);
    else map.set(parsed.surface, [parsed.split]);
  }
  return map;
}
