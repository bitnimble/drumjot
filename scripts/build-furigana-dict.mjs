/**
 * Build the trimmed per-kanji furigana split table from JmdictFurigana.
 *
 * Source: https://github.com/Doublevil/JmdictFurigana (releases →
 * `JmdictFurigana.txt`). That file maps every JMdict headword to its
 * per-character ruby segmentation; we only need the slice that lets us
 * split a *compound kanji reading* across the individual kanji, for the
 * case where the lyrics forced-aligner cut a compound across chips and
 * kuromoji's token-level reading can't be divided (盲目的 → 盲/もう 目/もく
 * 的/てき). See `src/lyrics/furigana.ts`.
 *
 * Filtering keeps only entries that are (a) entirely kanji in the surface
 * (okurigana words are already handled by the kana-anchored fitter) and
 * (b) split into ≥2 ruby spans (a single whole-span jukujikun like
 * 今日→きょう can't be divided, so it earns nothing here and is dropped, * the annotator's standalone fallback covers it). The spans are
 * re-expressed as ordered `<utf16-len>,<reading>` pairs that tile the
 * surface, so the runtime loader never has to reason about code-point vs
 * UTF-16 indices.
 *
 * Output: `public/jmdict-furigana/furigana.txt.gz`, committed (JmdictFurigana
 * isn't an npm dep, so unlike the kuromoji dict it can't be regenerated
 * from node_modules at install time). Regenerate with:
 *   bun scripts/build-furigana-dict.mjs path/to/JmdictFurigana.txt
 *
 * Licence: JmdictFurigana is CC BY-SA (same as JMdict/EDRDG); see the
 * emitted NOTICE. Attribution is mandatory.
 */
import { gzipSync } from 'node:zlib';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, '..');
const OUT_DIR = resolve(REPO, 'frontend/public/jmdict-furigana');

/** Same Han ranges as `src/lyrics/furigana.ts::isKanjiCp`. */
function isKanjiCp(cp) {
  return (
    (cp >= 0x4e00 && cp <= 0x9fff) ||
    (cp >= 0x3400 && cp <= 0x4dbf) ||
    (cp >= 0xf900 && cp <= 0xfaff) ||
    (cp >= 0x20000 && cp <= 0x2a6df)
  );
}

function allKanji(s) {
  for (const ch of s) {
    const cp = ch.codePointAt(0);
    if (cp === undefined || !isKanjiCp(cp)) return false;
  }
  return s.length > 0;
}

/** Parse one `start[-end]:reading` span; returns {start, end, reading}
 *  with inclusive code-point indices, or null if malformed. */
function parseSpan(span) {
  const colon = span.indexOf(':');
  if (colon < 0) return null;
  const idx = span.slice(0, colon);
  const reading = span.slice(colon + 1);
  if (!reading) return null;
  const dash = idx.indexOf('-');
  if (dash < 0) {
    const n = Number(idx);
    if (!Number.isInteger(n)) return null;
    return { start: n, end: n, reading };
  }
  const a = Number(idx.slice(0, dash));
  const b = Number(idx.slice(dash + 1));
  if (!Number.isInteger(a) || !Number.isInteger(b) || b < a) return null;
  return { start: a, end: b, reading };
}

function main() {
  const src = process.argv[2];
  if (!src) {
    console.error('usage: bun scripts/build-furigana-dict.mjs <JmdictFurigana.txt>');
    process.exit(2);
  }
  const text = readFileSync(src, 'utf8');
  const outLines = [];
  let kept = 0;
  let scanned = 0;
  for (const raw of text.split('\n')) {
    const line = raw.trimEnd();
    if (!line) continue;
    scanned++;
    const [surface, _kana, furi] = line.split('|');
    if (!surface || !furi) continue;
    if (!allKanji(surface)) continue;

    const spans = furi.split(';').map(parseSpan);
    if (spans.some((s) => s === null) || spans.length < 2) continue;

    // Code-point view of the surface so span indices (which are
    // code-point positions) extract the right glyphs even for astral
    // kanji; the emitted lengths are UTF-16 units, matching runtime.
    const cps = Array.from(surface);
    spans.sort((a, b) => a.start - b.start);
    // Spans must tile [0, cps.length) with no gaps/overlaps; otherwise we
    // can't unambiguously re-derive the bases, so skip the entry.
    let cursor = 0;
    let ok = true;
    const pairs = [];
    for (const s of spans) {
      if (s.start !== cursor || s.end >= cps.length) {
        ok = false;
        break;
      }
      const base = cps.slice(s.start, s.end + 1).join('');
      pairs.push(`${base.length},${s.reading}`);
      cursor = s.end + 1;
    }
    if (!ok || cursor !== cps.length) continue;

    outLines.push(`${surface}|${pairs.join(';')}`);
    kept++;
  }

  // Stable order keeps the gzipped asset byte-identical across rebuilds.
  outLines.sort();
  const body = outLines.join('\n') + '\n';
  const gz = gzipSync(Buffer.from(body, 'utf8'), { level: 9 });

  mkdirSync(OUT_DIR, { recursive: true });
  writeFileSync(resolve(OUT_DIR, 'furigana.txt.gz'), gz);
  writeFileSync(
    resolve(OUT_DIR, 'NOTICE'),
    [
      'This directory contains a trimmed, reformatted derivative of',
      'JmdictFurigana (https://github.com/Doublevil/JmdictFurigana),',
      'itself derived from JMdict/EDICT (https://www.edrdg.org/).',
      '',
      'Licensed under Creative Commons Attribution-ShareAlike (CC BY-SA),',
      'the same licence as JMdict. © James William Breen and the',
      'Electronic Dictionary Research and Development Group.',
      '',
      'Only all-kanji headwords with a per-kanji ruby split are kept, re-',
      'expressed as `surface|<utf16len>,<reading>;...`. Regenerate with',
      'scripts/build-furigana-dict.mjs.',
      '',
    ].join('\n'),
  );

  const raw = body.length;
  console.log(
    `scanned ${scanned} lines → kept ${kept} entries\n` +
      `runtime text: ${(raw / 1e6).toFixed(2)} MB raw, ` +
      `${(gz.length / 1e6).toFixed(2)} MB gzipped`,
  );
}

main();
