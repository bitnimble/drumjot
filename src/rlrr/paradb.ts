/**
 * ParaDB map-pack (`.zip`) loading.
 *
 * A ParaDB / Paradiddle chart is distributed as a `.zip` containing:
 *  - one or more `.rlrr` files (one per difficulty), each a JSON doc
 *    whose `audioFileData` names the backing audio,
 *  - the audio tracks referenced by those names (typically a drumless
 *    "song" track and a drums-only track), and usually a cover image.
 *
 * We unzip entirely in memory using the browser's `DecompressionStream`
 * (no zip dependency) — a minimal central-directory reader locates the
 * entries and we inflate only the ones we actually need: the chosen
 * `.rlrr` plus its audio tracks. Cover images and the difficulties we
 * don't load are never decompressed.
 *
 * Difficulty selection: a pack often ships Easy/Medium/Hard/Expert
 * `.rlrr` files. We load the highest `recordingMetadata.complexity`
 * (the most complete chart). Packs sometimes give every difficulty the
 * same `complexity`, so ties are broken by the difficulty word in the
 * filename (Expert > Hard > Medium > Easy), then by first entry.
 *
 * Audio tracks: every entry of `audioFileData.songTracks` then every
 * entry of `drumTracks` is extracted as an independent track (a pack
 * may ship several of either). They carry no music-vs-drums semantics —
 * the caller loads each as its own track. Audio entries are matched to the
 * referenced names by basename, case-insensitively, since pack authors
 * vary the path casing/folders. De-dup is on the resolved file, so the
 * same physical file referenced twice (e.g. once per array) loads once,
 * but every distinct track in the arrays loads.
 */
import { Jot } from 'src/dsl';
import {
  inflateEntry,
  readCentralDirectory,
  ZipEntry,
  zipEntryBasename,
} from 'src/zip';
import { rlrrToJot, RlrrToJotOptions } from './rlrr_to_jot';
import { RlrrFile } from './schema';

export type ParadbTrack = {
  file: File;
  /**
   * True for tracks that came from `audioFileData.drumTracks`. They
   * still load as normal audio tracks, but the caller defaults them to
   * muted: when practising along you want the backing music audible and
   * the reference drum track silent (you're playing the drums). A file
   * referenced by both arrays counts as a song track (not muted).
   */
  defaultMuted: boolean;
  /**
   * Track role for the per-row overflow menu. ParaDB's `audioFileData`
   * doesn't formally distinguish drumless backing from full-mix backing,
   * so `songTracks` entries are always tagged `'full-mix'` (lets the
   * user re-run drum isolation on packs whose stem separation was poor)
   * and `drumTracks` entries are tagged `'drums'`.
   */
  role: 'full-mix' | 'drums';
};

export type ParadbMap = {
  jot: Jot;
  /** Source `.rlrr` filename within the pack (for diagnostics/title). */
  rlrrName: string;
  /**
   * Every audio track the chart references — all of `songTracks`
   * followed by all of `drumTracks`, in that order. The caller loads
   * each as its own track; `defaultMuted` flags the drum tracks.
   * Duplicate references collapse to one entry.
   */
  audioTracks: ParadbTrack[];
};

export type LoadParadbOptions = RlrrToJotOptions;

/**
 * Load a ParaDB `.zip` map pack: convert the chosen `.rlrr` to a Jot
 * and extract the referenced audio tracks as `File`s ready for
 * `jotPlayer.loadAudioTrack`. Throws with a human-readable message on a
 * malformed pack so callers can surface it directly.
 */
export async function loadParadbZip(
  file: File,
  options: LoadParadbOptions = {},
): Promise<ParadbMap> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const entries = readCentralDirectory(bytes);
  if (entries.length === 0) {
    throw new Error('Not a valid zip archive (no central directory found).');
  }

  const rlrrEntries = entries.filter((e) => e.name.toLowerCase().endsWith('.rlrr'));
  if (rlrrEntries.length === 0) {
    throw new Error('No .rlrr chart found in the ParaDB pack.');
  }

  // Inflate every candidate .rlrr so we can pick by difficulty. These
  // are small JSON docs; the heavy audio entries stay compressed until
  // we know which two we need.
  const candidates: { entry: ZipEntry; rlrr: RlrrFile }[] = [];
  for (const entry of rlrrEntries) {
    const raw = await inflateEntry(bytes, entry);
    let rlrr: RlrrFile;
    try {
      rlrr = JSON.parse(decodeRlrrText(raw)) as RlrrFile;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      throw new Error(`Could not parse ${entry.name}: ${message}`);
    }
    candidates.push({ entry, rlrr });
  }

  const chosen = candidates.reduce((best, cur) => {
    const byComplexity = complexityOf(cur.rlrr) - complexityOf(best.rlrr);
    if (byComplexity !== 0) return byComplexity > 0 ? cur : best;
    // Same complexity: prefer the harder-named file (Expert > Hard >
    // Medium > Easy), keeping the earlier entry on a full tie.
    return difficultyRank(cur.entry.name) > difficultyRank(best.entry.name)
      ? cur
      : best;
  });
  const rlrr = chosen.rlrr;
  const jot = rlrrToJot(rlrr, options);

  const refs: { ref: string; isDrums: boolean }[] = [
    ...(rlrr.audioFileData?.songTracks ?? []).map((ref) => ({ ref, isDrums: false })),
    ...(rlrr.audioFileData?.drumTracks ?? []).map((ref) => ({ ref, isDrums: true })),
  ];
  const audioTracks: ParadbTrack[] = [];
  // De-dupe on the resolved zip entry, not the reference string, so
  // every distinct `songTracks` / `drumTracks` file loads (even two
  // that share a basename in different folders) while the same physical
  // file referenced twice — e.g. once in each array — isn't loaded
  // twice. Song tracks come first, so a file in both arrays keeps its
  // song (unmuted) classification.
  const seen = new Set<string>();
  for (const { ref, isDrums } of refs) {
    const entry = resolveEntry(entries, ref);
    if (seen.has(entry.name)) continue;
    seen.add(entry.name);
    audioTracks.push({
      file: await inflateToFile(bytes, entry),
      defaultMuted: isDrums,
      role: isDrums ? 'drums' : 'full-mix',
    });
  }

  return { jot, rlrrName: chosen.entry.name, audioTracks };
}

/**
 * Decode `.rlrr` bytes to text. Paradiddle (a Unity/Windows app) writes
 * these files in several encodings: UTF-8, UTF-8 with BOM, or UTF-16
 * (LE/BE), sometimes BOM-less. A plain UTF-8 `TextDecoder` turns a
 * UTF-16 doc into U+FFFD at byte 0, so `JSON.parse` fails on the very
 * first character even though a text editor (which sniffs the encoding)
 * opens it fine. Pick the decoder from the BOM, then fall back to a
 * NUL-pattern heuristic for the BOM-less UTF-16 case.
 */
function decodeRlrrText(raw: Uint8Array): string {
  if (raw.length >= 3 && raw[0] === 0xef && raw[1] === 0xbb && raw[2] === 0xbf) {
    // UTF-8 BOM: the default decoder strips it (ignoreBOM === false).
    return new TextDecoder('utf-8').decode(raw);
  }
  if (raw.length >= 2 && raw[0] === 0xff && raw[1] === 0xfe) {
    return new TextDecoder('utf-16le').decode(raw);
  }
  if (raw.length >= 2 && raw[0] === 0xfe && raw[1] === 0xff) {
    return new TextDecoder('utf-16be').decode(raw);
  }
  // BOM-less UTF-16: an ASCII first char (`{` or whitespace) leaves a
  // NUL in the other byte of the code unit. UTF-8 JSON never has NULs.
  if (raw.length >= 2 && raw[0] !== 0 && raw[1] === 0) {
    return new TextDecoder('utf-16le').decode(raw);
  }
  if (raw.length >= 2 && raw[0] === 0 && raw[1] !== 0) {
    return new TextDecoder('utf-16be').decode(raw);
  }
  return new TextDecoder('utf-8').decode(raw);
}

function complexityOf(rlrr: RlrrFile): number {
  return rlrr.recordingMetadata?.complexity ?? 0;
}

/**
 * Rank the difficulty named in an `.rlrr` filename so equal-complexity
 * packs still pick the most complete chart. Higher is harder; an
 * unrecognised name ranks below Easy so a named difficulty always wins.
 */
function difficultyRank(name: string): number {
  const n = name.toLowerCase();
  if (n.includes('expert')) return 4;
  if (n.includes('hard')) return 3;
  if (n.includes('medium')) return 2;
  if (n.includes('easy')) return 1;
  return 0;
}

/**
 * Resolve an `audioFileData` reference to its zip entry. Pack authors
 * store these as bare names, `Songs/foo.ogg`, mixed case, etc. Prefer
 * an exact (normalized, case-insensitive) full-path match so two song
 * tracks that differ only by folder still resolve to distinct files;
 * fall back to a basename match for the common bare-name case. Throws
 * if the referenced file isn't in the pack.
 */
function resolveEntry(entries: ZipEntry[], ref: string): ZipEntry {
  const norm = (s: string) => s.replace(/\\/g, '/').toLowerCase();
  const fullRef = norm(ref);
  const exact = entries.find((e) => norm(e.name) === fullRef);
  if (exact) return exact;
  const wanted = zipEntryBasename(ref).toLowerCase();
  const entry = entries.find((e) => zipEntryBasename(e.name).toLowerCase() === wanted);
  if (!entry) {
    throw new Error(`Audio "${ref}" referenced by the chart is missing from the pack.`);
  }
  return entry;
}

/** Inflate a resolved entry into a typed `File` ready for playback. */
async function inflateToFile(bytes: Uint8Array, entry: ZipEntry): Promise<File> {
  const data = await inflateEntry(bytes, entry);
  const name = zipEntryBasename(entry.name);
  // Type the Blob so `decodeAudioData` / <audio> get a codec hint;
  // extension-based is enough here and matches how the picker labels it.
  return new File([data as BlobPart], name, { type: mimeForExt(name) });
}

function mimeForExt(name: string): string {
  const ext = name.slice(name.lastIndexOf('.') + 1).toLowerCase();
  switch (ext) {
    case 'ogg':
    case 'oga':
      return 'audio/ogg';
    case 'mp3':
      return 'audio/mpeg';
    case 'wav':
      return 'audio/wav';
    case 'flac':
      return 'audio/flac';
    case 'm4a':
    case 'aac':
      return 'audio/aac';
    default:
      return 'application/octet-stream';
  }
}
