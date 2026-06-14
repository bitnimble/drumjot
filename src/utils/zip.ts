/**
 * Minimal in-memory zip reader shared by the .rlrr (ParaDB) and debug-bundle
 * loaders.
 *
 * Browsers ship `DecompressionStream('deflate-raw')` so we don't need a
 * zip dependency — only a small parser over the End Of Central Directory
 * record and a per-entry inflater. Reading the central directory (not
 * scanning local headers) is robust to zips written with streaming data
 * descriptors that omit sizes from the local header.
 *
 * This module owns the *bytes-level* contract only; format-specific
 * concerns (which entries to load, how to decode their contents) live in
 * the callers (`src/rlrr/paradb.ts`, `src/debug_zip.ts`).
 */

/** One entry from the zip's central directory. */
export type ZipEntry = {
  name: string;
  /** 0 = stored, 8 = deflate. Other methods are unsupported. */
  method: number;
  compressedSize: number;
  /** Byte offset of the entry's local file header. */
  localHeaderOffset: number;
};

const EOCD_SIG = 0x06054b50;
const CEN_SIG = 0x02014b50;
const LOC_SIG = 0x04034b50;

/**
 * Parse the End Of Central Directory record and walk the central
 * directory, returning every entry's name, compression method, size,
 * and local-header offset. Returns an empty array on a truncated /
 * malformed file rather than throwing — the caller is expected to handle
 * "no entries" itself.
 */
export function readCentralDirectory(bytes: Uint8Array): ZipEntry[] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const eocd = findEocd(view);
  if (eocd < 0) return [];

  const total = view.getUint16(eocd + 10, true);
  let ptr = view.getUint32(eocd + 16, true); // central directory offset

  const entries: ZipEntry[] = [];
  for (let i = 0; i < total; i++) {
    if (ptr + 46 > bytes.length || view.getUint32(ptr, true) !== CEN_SIG) {
      break; // truncated / unexpected — return what we have
    }
    const method = view.getUint16(ptr + 10, true);
    const compressedSize = view.getUint32(ptr + 20, true);
    const nameLen = view.getUint16(ptr + 28, true);
    const extraLen = view.getUint16(ptr + 30, true);
    const commentLen = view.getUint16(ptr + 32, true);
    const localHeaderOffset = view.getUint32(ptr + 42, true);
    const name = new TextDecoder().decode(bytes.subarray(ptr + 46, ptr + 46 + nameLen));
    if (!name.endsWith('/')) {
      entries.push({ name, method, compressedSize, localHeaderOffset });
    }
    ptr += 46 + nameLen + extraLen + commentLen;
  }
  return entries;
}

/** Scan backwards for the EOCD signature (the trailing comment, if any). */
function findEocd(view: DataView): number {
  const len = view.byteLength;
  const min = Math.max(0, len - (22 + 0xffff)); // 22-byte EOCD + max comment
  for (let i = len - 22; i >= min; i--) {
    if (view.getUint32(i, true) === EOCD_SIG) return i;
  }
  return -1;
}

/**
 * Inflate (or copy, for stored) a single entry's bytes. Throws if the
 * local header signature is wrong or the compression method is anything
 * other than stored (0) or deflate (8).
 */
export async function inflateEntry(
  bytes: Uint8Array,
  entry: ZipEntry
): Promise<Uint8Array> {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const lh = entry.localHeaderOffset;
  if (view.getUint32(lh, true) !== LOC_SIG) {
    throw new Error(`Corrupt zip: bad local header for ${entry.name}.`);
  }
  // The local header's name/extra lengths can differ from the central
  // directory's (extra fields are often rewritten), so read them here.
  const nameLen = view.getUint16(lh + 26, true);
  const extraLen = view.getUint16(lh + 28, true);
  const dataStart = lh + 30 + nameLen + extraLen;
  const data = bytes.subarray(dataStart, dataStart + entry.compressedSize);

  if (entry.method === 0) return data.slice(); // stored
  if (entry.method !== 8) {
    throw new Error(`Unsupported zip compression method ${entry.method} for ${entry.name}.`);
  }
  const stream = new Blob([data as BlobPart])
    .stream()
    .pipeThrough(new DecompressionStream('deflate-raw'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** `path/to/file.ext` -> `file.ext`. Tolerates backslashes from Windows-authored zips. */
export function zipEntryBasename(path: string): string {
  const norm = path.replace(/\\/g, '/');
  const slash = norm.lastIndexOf('/');
  return slash >= 0 ? norm.slice(slash + 1) : norm;
}
