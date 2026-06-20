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

// ---------- Writer ----------

/** One entry to write into a zip via {@link writeZip}. */
export type ZipWriteEntry = {
  /** Entry path inside the archive (forward slashes), e.g. `audio/0.mp3`. */
  name: string;
  /** Uncompressed bytes. */
  data: Uint8Array;
  /** Deflate the entry (method 8) when true; store it uncompressed (method 0)
   *  when false/omitted. Leave off for already-compressed payloads (MP3 /
   *  FLAC audio) where deflate only burns CPU; turn on for JSON / text. */
  compress?: boolean;
};

/**
 * Write a minimal, spec-valid zip archive that {@link readCentralDirectory} +
 * {@link inflateEntry} read back. The counterpart to the reader above: same
 * "no zip dependency" approach, using `CompressionStream('deflate-raw')` for
 * the compressed entries. Local headers carry real sizes + CRC32 (no data
 * descriptors), so the archive is also openable by standard tools.
 */
export async function writeZip(entries: ZipWriteEntry[]): Promise<Uint8Array> {
  const prepared = await Promise.all(
    entries.map(async (e) => {
      const crc = crc32(e.data);
      const body = e.compress ? await deflateRaw(e.data) : e.data;
      return {
        nameBytes: new TextEncoder().encode(e.name),
        body,
        crc,
        method: e.compress ? 8 : 0,
        uncompSize: e.data.length,
      };
    })
  );

  const chunks: Uint8Array[] = [];
  const central: Uint8Array[] = [];
  let offset = 0;

  for (const e of prepared) {
    const local = new Uint8Array(30 + e.nameBytes.length);
    const lv = new DataView(local.buffer);
    lv.setUint32(0, LOC_SIG, true);
    lv.setUint16(4, 20, true); // version needed
    lv.setUint16(6, 0, true); // flags
    lv.setUint16(8, e.method, true);
    lv.setUint16(10, 0, true); // mod time
    lv.setUint16(12, 0, true); // mod date
    lv.setUint32(14, e.crc, true);
    lv.setUint32(18, e.body.length, true); // compressed size
    lv.setUint32(22, e.uncompSize, true);
    lv.setUint16(26, e.nameBytes.length, true);
    lv.setUint16(28, 0, true); // extra len
    local.set(e.nameBytes, 30);

    const localOffset = offset;
    chunks.push(local, e.body);
    offset += local.length + e.body.length;

    const cen = new Uint8Array(46 + e.nameBytes.length);
    const cv = new DataView(cen.buffer);
    cv.setUint32(0, CEN_SIG, true);
    cv.setUint16(4, 20, true); // version made by
    cv.setUint16(6, 20, true); // version needed
    cv.setUint16(8, 0, true); // flags
    cv.setUint16(10, e.method, true);
    cv.setUint16(12, 0, true); // mod time
    cv.setUint16(14, 0, true); // mod date
    cv.setUint32(16, e.crc, true);
    cv.setUint32(20, e.body.length, true);
    cv.setUint32(24, e.uncompSize, true);
    cv.setUint16(28, e.nameBytes.length, true);
    cv.setUint16(30, 0, true); // extra len
    cv.setUint16(32, 0, true); // comment len
    cv.setUint16(34, 0, true); // disk start
    cv.setUint16(36, 0, true); // internal attrs
    cv.setUint32(38, 0, true); // external attrs
    cv.setUint32(42, localOffset, true);
    cen.set(e.nameBytes, 46);
    central.push(cen);
  }

  const cdOffset = offset;
  let cdSize = 0;
  for (const c of central) {
    chunks.push(c);
    cdSize += c.length;
    offset += c.length;
  }

  const eocd = new Uint8Array(22);
  const ev = new DataView(eocd.buffer);
  ev.setUint32(0, EOCD_SIG, true);
  ev.setUint16(8, prepared.length, true); // entries on this disk
  ev.setUint16(10, prepared.length, true); // total entries
  ev.setUint32(12, cdSize, true);
  ev.setUint32(16, cdOffset, true);
  chunks.push(eocd);

  return concatBytes(chunks, offset + eocd.length);
}

async function deflateRaw(data: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([new Uint8Array(data)])
    .stream()
    .pipeThrough(new CompressionStream('deflate-raw'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function concatBytes(chunks: readonly Uint8Array[], total: number): Uint8Array {
  const out = new Uint8Array(total);
  let at = 0;
  for (const c of chunks) {
    out.set(c, at);
    at += c.length;
  }
  return out;
}

let CRC_TABLE: Uint32Array | undefined;

/** Standard zip CRC-32 (reflected, poly 0xEDB88320). */
function crc32(bytes: Uint8Array): number {
  let table = CRC_TABLE;
  if (!table) {
    table = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      table[n] = c >>> 0;
    }
    CRC_TABLE = table;
  }
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) crc = table[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}
