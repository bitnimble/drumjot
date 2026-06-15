import { describe, expect, it } from 'bun:test';
import { loadParadbZip } from 'src/schema/rlrr/paradb';
import { RlrrFile } from 'src/schema/rlrr/schema';

// ---------- minimal zip writer (mirrors the reader's assumptions) ----------

type InFile = { name: string; data: Uint8Array; deflate?: boolean };

async function deflateRaw(data: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([data as BlobPart])
    .stream()
    .pipeThrough(new CompressionStream('deflate-raw'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/**
 * Build a spec-shaped zip with stored and/or deflate entries. CRC fields
 * are left zero — the reader (and `DecompressionStream`) never validate
 * them, which is exactly the property this exercises.
 */
async function makeZip(files: InFile[]): Promise<Uint8Array> {
  const enc = new TextEncoder();
  const locals: Uint8Array[] = [];
  const centrals: Uint8Array[] = [];
  let offset = 0;

  for (const f of files) {
    const nameBytes = enc.encode(f.name);
    const method = f.deflate ? 8 : 0;
    const body = f.deflate ? await deflateRaw(f.data) : f.data;

    const lh = new Uint8Array(30 + nameBytes.length);
    const lv = new DataView(lh.buffer);
    lv.setUint32(0, 0x04034b50, true);
    lv.setUint16(4, 20, true);
    lv.setUint16(8, method, true);
    lv.setUint32(18, body.length, true);
    lv.setUint32(22, f.data.length, true);
    lv.setUint16(26, nameBytes.length, true);
    lh.set(nameBytes, 30);
    locals.push(lh, body);

    const ch = new Uint8Array(46 + nameBytes.length);
    const cv = new DataView(ch.buffer);
    cv.setUint32(0, 0x02014b50, true);
    cv.setUint16(4, 20, true);
    cv.setUint16(6, 20, true);
    cv.setUint16(10, method, true);
    cv.setUint32(20, body.length, true);
    cv.setUint32(24, f.data.length, true);
    cv.setUint16(28, nameBytes.length, true);
    cv.setUint32(42, offset, true);
    ch.set(nameBytes, 46);
    centrals.push(ch);

    offset += lh.length + body.length;
  }

  const cdParts = centrals;
  const cdSize = cdParts.reduce((n, p) => n + p.length, 0);
  const cdOffset = offset;

  const eocd = new Uint8Array(22);
  const ev = new DataView(eocd.buffer);
  ev.setUint32(0, 0x06054b50, true);
  ev.setUint16(8, files.length, true);
  ev.setUint16(10, files.length, true);
  ev.setUint32(12, cdSize, true);
  ev.setUint32(16, cdOffset, true);

  const all = [...locals, ...cdParts, eocd];
  const total = all.reduce((n, p) => n + p.length, 0);
  const out = new Uint8Array(total);
  let p = 0;
  for (const part of all) {
    out.set(part, p);
    p += part.length;
  }
  return out;
}

function rlrr(
  complexity: number,
  song: string | string[],
  drums: string | string[],
): RlrrFile {
  return {
    version: 0.7,
    recordingMetadata: { title: '', complexity },
    audioFileData: {
      songTracks: Array.isArray(song) ? song : [song],
      drumTracks: Array.isArray(drums) ? drums : [drums],
    },
    instruments: [],
    events: [{ name: 'BP_Snare_C_1', vel: 90, loc: 0, time: 0.5 }],
    bpmEvents: [{ bpm: 120, time: 0 }],
  };
}

function zipFile(bytes: Uint8Array, name = 'pack.zip'): File {
  return new File([bytes as BlobPart], name, { type: 'application/zip' });
}

const SONG = new Uint8Array([1, 2, 3, 4, 5]);
const DRUMS = new Uint8Array([9, 8, 7, 6]);
const enc = (o: unknown) => new TextEncoder().encode(JSON.stringify(o));

/** UTF-16LE bytes (Paradiddle on Windows writes some .rlrr like this). */
function utf16le(o: unknown, bom: boolean): Uint8Array {
  const s = JSON.stringify(o);
  const out = new Uint8Array((bom ? 1 : 0) * 2 + s.length * 2);
  const v = new DataView(out.buffer);
  let p = 0;
  if (bom) {
    v.setUint16(p, 0xfeff, true);
    p += 2;
  }
  for (let i = 0; i < s.length; i++, p += 2) v.setUint16(p, s.charCodeAt(i), true);
  return out;
}

/** UTF-8 with a leading byte-order mark. */
function utf8Bom(o: unknown): Uint8Array {
  const body = new TextEncoder().encode(JSON.stringify(o));
  const out = new Uint8Array(3 + body.length);
  out.set([0xef, 0xbb, 0xbf], 0);
  out.set(body, 3);
  return out;
}

// ---------- tests ----------

describe('loadParadbZip', () => {
  async function bytesOf(f: File): Promise<Uint8Array> {
    return new Uint8Array(await f.arrayBuffer());
  }

  it('extracts every track, songTracks before drumTracks (stored + deflate)', async () => {
    const bytes = await makeZip([
      { name: 'Song_Expert.rlrr', data: enc(rlrr(4, 'song.ogg', 'Drums/drums.ogg')) },
      { name: 'song.ogg', data: SONG },
      { name: 'Drums/drums.ogg', data: DRUMS, deflate: true },
    ]);

    const map = await loadParadbZip(zipFile(bytes));

    expect(map.rlrrName).toBe('Song_Expert.rlrr');
    expect(map.jot.voices.length).toBeGreaterThan(0);

    expect(map.audioTracks).toHaveLength(2);
    expect(map.audioTracks[0].file.name).toBe('song.ogg');
    expect(map.audioTracks[0].file.type).toBe('audio/ogg');
    expect(map.audioTracks[0].defaultMuted).toBe(false); // song track
    expect(await bytesOf(map.audioTracks[0].file)).toEqual(SONG);

    // The drums entry was deflate-compressed: round-trips via
    // DecompressionStream('deflate-raw'). Drum tracks default to muted.
    expect(map.audioTracks[1].file.name).toBe('drums.ogg');
    expect(map.audioTracks[1].defaultMuted).toBe(true);
    expect(await bytesOf(map.audioTracks[1].file)).toEqual(DRUMS);
  });

  it('extracts multiple song tracks and de-dupes repeated refs', async () => {
    const bytes = await makeZip([
      {
        name: 'chart.rlrr',
        // Two distinct song tracks; the drum ref repeats song_b.
        data: enc(rlrr(2, ['song_a.ogg', 'song_b.ogg'], ['song_b.ogg', 'kit.ogg'])),
      },
      { name: 'song_a.ogg', data: SONG },
      { name: 'song_b.ogg', data: DRUMS },
      { name: 'kit.ogg', data: new Uint8Array([42]) },
    ]);

    const map = await loadParadbZip(zipFile(bytes));
    expect(map.audioTracks.map((t) => t.file.name)).toEqual([
      'song_a.ogg',
      'song_b.ogg',
      'kit.ogg',
    ]);
    // song_b is in both arrays but first as a song track ⇒ not muted.
    expect(map.audioTracks.map((t) => t.defaultMuted)).toEqual([false, false, true]);
  });

  it('loads every songTracks entry, even ones sharing a basename across folders', async () => {
    const bytes = await makeZip([
      {
        name: 'chart.rlrr',
        data: enc(rlrr(2, ['stems/song.ogg', 'extra/song.ogg'], ['drums.ogg'])),
      },
      { name: 'stems/song.ogg', data: SONG },
      { name: 'extra/song.ogg', data: new Uint8Array([7, 7, 7]) },
      { name: 'drums.ogg', data: DRUMS },
    ]);

    const map = await loadParadbZip(zipFile(bytes));
    // Both same-named song tracks resolve to their own folder's file
    // (exact-path match) rather than collapsing to one.
    expect(map.audioTracks).toHaveLength(3);
    expect(await bytesOf(map.audioTracks[0].file)).toEqual(SONG);
    expect(await bytesOf(map.audioTracks[1].file)).toEqual(new Uint8Array([7, 7, 7]));
    expect(await bytesOf(map.audioTracks[2].file)).toEqual(DRUMS);
  });

  it('picks the highest-complexity difficulty', async () => {
    const bytes = await makeZip([
      { name: 'Easy.rlrr', data: enc(rlrr(1, 'song.ogg', 'drums.ogg')) },
      { name: 'Expert.rlrr', data: enc(rlrr(4, 'song.ogg', 'drums.ogg')) },
      { name: 'Medium.rlrr', data: enc(rlrr(2, 'song.ogg', 'drums.ogg')) },
      { name: 'song.ogg', data: SONG },
      { name: 'drums.ogg', data: DRUMS },
    ]);

    const map = await loadParadbZip(zipFile(bytes));
    expect(map.rlrrName).toBe('Expert.rlrr');
  });

  it('breaks complexity ties by filename difficulty (Expert > Hard > Medium > Easy)', async () => {
    const bytes = await makeZip([
      { name: 'Song_Easy.rlrr', data: enc(rlrr(3, 'song.ogg', 'drums.ogg')) },
      { name: 'Song_Expert.rlrr', data: enc(rlrr(3, 'song.ogg', 'drums.ogg')) },
      { name: 'Song_Hard.rlrr', data: enc(rlrr(3, 'song.ogg', 'drums.ogg')) },
      { name: 'song.ogg', data: SONG },
      { name: 'drums.ogg', data: DRUMS },
    ]);

    const map = await loadParadbZip(zipFile(bytes));
    expect(map.rlrrName).toBe('Song_Expert.rlrr');
  });

  it('matches audio tracks by basename, ignoring case and folders', async () => {
    const bytes = await makeZip([
      { name: 'chart.rlrr', data: enc(rlrr(3, 'SONG.OGG', 'kit.ogg')) },
      { name: 'audio/song.ogg', data: SONG },
      { name: 'audio/Kit.OGG', data: DRUMS },
    ]);

    const map = await loadParadbZip(zipFile(bytes));
    expect(await bytesOf(map.audioTracks[0].file)).toEqual(SONG);
    expect(await bytesOf(map.audioTracks[1].file)).toEqual(DRUMS);
  });

  it('parses .rlrr written as UTF-16LE with a BOM', async () => {
    const bytes = await makeZip([
      { name: 'chart.rlrr', data: utf16le(rlrr(3, 'song.ogg', 'drums.ogg'), true) },
      { name: 'song.ogg', data: SONG },
      { name: 'drums.ogg', data: DRUMS },
    ]);
    const map = await loadParadbZip(zipFile(bytes));
    expect(map.jot.voices.length).toBeGreaterThan(0);
    expect(map.audioTracks).toHaveLength(2);
  });

  it('parses BOM-less UTF-16LE .rlrr (NUL-pattern heuristic)', async () => {
    const bytes = await makeZip([
      { name: 'chart.rlrr', data: utf16le(rlrr(3, 'song.ogg', 'drums.ogg'), false) },
      { name: 'song.ogg', data: SONG },
      { name: 'drums.ogg', data: DRUMS },
    ]);
    const map = await loadParadbZip(zipFile(bytes));
    expect(map.jot.voices.length).toBeGreaterThan(0);
  });

  it('parses .rlrr written as UTF-8 with a BOM', async () => {
    const bytes = await makeZip([
      { name: 'chart.rlrr', data: utf8Bom(rlrr(3, 'song.ogg', 'drums.ogg')) },
      { name: 'song.ogg', data: SONG },
      { name: 'drums.ogg', data: DRUMS },
    ]);
    const map = await loadParadbZip(zipFile(bytes));
    expect(map.jot.voices.length).toBeGreaterThan(0);
  });

  it('throws a helpful error when an .rlrr is absent', async () => {
    const bytes = await makeZip([{ name: 'readme.txt', data: SONG }]);
    await expect(loadParadbZip(zipFile(bytes))).rejects.toThrow(/No \.rlrr/);
  });

  it('throws when a referenced audio track is missing from the pack', async () => {
    const bytes = await makeZip([
      { name: 'chart.rlrr', data: enc(rlrr(3, 'gone.ogg', 'drums.ogg')) },
      { name: 'drums.ogg', data: DRUMS },
    ]);
    await expect(loadParadbZip(zipFile(bytes))).rejects.toThrow(/gone\.ogg/);
  });
});
