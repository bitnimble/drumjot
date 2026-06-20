import { describe, expect, it } from 'bun:test';
import {
  classifyByExtension,
  classifyZip,
  extractJotFromZip,
  fileExtension,
  planDrop,
} from 'src/editing/drag_drop/file_routing';

function fileNamed(name: string, content = ''): File {
  return new File([content], name);
}

/**
 * Build a minimal "stored" (uncompressed, method 0) zip with the given
 * entries. Enough for {@link classifyZip} (central-directory names only)
 * and {@link extractJotFromZip} (stored bytes copied verbatim) without a
 * compression dependency.
 */
function makeStoredZip(entries: { name: string; content?: string }[]): File {
  const enc = new TextEncoder();
  const parts: Uint8Array[] = [];
  const central: Uint8Array[] = [];
  let offset = 0;

  for (const { name, content = '' } of entries) {
    const nameBytes = enc.encode(name);
    const data = enc.encode(content);

    const local = new Uint8Array(30 + nameBytes.length + data.length);
    const lv = new DataView(local.buffer);
    lv.setUint32(0, 0x04034b50, true); // local file header sig
    lv.setUint16(8, 0, true); // method = stored
    lv.setUint32(18, data.length, true); // compressed size
    lv.setUint32(22, data.length, true); // uncompressed size
    lv.setUint16(26, nameBytes.length, true); // name length
    local.set(nameBytes, 30);
    local.set(data, 30 + nameBytes.length);

    const cen = new Uint8Array(46 + nameBytes.length);
    const cv = new DataView(cen.buffer);
    cv.setUint32(0, 0x02014b50, true); // central dir header sig
    cv.setUint16(10, 0, true); // method = stored
    cv.setUint32(20, data.length, true); // compressed size
    cv.setUint32(24, data.length, true); // uncompressed size
    cv.setUint16(28, nameBytes.length, true); // name length
    cv.setUint32(42, offset, true); // local header offset
    cen.set(nameBytes, 46);

    parts.push(local);
    central.push(cen);
    offset += local.length;
  }

  const cdOffset = offset;
  const cdSize = central.reduce((n, c) => n + c.length, 0);
  const eocd = new Uint8Array(22);
  const ev = new DataView(eocd.buffer);
  ev.setUint32(0, 0x06054b50, true); // EOCD sig
  ev.setUint16(8, entries.length, true); // entries on this disk
  ev.setUint16(10, entries.length, true); // total entries
  ev.setUint32(12, cdSize, true);
  ev.setUint32(16, cdOffset, true);

  return new File([...parts, ...central, eocd] as BlobPart[], 'bundle.zip');
}

describe('fileExtension', () => {
  it('lowercases and strips the dot', () => {
    expect(fileExtension('Song.JOT')).toBe('jot');
    expect(fileExtension('a.b.MID')).toBe('mid');
  });
  it('is empty for no extension or a trailing dot', () => {
    expect(fileExtension('README')).toBe('');
    expect(fileExtension('weird.')).toBe('');
  });
});

describe('classifyByExtension', () => {
  it('routes the known kinds', () => {
    expect(classifyByExtension(fileNamed('a.jot'))).toBe('jot');
    expect(classifyByExtension(fileNamed('a.mid'))).toBe('midi');
    expect(classifyByExtension(fileNamed('a.midi'))).toBe('midi');
    expect(classifyByExtension(fileNamed('a.lrc'))).toBe('lyrics');
    expect(classifyByExtension(fileNamed('a.txt'))).toBe('lyrics');
    expect(classifyByExtension(fileNamed('a.zip'))).toBe('zip');
    expect(classifyByExtension(fileNamed('a.mp3'))).toBe('audio');
    expect(classifyByExtension(fileNamed('a.wav'))).toBe('audio');
    expect(classifyByExtension(fileNamed('a.flac'))).toBe('audio');
  });
  it('is unknown for unrecognised extensions', () => {
    expect(classifyByExtension(fileNamed('a.pdf'))).toBe('unknown');
    expect(classifyByExtension(fileNamed('noext'))).toBe('unknown');
  });
});

describe('classifyZip', () => {
  it('detects a debug bundle by debug.json', async () => {
    const zip = makeStoredZip([{ name: 'debug.json' }, { name: 'foo.rlrr' }]);
    expect(await classifyZip(zip)).toBe('debug');
  });
  it('detects a ParaDB pack by a .rlrr entry', async () => {
    const zip = makeStoredZip([{ name: 'chart_Expert.rlrr' }, { name: 'song.ogg' }]);
    expect(await classifyZip(zip)).toBe('paradb');
  });
  it('detects a bare-jot archive by a .jot entry', async () => {
    const zip = makeStoredZip([{ name: 'pattern.jot', content: 'x' }]);
    expect(await classifyZip(zip)).toBe('jot');
  });
  it('is unknown for an archive matching no layout', async () => {
    const zip = makeStoredZip([{ name: 'notes.md' }]);
    expect(await classifyZip(zip)).toBe('unknown');
  });
});

describe('extractJotFromZip', () => {
  it('returns the inner .jot as a File with its DSL text', async () => {
    const zip = makeStoredZip([{ name: 'nested/groove.jot', content: 'title Test' }]);
    const file = await extractJotFromZip(zip);
    expect(file).not.toBeNull();
    expect(file!.name).toBe('groove.jot');
    expect(await file!.text()).toBe('title Test');
  });
  it('returns null when there is no .jot entry', async () => {
    const zip = makeStoredZip([{ name: 'debug.json' }]);
    expect(await extractJotFromZip(zip)).toBeNull();
  });
});

describe('planDrop', () => {
  it('routes additive audio + lyrics and keeps drop order', async () => {
    const plan = await planDrop([fileNamed('a.mp3'), fileNamed('b.lrc'), fileNamed('c.wav')]);
    expect(plan.documentLoad).toBeUndefined();
    expect(plan.additive.map((a) => a.kind)).toEqual(['audio', 'lyrics', 'audio']);
    expect(plan.unknown).toHaveLength(0);
  });

  it('takes the first document load and flags the rest as ignored', async () => {
    const plan = await planDrop([fileNamed('one.jot'), fileNamed('two.mid'), fileNamed('s.mp3')]);
    expect(plan.documentLoad?.kind).toBe('jot');
    expect(plan.ignoredDocumentLoads.map((d) => d.kind)).toEqual(['midi']);
    expect(plan.additive.map((a) => a.kind)).toEqual(['audio']);
  });

  it('collects unrecognised files', async () => {
    const plan = await planDrop([fileNamed('a.pdf'), fileNamed('b.mp3')]);
    expect(plan.unknown.map((f) => f.name)).toEqual(['a.pdf']);
    expect(plan.additive.map((a) => a.kind)).toEqual(['audio']);
  });

  it('resolves a debug-bundle zip to a single document load', async () => {
    const zip = makeStoredZip([{ name: 'debug.json' }]);
    const plan = await planDrop([zip]);
    expect(plan.documentLoad?.kind).toBe('debug');
  });

  it('extracts a bare-jot zip into a jot document load', async () => {
    const zip = makeStoredZip([{ name: 'groove.jot', content: 'title Z' }]);
    const plan = await planDrop([zip]);
    expect(plan.documentLoad?.kind).toBe('jot');
    expect(plan.documentLoad?.file.name).toBe('groove.jot');
  });

  it('routes an unrecognised zip to unknown', async () => {
    const zip = makeStoredZip([{ name: 'readme.md' }]);
    const plan = await planDrop([zip]);
    expect(plan.documentLoad).toBeUndefined();
    expect(plan.unknown).toHaveLength(1);
  });
});
