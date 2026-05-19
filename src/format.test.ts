import { describe, expect, it } from 'bun:test';
import { formatJot } from 'src/format';
import { Jot } from 'src/dsl';
import { parse } from 'src/parser';

/**
 * Deep-strip parser-synthesised `range`s so a `parse → format → parse`
 * comparison reflects only semantic structure, not byte offsets (which
 * legitimately shift when the document is reformatted).
 */
function stripRanges<T>(value: T): T {
  if (Array.isArray(value)) {
    return value.map(stripRanges) as unknown as T;
  }
  if (value && typeof value === 'object') {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value)) {
      if (k === 'range') continue;
      out[k] = stripRanges(v);
    }
    return out as T;
  }
  return value;
}

function roundTrip(src: string): { before: Jot; after: Jot; formatted: string } {
  const before = parse(src);
  const formatted = formatJot(before);
  const after = parse(formatted);
  return { before, after, formatted };
}

function expectStable(src: string): string {
  const { before, after, formatted } = roundTrip(src);
  expect(stripRanges(after)).toEqual(stripRanges(before));
  return formatted;
}

describe('formatJot', () => {
  it('puts each bar on its own line', () => {
    const out = expectStable('{{ time: "4/4" }}\n| k.s.kks. | k.s.k.s. |');
    const barLines = out.split('\n').filter((l) => l.trim().startsWith('|'));
    expect(barLines).toEqual(['| k . s . k k s . |', '| k . s . k . s . |']);
  });

  it('puts each pattern definition on its own line', () => {
    const out = expectStable('[Groove=(k.s.kks.)] [Fill=(k k k k)] | [Groove] [Fill] |');
    const lines = out.split('\n');
    expect(lines).toContain('[Groove=(k . s . k k s .)]');
    expect(lines).toContain('[Fill=(k k k k)]');
  });

  it('round-trips global metadata including title and time signature', () => {
    const out = expectStable(
      '{{ title: "Song", bpm: 120, time: "7/8", ' +
        'instrumentMapping: { k:{name:"Kick"}, s:{name:"Snare"} } }}\n' +
        '| k . s . k k s |'
    );
    expect(out.split('\n')[0]).toContain('title: "Song"');
    expect(out.split('\n')[0]).toContain('time: "7/8"');
  });

  it('round-trips voices joined by ||, on their own line', () => {
    const out = expectStable(
      '| h:c h:c h:c h:c |\n||\n| k . s . |'
    );
    expect(out.split('\n')).toContain('||');
  });

  it('round-trips an anacrusis on its own line above the bars', () => {
    const out = expectStable('{{ time: "4/4" }}\nk k k | s . k . s . k . |');
    const lines = out.split('\n');
    const anaIdx = lines.indexOf('k k k');
    expect(anaIdx).toBeGreaterThanOrEqual(0);
    expect(lines[anaIdx + 1].trim().startsWith('|')).toBe(true);
  });

  it('round-trips weights, repeats, rolls, modifiers, sticking', () => {
    expectStable('| k . s . (k+s k+s k+s)_4 |');
    expectStable('| (k.s.)*4 |');
    expectStable('| c~_8:o | c:k . . . k . s . |');
    expectStable('| s:fl@l k@r s@r:a . k@r s@l k@r k@r |');
  });

  it('round-trips a 3:4 polyrhythm', () => {
    expectStable('| (a a a)_4 + (b b b b)_4 |');
  });

  it('round-trips pattern substitutions', () => {
    expectStable('[Groove=(k.s.kks.)] | [Groove#5-8=(k+s k+s k+s)_4] |');
  });

  it('round-trips per-bar time-signature changes', () => {
    expectStable(
      '{{ time: "7/8" }}\n| k . s . k k s |\n{{ time: "4/4" }}\n| k . s . k . s . |'
    );
  });
});
