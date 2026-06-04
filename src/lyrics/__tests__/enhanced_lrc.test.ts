import { describe, expect, test } from 'bun:test';
import { LyricLine } from '../lrc';
import {
  parseEnhancedLrc,
  secondsToStamp,
  serializeEnhancedLrc,
} from '../enhanced_lrc';

/** Round-trip via the serialized form: serializing the parse of a
 *  document must reproduce the document byte-for-byte. Avoids
 *  float-equality pitfalls since serialization quantizes to ms. */
function reserialize(doc: string): string {
  const { lines, offsetSec } = parseEnhancedLrc(doc);
  return serializeEnhancedLrc(lines, { offsetSec });
}

describe('secondsToStamp', () => {
  test('formats mm:ss.ccc with zero-padding', () => {
    expect(secondsToStamp(12.34)).toBe('00:12.340');
    expect(secondsToStamp(13)).toBe('00:13.000');
    expect(secondsToStamp(65.5)).toBe('01:05.500');
    expect(secondsToStamp(0)).toBe('00:00.000');
  });

  test('clamps negative to zero', () => {
    expect(secondsToStamp(-1)).toBe('00:00.000');
  });
});

describe('serializeEnhancedLrc', () => {
  test('word-aligned line emits <start>text<end> per word', () => {
    const lines: LyricLine[] = [
      {
        startSec: 12.34,
        text: 'Hello world',
        words: [
          { startSec: 12.34, endSec: 12.9, text: 'Hello' },
          { startSec: 13.0, endSec: 13.45, text: 'world' },
        ],
      },
    ];
    expect(serializeEnhancedLrc(lines)).toBe(
      '[00:12.340]<00:12.340>Hello<00:12.900> <00:13.000>world<00:13.450>\n',
    );
  });

  test('line-only line stays plain LRC', () => {
    const lines: LyricLine[] = [{ startSec: 15, text: 'A plain line' }];
    expect(serializeEnhancedLrc(lines)).toBe('[00:15.000]A plain line\n');
  });

  test('non-zero offset emits a leading [offset:ms] header', () => {
    const lines: LyricLine[] = [{ startSec: 1, text: 'x' }];
    expect(serializeEnhancedLrc(lines, { offsetSec: 1.23 })).toBe(
      '[offset:1230]\n[00:01.000]x\n',
    );
    expect(serializeEnhancedLrc(lines, { offsetSec: -0.5 })).toBe(
      '[offset:-500]\n[00:01.000]x\n',
    );
    expect(serializeEnhancedLrc(lines, { offsetSec: 0 })).toBe(
      '[00:01.000]x\n',
    );
  });

  test('escapes backslash and angle brackets in word text', () => {
    const lines: LyricLine[] = [
      {
        startSec: 1,
        text: 'x<3 a\\b',
        words: [
          { startSec: 1, endSec: 1.2, text: 'x<3' },
          { startSec: 1.3, endSec: 1.5, text: 'a\\b' },
        ],
      },
    ];
    expect(serializeEnhancedLrc(lines)).toBe(
      '[00:01.000]<00:01.000>x\\<3<00:01.200> <00:01.300>a\\\\b<00:01.500>\n',
    );
  });
});

describe('parseEnhancedLrc', () => {
  test('parses word-aligned line into words with start + end', () => {
    const { lines, offsetSec } = parseEnhancedLrc(
      '[00:12.340]<00:12.340>Hello<00:12.900> <00:13.000>world<00:13.450>\n',
    );
    expect(offsetSec).toBe(0);
    expect(lines).toHaveLength(1);
    const w = lines[0].words!;
    expect(w).toHaveLength(2);
    expect(w[0].text).toBe('Hello');
    expect(w[0].startSec).toBeCloseTo(12.34, 6);
    expect(w[0].endSec).toBeCloseTo(12.9, 6);
    expect(w[1].text).toBe('world');
    expect(w[1].startSec).toBeCloseTo(13.0, 6);
    expect(w[1].endSec).toBeCloseTo(13.45, 6);
  });

  test('decodes the offset header without baking it into times', () => {
    const { lines, offsetSec } = parseEnhancedLrc(
      '[offset:-500]\n[00:01.000]<00:01.000>x<00:01.200>\n',
    );
    expect(offsetSec).toBeCloseTo(-0.5, 6);
    expect(lines[0].words![0].startSec).toBeCloseTo(1.0, 6);
  });

  test('recovers escaped word text', () => {
    const { lines } = parseEnhancedLrc(
      '[00:01.000]<00:01.000>x\\<3<00:01.200> <00:01.300>a\\\\b<00:01.500>\n',
    );
    const w = lines[0].words!;
    expect(w[0].text).toBe('x<3');
    expect(w[1].text).toBe('a\\b');
  });

  test('infers end times for start-only (foreign) enhanced LRC', () => {
    const { lines } = parseEnhancedLrc(
      '[00:01.000]<00:01.000>foo <00:01.500>bar\n',
    );
    const w = lines[0].words!;
    expect(w[0].text).toBe('foo');
    expect(w[0].endSec).toBeCloseTo(1.5, 6); // next word's start
    expect(w[1].text).toBe('bar');
    expect(w[1].endSec).toBeCloseTo(1.55, 6); // last word: start + epsilon
  });

  test('preserves instrumental-gap blank stamps', () => {
    const { lines } = parseEnhancedLrc('[00:34.120]\n');
    expect(lines).toHaveLength(1);
    expect(lines[0].text).toBe('');
    expect(lines[0].words).toBeUndefined();
  });

  test('back-compat: plain line-level LRC parses unchanged', () => {
    const { lines, offsetSec } = parseEnhancedLrc(
      '[00:12.34]Hello world\n[01:05.50]Second line\n',
    );
    expect(offsetSec).toBe(0);
    expect(lines).toHaveLength(2);
    expect(lines[0].text).toBe('Hello world');
    expect(lines[0].startSec).toBeCloseTo(12.34, 6);
    expect(lines[0].words).toBeUndefined();
    expect(lines[1].startSec).toBeCloseTo(65.5, 6);
  });

  test('sorts lines by start time', () => {
    const { lines } = parseEnhancedLrc('[00:30.000]later\n[00:10.000]earlier\n');
    expect(lines.map((l) => l.text)).toEqual(['earlier', 'later']);
  });
});

describe('round-trip stability', () => {
  test('serialize → parse → serialize is idempotent', () => {
    const docs = [
      // word-aligned with a silent gap between words
      '[00:12.340]<00:12.340>Hello<00:12.900> <00:13.000>world<00:13.450>\n',
      // mixed: word line, plain line, instrumental blank
      '[offset:1230]\n' +
        '[00:01.000]<00:01.000>one<00:01.400> <00:01.400>two<00:01.900>\n' +
        '[00:05.000]a plain line\n' +
        '[00:08.000]\n',
      // Japanese tokens (no inter-token spaces in the source surface)
      '[00:02.000]<00:02.000>君<00:02.300> <00:02.300>の<00:02.450> <00:02.450>名<00:02.800>\n',
      // escaped specials
      '[00:01.000]<00:01.000>x\\<3<00:01.200>\n',
      // negative offset, line-only
      '[offset:-250]\n[00:03.000]just text\n',
    ];
    for (const doc of docs) {
      expect(reserialize(doc)).toBe(doc);
    }
  });

  test('full LyricLine[] round-trips through serialize → parse', () => {
    const lines: LyricLine[] = [
      {
        startSec: 1.0,
        text: 'one two',
        words: [
          { startSec: 1.0, endSec: 1.4, text: 'one' },
          { startSec: 1.4, endSec: 1.9, text: 'two' },
        ],
      },
      { startSec: 5.0, text: 'plain line' },
    ];
    const { lines: out } = parseEnhancedLrc(
      serializeEnhancedLrc(lines, { offsetSec: 0.5 }),
    );
    expect(out).toHaveLength(2);
    expect(out[0].words).toHaveLength(2);
    expect(out[0].words![0]).toMatchObject({
      startSec: 1.0,
      endSec: 1.4,
      text: 'one',
    });
    expect(out[0].words![1]).toMatchObject({
      startSec: 1.4,
      endSec: 1.9,
      text: 'two',
    });
    expect(out[1].text).toBe('plain line');
    expect(out[1].words).toBeUndefined();
  });
});
