import { describe, expect, test } from 'bun:test';
import { activeLineIndexAt, parseLrc, stripLyricNoise } from '../lrc';

describe('parseLrc', () => {
  test('parses single-stamp lines in mm:ss.cc form', () => {
    const lines = parseLrc('[00:12.34]Hello world\n[01:05.50]Second line\n');
    expect(lines).toEqual([
      { startSec: 12.34, text: 'Hello world' },
      { startSec: 65.5, text: 'Second line' },
    ]);
  });

  test('accepts mm:ss.ccc (millisecond) precision', () => {
    const lines = parseLrc('[00:01.234]Ms precision');
    expect(lines[0].startSec).toBeCloseTo(1.234, 6);
  });

  test('expands multi-stamp lines into separate sorted entries', () => {
    const lines = parseLrc('[00:30.00][00:10.00]chorus line');
    expect(lines).toEqual([
      { startSec: 10, text: 'chorus line' },
      { startSec: 30, text: 'chorus line' },
    ]);
  });

  test('drops metadata-only lines that have no timestamp', () => {
    const lines = parseLrc('[ar:Artist]\n[ti:Title]\n[00:01.00]actual lyric\n');
    expect(lines).toEqual([{ startSec: 1, text: 'actual lyric' }]);
  });

  test('keeps empty-text stamps so instrumental gaps survive', () => {
    const lines = parseLrc('[00:00.00]\n[00:05.00]first words\n');
    expect(lines).toEqual([
      { startSec: 0, text: '' },
      { startSec: 5, text: 'first words' },
    ]);
  });

  test('returns lines sorted by startSec even when input is unordered', () => {
    const lines = parseLrc('[00:05.00]b\n[00:01.00]a\n[00:09.00]c\n');
    expect(lines.map((l) => l.startSec)).toEqual([1, 5, 9]);
  });

  test('returns an empty array on input with no parseable stamps', () => {
    expect(parseLrc('not an lrc file\njust prose\n')).toEqual([]);
    expect(parseLrc('')).toEqual([]);
  });

  test('skips malformed stamps inside an otherwise-good document', () => {
    // The second stamp has a malformed-fraction `xx`; the regex won't
    // match it. The valid stamp on the first line still goes through.
    const lines = parseLrc('[00:01.00]ok\n[malformed]nope\n[00:02.00]also ok\n');
    expect(lines).toEqual([
      { startSec: 1, text: 'ok' },
      { startSec: 2, text: 'also ok' },
    ]);
  });

  test('drops lines whose text is only a parenthetical aside', () => {
    const lines = parseLrc(
      '[00:01.00]On my own\n' +
        "[00:02.00](I'm screaming, I love you so)\n" +
        '[00:03.00]How did we get here?\n',
    );
    expect(lines).toEqual([
      { startSec: 1, text: 'On my own' },
      { startSec: 3, text: 'How did we get here?' },
    ]);
  });

  test('strips inline parenthetical asides and collapses whitespace', () => {
    const lines = parseLrc('[00:01.00]On my own (just me alone)\n');
    expect(lines).toEqual([{ startSec: 1, text: 'On my own' }]);
  });

  test('drops lines that are only music glyphs', () => {
    const lines = parseLrc('[00:01.00]hello\n[00:02.00]♪ ♪ ♪\n');
    expect(lines).toEqual([{ startSec: 1, text: 'hello' }]);
  });

  test('strips inline music glyphs but keeps the surrounding text', () => {
    const lines = parseLrc('[00:01.00]♪ hello ♪ world ♪\n');
    expect(lines).toEqual([{ startSec: 1, text: 'hello world' }]);
  });

  test('preserves originally-empty stamps as instrumental gaps', () => {
    const lines = parseLrc('[00:00.00]\n[00:05.00]first words\n');
    expect(lines).toEqual([
      { startSec: 0, text: '' },
      { startSec: 5, text: 'first words' },
    ]);
  });
});

describe('stripLyricNoise', () => {
  test('removes parenthetical asides', () => {
    expect(stripLyricNoise("(I'm screaming) hello")).toBe('hello');
  });

  test('removes music glyphs across both blocks', () => {
    // U+266A (eighth note) + U+1D11E (musical symbol G clef).
    expect(stripLyricNoise('hi ♪ \u{1d11e} there')).toBe('hi there');
  });

  test('collapses runs of whitespace introduced by stripping', () => {
    expect(stripLyricNoise('a   (foo)   b')).toBe('a b');
  });

  test('returns empty when every character is noise', () => {
    expect(stripLyricNoise('(echo) ♪ (♫)')).toBe('');
  });

  test('passes clean text through unchanged', () => {
    expect(stripLyricNoise('How did we get here?')).toBe('How did we get here?');
  });
});

describe('activeLineIndexAt', () => {
  const lines = [
    { startSec: 0, text: 'first' },
    { startSec: 10, text: 'second' },
    { startSec: 20, text: 'third' },
  ];

  test('returns undefined before the first line', () => {
    expect(activeLineIndexAt(lines, -1, 0)).toBeUndefined();
  });

  test('returns the active line when the playhead is inside its window', () => {
    expect(activeLineIndexAt(lines, 0, 0)).toBe(0);
    expect(activeLineIndexAt(lines, 5, 0)).toBe(0);
    expect(activeLineIndexAt(lines, 10, 0)).toBe(1);
    expect(activeLineIndexAt(lines, 19.99, 0)).toBe(1);
    expect(activeLineIndexAt(lines, 20, 0)).toBe(2);
  });

  test('past the final line, the final line stays active (no end stamp)', () => {
    expect(activeLineIndexAt(lines, 999, 0)).toBe(2);
  });

  test('positive offset shifts the active windows later in time', () => {
    // With +2s offset, line 1 (startSec=10) starts at audio t=12.
    expect(activeLineIndexAt(lines, 11, 2)).toBe(0);
    expect(activeLineIndexAt(lines, 12, 2)).toBe(1);
  });

  test('negative offset shifts the windows earlier', () => {
    // With -2s offset, line 1 starts at audio t=8.
    expect(activeLineIndexAt(lines, 7.99, -2)).toBe(0);
    expect(activeLineIndexAt(lines, 8, -2)).toBe(1);
  });
});
