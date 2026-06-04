import { describe, expect, test } from 'bun:test';
import { fitFurigana, hasKanji, toHiragana } from '../furigana';

describe('hasKanji', () => {
  test('detects kanji', () => {
    expect(hasKanji('君')).toBe(true);
    expect(hasKanji('取り引き')).toBe(true);
    expect(hasKanji('食べる')).toBe(true);
  });

  test('false for kana / latin / punctuation only', () => {
    expect(hasKanji('みず')).toBe(false);
    expect(hasKanji('コーヒー')).toBe(false);
    expect(hasKanji('hello world')).toBe(false);
    expect(hasKanji('！？…')).toBe(false);
  });

  test('detects astral (Extension B) kanji', () => {
    expect(hasKanji('𠮷')).toBe(true); // U+20BB7
  });
});

describe('toHiragana', () => {
  test('shifts katakana to hiragana', () => {
    expect(toHiragana('タベル')).toBe('たべる');
    expect(toHiragana('キミ')).toBe('きみ');
  });

  test('leaves the prolonged sound mark and non-katakana intact', () => {
    expect(toHiragana('コーヒー')).toBe('こーひー');
    expect(toHiragana('みず')).toBe('みず');
    expect(toHiragana('A！')).toBe('A！');
  });
});

describe('fitFurigana', () => {
  test('single okurigana run: reading sits over the kanji only', () => {
    expect(fitFurigana('食べる', 'たべる')).toEqual([
      { base: '食', reading: 'た' },
      { base: 'べる' },
    ]);
  });

  test('pure single kanji takes the whole reading', () => {
    expect(fitFurigana('君', 'きみ')).toEqual([{ base: '君', reading: 'きみ' }]);
  });

  test('kanji compound (jukugo) takes the whole reading', () => {
    expect(fitFurigana('可能', 'かのう')).toEqual([
      { base: '可能', reading: 'かのう' },
    ]);
  });

  test('jukujikun compound stays whole', () => {
    expect(fitFurigana('今日', 'きょう')).toEqual([
      { base: '今日', reading: 'きょう' },
    ]);
  });

  test('interleaved okurigana anchors each kanji run', () => {
    expect(fitFurigana('取り引き', 'とりひき')).toEqual([
      { base: '取', reading: 'と' },
      { base: 'り' },
      { base: '引', reading: 'ひ' },
      { base: 'き' },
    ]);
  });

  test('leading okurigana stays bare', () => {
    expect(fitFurigana('お前', 'おまえ')).toEqual([
      { base: 'お' },
      { base: '前', reading: 'まえ' },
    ]);
  });

  test('all-kana surface gets no furigana', () => {
    expect(fitFurigana('する', 'する')).toEqual([{ base: 'する' }]);
  });

  test('falls back to bare text when okurigana does not anchor', () => {
    // Reading lacks the surface okurigana `る`, so the fit cannot align.
    expect(fitFurigana('見る', 'みた')).toEqual([{ base: '見る' }]);
  });

  test('falls back to bare text on leftover reading', () => {
    // Reading longer than the kanji can absorb given the okurigana anchor.
    expect(fitFurigana('見る', 'みるある')).toEqual([{ base: '見る' }]);
  });
});
