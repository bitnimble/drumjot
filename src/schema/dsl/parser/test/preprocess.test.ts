import { describe, expect, it } from 'bun:test';
import { preprocessMacros } from 'src/schema/dsl/parser/preprocess';
import { ParseError } from 'src/schema/dsl/parser/errors';

describe('preprocessMacros', () => {
  it('returns input unchanged when no macros are present', () => {
    const src = '| k . s . k . s . |';
    const { text, macros } = preprocessMacros(src);
    expect(text).toBe(src);
    expect(macros).toEqual({});
  });

  it('strips macro definitions and substitutes references', () => {
    const src = '[$grv=k.s.kks.]([$grv])*4';
    const { text, macros } = preprocessMacros(src);
    expect(text).toBe('(k.s.kks.)*4');
    expect(macros).toEqual({ grv: 'k.s.kks.' });
  });

  it('resolves macros that reference other macros', () => {
    const src = '[$inner=k.s.][$outer=[$inner][$inner]][$outer]';
    const { text } = preprocessMacros(src);
    expect(text).toBe('k.s.k.s.');
  });

  it('handles macro bodies containing balanced brackets', () => {
    const src = '[$grp=(k+s)][$grp]';
    const { text } = preprocessMacros(src);
    expect(text).toBe('(k+s)');
  });

  it('throws on unknown macro references', () => {
    expect(() => preprocessMacros('[$missing]')).toThrow(ParseError);
  });

  it('detects non-converging macro substitution', () => {
    const src = '[$a=[$b]][$b=[$a]][$a]';
    expect(() => preprocessMacros(src)).toThrow(/did not converge/);
  });
});
