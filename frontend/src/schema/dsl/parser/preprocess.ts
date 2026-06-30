import { ParseError } from './errors';

/**
 * Macro pre-processing for the Drumjot DSL.
 *
 * Per SPEC.md, `[$name=...]` is a preprocessor substitution: the definition
 * is stripped from the source and every `[$name]` is replaced verbatim with
 * the raw definition text before the full parse runs. Macro bodies may be
 * any text fragment (not necessarily a complete group).
 */

export type Macros = Record<string, string>;

const MAX_PASSES = 64;
const DEF_RE = /^\[\$([A-Za-z][A-Za-z0-9_]*)=/;
const REF_RE = /^\[\$([A-Za-z][A-Za-z0-9_]*)\]/;

export type PreprocessResult = {
  /** Source text with all macro defs stripped and refs substituted. */
  text: string;
  /** Map of macro name -> raw definition body (for inspection / debugging). */
  macros: Macros;
};

export function preprocessMacros(src: string): PreprocessResult {
  const macros: Macros = {};
  let text = src;
  for (let pass = 0; pass < MAX_PASSES; pass++) {
    const next = applyPass(text, macros);
    if (next === text) return { text, macros };
    text = next;
  }
  throw new Error(
    `Macro expansion did not converge after ${MAX_PASSES} passes ` +
      `(circular macro definitions?)`
  );
}

function applyPass(src: string, macros: Macros): string {
  let out = '';
  let i = 0;
  while (i < src.length) {
    if (src[i] === '[' && src[i + 1] === '$') {
      const rest = src.substring(i);
      const defMatch = rest.match(DEF_RE);
      if (defMatch) {
        const name = defMatch[1];
        const bodyStart = i + defMatch[0].length;
        const closeIdx = findMatchingClose(src, bodyStart, i);
        macros[name] = src.substring(bodyStart, closeIdx);
        i = closeIdx + 1;
        continue;
      }
      const refMatch = rest.match(REF_RE);
      if (refMatch) {
        const name = refMatch[1];
        if (!(name in macros)) {
          throw new ParseError(`Unknown macro $${name}`, src, i);
        }
        out += macros[name];
        i += refMatch[0].length;
        continue;
      }
    }
    out += src[i];
    i++;
  }
  return out;
}

/**
 * Walk forward from `start` returning the index of the `]` that closes the
 * macro opened at `openIdx`. Bracket depth is tracked so nested `[` ... `]`
 * inside a macro body do not terminate it early.
 */
function findMatchingClose(src: string, start: number, openIdx: number): number {
  let depth = 1;
  for (let i = start; i < src.length; i++) {
    if (src[i] === '[') {
      depth++;
    } else if (src[i] === ']') {
      depth--;
      if (depth === 0) return i;
    }
  }
  throw new ParseError(`Unclosed macro definition`, src, openIdx);
}
