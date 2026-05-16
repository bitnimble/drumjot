import {
  Bar,
  Element,
  Group,
  Jot,
  Metadata,
  Modifier,
  Note,
  Pattern,
  PatternRef,
  PatternSubstitution,
  Simultaneity,
  Sticking,
  Voice,
} from 'src/dsl';
import { Cursor } from './cursor';
import { ParseError } from './errors';
import { parseMetadata } from './metadata';
import { preprocessMacros } from './preprocess';

/**
 * Entry point: parse a Drumjot DSL source string into a Jot.
 *
 * Steps:
 *   1. Macro preprocessing (textual `[$name=...]` / `[$name]` substitution).
 *   2. Recursive-descent parse of the resulting text into a Jot.
 */
export function parse(src: string): Jot {
  const { text } = preprocessMacros(src);
  const cursor = new Cursor(text);
  return parseJot(cursor);
}

// ---------- Top-level: voices, bars, patterns, global metadata ----------

type Item =
  | { kind: 'el'; el: Element }
  | { kind: 'bar' }
  | { kind: 'voice' };

function parseJot(c: Cursor): Jot {
  let globalMetadata: Metadata = {};
  const patterns: Record<string, Pattern> = {};
  const items: Item[] = [];

  while (true) {
    c.skipWs();
    if (c.eof()) break;

    if (c.match('{{')) {
      const meta = parseMetadata(c, true);
      globalMetadata = { ...globalMetadata, ...meta };
      continue;
    }
    if (c.match('||')) {
      c.advance(2);
      items.push({ kind: 'voice' });
      continue;
    }
    if (c.peek() === '|') {
      c.advance();
      items.push({ kind: 'bar' });
      continue;
    }
    if (c.peek() === '[' && isLikelyDefinition(c)) {
      const def = parseDefinition(c);
      patterns[def.name] = {
        name: def.name,
        silent: def.silent,
        elements: def.elements,
      };
      if (!def.silent) {
        const ref: PatternRef = { kind: 'patternRef', name: def.name };
        items.push({ kind: 'el', el: applySuffixesAndSimul(c, ref) });
      }
      continue;
    }
    items.push({ kind: 'el', el: parseElement(c) });
  }

  // Slice items into voices (split on `||`) and bars (split on `|`).
  const voices: Voice[] = [];
  let voiceItems: Array<Exclude<Item, { kind: 'voice' }>> = [];
  for (const it of items) {
    if (it.kind === 'voice') {
      voices.push(buildVoice(voiceItems));
      voiceItems = [];
    } else {
      voiceItems.push(it);
    }
  }
  voices.push(buildVoice(voiceItems));

  const title = typeof globalMetadata.title === 'string' ? globalMetadata.title : '';
  const restMeta: Metadata = { ...globalMetadata };
  delete (restMeta as Record<string, unknown>).title;

  const jot: Jot = {
    title,
    globalMetadata: restMeta,
    voices,
  };
  if (Object.keys(patterns).length > 0) jot.patterns = patterns;
  return jot;
}

function buildVoice(items: Array<Exclude<Item, { kind: 'voice' }>>): Voice {
  const bars: Bar[] = [];
  let anacrusis: Element[] | undefined;
  let current: Element[] = [];
  let seenBarSep = false;

  for (const it of items) {
    if (it.kind === 'bar') {
      if (!seenBarSep) {
        if (current.length > 0) anacrusis = current;
        current = [];
        seenBarSep = true;
      } else {
        // Skip empty bars: consecutive `|`s separated only by whitespace or
        // metadata blocks act as a single separator, not an empty bar.
        if (current.length > 0) {
          bars.push({ elements: current });
        }
        current = [];
      }
    } else {
      current.push(it.el);
    }
  }
  if (current.length > 0) {
    // Either trailing content after the last '|' (a final bar) or content
    // with no '|' anywhere (treat as a single bar for usability).
    bars.push({ elements: current });
  }

  const voice: Voice = { bars };
  if (anacrusis) voice.anacrusis = anacrusis;
  return voice;
}

// ---------- Pattern definitions ----------

/**
 * Look ahead to determine whether `[` starts a pattern definition (which is
 * only legal at the top level) versus a pattern reference. We tentatively
 * scan past optional `?`, an identifier, whitespace, and check for `=`.
 */
function isLikelyDefinition(c: Cursor): boolean {
  let i = c.pos + 1; // skip '['
  while (i < c.src.length && /\s/.test(c.src[i])) i++;
  if (c.src[i] === '?') {
    i++;
    while (i < c.src.length && /\s/.test(c.src[i])) i++;
  }
  if (!/[A-Za-z]/.test(c.src[i] ?? '')) return false;
  while (i < c.src.length && /[A-Za-z0-9_]/.test(c.src[i])) i++;
  while (i < c.src.length && /\s/.test(c.src[i])) i++;
  return c.src[i] === '=';
}

function parseDefinition(c: Cursor): {
  name: string;
  silent: boolean;
  elements: Element[];
} {
  c.consume('[');
  c.skipWs();
  let silent = false;
  if (c.peek() === '?') {
    silent = true;
    c.advance();
    c.skipWs();
  }
  const name = parseIdentifier(c);
  c.skipWs();
  c.consume('=');
  const elements = parseElementSequence(c, ']');
  c.consume(']');
  return { name, silent, elements };
}

function parseIdentifier(c: Cursor): string {
  if (!/[A-Za-z]/.test(c.peek())) {
    throw new ParseError(`Expected identifier`, c.src, c.pos);
  }
  const start = c.pos;
  let s = '';
  while (/[A-Za-z0-9_]/.test(c.peek())) {
    s += c.peek();
    c.advance();
  }
  if (s.length < 2) {
    throw new ParseError(`Identifier '${s}' must be at least 2 characters`, c.src, start);
  }
  return s;
}

// ---------- Element sequences, elements, suffixes, simultaneity ----------

function parseElementSequence(c: Cursor, terminator: string): Element[] {
  const out: Element[] = [];
  while (true) {
    c.skipWs();
    if (c.eof() || c.peek() === terminator) break;
    out.push(parseElement(c));
  }
  return out;
}

function parseElement(c: Cursor): Element {
  c.skipWs();
  const primary = parsePrimary(c);
  return applySuffixesAndSimul(c, primary);
}

/** Apply postfix attachments and a possible trailing `+ rhs` simultaneity. */
function applySuffixesAndSimul(c: Cursor, primary: Element): Element {
  let el = parseSuffixes(c, primary);
  c.skipWs();
  if (c.peek() === '+') {
    c.advance();
    c.skipWs();
    const right = parseElement(c);
    el = mergeSimul(el, right);
  }
  return el;
}

function mergeSimul(a: Element, b: Element): Simultaneity {
  const aEls = a.kind === 'simul' ? a.elements : [a];
  const bEls = b.kind === 'simul' ? b.elements : [b];
  return { kind: 'simul', elements: [...aEls, ...bEls] };
}

function parsePrimary(c: Cursor): Element {
  c.skipWs();
  const ch = c.peek();
  if (/[a-z]/.test(ch)) {
    const note: Note = { kind: 'note', pitch: ch };
    c.advance();
    return note;
  }
  if (ch === '.') {
    c.advance();
    return { kind: 'rest' };
  }
  if (ch === '(') return parseGroup(c);
  if (ch === '[') return parsePatternRef(c);
  throw new ParseError(`Unexpected character '${ch || 'EOF'}'`, c.src, c.pos);
}

function parseGroup(c: Cursor): Group {
  c.consume('(');
  const elements = parseElementSequence(c, ')');
  c.consume(')');
  return { kind: 'group', elements };
}

function parsePatternRef(c: Cursor): PatternRef {
  c.consume('[');
  c.skipWs();
  const name = parseIdentifier(c);
  c.skipWs();
  const ref: PatternRef = { kind: 'patternRef', name };
  if (c.peek() === '#' || c.peek() === ',') {
    ref.substitutions = parseSubstitutions(c);
    c.skipWs();
  }
  c.consume(']');
  return ref;
}

function parseSubstitutions(c: Cursor): PatternSubstitution[] {
  const subs: PatternSubstitution[] = [];
  while (true) {
    c.skipWs();
    if (c.peek() === ',') {
      c.advance();
      c.skipWs();
    }
    if (c.peek() !== '#') break;
    subs.push(parseOneSubstitution(c));
  }
  return subs;
}

function parseOneSubstitution(c: Cursor): PatternSubstitution {
  const path: PatternSubstitution['path'] = [];
  while (c.peek() === '#') {
    c.advance();
    c.skipWs();
    const first = parseInteger(c);
    if (c.peek() === '-') {
      c.advance();
      c.skipWs();
      const last = parseInteger(c);
      path.push([first, last]);
    } else {
      path.push(first);
    }
    c.skipWs();
  }
  c.consume('=');
  c.skipWs();
  const replacement = parseElement(c);
  return { path, replacement };
}

function parseInteger(c: Cursor): number {
  c.skipWs();
  let s = '';
  while (/[0-9]/.test(c.peek())) {
    s += c.peek();
    c.advance();
  }
  if (!s) {
    throw new ParseError(`Expected integer`, c.src, c.pos);
  }
  return parseInt(s, 10);
}

// ---------- Suffixes: `:mod`, `@stick`, `_N`, `*N`, `~`, `{meta}` ----------

const MULTI_MODS = new Set<Modifier>(['fl', 'dr', 'rf']);

function parseSuffixes(c: Cursor, el: Element): Element {
  while (true) {
    c.skipWs();
    const ch = c.peek();
    if (ch === ':') {
      const mod = parseModifier(c);
      if (el.kind === 'note' || el.kind === 'group') {
        el.modifiers = (el.modifiers ?? []).concat(mod);
      } else {
        throw new ParseError(
          `Modifier ':${mod}' cannot attach to a ${el.kind}`,
          c.src,
          c.pos
        );
      }
    } else if (ch === '@') {
      const stk = parseSticking(c);
      if (el.kind === 'note') {
        el.sticking = stk;
      } else {
        throw new ParseError(
          `Sticking '@${stk}' can only attach to a note`,
          c.src,
          c.pos
        );
      }
    } else if (ch === '_') {
      c.advance();
      const n = parseInteger(c);
      (el as { weight?: number }).weight = n;
    } else if (ch === '*') {
      c.advance();
      const n = parseInteger(c);
      if (el.kind === 'simul') {
        throw new ParseError(
          `Repeat '*${n}' cannot attach to a simultaneity; wrap in (...)`,
          c.src,
          c.pos
        );
      }
      (el as { repeat?: number }).repeat = n;
    } else if (ch === '~') {
      c.advance();
      if (el.kind !== 'note' && el.kind !== 'group') {
        throw new ParseError(
          `Roll '~' can only attach to a note or group`,
          c.src,
          c.pos
        );
      }
      el.roll = true;
    } else if (ch === '{' && c.peek(1) !== '{') {
      if (el.kind !== 'note' && el.kind !== 'group') {
        throw new ParseError(
          `Metadata '{...}' can only attach to a note or group`,
          c.src,
          c.pos
        );
      }
      el.metadata = parseMetadata(c, false);
    } else {
      break;
    }
  }
  return el;
}

function parseModifier(c: Cursor): Modifier {
  c.consume(':');
  c.skipWs();
  const c1 = c.peek(0);
  const c2 = c.peek(1);
  if (c1 && c2 && /[a-z]/.test(c1) && /[a-z]/.test(c2) && MULTI_MODS.has((c1 + c2) as Modifier)) {
    c.advance(2);
    return (c1 + c2) as Modifier;
  }
  if (c1 && /[a-z]/.test(c1)) {
    c.advance();
    return c1 as Modifier;
  }
  throw new ParseError(`Expected modifier after ':'`, c.src, c.pos);
}

function parseSticking(c: Cursor): Sticking {
  c.consume('@');
  c.skipWs();
  const c1 = c.peek(0);
  const c2 = c.peek(1);
  if ((c1 === 'r' || c1 === 'l') && c2 === 'f') {
    c.advance(2);
    return (c1 + c2) as Sticking;
  }
  if (c1 === 'r' || c1 === 'l') {
    c.advance();
    return c1 as Sticking;
  }
  throw new ParseError(`Expected sticking (r, l, rf, lf) after '@'`, c.src, c.pos);
}
