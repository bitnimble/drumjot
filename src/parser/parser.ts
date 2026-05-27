import {
  Bar,
  BarTempoSource,
  BpmTransition,
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
import { hoistTempoEvents } from './hoist_tempo';
import { parseMetadata } from './metadata';
import { preprocessMacros } from './preprocess';

/**
 * Entry point: parse a Drumjot DSL source string into a Jot.
 *
 * Steps:
 *   1. Macro preprocessing (textual `[$name=...]` / `[$name]` substitution).
 *   2. Recursive-descent parse of the resulting text into a Jot.
 *   3. Hoist every `bpm` declaration (global, bar-opening, mid-bar, group,
 *      note) into `jot.tempoEvents`; strip `bpm` from element/bar
 *      metadata. After this step `jot.tempoEvents` is the single
 *      runtime source of truth for tempo and no `metadata.bpm` survives
 *      anywhere except `jot.globalMetadata.bpm` (the initial tempo).
 */
export function parse(src: string): Jot {
  const { text } = preprocessMacros(src);
  const cursor = new Cursor(text);
  const jot = parseJot(cursor);
  hoistTempoEvents(jot);
  return jot;
}

// ---------- Top-level: voices, bars, patterns, global metadata ----------

/**
 * Subset of `Metadata` that propagates from inline `{{...}}` blocks down to
 * each bar's `bar.metadata`. The DSL spec says these "remain in effect until
 * the next override", so a bar that's parsed after such a block carries
 * the effective values for layout and renderers to use directly.
 *
 * Other metadata keys (mapping, comment, custom user keys) stay only on
 * `jot.globalMetadata` - per-bar duplication adds no value for them and
 * would bloat the AST.
 */
type BarMeta = Pick<Metadata, 'time' | 'bpm'>;

type Item =
  | { kind: 'el'; el: Element }
  | { kind: 'bar'; activeMeta: BarMeta; pos: number }
  | { kind: 'voice' }
  /**
   * A mid-track `{{bpm: X}}`. `buildVoice` attaches each marker to the
   * next element pushed into the current bar (its index in
   * `bar.elements`). Markers seen before the first `|` (anacrusis
   * section) are dropped; the global / bar-opening tempo path already
   * carries those via `globalMetadata.bpm` and the `barActive` snapshot.
   */
  | { kind: 'tempoMarker'; bpm: number | BpmTransition };

function parseJot(c: Cursor): Jot {
  let globalMetadata: Metadata = {};
  // Running snapshot of values that propagate to the bar level. Updated by
  // every inline `{{...}}` block we encounter; the snapshot at each `|`
  // becomes the opening bar's `bar.metadata` so the AST is the source of
  // truth for per-bar tempo/time information.
  let barActive: BarMeta = {};
  const patterns: Record<string, Pattern> = {};
  const items: Item[] = [];

  while (true) {
    c.skipWs();
    if (c.eof()) break;

    if (c.match('{{')) {
      const meta = parseMetadata(c, true);
      globalMetadata = { ...globalMetadata, ...meta };
      if (meta.time !== undefined) barActive = { ...barActive, time: meta.time };
      if (meta.bpm !== undefined) {
        barActive = { ...barActive, bpm: meta.bpm };
        // Emit a marker so a `{{bpm}}` between elements in the same bar
        // anchors at the next element's onset. Markers before the first
        // `|` are silently dropped by `buildVoice`.
        items.push({ kind: 'tempoMarker', bpm: meta.bpm });
      }
      continue;
    }
    if (c.match('||')) {
      c.advance(2);
      items.push({ kind: 'voice' });
      continue;
    }
    if (c.peek() === '|') {
      const pos = c.pos;
      c.advance();
      items.push({ kind: 'bar', activeMeta: { ...barActive }, pos });
      continue;
    }
    if (c.peek() === '[' && isLikelyDefinition(c)) {
      const def = parseDefinition(c);
      patterns[def.name] = {
        name: def.name,
        elements: def.elements,
      };
      continue;
    }
    items.push({ kind: 'el', el: parseElement(c) });
  }

  // Slice items into voices (split on `||`) and bars (split on `|`).
  const srcLength = c.src.length;
  const voices: Voice[] = [];
  let voiceItems: Array<Exclude<Item, { kind: 'voice' }>> = [];
  for (const it of items) {
    if (it.kind === 'voice') {
      voices.push(buildVoice(voiceItems, srcLength));
      voiceItems = [];
    } else {
      voiceItems.push(it);
    }
  }
  voices.push(buildVoice(voiceItems, srcLength));

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

function buildVoice(
  items: Array<Exclude<Item, { kind: 'voice' }>>,
  srcLength: number
): Voice {
  const bars: Bar[] = [];
  let anacrusis: Element[] | undefined;
  let current: Element[] = [];
  let seenBarSep = false;
  // The metadata that was active when the *current* bar opened. Each `|`
  // closes the prior bar (using `barOpeningMeta`) and starts a new one
  // (whose meta is the active snapshot recorded on this `|` item).
  let barOpeningMeta: BarMeta = {};
  // Source position of the `|` that opened the current bar. `null` until
  // the first `|` is seen; the linter degrades gracefully when this is
  // absent (e.g. inline jots with no bar separator at all).
  let barOpeningPos: number | null = null;
  // Source range for the voice as a whole; set lazily when we see the
  // first item, extended as we go.
  let voiceStart: number | null = null;
  let voiceEnd = 0;
  // Mid-bar `{{bpm}}` markers issued since the last element was pushed,
  // waiting for the next element to anchor against. They survive `|`
  // boundaries (a marker between bars attaches to the first element of
  // the next bar at beat 0); they're dropped at the start of the voice
  // (before the first `|`) since the anacrusis isn't part of the
  // bar-indexed tempo timeline.
  let pendingTempoMarkers: Array<number | BpmTransition> = [];
  // BarTempoSources accumulated for the bar currently being assembled
  // in `current`. Flushed onto the bar at `commit`, then reset.
  let barTempoSources: BarTempoSource[] = [];

  const commit = (
    els: Element[],
    meta: BarMeta,
    start: number | null,
    end: number,
    sources: BarTempoSource[]
  ) => {
    if (els.length === 0) return;
    const bar: Bar = { elements: els };
    if (hasAnyMeta(meta)) bar.metadata = meta as Metadata;
    if (start !== null) bar.range = { start, end };
    if (sources.length > 0) bar.tempoSources = sources;
    bars.push(bar);
  };

  for (const it of items) {
    if (it.kind === 'bar') {
      if (voiceStart === null) voiceStart = it.pos;
      voiceEnd = Math.max(voiceEnd, it.pos);
      if (!seenBarSep) {
        if (current.length > 0) anacrusis = current;
        current = [];
        // Tempo markers before the first `|` are dropped; they live in
        // the anacrusis section, which the bar-indexed tempo timeline
        // doesn't represent. `globalMetadata.bpm` + the `barActive`
        // snapshot still carry their effect to bar 0 via the closing `|`.
        pendingTempoMarkers = [];
        barTempoSources = [];
        seenBarSep = true;
      } else {
        commit(current, barOpeningMeta, barOpeningPos, it.pos, barTempoSources);
        current = [];
        barTempoSources = [];
      }
      barOpeningMeta = it.activeMeta;
      barOpeningPos = it.pos;
    } else if (it.kind === 'tempoMarker') {
      // Markers before the first `|` are dropped (anacrusis section).
      if (seenBarSep) pendingTempoMarkers.push(it.bpm);
    } else {
      const range = elementRange(it.el);
      if (range) {
        if (voiceStart === null) voiceStart = range.start;
        voiceEnd = Math.max(voiceEnd, range.end);
      }
      if (seenBarSep && pendingTempoMarkers.length > 0) {
        const elementIndex = current.length;
        for (const bpm of pendingTempoMarkers) {
          barTempoSources.push({ elementIndex, bpm });
        }
        pendingTempoMarkers = [];
      }
      current.push(it.el);
    }
  }
  if (current.length > 0) {
    // Either trailing content after the last '|' (a final bar) or content
    // with no '|' anywhere (treat as a single bar for usability).
    commit(current, barOpeningMeta, barOpeningPos, srcLength, barTempoSources);
  }

  const voice: Voice = { bars };
  if (anacrusis) voice.anacrusis = anacrusis;
  if (voiceStart !== null) {
    voice.range = { start: voiceStart, end: Math.max(voiceEnd, srcLength) };
  }
  return voice;
}

function elementRange(el: Element): { start: number; end: number } | undefined {
  if (el.kind === 'note' || el.kind === 'group') return el.range;
  return undefined;
}

function hasAnyMeta(m: BarMeta): boolean {
  return m.time !== undefined || m.bpm !== undefined;
}

// ---------- Pattern definitions ----------

/**
 * Look ahead to determine whether `[` starts a pattern definition (which is
 * only legal at the top level) versus a pattern reference. We tentatively
 * scan past an identifier and whitespace and check for `=`.
 */
function isLikelyDefinition(c: Cursor): boolean {
  let i = c.pos + 1; // skip '['
  while (i < c.src.length && /\s/.test(c.src[i])) i++;
  if (!/[A-Za-z]/.test(c.src[i] ?? '')) return false;
  while (i < c.src.length && /[A-Za-z0-9_]/.test(c.src[i])) i++;
  while (i < c.src.length && /\s/.test(c.src[i])) i++;
  return c.src[i] === '=';
}

function parseDefinition(c: Cursor): {
  name: string;
  elements: Element[];
} {
  c.consume('[');
  c.skipWs();
  const name = parseIdentifier(c);
  c.skipWs();
  c.consume('=');
  const elements = parseElementSequence(c, ']');
  c.consume(']');
  return { name, elements };
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
  const start = c.pos;
  const primary = parsePrimary(c);
  return applySuffixesAndSimul(c, primary, start);
}

/**
 * Apply postfix attachments and a possible trailing `+ rhs` simultaneity.
 * When `start` is supplied and the post-suffix element is a Note/Group,
 * attach its source range BEFORE any simultaneity merge — otherwise the
 * left operand of `k + s + h` ends up wrapped inside a Simultaneity and
 * never gets its own range (lint rules that anchor on the first-stacked
 * note then report "(no position)").
 */
function applySuffixesAndSimul(c: Cursor, primary: Element, start?: number): Element {
  let el = parseSuffixes(c, primary);
  if (start !== undefined && (el.kind === 'note' || el.kind === 'group')) {
    el.range = { start, end: c.pos };
  }
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
