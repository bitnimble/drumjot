import { Metadata, TimeSignature } from 'src/dsl';
import { Cursor } from './cursor';
import { ParseError } from './errors';

/**
 * Parse a metadata block. JSON-ish syntax with unquoted identifier keys,
 * double-quoted strings, numbers, nested objects and arrays. Bare
 * identifiers as values are returned as their string form (e.g. `vol: ff`
 * yields `"ff"`), matching how DSL examples express enum-like values.
 */
export function parseMetadata(c: Cursor, isGlobal: boolean): Metadata {
  if (isGlobal) {
    c.consume('{{');
  } else {
    c.consume('{');
  }
  const out: Record<string, unknown> = {};
  c.skipWs();
  while (true) {
    c.skipWs();
    if (c.peek() === '}') break;
    if (c.eof()) {
      throw new ParseError(`Unexpected EOF in metadata block`, c.src, c.pos);
    }
    const key = parseMetaIdentifier(c);
    c.skipWs();
    c.consume(':');
    c.skipWs();
    const raw = parseMetaValue(c);
    out[key] = normalizeMetadataValue(key, raw, c);
    c.skipWs();
    if (c.peek() === ',') {
      c.advance();
      c.skipWs();
    } else if (c.peek() !== '}') {
      throw new ParseError(`Expected ',' or '}' in metadata`, c.src, c.pos);
    }
  }
  if (isGlobal) {
    c.consume('}}');
  } else {
    c.consume('}');
  }
  return out as Metadata;
}

function parseMetaIdentifier(c: Cursor): string {
  let s = '';
  while (c.pos < c.src.length && /[A-Za-z0-9_]/.test(c.peek())) {
    s += c.peek();
    c.advance();
  }
  if (!s) {
    throw new ParseError(`Expected identifier in metadata`, c.src, c.pos);
  }
  return s;
}

function parseMetaValue(c: Cursor): unknown {
  c.skipWs();
  const ch = c.peek();
  if (ch === '"') return parseString(c);
  if (ch === '{') return parseObject(c);
  if (ch === '[') return parseArray(c);
  if (ch === '-' || /[0-9]/.test(ch)) return parseNumber(c);
  if (/[A-Za-z_]/.test(ch)) return parseMetaIdentifier(c);
  throw new ParseError(`Unexpected character '${ch || 'EOF'}' in metadata value`, c.src, c.pos);
}

function parseString(c: Cursor): string {
  c.consume('"');
  let s = '';
  while (!c.eof() && c.peek() !== '"') {
    if (c.peek() === '\\') {
      c.advance();
      const esc = c.peek();
      const map: Record<string, string> = {
        n: '\n',
        t: '\t',
        r: '\r',
        '"': '"',
        '\\': '\\',
        '/': '/',
      };
      s += map[esc] ?? esc;
      c.advance();
    } else {
      s += c.peek();
      c.advance();
    }
  }
  c.consume('"');
  return s;
}

function parseNumber(c: Cursor): number {
  let s = '';
  if (c.peek() === '-') {
    s += '-';
    c.advance();
  }
  while (/[0-9.]/.test(c.peek())) {
    s += c.peek();
    c.advance();
  }
  const n = Number(s);
  if (!Number.isFinite(n)) {
    throw new ParseError(`Invalid number '${s}'`, c.src, c.pos);
  }
  return n;
}

function parseObject(c: Cursor): Record<string, unknown> {
  c.consume('{');
  c.skipWs();
  const out: Record<string, unknown> = {};
  while (true) {
    c.skipWs();
    if (c.peek() === '}') break;
    if (c.eof()) {
      throw new ParseError(`Unexpected EOF in object`, c.src, c.pos);
    }
    const key = parseMetaIdentifier(c);
    c.skipWs();
    c.consume(':');
    c.skipWs();
    out[key] = parseMetaValue(c);
    c.skipWs();
    if (c.peek() === ',') {
      c.advance();
      c.skipWs();
    } else if (c.peek() !== '}') {
      throw new ParseError(`Expected ',' or '}'`, c.src, c.pos);
    }
  }
  c.consume('}');
  return out;
}

function parseArray(c: Cursor): unknown[] {
  c.consume('[');
  c.skipWs();
  const out: unknown[] = [];
  while (true) {
    c.skipWs();
    if (c.peek() === ']') break;
    out.push(parseMetaValue(c));
    c.skipWs();
    if (c.peek() === ',') {
      c.advance();
      c.skipWs();
    } else if (c.peek() !== ']') {
      throw new ParseError(`Expected ',' or ']'`, c.src, c.pos);
    }
  }
  c.consume(']');
  return out;
}

/**
 * Coerce well-known top-level metadata keys into their typed forms. `time`
 * arrives as a string ("4/4") in the DSL but is exposed as a structured
 * TimeSignature object on the Metadata type.
 */
function normalizeMetadataValue(key: string, value: unknown, c: Cursor): unknown {
  if (key === 'time' && typeof value === 'string') {
    const m = /^(\d+)\s*\/\s*(\d+)$/.exec(value);
    if (!m) {
      throw new ParseError(`Invalid time signature '${value}'`, c.src, c.pos);
    }
    const ts: TimeSignature = { count: Number(m[1]), unit: Number(m[2]) };
    return ts;
  }
  return value;
}
