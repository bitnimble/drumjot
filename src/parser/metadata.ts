import { Instrument, Metadata, TimeSignature, Volume } from 'src/dsl/dsl';
import {
  ALL_DRUM_INSTRUMENT_KINDS,
  DrumInstrumentKind,
  defaultKindForPitch,
} from 'src/instruments/instruments';
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
 * TimeSignature object on the Metadata type. `instrumentMapping` entries get
 * their `kind` field auto-filled from the pitch letter (or from an explicit
 * `kind:` in the DSL, if the user supplied one) so downstream consumers can
 * rely on a first-class instrument taxonomy.
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
  if (key === 'instrumentMapping' && value && typeof value === 'object') {
    return normalizeInstrumentMapping(value as Record<string, unknown>, c);
  }
  return value;
}

/**
 * Per-pitch fill of the `kind` field. If the DSL supplied `kind` explicitly,
 * validate it against the enum; otherwise look up the default for the pitch
 * letter (`k → kick`, `s → snare`, ...) and fall back to `custom`.
 */
function normalizeInstrumentMapping(
  raw: Record<string, unknown>,
  c: Cursor
): Record<string, Instrument> {
  const out: Record<string, Instrument> = {};
  for (const [pitch, entryRaw] of Object.entries(raw)) {
    if (!entryRaw || typeof entryRaw !== 'object') {
      throw new ParseError(
        `instrumentMapping['${pitch}'] must be an object`,
        c.src,
        c.pos
      );
    }
    const entry = entryRaw as Record<string, unknown>;
    let kind: DrumInstrumentKind;
    if (typeof entry.kind === 'string') {
      if (!(ALL_DRUM_INSTRUMENT_KINDS as readonly string[]).includes(entry.kind)) {
        throw new ParseError(
          `Unknown instrument kind '${entry.kind}' for pitch '${pitch}'; ` +
            `expected one of: ${ALL_DRUM_INSTRUMENT_KINDS.join(', ')}`,
          c.src,
          c.pos
        );
      }
      kind = entry.kind as DrumInstrumentKind;
    } else {
      kind = defaultKindForPitch(pitch);
    }
    const instrument: Instrument = { kind };
    if (typeof entry.name === 'string') instrument.name = entry.name;
    if (typeof entry.limb === 'string') {
      const limb = entry.limb;
      if (limb === 'lh' || limb === 'rh' || limb === 'lf' || limb === 'rf') {
        instrument.limb = limb;
      }
    }
    if (entry.midi && typeof entry.midi === 'object') {
      const midi = entry.midi as Record<string, unknown>;
      if (typeof midi.note === 'number') {
        instrument.midi = { note: midi.note };
        if (typeof midi.vol === 'string' && isVolume(midi.vol)) {
          instrument.midi.vol = midi.vol;
        }
      }
    }
    out[pitch] = instrument;
  }
  return out;
}

const VOLUMES: ReadonlySet<string> = new Set(['pp', 'p', 'mp', 'mf', 'f', 'ff']);
function isVolume(v: string): v is Volume {
  return VOLUMES.has(v);
}
