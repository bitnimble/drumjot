/**
 * Jot formatter — the inverse of `src/parser`.
 *
 * Takes an in-memory {@link Jot} and renders it back to DSL source text,
 * formatted for human readability. The output is intended to re-parse to a
 * structurally identical Jot (modulo source `range`s, which the parser
 * synthesises and a formatted document re-derives on the next parse).
 *
 * Readability rules:
 *   - Global metadata block on its own line.
 *   - Each pattern definition on its own line.
 *   - Each bar on its own line, wrapped in `| ... |`.
 *   - Each `||` voice separator on its own line.
 *   - An anacrusis (pickup) sits on its own line above its voice's bars.
 *
 * The formatter is deliberately minimal: it emits a single canonical suffix
 * ordering (`:mods @stick ~ _weight *repeat {meta}`) rather than trying to
 * preserve whatever order the source happened to use, since the parser
 * accepts suffixes in any order and the grammar is whitespace-insensitive.
 */

import {
  Element,
  Jot,
  Metadata,
  PatternSubstitution,
  TimeSignature,
  Voice,
} from 'src/dsl';

/** Format a whole Jot as DSL source text. */
export function formatJot(jot: Jot): string {
  const lines: string[] = [];

  // Global metadata. The parser lifts `title` out into `jot.title`, so fold
  // it back in here as the leading key for a faithful round-trip.
  const meta: Metadata = { ...jot.globalMetadata };
  if (jot.title) {
    const withTitle: Metadata = { title: jot.title };
    Object.assign(withTitle, meta);
    if (hasMetaKeys(withTitle)) lines.push(formatMetadata(withTitle, true));
  } else if (hasMetaKeys(meta)) {
    lines.push(formatMetadata(meta, true));
  }

  // Pattern definitions, one per line.
  for (const pattern of Object.values(jot.patterns ?? {})) {
    lines.push(`[${pattern.name}=${formatSequence(pattern.elements)}]`);
  }

  // The parser snapshots the active `time`/`bpm` onto every bar, so a bar's
  // metadata only deserves its own `{{...}}` line when it *changes* the
  // value already in effect. `time`/`bpm` propagate textually across the
  // whole document (including past `||`), so this state is voice-spanning.
  const active: { time?: unknown; bpm?: unknown } = {
    time: jot.globalMetadata.time,
    bpm: jot.globalMetadata.bpm,
  };

  // Voices, separated by `||` on its own line.
  jot.voices.forEach((voice, i) => {
    if (i > 0) lines.push('||');
    lines.push(...formatVoice(voice, active));
  });

  return lines.join('\n') + '\n';
}

// ---------- Voices & bars ----------

function formatVoice(
  voice: Voice,
  active: { time?: unknown; bpm?: unknown }
): string[] {
  const lines: string[] = [];
  if (voice.anacrusis && voice.anacrusis.length > 0) {
    // Content before the first `|` is the anacrusis; emit it unwrapped.
    lines.push(formatSequence(voice.anacrusis));
  }
  for (const b of voice.bars) {
    const delta: Metadata = {};
    if (b.metadata?.time !== undefined && !sameValue(b.metadata.time, active.time)) {
      delta.time = b.metadata.time;
      active.time = b.metadata.time;
    }
    if (b.metadata?.bpm !== undefined && !sameValue(b.metadata.bpm, active.bpm)) {
      delta.bpm = b.metadata.bpm;
      active.bpm = b.metadata.bpm;
    }
    if (hasMetaKeys(delta)) lines.push(formatMetadata(delta, true));
    lines.push(`| ${formatSequence(b.elements)} |`);
  }
  return lines;
}

/** Structural equality, enough for `time`/`bpm` (scalars or small objects). */
function sameValue(a: unknown, b: unknown): boolean {
  return a === b || JSON.stringify(a) === JSON.stringify(b);
}

// ---------- Elements ----------

function formatSequence(els: Element[]): string {
  return els.map(formatElement).join(' ');
}

function formatElement(el: Element): string {
  switch (el.kind) {
    case 'note': {
      let s = el.pitch;
      for (const m of el.modifiers ?? []) s += `:${m}`;
      if (el.sticking) s += `@${el.sticking}`;
      if (el.roll) s += '~';
      s += weightRepeat(el);
      if (el.metadata && hasMetaKeys(el.metadata)) {
        s += formatMetadata(el.metadata, false);
      }
      return s;
    }
    case 'rest':
      return '.' + weightRepeat(el);
    case 'simul': {
      const joined = el.elements.map(formatElement).join('+');
      // A `+` chain can't itself carry a weight in surface syntax, so wrap
      // it in a group when one is present (semantically equivalent).
      return el.weight !== undefined && el.weight !== 1
        ? `(${joined})_${el.weight}`
        : joined;
    }
    case 'group': {
      let s = `(${formatSequence(el.elements)})`;
      for (const m of el.modifiers ?? []) s += `:${m}`;
      if (el.roll) s += '~';
      s += weightRepeat(el);
      if (el.metadata && hasMetaKeys(el.metadata)) {
        s += formatMetadata(el.metadata, false);
      }
      return s;
    }
    case 'patternRef': {
      const subs = (el.substitutions ?? [])
        .map(formatSubstitution)
        .join(', ');
      let s = `[${el.name}${subs}]`;
      s += weightRepeat(el);
      return s;
    }
  }
}

/** Shared `_weight` / `*repeat` suffix, omitting the no-op defaults of 1. */
function weightRepeat(el: { weight?: number; repeat?: number }): string {
  let s = '';
  if (el.weight !== undefined && el.weight !== 1) s += `_${el.weight}`;
  if (el.repeat !== undefined && el.repeat !== 1) s += `*${el.repeat}`;
  return s;
}

function formatSubstitution(sub: PatternSubstitution): string {
  const path = sub.path
    .map((p) => (typeof p === 'number' ? `#${p}` : `#${p[0]}-${p[1]}`))
    .join('');
  return `${path}=${formatElement(sub.replacement)}`;
}

// ---------- Metadata ----------

function hasMetaKeys(m: Metadata): boolean {
  return Object.keys(m).length > 0;
}

/** Render a metadata block: `{{ ... }}` when global, `{ ... }` otherwise. */
function formatMetadata(m: Metadata, global: boolean): string {
  const open = global ? '{{ ' : '{ ';
  const close = global ? ' }}' : ' }';
  return open + formatObjectBody(m) + close;
}

function formatObjectBody(obj: Record<string, unknown>): string {
  return Object.entries(obj)
    .map(([k, v]) => `${k}: ${formatValue(k, v)}`)
    .join(', ');
}

/**
 * Serialise one metadata value. Mirrors the parser's JSON-ish grammar:
 * unquoted identifier keys, double-quoted strings, bare numbers/booleans.
 * The `key` is needed because `time` is exposed as a structured
 * {@link TimeSignature} but written as a `"count/unit"` string in source.
 */
function formatValue(key: string, v: unknown): string {
  if (key === 'time' && isTimeSignature(v)) {
    return `"${v.count}/${v.unit}"`;
  }
  if (v === null || v === undefined) return '""';
  if (typeof v === 'string') return quote(v);
  if (typeof v === 'number' || typeof v === 'boolean') return String(v);
  if (Array.isArray(v)) {
    return `[${v.map((x) => formatValue('', x)).join(', ')}]`;
  }
  if (typeof v === 'object') {
    return `{ ${formatObjectBody(v as Record<string, unknown>)} }`;
  }
  return quote(String(v));
}

function isTimeSignature(v: unknown): v is TimeSignature {
  return (
    typeof v === 'object' &&
    v !== null &&
    typeof (v as TimeSignature).count === 'number' &&
    typeof (v as TimeSignature).unit === 'number'
  );
}

function quote(s: string): string {
  const escaped = s
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"')
    .replace(/\n/g, '\\n')
    .replace(/\t/g, '\\t')
    .replace(/\r/g, '\\r');
  return `"${escaped}"`;
}
