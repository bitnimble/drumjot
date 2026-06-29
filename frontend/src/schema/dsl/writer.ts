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
 *   - Each `||` layer separator on its own line.
 *   - An anacrusis (pickup) sits on its own line above its layer's bars.
 *
 * The formatter is deliberately minimal: it emits a single canonical suffix
 * ordering (`:mods @stick ~ _weight *repeat {meta}`) rather than trying to
 * preserve whatever order the source happened to use, since the parser
 * accepts suffixes in any order and the grammar is whitespace-insensitive.
 */

import {
  BpmTransition,
  Element,
  Jot,
  Metadata,
  PatternSubstitution,
  TempoEvent,
  TimeSignature,
  Layer,
} from 'src/schema/dsl/dsl';

/** Format a whole Jot as DSL source text. */
export function writeDsl(jot: Jot): string {
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

  // The parser snapshots the active `time` onto every bar, so a bar's
  // metadata only deserves its own `{{...}}` line when it *changes* the
  // value already in effect. `time` propagates textually across the
  // whole document (including past `||`), so this state is layer-spanning.
  // `bpm` follows the same propagation rule but comes from
  // `jot.tempoEvents` (the post-parse SoT) rather than per-bar metadata.
  const active: { time?: unknown; bpm?: unknown } = {
    time: jot.globalMetadata.time,
    bpm: jot.globalMetadata.bpm,
  };

  // Tempo events live at the Jot level and feed only into layer 0's
  // output (tempo is global; layers 1+ would emit duplicates).
  const eventsByBar = new Map<number, TempoEvent[]>();
  for (const ev of jot.tempoEvents ?? []) {
    const arr = eventsByBar.get(ev.barIndex) ?? [];
    arr.push(ev);
    eventsByBar.set(ev.barIndex, arr);
  }
  for (const arr of eventsByBar.values()) arr.sort((a, b) => a.beat - b.beat);

  // Layers, separated by `||` on its own line.
  jot.layers.forEach((layer, i) => {
    if (i > 0) lines.push('||');
    lines.push(...formatLayer(layer, active, i === 0 ? eventsByBar : undefined));
  });

  return lines.join('\n') + '\n';
}

// ---------- Layers & bars ----------

function formatLayer(
  layer: Layer,
  active: { time?: unknown; bpm?: unknown },
  eventsByBar: Map<number, TempoEvent[]> | undefined
): string[] {
  const lines: string[] = [];
  if (layer.anacrusis && layer.anacrusis.length > 0) {
    // Content before the first `|` is the anacrusis; emit it unwrapped.
    lines.push(formatSequence(layer.anacrusis));
  }
  for (let i = 0; i < layer.bars.length; i++) {
    const b = layer.bars[i];
    const delta: Metadata = {};
    if (b.metadata?.time !== undefined && !sameValue(b.metadata.time, active.time)) {
      delta.time = b.metadata.time;
      active.time = b.metadata.time;
    }
    const events = eventsByBar?.get(i) ?? [];
    // Beat-0 tempo events render as `{{ bpm: X }}` on its own line,
    // before the bar; same shape as the parser's bar-aligned input.
    const downbeatEvent = events.find((e) => e.beat === 0);
    if (downbeatEvent && !sameValue(downbeatEvent.bpm, active.bpm)) {
      delta.bpm = downbeatEvent.bpm;
      active.bpm = downbeatEvent.bpm;
    }
    if (hasMetaKeys(delta)) lines.push(formatMetadata(delta, true));

    // Mid-bar tempo events (`beat > 0`) splice `{{ bpm: X }}` between
    // elements at the matching beat position. The parser re-anchors
    // these to the next element on re-parse, so a marker emitted before
    // element k re-hoists to `elementBeats[k]`.
    const midBarEvents = events.filter((e) => e.beat > 0);
    if (midBarEvents.length === 0) {
      lines.push(`| ${formatSequence(b.elements)} |`);
    } else {
      const time = (active.time ?? { count: 4, unit: 4 }) as TimeSignature;
      const beats = (time.count * 4) / time.unit;
      const elementBeats = computeElementBeats(b.elements, beats);
      const tokens: string[] = [];
      let evIdx = 0;
      const emitEvent = (bpm: number | BpmTransition) => {
        if (sameValue(bpm, active.bpm)) return;
        tokens.push(formatMetadata({ bpm }, true));
        active.bpm = bpm;
      };
      for (let k = 0; k < b.elements.length; k++) {
        const beat = elementBeats[k];
        while (evIdx < midBarEvents.length && midBarEvents[evIdx].beat <= beat) {
          emitEvent(midBarEvents[evIdx].bpm);
          evIdx++;
        }
        tokens.push(formatElement(b.elements[k]));
      }
      while (evIdx < midBarEvents.length) {
        emitEvent(midBarEvents[evIdx].bpm);
        evIdx++;
      }
      lines.push(`| ${tokens.join(' ')} |`);
    }
  }
  return lines;
}

/** Onset beats (within bar) of each top-level element, accounting for
 *  `_N` weights. Index-aligned with `els`. */
function computeElementBeats(els: Element[], totalBeats: number): number[] {
  const weights = els.map((e) => (e as { weight?: number }).weight ?? 1);
  const totalWeight = weights.reduce((a, b) => a + b, 0) || 1;
  const out: number[] = new Array(els.length);
  let cursor = 0;
  for (let i = 0; i < els.length; i++) {
    out[i] = cursor;
    cursor += (weights[i] / totalWeight) * totalBeats;
  }
  return out;
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
      let s = el.lane;
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
