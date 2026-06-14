/**
 * Tempo-hoist post-parse pass.
 *
 * Walks the parsed jot once, harvests every bpm declaration (global,
 * bar-opening, mid-bar marker, group, note), and emits a single canonical
 * `jot.tempoEvents` list. Every `bpm` field is stripped from per-element
 * and per-bar metadata so the runtime has only one place to look.
 * `globalMetadata.bpm` is preserved as the song's initial tempo (the
 * value in force before any tempoEvent fires); it's set to the bpm
 * effective at (bar 0, beat 0) when the parser stamped a different
 * value on every bar.
 *
 * No-op tempo declarations (same value as the currently-effective tempo)
 * are dropped, which keeps the event list compact in the common case
 * where the parser stamps every bar's opening with the same active bpm.
 */
import {
  Bar,
  BpmTransition,
  Element,
  Group,
  Jot,
  Pattern,
  TempoEvent,
} from 'src/dsl/dsl';
import { resolveBpm, DEFAULT_BPM } from 'src/tempo/tempo';

type BpmField = number | BpmTransition;

/**
 * In-place: populate `jot.tempoEvents` from every bpm source on the AST
 * and strip the originating fields. Called once by the parser at the
 * end of `parse()`.
 */
export function hoistTempoEvents(jot: Jot): void {
  const events: TempoEvent[] = [];
  let currentBpm = resolveBpm(jot.globalMetadata.bpm, DEFAULT_BPM);
  let initialBpmCaptured = false;

  // The initial tempo: the very first bpm value encountered (in voice 0,
  // bar 0). If no bar carries an opening bpm, the existing
  // `globalMetadata.bpm` (else default) stands.
  const setInitial = (bpm: BpmField) => {
    if (initialBpmCaptured) return;
    const resolved = resolveBpm(bpm, currentBpm);
    jot.globalMetadata.bpm = bpm;
    currentBpm = resolved;
    initialBpmCaptured = true;
  };

  const considerChange = (barIndex: number, beat: number, bpm: BpmField) => {
    const resolved = resolveBpm(bpm, currentBpm);
    if (resolved === currentBpm) return; // no-op
    events.push({ barIndex, beat, bpm });
    currentBpm = resolved;
  };

  // We canonicalize tempo against voice 0. Voices 1+ share the same bar
  // grid (the parser's `barActive` propagates across `||`), and any
  // genuinely per-voice element-level bpm is uncommon and would already
  // be ignored by today's downstream MIDI path. Strip bpm from those
  // voices defensively so the post-hoist invariant ("no bpm on element
  // metadata") holds everywhere.
  const voice0 = jot.voices[0];
  if (voice0) {
    const bars = voice0.bars;

    // Pre-pass: lift bar 0's opening bpm into globalMetadata.bpm as the
    // initial tempo. The parser stamps every bar with its barActive
    // snapshot, so bar 0's metadata.bpm (if set) is the first bpm in
    // force on the bar timeline.
    if (bars.length > 0 && bars[0].metadata?.bpm !== undefined) {
      setInitial(bars[0].metadata.bpm);
    }

    for (let i = 0; i < bars.length; i++) {
      const bar = bars[i];
      processBarTempo(bar, i, considerChange, jot.patterns ?? {});
    }
  }

  for (let v = 1; v < jot.voices.length; v++) {
    for (const bar of jot.voices[v].bars) {
      stripBarTempo(bar, jot.patterns ?? {});
    }
  }

  // Patterns may contain bpm-bearing notes/groups. Each pattern usage
  // emits its own tempo changes via the expansion above; the pattern
  // body itself should not retain the bpm fields after hoist (a
  // follow-up edit shouldn't re-introduce them).
  for (const pat of Object.values(jot.patterns ?? {})) {
    stripBpmFromElements(pat.elements);
  }

  if (events.length > 0) jot.tempoEvents = events;
}

function processBarTempo(
  bar: Bar,
  barIndex: number,
  considerChange: (barIndex: number, beat: number, bpm: BpmField) => void,
  patterns: Record<string, Pattern>,
): void {
  // Bar opening (beat 0): bar.metadata.bpm.
  if (bar.metadata?.bpm !== undefined) {
    considerChange(barIndex, 0, bar.metadata.bpm);
  }

  // Mid-bar markers (`{{bpm}}` between elements) carry their anchor
  // element's index; convert to beat-within-bar by walking weights.
  // Element-level `{bpm}` on notes/groups: same conversion, anchored at
  // that element's onset within the bar (expansion-aware).
  const beats = computeBarBeats(bar);
  if (beats > 0) {
    // Build a beat-indexed list of bpm anchors, then emit in beat order.
    type Anchor = { beat: number; bpm: BpmField; order: number };
    const anchors: Anchor[] = [];

    // Expansion preserves source-order so a `BarTempoSource` pointing at
    // `bar.elements[k]` lines up with the same logical position in the
    // expansion if `bar.elements[k]` itself is a note/rest/simul/group
    // (no patternRef can sit at that index unless the user authored one
    //; and we still hit the expansion through the index lookup below,
    // resolved against the original elements with weight aggregation).
    const elementBeats = computeElementBeats(bar.elements, beats, patterns);

    if (bar.tempoSources) {
      for (let i = 0; i < bar.tempoSources.length; i++) {
        const src = bar.tempoSources[i];
        const beat = elementBeats[src.elementIndex] ?? 0;
        anchors.push({ beat, bpm: src.bpm, order: i });
      }
    }

    // Walk expanded elements to find note/group-level bpm metadata.
    // Anchors here use a larger `order` so a tempoSource at the same
    // beat is processed first (the marker arrived earlier in source).
    let orderCounter = (bar.tempoSources?.length ?? 0);
    visitExpanded(bar.elements, 0, beats, patterns, (beat, bpmField) => {
      anchors.push({ beat, bpm: bpmField, order: orderCounter++ });
    });

    anchors.sort((a, b) => {
      if (a.beat !== b.beat) return a.beat - b.beat;
      return a.order - b.order;
    });

    for (const a of anchors) {
      considerChange(barIndex, a.beat, a.bpm);
    }
  }

  stripBarTempo(bar, patterns);
}

function stripBarTempo(bar: Bar, patterns: Record<string, Pattern>): void {
  if (bar.metadata) {
    delete (bar.metadata as { bpm?: unknown }).bpm;
    if (Object.keys(bar.metadata).length === 0) delete bar.metadata;
  }
  delete bar.tempoSources;
  stripBpmFromElements(bar.elements);
  // Defensive: pattern bodies are stripped separately in the top-level
  // pass, but if a hand-built bar happens to carry a patternRef whose
  // pattern was already stripped, this is a no-op.
  void patterns;
}

function stripBpmFromElements(els: Element[]): void {
  for (const el of els) {
    if (el.kind === 'note' || el.kind === 'group') {
      if (el.metadata) {
        delete (el.metadata as { bpm?: unknown }).bpm;
        if (Object.keys(el.metadata).length === 0) delete el.metadata;
      }
    }
    if (el.kind === 'group') stripBpmFromElements(el.elements);
    if (el.kind === 'simul') stripBpmFromElements(el.elements);
  }
}

/**
 * Bar length in quarter notes derived from the bar's effective time
 * signature. Used standalone (without the full RenderedJot pipeline) so
 * the hoist can compute beat positions without a circular dep on jot.ts.
 */
function computeBarBeats(bar: Bar): number {
  const time = bar.metadata?.time;
  // The parser propagates `time` onto every bar's metadata snapshot,
  // same as `bpm`, so a bar without a time field inherits the
  // global / preceding active value (which the parser already
  // captured). Default to 4/4 when entirely absent.
  if (time) return (time.count * 4) / time.unit;
  return 4;
}

/**
 * Onset beats (within bar) of each top-level element in `els`,
 * accounting for `_N` weights. Index-aligned with `els`. A `patternRef`
 * is treated as a single slot for the purposes of source-index lookup;
 * the expansion below visits its body recursively for note-level bpm.
 */
function computeElementBeats(
  els: Element[],
  totalBeats: number,
  _patterns: Record<string, Pattern>,
): number[] {
  const weights = els.map(elementSlotWeight);
  const totalWeight = weights.reduce((a, b) => a + b, 0) || 1;
  const out: number[] = new Array(els.length);
  let cursor = 0;
  for (let i = 0; i < els.length; i++) {
    out[i] = cursor;
    cursor += (weights[i] / totalWeight) * totalBeats;
  }
  return out;
}

function elementSlotWeight(el: Element): number {
  return (el as { weight?: number }).weight ?? 1;
}

/**
 * Walk an element tree distributed across `[startBeat, startBeat+span)`
 * and emit (beat, bpm) anchors for any note/group carrying
 * `metadata.bpm`. Pattern references expand into their body; the body
 * is laid out in the patternRef's slot's beat range. Repeats (`*N`)
 * unroll into sibling copies (their bpm fires on each iteration).
 */
function visitExpanded(
  els: Element[],
  startBeat: number,
  totalBeats: number,
  patterns: Record<string, Pattern>,
  emit: (beat: number, bpm: BpmField) => void,
): void {
  const expanded = expandForVisit(els, patterns);
  const weights = expanded.map(elementSlotWeight);
  const totalWeight = weights.reduce((a, b) => a + b, 0) || 1;
  let cursor = startBeat;
  for (let i = 0; i < expanded.length; i++) {
    const el = expanded[i];
    const span = (weights[i] / totalWeight) * totalBeats;
    visitElement(el, cursor, span, patterns, emit);
    cursor += span;
  }
}

function visitElement(
  el: Element,
  beat: number,
  span: number,
  patterns: Record<string, Pattern>,
  emit: (beat: number, bpm: BpmField) => void,
): void {
  if (el.kind === 'note' || el.kind === 'group') {
    const bpmField = (el.metadata as { bpm?: BpmField } | undefined)?.bpm;
    if (bpmField !== undefined) emit(beat, bpmField);
  }
  if (el.kind === 'group') {
    visitExpanded(el.elements, beat, span, patterns, emit);
  } else if (el.kind === 'simul') {
    for (const child of el.elements) visitElement(child, beat, span, patterns, emit);
  }
}

/**
 * Pattern expansion + repeat unroll, mirroring `jot.ts::expandElements`
 * but inlined here so this module stays free of layout deps.
 */
function expandForVisit(els: Element[], patterns: Record<string, Pattern>): Element[] {
  const out: Element[] = [];
  for (const el of els) {
    if (el.kind === 'patternRef') {
      const pat = patterns[el.name];
      const body = pat ? expandForVisit(pat.elements, patterns) : [];
      const wrapper: Group = {
        kind: 'group',
        elements: body,
        weight: el.weight,
      };
      const repeat = el.repeat ?? 1;
      for (let i = 0; i < Math.max(1, repeat); i++) out.push(wrapper);
    } else {
      const repeat = (el as { repeat?: number }).repeat ?? 1;
      const cleaned: Element =
        el.kind === 'group'
          ? { ...el, elements: expandForVisit(el.elements, patterns) }
          : el.kind === 'simul'
            ? { ...el, elements: expandForVisit(el.elements, patterns) }
            : el;
      for (let i = 0; i < Math.max(1, repeat); i++) out.push(cleaned);
    }
  }
  return out;
}

