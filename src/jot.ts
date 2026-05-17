import { makeAutoObservable } from 'mobx';
import { computedFn } from 'mobx-utils';
import {
  Bar,
  Element,
  Group,
  Instrument,
  Jot,
  Metadata,
  Modifier,
  Note,
  Pattern,
  PatternRef,
  PatternSubstitution,
  Rest,
  Simultaneity,
  TimeSignature,
  Voice,
} from 'src/dsl';

/** Branded pixel scalar to avoid mixing pixel and beat measurements. */
export type Pixels = number & { __pixels: never };
export const px = (n: number) => n as Pixels;

// ---------- Layout config ----------

export class ViewConfig {
  /** Pixel width of one whole bar at default zoom. */
  barWidth = px(640);
  /** Vertical height of one rendered pitch track. */
  trackHeight = px(36);
  /** Padding above/below each voice block. */
  voicePadding = px(12);
  /** Note dot diameter. */
  noteDiameter = px(14);
  /**
   * Horizontal padding inside each bar before the first note (and after the
   * last). The note area is `barWidth - barNotePaddingLeft - barNotePaddingRight`;
   * notes and pattern brackets are positioned within this inner area to mimic
   * conventional engraved notation where the first note sits inside the bar
   * line rather than directly on it.
   */
  barNotePaddingLeft = px(14);
  barNotePaddingRight = px(8);
  /** Palette used when a pitch has no explicit colour. */
  palette: string[] = [
    '#FF8C55',
    '#5BA8E8',
    '#7BC74D',
    '#C77DFF',
    '#FFD166',
    '#EF476F',
    '#06AED5',
    '#8D6E63',
  ];

  constructor() {
    makeAutoObservable(this);
  }
}

// ---------- Resolved (laid out) types ----------

export type ResolvedNote = {
  source: Note;
  pitch: string;
  /**
   * Modifiers attached to this note, narrowed to the {@link Modifier}
   * union so consumers can pass literal modifier strings to `.has()`
   * without casting and so misspelled modifiers fail at compile time.
   */
  modifiers: ReadonlySet<Modifier>;
  sticking?: Note['sticking'];
  roll: boolean;
  /** Position within bar, in beats (0 = bar start, bar.length = bar end). */
  beat: number;
  /** Onset duration, in beats. */
  duration: number;
  /** Pixel x within the bar. */
  x: Pixels;
  width: Pixels;
};

export type ResolvedTrack = {
  pitch: string;
  /** The Instrument resolved from `globalMetadata.instrumentMapping[pitch]`. */
  instrument: Instrument;
  color: string;
  notes: ResolvedNote[];
};

export type PatternSpan = {
  name: string;
  /** Position within the bar in beats. */
  startBeat: number;
  endBeat: number;
  /** Pixel x within the bar (aligns with the padded note area). */
  x: Pixels;
  /** Pixel width within the bar (aligns with the padded note area). */
  width: Pixels;
  /**
   * True for the first non-silent occurrence of the pattern across the whole
   * jot (in voice-then-bar source order). Silent patterns never have a
   * definition span - they're only rendered through their usages.
   */
  isDefinition: boolean;
};

export type ResolvedBar = {
  source: Bar;
  /** Pixel x relative to start of the voice. */
  x: Pixels;
  width: Pixels;
  time: TimeSignature;
  /** Bar length in beats (count * 4/unit, in quarter notes). */
  beats: number;
  /** Per-pitch track lanes within this bar. */
  tracks: Record<string, ResolvedTrack>;
  /**
   * Pattern usages within this bar, in source order. Renderer draws outlines
   * around each span.
   */
  patternSpans: PatternSpan[];
  /** Bar number within the voice (1-based; anacrusis is bar 0). */
  index: number;
};

export type ResolvedVoice = {
  source: Voice;
  /** All bars including anacrusis if present (anacrusis is bar 0). */
  bars: ResolvedBar[];
  /**
   * Pitches that appear at least once in this voice, ordered for rendering:
   * mapped pitches first in `globalMetadata.instrumentMapping` declaration
   * order, then any unmapped pitches in first-seen order. This is the lane
   * order in the rendered output (top to bottom).
   */
  pitches: string[];
  width: Pixels;
};

export type ResolvedJot = {
  title: string;
  voices: ResolvedVoice[];
  /** Maximum width across all voices, in pixels. */
  width: Pixels;
  globalMetadata: Metadata;
};

// ---------- Beat length helpers ----------

/** Returns the bar's length in quarter notes given a time signature. */
export function barBeats(time: TimeSignature): number {
  return (time.count * 4) / time.unit;
}

const DEFAULT_TIME: TimeSignature = { count: 4, unit: 4 };

// ---------- RenderedJot: MobX-observable layout container ----------

export class RenderedJot {
  private viewConfig: ViewConfig;

  constructor(public source: Jot, viewConfig?: ViewConfig) {
    this.viewConfig = viewConfig ?? new ViewConfig();
    makeAutoObservable(this, { source: false });
  }

  get title() {
    return this.source.title;
  }

  get globalMetadata(): Metadata {
    return this.source.globalMetadata;
  }

  get config() {
    return this.viewConfig;
  }

  get resolved(): ResolvedJot {
    return this.layoutJot(this.source);
  }

  // ----- Layout pipeline -----

  private layoutJot = computedFn((jot: Jot): ResolvedJot => {
    const voices = jot.voices.map((v) => this.layoutVoice(v, jot));

    // Mark the first non-silent occurrence of each pattern as its definition.
    // Subsequent occurrences are usages; silent patterns have no inline def
    // (their pattern body lives in `jot.patterns` only).
    const seenPattern = new Set<string>();
    for (const voice of voices) {
      for (const bar of voice.bars) {
        for (const span of bar.patternSpans) {
          if (seenPattern.has(span.name)) continue;
          seenPattern.add(span.name);
          const pattern = jot.patterns?.[span.name];
          if (pattern && !pattern.silent) span.isDefinition = true;
        }
      }
    }

    const width = px(Math.max(0, ...voices.map((v) => v.width)));
    return {
      title: jot.title,
      voices,
      width,
      globalMetadata: jot.globalMetadata,
    };
  });

  private layoutVoice = (voice: Voice, jot: Jot): ResolvedVoice => {
    const patterns = jot.patterns ?? {};
    const globalTime = jot.globalMetadata.time ?? DEFAULT_TIME;
    const instrumentMap = jot.globalMetadata.instrumentMapping ?? {};

    const bars: ResolvedBar[] = [];
    const pitchOrder: string[] = [];
    let cursor = 0 as Pixels;
    let activeTime = globalTime;

    const noteSeenForPitch = (p: string) => {
      if (!pitchOrder.includes(p)) pitchOrder.push(p);
    };

    // Anacrusis (bar 0) - treated as a single short "bar" sized to its content.
    if (voice.anacrusis && voice.anacrusis.length > 0) {
      const elements = expandElements(voice.anacrusis, patterns);
      const beats = sumWeights(elements); // anacrusis is unconstrained
      const widthPx = px(this.beatsToPx(beats, activeTime));
      const { tracks, patternSpans } = this.layoutBarContents(
        elements,
        beats,
        widthPx,
        instrumentMap,
        activeTime
      );
      Object.keys(tracks).forEach(noteSeenForPitch);
      bars.push({
        source: { elements: voice.anacrusis },
        x: cursor,
        width: widthPx,
        time: activeTime,
        beats,
        tracks,
        patternSpans,
        index: 0,
      });
      cursor = px(cursor + widthPx);
    }

    let barIndex = 1;
    for (const b of voice.bars) {
      // Per-bar metadata can override time signature.
      const barTime = b.metadata?.time ?? activeTime;
      activeTime = barTime;
      const beats = barBeats(barTime);
      const widthPx = px(this.beatsToPx(beats, barTime));
      const elements = expandElements(b.elements, patterns);
      const { tracks, patternSpans } = this.layoutBarContents(
        elements,
        beats,
        widthPx,
        instrumentMap,
        barTime
      );
      Object.keys(tracks).forEach(noteSeenForPitch);
      bars.push({
        source: b,
        x: cursor,
        width: widthPx,
        time: barTime,
        beats,
        tracks,
        patternSpans,
        index: barIndex++,
      });
      cursor = px(cursor + widthPx);
    }

    // Voice lane order. The author's instrumentMapping (when provided) is
    // authoritative: every mapped pitch that actually appears in the voice
    // comes first, in declaration order. Any pitches without an entry fall
    // back to the order they were first seen in the source. Authoring
    // `instrumentMapping: { h, s, k }` renders hi-hat on top, snare in the
    // middle, kick on the bottom.
    const orderedPitches: string[] = [];
    const seen = new Set(pitchOrder);
    for (const mapped of Object.keys(instrumentMap)) {
      if (seen.has(mapped) && !orderedPitches.includes(mapped)) {
        orderedPitches.push(mapped);
      }
    }
    for (const p of pitchOrder) {
      if (!orderedPitches.includes(p)) orderedPitches.push(p);
    }

    // Now that the voice-level lane order is known, assign a stable color
    // to every track of a given pitch. Without this pass, colors would
    // flicker per bar because `layoutBarContents` doesn't see lane order.
    this.assignTrackColors(bars, orderedPitches);

    return {
      source: voice,
      bars,
      pitches: orderedPitches,
      width: cursor,
    };
  };

  /**
   * Walk every bar's tracks and assign each pitch a color from the palette
   * based on its index in the voice-level lane order. The result is that
   * the same pitch is always the same colour for the whole voice, no
   * matter which bar fires it first.
   */
  private assignTrackColors(bars: ResolvedBar[], orderedPitches: string[]): void {
    const palette = this.viewConfig.palette;
    if (palette.length === 0) return;
    for (const bar of bars) {
      for (const pitch of Object.keys(bar.tracks)) {
        const idx = orderedPitches.indexOf(pitch);
        const slot = idx >= 0 ? idx : 0;
        bar.tracks[pitch].color = palette[slot % palette.length];
      }
    }
  }

  /**
   * Lay out a bar's elements (post pattern-expansion) into per-pitch tracks
   * with absolute pixel positions within the bar.
   *
   * Note that the `color` of each emitted track is intentionally a
   * placeholder here. We don't know the final voice-level lane order until
   * after every bar has been laid out, so `layoutVoice` does a second pass
   * to assign stable colors per pitch (see `assignTrackColors`). If we
   * picked palette indices here, the colors would flicker between bars
   * based on which pitch happened to fire first in each bar.
   */
  private layoutBarContents(
    elements: Element[],
    beats: number,
    barWidthPx: Pixels,
    instrumentMap: Record<string, Instrument>,
    _time: TimeSignature
  ): { tracks: Record<string, ResolvedTrack>; patternSpans: PatternSpan[] } {
    const flatNotes: Array<{ note: Note; beat: number; duration: number }> = [];
    const patternSpans: PatternSpan[] = [];

    // Walk the element tree and place every note onto the timeline in beats.
    const visit = (els: Element[], startBeat: number, totalBeats: number) => {
      const weights = els.map(elementWeight);
      const totalWeight = weights.reduce((a, b) => a + b, 0) || 1;
      let cursor = startBeat;
      for (let i = 0; i < els.length; i++) {
        const el = els[i];
        const span = (weights[i] / totalWeight) * totalBeats;
        placeElement(el, cursor, span);
        cursor += span;
      }
    };

    const placeElement = (el: Element, atBeat: number, span: number) => {
      switch (el.kind) {
        case 'note':
          flatNotes.push({ note: el, beat: atBeat, duration: span });
          return;
        case 'rest':
          return;
        case 'simul':
          // All children share the same onset; they get the full span.
          for (const child of el.elements) {
            placeElement(child, atBeat, span);
          }
          return;
        case 'group':
          if (el.patternSource) {
            // Record the pattern usage span; `isDefinition` is assigned in a
            // second pass once we know the global source order, and the pixel
            // x/width are filled in below once `pxPerBeat` is known.
            patternSpans.push({
              name: el.patternSource.name,
              startBeat: atBeat,
              endBeat: atBeat + span,
              x: px(0),
              width: px(0),
              isDefinition: false,
            });
          }
          visit(el.elements, atBeat, span);
          return;
        case 'patternRef':
          // Pattern refs should already have been expanded; ignore otherwise.
          return;
      }
    };

    visit(elements, 0, beats);

    // Map beats -> pixel x using the padded inner note area so the first
    // note sits inside the bar line instead of being centered on it.
    const padLeft = this.viewConfig.barNotePaddingLeft;
    const padRight = this.viewConfig.barNotePaddingRight;
    const noteArea = Math.max(1, barWidthPx - padLeft - padRight);
    const pxPerBeat = noteArea / (beats || 1);

    // Fill in the pixel positions on pattern spans now that we know pxPerBeat.
    for (const span of patternSpans) {
      span.x = px(padLeft + span.startBeat * pxPerBeat);
      span.width = px((span.endBeat - span.startBeat) * pxPerBeat);
    }

    // Group flat notes by pitch into tracks. Colors are placeholders here;
    // `assignTrackColors` rewrites them once the voice-level lane order is
    // known so the same pitch keeps a stable color across all bars.
    const tracks: Record<string, ResolvedTrack> = {};

    for (const { note, beat, duration } of flatNotes) {
      const pitch = note.pitch;
      let track = tracks[pitch];
      if (!track) {
        const instrument = instrumentMap[pitch] ?? {};
        track = {
          pitch,
          instrument,
          color: '',
          notes: [],
        };
        tracks[pitch] = track;
      }
      track.notes.push({
        source: note,
        pitch,
        modifiers: new Set<Modifier>(note.modifiers ?? []),
        sticking: note.sticking,
        roll: !!note.roll,
        beat,
        duration,
        x: px(padLeft + beat * pxPerBeat),
        width: px(duration * pxPerBeat),
      });
    }

    return { tracks, patternSpans };
  }

  private beatsToPx(beats: number, _time: TimeSignature): number {
    // Default scale: one 4/4 bar == viewConfig.barWidth. Other bar lengths
    // scale proportionally so an 8th-note keeps the same pixel width across
    // time signatures.
    return (beats / 4) * this.viewConfig.barWidth;
  }
}

// ---------- Element weights ----------

/**
 * Effective layout-weight of an element after expansion (one slot per element).
 * Note: expansion unrolls repeats into sibling copies, so by the time the
 * layout reads weights every element's `repeat` is implicitly 1.
 */
export function elementWeight(el: Element): number {
  switch (el.kind) {
    case 'note':
    case 'rest':
    case 'simul':
    case 'group':
    case 'patternRef':
      return el.weight ?? 1;
  }
}

function sumWeights(els: Element[]): number {
  return els.reduce((a, e) => a + elementWeight(e), 0);
}

// ---------- Pattern expansion ----------

/**
 * Expand pattern references and unroll `*N` repeats at every nesting level.
 * After expansion the element tree contains only notes, rests, simultaneities
 * and groups; pattern refs are inlined and no element has `repeat > 1`.
 */
export function expandElements(
  els: Element[],
  patterns: Record<string, Pattern>
): Element[] {
  const out: Element[] = [];
  for (const el of els) {
    for (const e of expandElement(el, patterns)) out.push(e);
  }
  return out;
}

function expandElement(el: Element, patterns: Record<string, Pattern>): Element[] {
  switch (el.kind) {
    case 'note':
    case 'rest':
      return unroll(el);
    case 'simul':
      return [{ ...el, elements: expandElements(el.elements, patterns) }];
    case 'group':
      return unroll({ ...el, elements: expandElements(el.elements, patterns) });
    case 'patternRef': {
      const pattern = patterns[el.name];
      // Tag the expanded group with `patternSource` so the renderer can draw
      // an outline + label for every usage (including unrolled `*N` copies).
      if (!pattern) {
        return unroll({
          kind: 'group',
          elements: [],
          weight: el.weight,
          repeat: el.repeat,
          patternSource: { name: el.name },
        });
      }
      let elements = expandElements(pattern.elements, patterns);
      if (el.substitutions) {
        elements = applySubstitutions(elements, el.substitutions);
      }
      return unroll({
        kind: 'group',
        elements,
        weight: el.weight,
        repeat: el.repeat,
        patternSource: { name: el.name },
      });
    }
  }
}

function unroll<T extends Element & { repeat?: number }>(el: T): Element[] {
  const repeat = el.repeat ?? 1;
  if (repeat <= 1) {
    const { repeat: _r, ...rest } = el;
    return [rest as Element];
  }
  const copies: Element[] = [];
  for (let i = 0; i < repeat; i++) {
    const { repeat: _r, ...rest } = el;
    copies.push(rest as Element);
  }
  return copies;
}

function applySubstitutions(
  elements: Element[],
  subs: PatternSubstitution[]
): Element[] {
  let result = elements;
  for (const sub of subs) {
    result = applySubstitution(result, sub.path, sub.replacement);
  }
  return result;
}

function applySubstitution(
  elements: Element[],
  path: Array<number | [number, number]>,
  replacement: Element
): Element[] {
  if (path.length === 0) return elements;
  const [head, ...rest] = path;
  const copy = elements.slice();

  if (rest.length === 0) {
    if (typeof head === 'number') {
      const idx = head - 1;
      if (idx >= 0 && idx < copy.length) copy[idx] = replacement;
    } else {
      const [start, end] = head;
      const s = Math.max(0, start - 1);
      const e = Math.min(copy.length, end);
      copy.splice(s, e - s, replacement);
    }
    return copy;
  }

  // Descend into a group at position `head`.
  const idx = (typeof head === 'number' ? head : head[0]) - 1;
  const target = copy[idx];
  if (target && target.kind === 'group') {
    copy[idx] = { ...target, elements: applySubstitution(target.elements, rest, replacement) };
  }
  return copy;
}

// ---------- Misc helpers ----------

/** Convenience type guards. */
export const isNote = (el: Element): el is Note => el.kind === 'note';
export const isRest = (el: Element): el is Rest => el.kind === 'rest';
export const isGroup = (el: Element): el is Group => el.kind === 'group';
export const isSimul = (el: Element): el is Simultaneity => el.kind === 'simul';
export const isPatternRef = (el: Element): el is PatternRef => el.kind === 'patternRef';
