import { makeAutoObservable } from 'mobx';
import { computedFn } from 'mobx-utils';
import {
  Bar,
  Element,
  Group,
  Jot,
  Metadata,
  Note,
  NoteMapping,
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
  modifiers: ReadonlySet<string>;
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
  mapping: NoteMapping;
  color: string;
  notes: ResolvedNote[];
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
  /** Bar number within the voice (1-based; anacrusis is bar 0). */
  index: number;
};

export type ResolvedVoice = {
  source: Voice;
  /** All bars including anacrusis if present (anacrusis is bar 0). */
  bars: ResolvedBar[];
  /** Pitches that appear at least once in this voice, in first-seen order. */
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
    const globalMapping = jot.globalMetadata.mapping ?? {};

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
      const tracks = this.layoutBarContents(elements, beats, widthPx, globalMapping, activeTime);
      Object.keys(tracks).forEach(noteSeenForPitch);
      bars.push({
        source: { elements: voice.anacrusis },
        x: cursor,
        width: widthPx,
        time: activeTime,
        beats,
        tracks,
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
      const tracks = this.layoutBarContents(elements, beats, widthPx, globalMapping, barTime);
      Object.keys(tracks).forEach(noteSeenForPitch);
      bars.push({
        source: b,
        x: cursor,
        width: widthPx,
        time: barTime,
        beats,
        tracks,
        index: barIndex++,
      });
      cursor = px(cursor + widthPx);
    }

    return {
      source: voice,
      bars,
      pitches: pitchOrder,
      width: cursor,
    };
  };

  /**
   * Lay out a bar's elements (post pattern-expansion) into per-pitch tracks
   * with absolute pixel positions within the bar.
   */
  private layoutBarContents(
    elements: Element[],
    beats: number,
    barWidthPx: Pixels,
    globalMapping: Record<string, NoteMapping>,
    _time: TimeSignature
  ): Record<string, ResolvedTrack> {
    const flatNotes: Array<{ note: Note; beat: number; duration: number }> = [];

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
          visit(el.elements, atBeat, span);
          return;
        case 'patternRef':
          // Pattern refs should already have been expanded; ignore otherwise.
          return;
      }
    };

    visit(elements, 0, beats);

    // Group flat notes by pitch into tracks.
    const tracks: Record<string, ResolvedTrack> = {};
    const pxPerBeat = barWidthPx / (beats || 1);

    let paletteIndex = 0;
    const nextColor = () =>
      this.viewConfig.palette[paletteIndex++ % this.viewConfig.palette.length];

    for (const { note, beat, duration } of flatNotes) {
      const pitch = note.pitch;
      let track = tracks[pitch];
      if (!track) {
        const mapping = globalMapping[pitch] ?? {};
        track = {
          pitch,
          mapping,
          color: nextColor(),
          notes: [],
        };
        tracks[pitch] = track;
      }
      track.notes.push({
        source: note,
        pitch,
        modifiers: new Set(note.modifiers ?? []),
        sticking: note.sticking,
        roll: !!note.roll,
        beat,
        duration,
        x: px(beat * pxPerBeat),
        width: px(duration * pxPerBeat),
      });
    }

    return tracks;
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
      if (!pattern) {
        return unroll({
          kind: 'group',
          elements: [],
          weight: el.weight,
          repeat: el.repeat,
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
