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
  barWidth = px(448);
  /** Vertical height of one rendered pitch track. */
  trackHeight = px(36);
  /** Padding above/below each voice block. */
  voicePadding = px(12);
  /** Note dot diameter. */
  noteDiameter = px(14);
  /**
   * Constant horizontal offset applied to every note from its bar's left
   * edge, so the first note sits inside the bar line rather than centred on
   * it (mimicking conventional engraved notation). It is a pure offset, not
   * a subtraction from the note area: one beat is always `barWidth / beats`
   * px wide, so the gap across a bar boundary stays exactly one note step
   * (the trailing space before the next bar line works out to
   * `oneStep - barNotePaddingLeft`, and the next bar's first note is itself
   * `barNotePaddingLeft` in, summing back to one full step).
   */
  barNotePaddingLeft = px(14);
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
  /**
   * True when the onset lands on the binary (dyadic) grid — i.e. an
   * integer multiple of 1/2^m of a quarter note (whole/half/quarter/
   * eighth/sixteenth/...). False for triplet/quintuplet/swing positions
   * and anything else that isn't a "standard straight note". The
   * renderer flags non-straight notes that aren't already covered by a
   * tuplet bracket.
   */
  straight: boolean;
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
};

/**
 * A tuplet (triplet, quintuplet, sextuplet, swing group, ...) detected
 * from a non-pattern `group` whose internal subdivision lands off the
 * binary grid. The renderer draws a bracket spanning `[startBeat,
 * endBeat)` with `count` shown above it, the conventional engraved-
 * notation indicator that the enclosed notes are not straight.
 */
export type TupletSpan = {
  /** Slot count shown in the bracket (3 = triplet, 5 = quintuplet, ...). */
  count: number;
  /** Position within the bar in beats. */
  startBeat: number;
  endBeat: number;
  /** Pixel x within the bar (aligns with the padded note area). */
  x: Pixels;
  /** Pixel width within the bar (aligns with the padded note area). */
  width: Pixels;
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
  /**
   * Tuplet brackets within this bar, in source order. Renderer draws a
   * numbered bracket over each so triplets / non-straight subdivisions
   * are visually obvious.
   */
  tupletSpans: TupletSpan[];
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
  /**
   * Empty pixel space reserved before the first bar, representing the
   * recording's lead-in (`globalMetadata.startOffset` seconds of
   * silence / non-drum intro before jot-time 0). Scaled at the same
   * pixels-per-second as the bars so the drum notation lines up with a
   * loaded audio-track waveform: audio second `startOffset` (where the
   * drums actually enter) sits exactly at bar 1's left edge. `0` when
   * there is no offset.
   */
  leadInPx: Pixels;
  /** Lead-in duration in seconds (mirrors `globalMetadata.startOffset`). */
  leadInSec: number;
  /**
   * Constant horizontal offset (px) the note grid is shifted right of
   * each bar's left edge — `viewConfig.barNotePaddingLeft`, the
   * engraving inset applied to every note/pattern/tuplet position (see
   * {@link ViewConfig.barNotePaddingLeft}). Barlines are time-anchored
   * but notes are drawn `notePadPx` inside them, so the playback
   * playhead, the audio-track waveform and click-to-seek must apply the
   * SAME offset or the score reads a constant `notePadPx` px ahead of
   * where its onsets actually sound. Surfaced here so the playback
   * layer (which only sees the `RenderedJot`, not the `ViewConfig`) can
   * stay in the note grid's coordinate frame.
   */
  notePadPx: Pixels;
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

/**
 * Onset density (note onsets per quarter-note beat) that the static
 * `ViewConfig.barWidth` was tuned for — a 4/4 bar of straight eighths
 * (8 onsets / 4 beats). A song at this density gets `densityFactor = 1`
 * and renders exactly as before; sparser songs are compressed and
 * busier ones expanded so on-screen note spacing stays roughly
 * constant regardless of time signature or how many notes a bar holds.
 */
const REFERENCE_ONSETS_PER_BEAT = 2;
/**
 * Clamp on the density factor so a near-empty score isn't a sliver and
 * a blast-beat isn't kilometres wide. The asymmetric upper bound keeps
 * dense bars readable without runaway width.
 */
const MIN_DENSITY_FACTOR = 0.4;
const MAX_DENSITY_FACTOR = 1.6;

/**
 * Count note onsets (horizontal columns) in an already-expanded element
 * list. Rests contribute nothing; a simultaneity is one column; groups
 * recurse. Used only as a density heuristic, so weights/timing are
 * irrelevant — just how many things you'd see across the bar.
 */
function countOnsets(els: Element[]): number {
  let n = 0;
  for (const el of els) {
    switch (el.kind) {
      case 'note':
        n += 1;
        break;
      case 'rest':
      case 'patternRef': // already expanded away upstream
        break;
      case 'simul':
        if (el.elements.some((c) => c.kind !== 'rest')) n += 1;
        break;
      case 'group':
        n += countOnsets(el.elements);
        break;
    }
  }
  return n;
}

/**
 * Whole-jot onset density: the maximum, across voices, of
 * onsets-per-beat. We take the max (not the mean) because the bars are
 * a shared horizontal grid — the busiest voice is the one that needs
 * the room; sparser voices just carry more whitespace, which is fine.
 * Mirrors `layoutVoice`'s running time-signature logic so the beat
 * totals line up with what actually gets laid out.
 */
function measureOnsetDensity(jot: Jot): number {
  const patterns = jot.patterns ?? {};
  const globalTime = jot.globalMetadata.time ?? DEFAULT_TIME;
  let maxRatio = 0;
  for (const voice of jot.voices) {
    let onsets = 0;
    let beats = 0;
    let activeTime = globalTime;
    if (voice.anacrusis && voice.anacrusis.length > 0) {
      const els = expandElements(voice.anacrusis, patterns);
      beats += sumWeights(els);
      onsets += countOnsets(els);
    }
    for (const b of voice.bars) {
      const barTime = b.metadata?.time ?? activeTime;
      activeTime = barTime;
      beats += barBeats(barTime);
      onsets += countOnsets(expandElements(b.elements, patterns));
    }
    if (beats > 0) maxRatio = Math.max(maxRatio, onsets / beats);
  }
  return maxRatio;
}

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
    // Derive a per-song horizontal scale from note density so a sparse
    // score (e.g. a quarter-note 2/4 tune) doesn't render as wide as a
    // 16th-note blast beat. Threaded through layout rather than stored
    // as observable state so this stays a pure function of `jot`.
    const density = measureOnsetDensity(jot);
    const densityFactor =
      density > 0
        ? Math.max(
            MIN_DENSITY_FACTOR,
            Math.min(MAX_DENSITY_FACTOR, density / REFERENCE_ONSETS_PER_BEAT)
          )
        : 1;
    const voices = jot.voices.map((v) => this.layoutVoice(v, jot, densityFactor));
    const width = px(Math.max(0, ...voices.map((v) => v.width)));
    return {
      title: jot.title,
      voices,
      width,
      globalMetadata: jot.globalMetadata,
    };
  });

  private layoutVoice = (
    voice: Voice,
    jot: Jot,
    densityFactor: number
  ): ResolvedVoice => {
    const patterns = jot.patterns ?? {};
    const globalTime = jot.globalMetadata.time ?? DEFAULT_TIME;
    const instrumentMap = jot.globalMetadata.instrumentMapping ?? {};

    // Lead-in: reserve empty space before bar 1 for the recording's
    // pre-roll (startOffset seconds before the first beat). Scaled at
    // the SAME pixels-per-second the bars use — one beat is
    // `barWidth*densityFactor/4` px wide and lasts `60/bpm` s, so
    // px/s = (barWidth*df/4)*(bpm/60) — which is exactly what
    // `buildTimeline` + the waveform mapping assume. That equivalence
    // is what makes the drum notes line up with the audio-track waveform.
    const rawOffset = jot.globalMetadata.startOffset;
    const leadInSec =
      typeof rawOffset === 'number' && rawOffset > 0 ? rawOffset : 0;
    const bpmField = jot.globalMetadata.bpm;
    const bpm = typeof bpmField === 'number' && bpmField > 0 ? bpmField : 120;
    const pxPerSecond =
      ((this.viewConfig.barWidth * densityFactor) / 4) * (bpm / 60);
    const leadInPx = px(leadInSec * pxPerSecond);

    const bars: ResolvedBar[] = [];
    const pitchOrder: string[] = [];
    let cursor = leadInPx;
    let activeTime = globalTime;

    const noteSeenForPitch = (p: string) => {
      if (!pitchOrder.includes(p)) pitchOrder.push(p);
    };

    // Anacrusis (bar 0) - treated as a single short "bar" sized to its content.
    if (voice.anacrusis && voice.anacrusis.length > 0) {
      const elements = expandElements(voice.anacrusis, patterns);
      const beats = sumWeights(elements); // anacrusis is unconstrained
      const widthPx = px(this.beatsToPx(beats, densityFactor));
      const { tracks, patternSpans, tupletSpans } = this.layoutBarContents(
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
        tupletSpans,
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
      const widthPx = px(this.beatsToPx(beats, densityFactor));
      const elements = expandElements(b.elements, patterns);
      const { tracks, patternSpans, tupletSpans } = this.layoutBarContents(
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
        tupletSpans,
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
      leadInPx,
      leadInSec,
      notePadPx: this.viewConfig.barNotePaddingLeft,
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
  ): {
    tracks: Record<string, ResolvedTrack>;
    patternSpans: PatternSpan[];
    tupletSpans: TupletSpan[];
  } {
    const flatNotes: Array<{ note: Note; beat: number; duration: number }> = [];
    const patternSpans: PatternSpan[] = [];
    const tupletSpans: TupletSpan[] = [];

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
            // Record the pattern usage span; pixel x/width are filled in
            // below once `pxPerBeat` is known.
            patternSpans.push({
              name: el.patternSource.name,
              startBeat: atBeat,
              endBeat: atBeat + span,
              x: px(0),
              width: px(0),
            });
          } else if (el.elements.length > 1 && span > 0) {
            // Tuplet detection. Subdividing this group's span among its
            // children produces interior boundaries at cumulative
            // weight fractions. If any of those fractions is not a
            // dyadic rational (k / 2^m) the group isn't a straight
            // binary subdivision — it's a triplet, quintuplet, swing
            // group, etc. The test is on the local *fraction*, not the
            // absolute beat, so a straight pair nested inside a triplet
            // doesn't itself get bracketed (the triplet's own bracket
            // covers it).
            const ws = el.elements.map(elementWeight);
            const tw = ws.reduce((a, b) => a + b, 0) || 1;
            let acc = 0;
            let nonStraight = false;
            for (let i = 0; i < ws.length - 1; i++) {
              acc += ws[i];
              if (!isDyadic(acc / tw)) {
                nonStraight = true;
                break;
              }
            }
            if (nonStraight) {
              // The bracket should run from the first slot's onset to
              // the *last* slot's onset (where the note dots are
              // centred), not to the end of the group's rhythmic
              // duration — otherwise the right leg lands a full slot
              // past the final note.
              const lastOnsetFrac = (tw - ws[ws.length - 1]) / tw;
              tupletSpans.push({
                count: el.elements.length,
                startBeat: atBeat,
                endBeat: atBeat + lastOnsetFrac * span,
                x: px(0),
                width: px(0),
              });
            }
          }
          visit(el.elements, atBeat, span);
          return;
        case 'patternRef':
          // Pattern refs should already have been expanded; ignore otherwise.
          return;
      }
    };

    visit(elements, 0, beats);

    // Map beats -> pixel x on a continuous grid: one beat is always
    // `barWidthPx / beats`, so two onsets `Δbeats` apart are the same
    // pixel distance apart everywhere — including across a bar boundary,
    // where the previous bar's trailing space plus the next bar's
    // `padLeft` offset sum to exactly one note step. `padLeft` is a pure
    // constant offset (first note inside the bar line), NOT subtracted
    // from the area, which is what previously inflated boundary gaps.
    const padLeft = this.viewConfig.barNotePaddingLeft;
    const pxPerBeat = barWidthPx / (beats || 1);

    // Fill in the pixel positions on pattern + tuplet spans now that we
    // know pxPerBeat.
    for (const span of patternSpans) {
      span.x = px(padLeft + span.startBeat * pxPerBeat);
      span.width = px((span.endBeat - span.startBeat) * pxPerBeat);
    }
    for (const span of tupletSpans) {
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
        straight: isDyadic(beat),
        duration,
        x: px(padLeft + beat * pxPerBeat),
        width: px(duration * pxPerBeat),
      });
    }

    return { tracks, patternSpans, tupletSpans };
  }

  private beatsToPx(beats: number, densityFactor: number): number {
    // Base scale: one 4/4 bar == viewConfig.barWidth. Other bar lengths
    // scale proportionally so an 8th-note keeps the same pixel width
    // across time signatures. `densityFactor` (a whole-song constant
    // derived from onset density) then compresses sparse scores and
    // expands busy ones so on-screen note spacing stays roughly
    // constant; the zoom slider still multiplies on top via
    // viewConfig.barWidth.
    return (beats / 4) * this.viewConfig.barWidth * densityFactor;
  }
}

// ---------- Straightness ----------

/**
 * True when `x` is a dyadic rational — an integer multiple of 1/2^m for
 * some small m. Used two ways:
 *
 *  - on a note's beat position within the bar: a "standard straight
 *    note" sits on the binary grid (whole/half/quarter/eighth/16th/...);
 *  - on a group's cumulative weight fractions: a straight subdivision
 *    splits at dyadic fractions, a tuplet (triplet/quintuplet/swing)
 *    does not.
 *
 * m runs up to 6 (down to 1/64-of-a-quarter), which comfortably covers
 * everything the renderer needs while still rejecting thirds, fifths,
 * sevenths, etc. The tolerance is on the scaled value so float drift
 * from weight division (e.g. 1/3 = 0.3333…) doesn't read as straight.
 */
function isDyadic(x: number): boolean {
  if (!Number.isFinite(x)) return false;
  const a = Math.abs(x);
  for (let m = 0; m <= 6; m++) {
    const scaled = a * (1 << m);
    if (Math.abs(scaled - Math.round(scaled)) < 1e-4) return true;
  }
  return false;
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
