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
import { BarTempos, buildBarTempos, initialBpm } from 'src/tempo';
import { buildTimeline, JotTimeline, pickDominantBpmAndTime } from 'src/playback/timeline';
import { PICKER_PALETTE } from 'src/tracks';

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
   * Horizontal offset applied to every note from its bar's left edge,
   * expressed as a fraction of a quarter-note beat (so it scales with
   * zoom). Acts as `P/2`, where `P` is the gap across a barline between
   * the last subbeat of bar N and the first subbeat of bar N+1, set to
   * exactly the inter-subbeat distance at the finest grid (48ths,
   * `pxPerBeat/12`), so a 48th sitting on the last slot of bar N lands
   * at `barline - P/2` and the next bar's downbeat at `barline + P/2`.
   * The barline becomes a true separator rather than something the
   * last 48th overflows past, which both fixes the visual (the last
   * subbeat used to look like it belonged to the next bar) and the hit
   * region (an overflowing note got covered by bar N+1 and clicks
   * landed on the seek handler instead of the note).
   *
   * Default `1/24`: half of `1/12` (the 48ths inter-note distance in
   * beats) so `padLeft = pxPerBeat/24 = P/2`.
   */
  barNotePaddingBeats = 1 / 24;
  /** Palette used when a pitch has no explicit colour. Mirrors the
   *  shared {@link PICKER_PALETTE} so a colour the user picks from the
   *  picker's first swatch row matches a lane that landed on the same
   *  palette slot. */
  palette: string[] = [...PICKER_PALETTE];

  constructor() {
    makeAutoObservable(this);
  }
}

// ---------- Resolved (laid out) types ----------

// Each `Resolved*` type is the corresponding zoom-invariant `Structural*`
// type (defined below) plus the pixel fields materialised by the layout
// pass. Extending — rather than re-declaring — keeps the beat-only base
// properties (source, pitch, beat, …) defined in exactly one place; the
// pixel fields (x / width / notePadPx) live only on the resolved side.

export type ResolvedNote = StructuralNote & {
  /** Pixel x within the bar. */
  x: Pixels;
  width: Pixels;
};

export type ResolvedTrack = Omit<StructuralTrack, 'notes'> & {
  notes: ResolvedNote[];
};

export type PatternSpan = StructuralPatternSpan & {
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
export type TupletSpan = StructuralTupletSpan & {
  /** Pixel x within the bar (aligns with the padded note area). */
  x: Pixels;
  /** Pixel width within the bar (aligns with the padded note area). */
  width: Pixels;
};

export type ResolvedBar = Omit<StructuralBar, 'tracks' | 'patternSpans' | 'tupletSpans'> & {
  /** Pixel x relative to start of the voice. */
  x: Pixels;
  width: Pixels;
  /** Per-pitch track lanes within this bar. */
  tracks: Record<string, ResolvedTrack>;
  patternSpans: PatternSpan[];
  tupletSpans: TupletSpan[];
};

export type ResolvedVoice = Omit<StructuralVoice, 'bars'> & {
  bars: ResolvedBar[];
  width: Pixels;
  /**
   * Horizontal offset (px) the note grid is shifted right of each
   * bar's left edge: the engraving inset applied to every
   * note/pattern/tuplet position (see {@link
   * ViewConfig.barNotePaddingBeats}). Scales with zoom because the
   * source config is in beats; the value here is materialised in pixels
   * at the current `pxPerBeat`. Barlines are time-anchored but notes
   * are drawn `notePadPx` inside them, so the playback playhead, the
   * audio-track waveform and click-to-seek must apply the SAME offset
   * or the score reads a constant `notePadPx` px ahead of where its
   * onsets actually sound. Surfaced here so the playback layer (which
   * only sees the `RenderedJot`, not the `ViewConfig`) can stay in the
   * note grid's coordinate frame.
   */
  notePadPx: Pixels;
};

export type ResolvedJot = Omit<JotStructure, 'voices' | 'densityFactor'> & {
  voices: ResolvedVoice[];
  /** Maximum width across all voices, in pixels. */
  width: Pixels;
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
 * Pitches that appear anywhere inside an expanded element tree. Used to
 * tag a {@link PatternSpan} with the set of pitches its body actually
 * plays, so the mixer can draw the bracket only on rows whose pitch is
 * inside it. Pattern bodies are fully expanded before the structural
 * pass (see {@link expandElement}), so a `patternRef` here would be a
 * bug — silently skip it.
 */
function collectPitches(els: Element[]): Set<string> {
  const out = new Set<string>();
  const walk = (es: Element[]) => {
    for (const e of es) {
      switch (e.kind) {
        case 'note':
          out.add(e.pitch);
          break;
        case 'simul':
        case 'group':
          walk(e.elements);
          break;
        case 'rest':
        case 'patternRef':
          break;
      }
    }
  };
  walk(els);
  return out;
}

/**
 * Walk every patternSpan across every voice/bar in source order and
 * assign each unique pattern name the next color slot. Same name → same
 * index everywhere, including across voices. The renderer wraps the
 * index modulo the pattern palette, so a jot with more unique patterns
 * than colors quietly recycles instead of failing.
 */
function assignPatternColorIndices(voices: StructuralVoice[]): void {
  const indexByName = new Map<string, number>();
  for (const voice of voices) {
    for (const bar of voice.bars) {
      for (const span of bar.patternSpans) {
        let idx = indexByName.get(span.name);
        if (idx === undefined) {
          idx = indexByName.size;
          indexByName.set(span.name, idx);
        }
        span.colorIndex = idx;
      }
    }
  }
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

// ---------- Structural cache (beat-only, zoom-invariant) ----------
//
// These mirror the public `Resolved*` shapes but drop every field that
// requires a pixel multiplier (x, width, leadInPx, notePadPx). They're
// what `structureForJot` produces — a single heavy walk per source jot,
// cached for the lifetime of that jot reference. The downstream pixel
// pass turns these into the final `ResolvedJot` by multiplying beats
// into pixels, which is the cheap part of the pipeline. Splitting it
// this way means zoom (which only changes `viewConfig.barWidth`) skips
// the structural walk entirely.

export type StructuralNote = {
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
};

export type StructuralTrack = {
  pitch: string;
  /** The Instrument resolved from `globalMetadata.instrumentMapping[pitch]`. */
  instrument: Instrument;
  color: string;
  notes: StructuralNote[];
};

export type StructuralPatternSpan = {
  name: string;
  /** Position within the bar in beats. */
  startBeat: number;
  endBeat: number;
  /**
   * Pitches the pattern actually plays (collected from the expanded
   * pattern body). The mixer uses this to draw the bracket only on rows
   * whose pitch participates — intermediate rows that don't, the bracket
   * skips through.
   */
  pitches: ReadonlySet<string>;
  /**
   * Stable 0-based color slot for this pattern name, assigned in
   * first-seen order across the jot. The renderer wraps it modulo the
   * pattern palette length, so every usage of the same pattern shows
   * the same color.
   */
  colorIndex: number;
};

export type StructuralTupletSpan = {
  /** Slot count shown in the bracket (3 = triplet, 5 = quintuplet, ...). */
  count: number;
  /** Position within the bar in beats. */
  startBeat: number;
  endBeat: number;
};

export type StructuralBar = {
  source: Bar;
  time: TimeSignature;
  /** Bar length in beats (count * 4/unit, in quarter notes). */
  beats: number;
  /** Per-pitch track lanes within this bar. */
  tracks: Record<string, StructuralTrack>;
  /**
   * Pattern usages within this bar, in source order. Renderer draws outlines
   * around each span.
   */
  patternSpans: StructuralPatternSpan[];
  /**
   * Tuplet brackets within this bar, in source order. Renderer draws a
   * numbered bracket over each so triplets / non-straight subdivisions
   * are visually obvious.
   */
  tupletSpans: StructuralTupletSpan[];
  /** Bar number within the voice (1-based; anacrusis is bar 0). */
  index: number;
};

export type StructuralVoice = {
  source: Voice;
  /** All bars including anacrusis if present (anacrusis is bar 0). */
  bars: StructuralBar[];
  /**
   * Pitches that appear at least once in this voice, ordered for rendering:
   * mapped pitches first in `globalMetadata.instrumentMapping` declaration
   * order, then any unmapped pitches in first-seen order. This is the lane
   * order in the rendered output (top to bottom).
   */
  pitches: string[];
};

export type JotStructure = {
  title: string;
  voices: StructuralVoice[];
  globalMetadata: Metadata;
  /**
   * Whole-song horizontal scale derived from onset density. Same
   * meaning as the local `densityFactor` the old layout threaded
   * through every call — promoted onto the structural cache so the
   * pixel pass can read it without re-measuring.
   */
  densityFactor: number;
};

// ---------- RenderedJot: MobX-observable layout container ----------

export class RenderedJot {
  private viewConfig: ViewConfig;

  constructor(public source: Jot, viewConfig?: ViewConfig) {
    this.viewConfig = viewConfig ?? new ViewConfig();
    makeAutoObservable(this, { source: false });
  }

  /**
   * Palette colour assigned to a pitch for this jot's voice ordering,
   * before any per-instrument override is layered on top. Used as the
   * fallback by {@link InstrumentTrack}'s colour getter so a track with
   * no user override picks up whatever the jot's palette would have
   * painted. Returns undefined when the pitch doesn't appear in any
   * voice (rare; caller falls back to a neutral grey).
   *
   * Reads the structural cache, not the resolved layout, so a wheel-
   * driven zoom tick doesn't trigger a palette re-lookup.
   */
  defaultPaletteColorFor(pitch: string): string | undefined {
    const palette = this.viewConfig.palette;
    if (palette.length === 0) return undefined;
    for (const voice of this.structure.voices) {
      const idx = voice.pitches.indexOf(pitch);
      if (idx >= 0) return palette[idx % palette.length];
    }
    return undefined;
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

  /**
   * Beat-grid offset (in quarter-note beats) applied to every drum note
   * before rendering / playback. Corrects a consistent beat-detection
   * error from transcription — e.g. a groove that landed 1.5 beats late
   * in every bar — by sliding all notes across the (fixed) bar grid until
   * they line up with the beat. `0` = exactly as transcribed. Observable,
   * so changing it reflows the score and (via the player) reschedules
   * playback live. See {@link applyDrumOffset}.
   *
   * Deliberately independent of the player's `drumsT0Sec` (audio offset).
   * The two sliders fix different problems and shouldn't auto-couple:
   * - `drumOffsetBeats` shifts every note by Δ beats in jot time, so the
   *   scheduled synth fires at `media = (jot + Δs) + drumsT0Sec`. The
   *   recorded audio's media positions don't move. If the user is
   *   correcting a transcription error (notation Δ beats off the actual
   *   audio drums), the synth-vs-audio alignment now matches exactly.
   * - `drumsT0Sec` shifts the audio relative to the notation (visually
   *   and audibly). Used when the audio file's load offset is off.
   *
   * Auto-shifting `drumsT0Sec` whenever `drumOffsetBeats` changed would
   * defeat the transcription-correction use case — the synth would move
   * with the notes, but so would the audio, preserving the very mismatch
   * the user is trying to remove. Both knobs can be non-zero at once;
   * keeping them orthogonal is the feature.
   */
  drumOffsetBeats = 0;

  /**
   * Zero point for the beat-grid offset control. The shift actually
   * applied to the score is `drumOffsetBeats - drumOffsetBeatsBaseline`,
   * so when both fields hold the same value the rendering matches the
   * raw source. Hydrated when loading a transcriber debug bundle: the
   * downbeat detector's `beat_alignment_offset_sec` becomes the baseline
   * (and the initial control value) so the user sees the alignment the
   * pipeline applied while the notes stay where the MIDI placed them.
   * Resetting the control to 0 then exposes the pre-alignment positions.
   */
  drumOffsetBeatsBaseline = 0;

  setDrumOffset(beats: number) {
    this.drumOffsetBeats = Number.isFinite(beats) ? beats : 0;
  }

  setDrumOffsetBaseline(beats: number) {
    this.drumOffsetBeatsBaseline = Number.isFinite(beats) ? beats : 0;
  }

  /** Net beat shift applied to the source — the user-visible control
   * value minus its baseline (see {@link drumOffsetBeatsBaseline}). */
  get effectiveDrumOffsetBeats(): number {
    return this.drumOffsetBeats - this.drumOffsetBeatsBaseline;
  }

  get resolved(): ResolvedJot {
    const base = this.layoutJot(this.source);
    const effective = this.effectiveDrumOffsetBeats;
    return effective === 0 ? base : applyDrumOffset(base, effective);
  }

  /**
   * Stable beat-coord layout used by React components. Identity is
   * preserved across zoom changes: when only `viewConfig.barWidth`
   * mutates, every sub-array and sub-object here keeps the same
   * reference, so observer components don't re-render their subtree.
   * Reads `drumOffsetBeats` so a beat-offset adjustment still reflows
   * the rendering.
   */
  get structure(): JotStructure {
    const base = this.structureForJot(this.source);
    const effective = this.effectiveDrumOffsetBeats;
    return effective === 0 ? base : applyDrumOffsetStructure(base, effective);
  }

  /**
   * Quarter-note-beat to pixel multiplier currently in effect. The
   * single number that drives every position in the rendered score —
   * exposed as a `--px-per-beat` CSS custom property by the renderer
   * so bars/notes/brackets can compute their layout from `var(--*)`
   * + `calc()` instead of re-rendering on every zoom tick.
   */
  get pxPerBeat(): number {
    return ((this.viewConfig.barWidth as number) * this.structure.densityFactor) / 4;
  }

  // ----- Derived structural views (MobX-cached) -----
  //
  // These getters expose structural data that consumers used to compute
  // ad-hoc in render. They all read from `this.structure` and/or
  // `this.source` (both stable per-RenderedJot), so MobX caches them on
  // the observable dependency graph and they only recompute when the
  // underlying structure changes; not on every zoom tick or React
  // render. See `makeAutoObservable(this, { source: false })` in the
  // constructor; `get x()` becomes a `computed` automatically.

  /**
   * Playback timeline for this jot; per-bar `startSec` / `durationSec`
   * keyed off `tempoEvents`. Delegated to {@link buildTimeline}, which
   * only reads `this.structure` and `this.source`, so the cached value
   * survives zoom changes (viewConfig.barWidth doesn't participate).
   */
  get timeline(): JotTimeline {
    return buildTimeline(this);
  }

  /** Total quarter-note beats across the primary voice (bar 0 + drum bars
   * + lead-in). Returns 0 when the jot has no voice. */
  get voiceBeats(): number {
    const bars = this.structure.voices[0]?.bars;
    if (!bars) return 0;
    let total = 0;
    for (const b of bars) total += b.beats;
    return total;
  }

  /**
   * First (canonical) voice of the structural cache. All voices in a Jot
   * share the same bar grid; consumers that need bars / pitches read
   * through this rather than reaching into `structure.voices[0]`
   * directly.
   */
  get primaryStructuralVoice(): StructuralVoice | undefined {
    return this.structure.voices[0];
  }

  /** Per-bar tempo layout (segments + durations) for the primary voice.
   * Delegates to {@link buildBarTempos}; returns `[]` when no voice. */
  get barTempos(): readonly BarTempos[] {
    const bars = this.structure.voices[0]?.bars;
    if (!bars) return [];
    return buildBarTempos(this.source, bars);
  }

  /** Dominant bpm and time signature across the song, excluding the
   * pre-drum lead-in. Delegates to {@link pickDominantBpmAndTime}. */
  get dominantBpmAndTime(): {
    dominantBpm: number | undefined;
    dominantTime: TimeSignature | undefined;
  } {
    return pickDominantBpmAndTime(this);
  }

  /**
   * Aggregated legend entries; unique `[pitch, { color, name }]` pairs
   * walked across every voice/bar in first-seen order. The Legend
   * component used to do this walk on every render; reading it here
   * lifts the walk onto MobX's computed graph so observer components
   * share the cache. Frozen so consumers can't mutate the cached array.
   */
  get legendPitches(): ReadonlyArray<readonly [string, { color: string; name?: string }]> {
    const seen = new Map<string, { color: string; name?: string }>();
    for (const voice of this.structure.voices) {
      for (const bar of voice.bars) {
        for (const pitch of Object.keys(bar.tracks)) {
          if (!seen.has(pitch)) {
            const track = bar.tracks[pitch];
            seen.set(pitch, { color: track.color, name: track.instrument.name });
          }
        }
      }
    }
    return Object.freeze(Array.from(seen.entries()));
  }

  /**
   * Ordered union of pitches across all voices in their structural
   * (declaration) order; first-seen wins. NOT mixer-sorted; the store's
   * `collectJotPitches` applies the mixer sort separately. Used to drive
   * `trackOrder` sync.
   */
  get jotPitches(): readonly string[] {
    const out: string[] = [];
    for (const voice of this.structure.voices) {
      for (const p of voice.pitches) {
        if (!out.includes(p)) out.push(p);
      }
    }
    return out;
  }

  // ----- Layout pipeline -----
  //
  // Two passes, deliberately separated:
  //
  //   structureForJot(jot)  → JotStructure   // heavy walk, computedFn(jot)
  //   layoutJot(jot)        → ResolvedJot    // cheap pixel multiply
  //
  // `structureForJot` does every zoom-invariant operation: pattern
  // expansion, the element-tree walk that places notes onto beat coords,
  // tuplet/pattern-span detection, pitch ordering, color assignment, and
  // the onset-density measurement that drives `densityFactor`. It's
  // memoized by `jot` identity via `computedFn`, so a wheel-tick that
  // only mutates `viewConfig.barWidth` returns the cached structure
  // without redoing any of that work.
  //
  // `layoutJot` takes the cached structure and multiplies beats into
  // pixels — the only step that actually depends on the zoom slider.
  // The output is the same `ResolvedJot` shape consumers (timeline.ts,
  // audio_tracks.ts, react components) have always seen, so this is
  // internally invisible.

  private layoutJot(jot: Jot): ResolvedJot {
    const structure = this.structureForJot(jot);
    const barWidth = this.viewConfig.barWidth as number;
    const pxPerBeat = (barWidth * structure.densityFactor) / 4;
    const padLeft = px(this.viewConfig.barNotePaddingBeats * pxPerBeat);
    const voices = structure.voices.map((sv) =>
      this.pixelVoice(sv, pxPerBeat, padLeft)
    );
    const width = px(Math.max(0, ...voices.map((v) => v.width as number)));
    return {
      title: structure.title,
      voices,
      width,
      globalMetadata: structure.globalMetadata,
    };
  }

  /**
   * Heavy, zoom-invariant pass. Memoized on `jot` identity by
   * `computedFn`, so every wheel tick that only changes `barWidth`
   * skips it. Only reads `viewConfig.palette` (for color assignment);
   * the palette is never reassigned in practice, so the cache stays
   * warm across user interactions.
   */
  private structureForJot = computedFn((jot: Jot): JotStructure => {
    const density = measureOnsetDensity(jot);
    const densityFactor =
      density > 0
        ? Math.max(
            MIN_DENSITY_FACTOR,
            Math.min(MAX_DENSITY_FACTOR, density / REFERENCE_ONSETS_PER_BEAT)
          )
        : 1;
    const voices = jot.voices.map((v) => this.structureForVoice(v, jot));
    assignPatternColorIndices(voices);
    return {
      title: jot.title,
      voices,
      globalMetadata: jot.globalMetadata,
      densityFactor,
    };
  });

  /**
   * Per-pitch derived data needed by the mixer's InstrumentRow: the
   * primary voice's bars, the voice-wide and lead-in beat totals,
   * cumulative bar-start offsets used to position absolute overlays,
   * and the row's display colour / instrument name. Memoised on `pitch` via
   * `computedFn` so each row reads its slice from the MobX cache instead
   * of recomputing every render. Reads only `this.structure`, so the
   * cache survives zoom changes.
   */
  barsForPitch = computedFn((pitch: string): {
    bars: readonly StructuralBar[];
    voiceBeats: number;
    leadInBarsBeats: number;
    barBeatStart: readonly number[];
    startBeats: readonly number[];
    pitchColor: string;
    instrumentName: string | undefined;
  } => {
    const voice = this.structure.voices[0];
    const bars = voice?.bars ?? [];
    let voiceBeats = 0;
    let leadInBarsBeats = 0;
    let countedLeadIn = true;
    const barBeatStart: number[] = new Array(bars.length);
    let cursor = 0;
    for (let i = 0; i < bars.length; i++) {
      barBeatStart[i] = cursor;
      const b = bars[i];
      cursor += b.beats;
      voiceBeats += b.beats;
      if (countedLeadIn) {
        if (b.index < 0) leadInBarsBeats += b.beats;
        else countedLeadIn = false;
      }
    }
    let pitchColor = '';
    let instrumentName: string | undefined;
    for (const b of bars) {
      const t = b.tracks[pitch];
      if (!t) continue;
      if (!pitchColor && t.color) pitchColor = t.color;
      if (instrumentName === undefined && t.instrument?.name !== undefined) {
        instrumentName = t.instrument.name;
      }
      if (pitchColor && instrumentName !== undefined) break;
    }
    return {
      bars,
      voiceBeats,
      leadInBarsBeats,
      barBeatStart,
      startBeats: barBeatStart,
      pitchColor,
      instrumentName,
    };
  });

  private structureForVoice(voice: Voice, jot: Jot): StructuralVoice {
    const patterns = jot.patterns ?? {};
    const globalTime = jot.globalMetadata.time ?? DEFAULT_TIME;
    const instrumentMap = jot.globalMetadata.instrumentMapping ?? {};
    const rawOffset = jot.globalMetadata.drumsT0Sec;
    const drumsT0Sec =
      typeof rawOffset === 'number' && rawOffset > 0 ? rawOffset : 0;
    // Initial tempo (the value in force before any tempoEvent fires), // what the synthetic lead-in bar runs at, since by construction it
    // precedes bar 0 / any tempoEvent anchored to it.
    const globalBpm = initialBpm(jot);

    const bars: StructuralBar[] = [];
    const pitchOrder: string[] = [];
    let activeTime = globalTime;

    const noteSeenForPitch = (p: string) => {
      if (!pitchOrder.includes(p)) pitchOrder.push(p);
    };

    // Anacrusis (bar 0) — single short "bar" sized to its content.
    if (voice.anacrusis && voice.anacrusis.length > 0) {
      const elements = expandElements(voice.anacrusis, patterns);
      const beats = sumWeights(elements); // anacrusis is unconstrained
      const { tracks, patternSpans, tupletSpans } = this.structureForBarContents(
        elements,
        beats,
        instrumentMap
      );
      Object.keys(tracks).forEach(noteSeenForPitch);
      bars.push({
        source: { elements: voice.anacrusis },
        time: activeTime,
        beats,
        tracks,
        patternSpans,
        tupletSpans,
        index: 0,
      });
    }

    // Bar numbering is drums-t0 anchored: bar 1 = first drum bar, with
    // any pre-drum lead-in bars getting negative indices and bar 0
    // reserved for the anacrusis (which IS drum content, so doesn't fall
    // in the negative range). The skip from -1 → 1 when there's no
    // anacrusis matches musical convention (no "bar 0" in most scoring
    // systems). `leadBars` on globalMetadata is the source-side hint of
    // how many leading `voice.bars` are pre-drum lead-in (set by
    // `from_midi.ts` when it absorbs a transcribed drumless intro); the
    // chrome-only path (`drumsT0Sec` set, `leadBars` absent; e.g. RLRR
    // imports and hand-authored DSL with just a pre-roll offset) is
    // handled by the synthetic-bar prepend below so the structure
    // always carries lead-in as real negative-indexed bars and the
    // renderer doesn't need a second code path.
    const rawLeadBars = jot.globalMetadata.leadBars;
    const leadBars =
      typeof rawLeadBars === 'number' && rawLeadBars > 0
        ? Math.min(rawLeadBars, voice.bars.length)
        : 0;
    for (let i = 0; i < voice.bars.length; i++) {
      const b = voice.bars[i];
      const barTime = b.metadata?.time ?? activeTime;
      activeTime = barTime;
      const beats = barBeats(barTime);
      const elements = expandElements(b.elements, patterns);
      const { tracks, patternSpans, tupletSpans } = this.structureForBarContents(
        elements,
        beats,
        instrumentMap
      );
      Object.keys(tracks).forEach(noteSeenForPitch);
      // Pre-drum bar: negative index counting up to -1. First drum bar:
      // index 1. Skip 0 (reserved for anacrusis if present).
      const index = i < leadBars ? i - leadBars : i - leadBars + 1;
      bars.push({
        source: b,
        time: barTime,
        beats,
        tracks,
        patternSpans,
        tupletSpans,
        index,
      });
    }

    // Chrome-only lead-in: `globalMetadata.drumsT0Sec` set without
    // `leadBars`, so no negative-indexed bars came in. Synthesize one
    // empty bar at `index = -1` covering the audio pre-roll so the
    // rendering / playback / waveform code all see lead-in as just
    // "the first negative-indexed bar(s)", with no separate chrome
    // path to maintain. Beat count is back-computed against the global
    // bpm (the bpm `buildTimeline` will carry into this synthetic bar,
    // since it precedes any explicit override) so `bar.beats * 60 /
    // currentBpm` equals `drumsT0Sec`; the time signature mirrors the
    // global so the (decorative) beat dividers look reasonable. Skipped
    // when leadBars already accounted for the pre-roll, when an
    // anacrusis already sits at the front of `bars`, or when a real
    // negative-indexed bar is already present (defence-in-depth; a
    // future code path that emits lead-in differently won't get
    // doubled up).
    const hasLeadInAlready = bars.some((b) => b.index < 0);
    if (drumsT0Sec > 0 && !hasLeadInAlready) {
      const synthBeats = (drumsT0Sec * globalBpm) / 60;
      const synthTime = bars.find((b) => b.index === 1)?.time ?? globalTime;
      const syntheticBar: StructuralBar = {
        source: { elements: [] },
        time: synthTime,
        beats: synthBeats,
        tracks: {},
        patternSpans: [],
        tupletSpans: [],
        index: -1,
      };
      // Slot it before any anacrusis (bar 0) too, visually the lead-in
      // always precedes the score's first audible content.
      bars.unshift(syntheticBar);
    }

    // Voice lane order: mapped pitches first in declaration order, then
    // any unmapped pitches in first-seen order.
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

    this.assignTrackColors(bars, orderedPitches);

    return {
      source: voice,
      bars,
      pitches: orderedPitches,
    };
  }

  /**
   * Walk a bar's element tree and emit per-pitch tracks, pattern spans
   * and tuplet spans in BEAT coordinates only. The downstream pixel
   * pass multiplies beats by `pxPerBeat` to get the final positions.
   */
  private structureForBarContents(
    elements: Element[],
    beats: number,
    instrumentMap: Record<string, Instrument>
  ): {
    tracks: Record<string, StructuralTrack>;
    patternSpans: StructuralPatternSpan[];
    tupletSpans: StructuralTupletSpan[];
  } {
    const flatNotes: Array<{ note: Note; beat: number; duration: number }> = [];
    const patternSpans: StructuralPatternSpan[] = [];
    const tupletSpans: StructuralTupletSpan[] = [];

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
          for (const child of el.elements) {
            placeElement(child, atBeat, span);
          }
          return;
        case 'group':
          if (el.patternSource) {
            patternSpans.push({
              name: el.patternSource.name,
              startBeat: atBeat,
              endBeat: atBeat + span,
              pitches: collectPitches(el.elements),
              // Placeholder — `structureForJot` assigns the real
              // (jot-wide, first-seen-order) index once all voices are
              // built, so the same pattern name gets the same color
              // across voices.
              colorIndex: 0,
            });
          } else if (el.elements.length > 1 && span > 0) {
            // Tuplet detection — see the dyadic-fraction comment on the
            // pre-refactor layout pass for the why.
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
              const lastOnsetFrac = (tw - ws[ws.length - 1]) / tw;
              tupletSpans.push({
                count: el.elements.length,
                startBeat: atBeat,
                endBeat: atBeat + lastOnsetFrac * span,
              });
            }
          }
          visit(el.elements, atBeat, span);
          return;
        case 'patternRef':
          return;
      }
    };

    visit(elements, 0, beats);

    const tracks: Record<string, StructuralTrack> = {};
    for (const { note, beat, duration } of flatNotes) {
      const pitch = note.pitch;
      let track = tracks[pitch];
      if (!track) {
        const instrument = instrumentMap[pitch] ?? {};
        track = { pitch, instrument, color: '', notes: [] };
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
      });
    }

    return { tracks, patternSpans, tupletSpans };
  }

  private assignTrackColors(
    bars: StructuralBar[],
    orderedPitches: string[]
  ): void {
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
   * Cheap pixel pass: take a structural voice and multiply beats into
   * pixels. Runs on every zoom tick but does no element-tree walking,
   * no pattern expansion, no tuplet detection.
   */
  private pixelVoice(
    sv: StructuralVoice,
    pxPerBeat: number,
    padLeft: Pixels
  ): ResolvedVoice {
    // Lead-in is materialised as negative-indexed bars in `sv.bars` by
    // `structureForVoice` (either real ones from `globalMetadata.leadBars`
    // or a single synthetic one back-computed from
    // `globalMetadata.drumsT0Sec`), so the pixel pass just lays bars
    // out left-to-right from x=0; no separate hatched-chrome offset
    // to reserve.
    const bars: ResolvedBar[] = new Array(sv.bars.length);
    let cursor = 0;
    for (let i = 0; i < sv.bars.length; i++) {
      const sb = sv.bars[i];
      const widthPx = px(sb.beats * pxPerBeat);
      const tracks: Record<string, ResolvedTrack> = {};
      for (const pitch of Object.keys(sb.tracks)) {
        const st = sb.tracks[pitch];
        const notes: ResolvedNote[] = new Array(st.notes.length);
        for (let j = 0; j < st.notes.length; j++) {
          const sn = st.notes[j];
          notes[j] = {
            source: sn.source,
            pitch: sn.pitch,
            modifiers: sn.modifiers,
            sticking: sn.sticking,
            roll: sn.roll,
            beat: sn.beat,
            straight: sn.straight,
            duration: sn.duration,
            x: px((padLeft as number) + sn.beat * pxPerBeat),
            width: px(sn.duration * pxPerBeat),
          };
        }
        tracks[pitch] = {
          pitch: st.pitch,
          instrument: st.instrument,
          color: st.color,
          notes,
        };
      }
      const patternSpans: PatternSpan[] = new Array(sb.patternSpans.length);
      for (let j = 0; j < sb.patternSpans.length; j++) {
        const s = sb.patternSpans[j];
        patternSpans[j] = {
          name: s.name,
          startBeat: s.startBeat,
          endBeat: s.endBeat,
          pitches: s.pitches,
          colorIndex: s.colorIndex,
          x: px((padLeft as number) + s.startBeat * pxPerBeat),
          width: px((s.endBeat - s.startBeat) * pxPerBeat),
        };
      }
      const tupletSpans: TupletSpan[] = new Array(sb.tupletSpans.length);
      for (let j = 0; j < sb.tupletSpans.length; j++) {
        const s = sb.tupletSpans[j];
        tupletSpans[j] = {
          count: s.count,
          startBeat: s.startBeat,
          endBeat: s.endBeat,
          x: px((padLeft as number) + s.startBeat * pxPerBeat),
          width: px((s.endBeat - s.startBeat) * pxPerBeat),
        };
      }
      bars[i] = {
        source: sb.source,
        x: px(cursor),
        width: widthPx,
        time: sb.time,
        beats: sb.beats,
        tracks,
        patternSpans,
        tupletSpans,
        index: sb.index,
      };
      cursor += widthPx as number;
    }
    return {
      source: sv.source,
      bars,
      pitches: sv.pitches,
      width: px(cursor),
      notePadPx: padLeft,
    };
  }
}

// ---------- Drum beat-grid offset ----------

/**
 * Shift every drum note by `offsetBeats` quarter-note beats relative to
 * the bar grid, re-bucketing each note into whichever bar now contains
 * it. Drives the "Beat offset" control, which corrects a consistent
 * beat-detection error from transcription (a groove that landed, say,
 * 1.5 beats late in every bar) by realigning the notes to the beat.
 *
 * The bar grid itself — bar count, time signatures, tempo, total
 * duration — is deliberately untouched: only note placement moves. That
 * keeps jot-time 0, the playback timeline, the playhead and the separate
 * audio-track offset all anchored, so this offset composes cleanly with
 * the audio one.
 *
 * Notes the shift pushes before the first beat or past the final bar are
 * dropped (no bar holds them); for the consistent, many-bar grooves this
 * targets that's at most a fraction of a bar at one end. Pattern / tuplet
 * brackets are cleared while a shift is active because a uniform shift
 * can leave a span straddling a barline.
 */
function applyDrumOffset(resolved: ResolvedJot, offsetBeats: number): ResolvedJot {
  return { ...resolved, voices: resolved.voices.map((v) => shiftVoice(v, offsetBeats)) };
}

/**
 * Beat-only analog of {@link applyDrumOffset}. Shifts every note across
 * the (fixed) bar grid by `offsetBeats` quarter-notes and rebuckets it
 * into the new owning bar. Operates purely in beats — no pixel info
 * involved — so the React components that consume `JotStructure` get
 * the shifted layout without us re-running the pixel pass first.
 */
function applyDrumOffsetStructure(
  structure: JotStructure,
  offsetBeats: number
): JotStructure {
  return {
    ...structure,
    voices: structure.voices.map((v) => shiftStructuralVoice(v, offsetBeats)),
  };
}

function shiftStructuralVoice(
  voice: StructuralVoice,
  offsetBeats: number
): StructuralVoice {
  const bars = voice.bars;
  if (bars.length === 0) return voice;

  const barStart: number[] = new Array(bars.length);
  let acc = 0;
  for (let i = 0; i < bars.length; i++) {
    barStart[i] = acc;
    acc += bars[i].beats;
  }
  const total = acc;
  if (total <= 0) return voice;

  // Empty clones preserving each bar's grid; notes re-added below.
  // Pattern + tuplet spans are cleared (a uniform shift can leave a
  // span straddling a barline), mirroring the ResolvedJot variant.
  const newBars: StructuralBar[] = bars.map((b) => ({
    ...b,
    tracks: {},
    patternSpans: [],
    tupletSpans: [],
  }));

  const locate = (absBeat: number): number => {
    for (let i = 0; i < bars.length; i++) {
      if (absBeat >= barStart[i] && absBeat < barStart[i] + bars[i].beats) return i;
    }
    return -1;
  };

  for (let i = 0; i < bars.length; i++) {
    const srcBar = bars[i];
    for (const pitch of Object.keys(srcBar.tracks)) {
      const srcTrack = srcBar.tracks[pitch];
      for (const note of srcTrack.notes) {
        const newAbs = barStart[i] + note.beat + offsetBeats;
        if (newAbs < 0 || newAbs >= total) continue;
        const j = locate(newAbs);
        if (j < 0) continue;
        const within = newAbs - barStart[j];
        const destBar = newBars[j];
        let destTrack = destBar.tracks[pitch];
        if (!destTrack) {
          destTrack = {
            pitch,
            instrument: srcTrack.instrument,
            color: srcTrack.color,
            notes: [],
          };
          destBar.tracks[pitch] = destTrack;
        }
        destTrack.notes.push({
          ...note,
          beat: within,
          straight: isDyadic(within),
        });
      }
    }
  }

  for (const bar of newBars) {
    for (const pitch of Object.keys(bar.tracks)) {
      bar.tracks[pitch].notes.sort((a, b) => a.beat - b.beat);
    }
  }

  return { ...voice, bars: newBars };
}

function shiftVoice(voice: ResolvedVoice, offsetBeats: number): ResolvedVoice {
  const bars = voice.bars;
  if (bars.length === 0) return voice;

  // Cumulative bar start (quarter-note beats) and the voice's total length.
  const barStart: number[] = new Array(bars.length);
  let acc = 0;
  for (let i = 0; i < bars.length; i++) {
    barStart[i] = acc;
    acc += bars[i].beats;
  }
  const total = acc;
  if (total <= 0) return voice;

  // pxPerBeat is uniform across the voice (see layoutBarContents), so a
  // note's within-bar pixel x can be recomputed with the same formula the
  // layout uses. Derive it from any bar with positive length.
  const ref = bars.find((b) => b.beats > 0);
  if (!ref) return voice;
  const pxPerBeat = (ref.width as number) / ref.beats;
  const padLeft = voice.notePadPx as number;

  // Empty clones preserving each bar's grid geometry; notes re-added below.
  const newBars: ResolvedBar[] = bars.map((b) => ({
    ...b,
    tracks: {},
    patternSpans: [],
    tupletSpans: [],
  }));

  const locate = (absBeat: number): number => {
    // Bars tile [start, start+beats) contiguously; linear scan is fine
    // (jots are typically well under ~64 bars).
    for (let i = 0; i < bars.length; i++) {
      if (absBeat >= barStart[i] && absBeat < barStart[i] + bars[i].beats) return i;
    }
    return -1;
  };

  for (let i = 0; i < bars.length; i++) {
    const srcBar = bars[i];
    for (const pitch of Object.keys(srcBar.tracks)) {
      const srcTrack = srcBar.tracks[pitch];
      for (const note of srcTrack.notes) {
        const newAbs = barStart[i] + note.beat + offsetBeats;
        if (newAbs < 0 || newAbs >= total) continue; // shifted off the grid
        const j = locate(newAbs);
        if (j < 0) continue;
        const within = newAbs - barStart[j];
        const destBar = newBars[j];
        let destTrack = destBar.tracks[pitch];
        if (!destTrack) {
          destTrack = {
            pitch,
            instrument: srcTrack.instrument,
            color: srcTrack.color,
            notes: [],
          };
          destBar.tracks[pitch] = destTrack;
        }
        destTrack.notes.push({
          ...note,
          beat: within,
          straight: isDyadic(within),
          x: px(padLeft + within * pxPerBeat),
        });
      }
    }
  }

  // Re-bucketing can interleave onsets from adjacent source bars; keep
  // each destination track's notes in onset order.
  for (const bar of newBars) {
    for (const pitch of Object.keys(bar.tracks)) {
      bar.tracks[pitch].notes.sort((a, b) => a.beat - b.beat);
    }
  }

  return { ...voice, bars: newBars };
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
export function isDyadic(x: number): boolean {
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
