/**
 * The laid-out jot model: the zoom-invariant `Structural*` types + the
 * pixel-bearing `Resolved*` types, and the `RenderedJot` container.
 *
 * `RenderedJot` is a thin composition root over the reactive Jot document:
 * its constructor converts the DSL `source` into a reactive model
 * ({@link dslToReactive}) and wires up the domain stores (structure /
 * palette / layout) + the {@link StructuralPresenter} that owns the
 * drum-offset and maps the stores' `Struct*` output onto the legacy
 * `Structural*` / `Resolved*` shapes consumers expect. Everything below is
 * delegation: structure / layout / offset → `StructuralPresenter`, the
 * pitch list → `PaletteStore`, tempo + timeline → their modules. The DSL
 * `source` is kept for metadata + tempo + the synthesised `note.source`.
 * Re-exported from `jot.ts`.
 */
import { computed, makeObservable } from 'mobx';
import { Bar, Instrument, Jot, Metadata, Modifier, Note, TimeSignature, Voice } from 'src/dsl/dsl';
import { BarTempos, buildBarTempos } from 'src/tempo/tempo';
import { buildTimeline, JotTimeline, pickDominantBpmAndTime } from 'src/jot_view/playback/timeline';
import { dslToReactive } from 'src/schema/from_dsl';
import type { Jot as ReactiveJot } from 'src/schema/schema';
import { StructureStore } from 'src/jot_view/structure/structure_store';
import { StructuralPresenter } from 'src/jot_view/structure/structural_presenter';
import { PaletteStore } from 'src/jot_view/palette/palette_store';
import { LayoutStore } from 'src/jot_view/viewport/layout_store';
import { Pixels, ViewConfig } from './view_config';

// ---------- Resolved (laid out) types ----------

export type ResolvedNote = StructuralNote & {
  /** Pixel x within the bar. */
  x: Pixels;
  width: Pixels;
};

export type ResolvedTrack = Omit<StructuralTrack, 'notes'> & {
  notes: ResolvedNote[];
};

export type PatternSpan = StructuralPatternSpan & {
  x: Pixels;
  width: Pixels;
};

export type TupletSpan = StructuralTupletSpan & {
  x: Pixels;
  width: Pixels;
};

export type ResolvedBar = Omit<StructuralBar, 'tracks' | 'patternSpans' | 'tupletSpans'> & {
  x: Pixels;
  width: Pixels;
  tracks: Record<string, ResolvedTrack>;
  patternSpans: PatternSpan[];
  tupletSpans: TupletSpan[];
};

export type ResolvedVoice = Omit<StructuralVoice, 'bars'> & {
  bars: ResolvedBar[];
  width: Pixels;
  /** Horizontal engraving inset (px) the note grid is shifted right of
   *  each bar's left edge. See {@link ViewConfig.barNotePaddingBeats}. */
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

// ---------- Structural cache (beat-only, zoom-invariant) ----------

export type StructuralNote = {
  /** Stable element id (selection / React keys / provenance key off it). */
  id: string;
  source: Note;
  pitch: string;
  modifiers: ReadonlySet<Modifier>;
  sticking?: Note['sticking'];
  roll: boolean;
  /** Position within bar, in beats (0 = bar start, bar.length = bar end). */
  beat: number;
  /** Onset lands on the binary (dyadic) grid. */
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
  startBeat: number;
  endBeat: number;
  pitches: ReadonlySet<string>;
  colorIndex: number;
};

export type StructuralTupletSpan = {
  count: number;
  startBeat: number;
  endBeat: number;
};

export type StructuralBar = {
  source: Bar;
  time: TimeSignature;
  /** Bar length in beats (count * 4/unit, in quarter notes). */
  beats: number;
  tracks: Record<string, StructuralTrack>;
  patternSpans: StructuralPatternSpan[];
  tupletSpans: StructuralTupletSpan[];
  /** Bar number within the voice (1-based; anacrusis is bar 0). */
  index: number;
};

export type StructuralVoice = {
  source: Voice;
  /** All bars including anacrusis if present (anacrusis is bar 0). */
  bars: StructuralBar[];
  /** Lane order: mapped pitches first, then unmapped in first-seen order. */
  pitches: string[];
};

export type JotStructure = {
  title: string;
  voices: StructuralVoice[];
  globalMetadata: Metadata;
  /** Whole-song horizontal scale derived from onset density. */
  densityFactor: number;
};

// ---------- RenderedJot: reactive-backed composition root ----------

export class RenderedJot {
  private viewConfig: ViewConfig;
  private reactive: ReactiveJot;
  private structureStore: StructureStore;
  private paletteStore: PaletteStore;
  private layoutStore: LayoutStore;
  private structural: StructuralPresenter;

  constructor(public source: Jot, viewConfig?: ViewConfig) {
    this.viewConfig = viewConfig ?? new ViewConfig();
    this.reactive = dslToReactive(source).model;
    this.structureStore = new StructureStore(() => this.reactive);
    this.paletteStore = new PaletteStore(
      this.structureStore,
      () => this.viewConfig.palette,
      () => this.reactive
    );
    this.layoutStore = new LayoutStore(
      this.structureStore,
      () => this.viewConfig.barWidth as number,
      () => this.viewConfig.barNotePaddingBeats
    );
    this.structural = new StructuralPresenter(
      this.structureStore,
      this.paletteStore,
      this.layoutStore,
      source,
      this.viewConfig
    );
    // Only the tempo/legend views are computed here; the structural /
    // layout / offset surface is delegated to `this.structural` and the
    // pitch list to `this.paletteStore`, each already cached there.
    makeObservable(this, {
      timeline: computed,
      barTempos: computed,
      dominantBpmAndTime: computed,
      legendPitches: computed,
    });
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

  defaultPaletteColorFor(pitch: string): string | undefined {
    const palette = this.viewConfig.palette;
    if (palette.length === 0) return undefined;
    const idx = this.paletteStore.jotPitches.indexOf(pitch);
    return idx >= 0 ? palette[idx % palette.length] : undefined;
  }

  // ----- structural / layout / offset (delegated to StructuralPresenter) -----

  get structure(): JotStructure {
    return this.structural.structure;
  }

  get resolved(): ResolvedJot {
    return this.structural.resolved;
  }

  get pxPerBeat(): number {
    return this.structural.pxPerBeat;
  }

  get voiceBeats(): number {
    return this.structural.voiceBeats;
  }

  get primaryStructuralVoice(): StructuralVoice | undefined {
    return this.structural.primaryStructuralVoice;
  }

  barsForPitch(pitch: string) {
    return this.structural.barsForPitch(pitch);
  }

  get drumOffsetBeats(): number {
    return this.structural.drumOffsetBeats;
  }

  get drumOffsetBeatsBaseline(): number {
    return this.structural.drumOffsetBeatsBaseline;
  }

  get effectiveDrumOffsetBeats(): number {
    return this.structural.effectiveDrumOffsetBeats;
  }

  setDrumOffset(beats: number) {
    this.structural.setDrumOffset(beats);
  }

  setDrumOffsetBaseline(beats: number) {
    this.structural.setDrumOffsetBaseline(beats);
  }

  // ----- tempo / timeline (thin wrappers over the tempo + timeline modules) -----

  get timeline(): JotTimeline {
    return buildTimeline(this);
  }

  get barTempos(): readonly BarTempos[] {
    const bars = this.structure.voices[0]?.bars;
    if (!bars) return [];
    return buildBarTempos(this.source, bars);
  }

  get dominantBpmAndTime(): {
    dominantBpm: number | undefined;
    dominantTime: TimeSignature | undefined;
  } {
    return pickDominantBpmAndTime(this);
  }

  // ----- palette-derived legend / pitch list -----

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

  get jotPitches(): readonly string[] {
    return this.paletteStore.jotPitches;
  }
}
