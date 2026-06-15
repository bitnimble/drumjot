/**
 * Structural view-model over the reactive document: exposes the store-native
 * `Struct*` layers (from {@link StructureStore}) with the interactive drum
 * beat-grid offset applied, plus the layout scale (`pxPerBeat`) and the
 * per-lane row data the mixer reads. Colours come from {@link PaletteStore}
 * and instruments from the global mapping; pixels live in `LayoutStore`.
 *
 * Owns the drum-offset state (the only mutable bit) and reads everything else
 * from the stores it composes. Satisfies {@link LaidOutJot} so it can back the
 * timeline maths directly.
 */
import { action, computed, makeObservable, observable } from 'mobx';
import { computedFn } from 'mobx-utils';
import { Instrument, Jot } from 'src/schema/dsl/dsl';
import { isDyadic } from 'src/schema/dsl/element_metrics';
import { resolveBpm } from 'src/schema/dsl/tempo';
import { ViewConfig } from 'src/editing/viewport/view_config';
import type { LaidOutJot } from 'src/editing/playback/timeline';
import {
  LEAD_IN_BAR_ID,
  type StructBar,
  type StructTrack,
  type StructLayer,
  type StructureStore,
} from './structure_store';
import type { PaletteStore } from 'src/editing/palette/palette_store';
import type { LayoutStore } from 'src/editing/viewport/layout_store';

export class StructuralPresenter implements LaidOutJot {
  drumOffsetBeats = 0;
  drumOffsetBeatsBaseline = 0;

  constructor(
    private readonly structureStore: StructureStore,
    private readonly paletteStore: PaletteStore,
    private readonly layoutStore: LayoutStore,
    readonly source: Jot,
    private readonly viewConfig: ViewConfig
  ) {
    makeObservable(this, {
      drumOffsetBeats: observable,
      drumOffsetBeatsBaseline: observable,
      setDrumOffset: action,
      setDrumOffsetBaseline: action,
      musicalLayers: computed,
      layers: computed,
      pxPerBeat: computed,
      layerBeats: computed,
      primaryLayer: computed,
    });
  }

  setDrumOffset(beats: number) {
    this.drumOffsetBeats = Number.isFinite(beats) ? beats : 0;
  }

  setDrumOffsetBaseline(beats: number) {
    this.drumOffsetBeatsBaseline = Number.isFinite(beats) ? beats : 0;
  }

  get effectiveDrumOffsetBeats(): number {
    return this.drumOffsetBeats - this.drumOffsetBeatsBaseline;
  }

  /** The shared layout config (part of the {@link LaidOutJot} surface this
   *  presenter satisfies, so it can back the timeline maths directly). */
  get config(): ViewConfig {
    return this.viewConfig;
  }

  /** The store-native `Struct*` layers, drum-offset applied. The MUSICAL
   *  structure (1:1 with the source bars): export, playback scheduling, and
   *  the tempo summary read this so a view-only lead-in never leaks into
   *  exported/scheduled content. View code reads {@link layers} instead. */
  get musicalLayers(): StructLayer[] {
    const base = this.structureStore.layers;
    const eff = this.effectiveDrumOffsetBeats;
    return eff === 0 ? base : applyOffsetToStructLayers(base, eff);
  }

  /** The structure as the SCORE renders it: {@link musicalLayers} plus a
   *  view-only "virtual" lead-in bar so the first note never clips at the
   *  left edge and an audio track with no pre-roll still has room. The
   *  virtual bar is sized so total lead-in is at least one full bar (using
   *  the `drumsT0Sec` audio pre-roll when that already exceeds a bar), and is
   *  prepended only when the song carries no real lead-in bar of its own.
   *  Tagged {@link LEAD_IN_BAR_ID}; transparent to the renderer, excluded
   *  from export/playback via {@link musicalLayers}. */
  get layers(): StructLayer[] {
    const drumsT0Sec = this.source.globalMetadata.drumsT0Sec ?? 0;
    const bpm = resolveBpm(this.source.globalMetadata.bpm, 120);
    return this.musicalLayers.map((layer) => withVirtualLeadIn(layer, drumsT0Sec, bpm));
  }

  get pxPerBeat(): number {
    return ((this.viewConfig.barWidth as number) * this.layoutStore.densityFactor) / 4;
  }

  get layerBeats(): number {
    const bars = this.layers[0]?.bars;
    if (!bars) return 0;
    let total = 0;
    for (const b of bars) total += b.beats;
    return total;
  }

  get primaryLayer(): StructLayer | undefined {
    return this.layers[0];
  }

  /** Instrument for a lane from the global mapping. */
  private instrumentFor(lane: string): Instrument {
    return this.source.globalMetadata.instrumentMapping?.[lane] ?? { kind: 'custom' };
  }

  barsForLane = computedFn(
    (
      lane: string
    ): {
      bars: readonly StructBar[];
      layerBeats: number;
      leadInBarsBeats: number;
      barBeatStart: readonly number[];
      startBeats: readonly number[];
      laneColor: string;
      instrumentName: string | undefined;
    } => {
      const layer = this.layers[0];
      const bars = layer?.bars ?? [];
      let layerBeats = 0;
      let leadInBarsBeats = 0;
      let countedLeadIn = true;
      const barBeatStart: number[] = new Array(bars.length);
      let cursor = 0;
      for (let i = 0; i < bars.length; i++) {
        barBeatStart[i] = cursor;
        const b = bars[i];
        cursor += b.beats;
        layerBeats += b.beats;
        if (countedLeadIn) {
          if (b.index < 0) leadInBarsBeats += b.beats;
          else countedLeadIn = false;
        }
      }
      // Colour + instrument are jot-wide functions of the lane (palette
      // slot + the instrument mapping), no longer per-track.
      const laneColor = this.paletteStore.colorForLane(lane);
      const instrumentName = this.instrumentFor(lane).name;
      return {
        bars,
        layerBeats,
        leadInBarsBeats,
        barBeatStart,
        startBeats: barBeatStart,
        laneColor,
        instrumentName,
      };
    }
  );
}

// ---------- View-only virtual lead-in ----------

/**
 * Prepend a view-only "virtual" lead-in bar to a layer when it has no real
 * lead-in bar of its own (no negative-indexed bar), so the first note isn't
 * clipped at the score's left edge. Sized to at least one full bar; when the
 * audio pre-roll (`drumsT0Sec`) already exceeds a bar, the bar covers the
 * whole pre-roll instead. The bar is empty and indexed -1, exactly like a
 * real lead-in bar, so the renderer / waveform / timeline treat it uniformly;
 * it carries {@link LEAD_IN_BAR_ID} so musical paths (`musicalLayers`) and the
 * tempo builder can tell it apart.
 */
function withVirtualLeadIn(layer: StructLayer, drumsT0Sec: number, bpm: number): StructLayer {
  if (layer.bars.length === 0) return layer;
  // A real lead-in (explicit `leadBars`) already gives the first note room.
  if (layer.bars.some((b) => b.index < 0)) return layer;
  const firstReal = layer.bars.find((b) => b.index === 1) ?? layer.bars[0];
  const oneBarBeats = (firstReal.tsCount * 4) / firstReal.tsUnit;
  const preRollBeats = (drumsT0Sec * bpm) / 60;
  const virtual: StructBar = {
    id: LEAD_IN_BAR_ID,
    index: -1,
    beats: Math.max(preRollBeats, oneBarBeats),
    tsCount: firstReal.tsCount,
    tsUnit: firstReal.tsUnit,
    anacrusis: false,
    tracks: {},
    patternSpans: [],
    tupletSpans: [],
  };
  return { ...layer, bars: [virtual, ...layer.bars] };
}

// ---------- Drum beat-grid offset ----------

function applyOffsetToStructLayers(layers: StructLayer[], offsetBeats: number): StructLayer[] {
  return layers.map((v) => shiftStructLayer(v, offsetBeats));
}

/** Slide every note across the fixed bar grid by `offsetBeats`, re-homing
 *  notes that cross a barline. Pattern/tuplet spans are cleared (their
 *  geometry no longer matches the shifted notes). */
function shiftStructLayer(layer: StructLayer, offsetBeats: number): StructLayer {
  const bars = layer.bars;
  if (bars.length === 0) return layer;

  const barStart: number[] = new Array(bars.length);
  let acc = 0;
  for (let i = 0; i < bars.length; i++) {
    barStart[i] = acc;
    acc += bars[i].beats;
  }
  const total = acc;
  if (total <= 0) return layer;

  const newBars: StructBar[] = bars.map((b) => ({
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
    for (const lane of Object.keys(srcBar.tracks)) {
      const srcTrack = srcBar.tracks[lane];
      for (const note of srcTrack.notes) {
        const newAbs = barStart[i] + note.beat + offsetBeats;
        if (newAbs < 0 || newAbs >= total) continue;
        const j = locate(newAbs);
        if (j < 0) continue;
        const within = newAbs - barStart[j];
        const destBar = newBars[j];
        let destTrack: StructTrack = destBar.tracks[lane];
        if (!destTrack) {
          destTrack = { lane, notes: [] };
          destBar.tracks[lane] = destTrack;
        }
        destTrack.notes.push({ ...note, beat: within, straight: isDyadic(within) });
      }
    }
  }

  for (const bar of newBars) {
    for (const lane of Object.keys(bar.tracks)) {
      bar.tracks[lane].notes.sort((a, b) => a.beat - b.beat);
    }
  }

  return { ...layer, bars: newBars };
}
