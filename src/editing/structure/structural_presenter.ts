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
 *
 * **Render-path granularity.** The score reads {@link barsForLane}, which (when
 * no beat-grid offset is applied, the common case) is built from the store's
 * per-(bar, lane) granular computeds: a stable geometry spine ({@link
 * viewGeometry}) plus {@link StructureStore.trackFor} / `spansFor`, each
 * structurally gated. So adding a note re-renders only the one bar+lane it
 * touched, and every other row's `barsForLane` keeps its cached value, no
 * re-render, no React reconciliation. `musicalLayers` / `layers` (the full
 * structure) stay for export / playback / tempo / density / lane-list
 * consumers; the render path never reads them.
 */
import { action, comparer, computed, makeObservable, observable } from 'mobx';
import { computedFn } from 'mobx-utils';
import { Instrument, TempoEvent } from 'src/schema/dsl/dsl';
import { isDyadic } from 'src/schema/dsl/element_metrics';
import { initialBpm, type TempoJot } from 'src/schema/dsl/tempo';
import { DEFAULT_GRID_DIVISION } from 'src/grid/grid';
import type { MutableJot } from 'src/schema/schema';
import type { JotDerivedRegistry } from 'src/schema/derived_fields';
import { ViewConfig } from 'src/editing/viewport/view_config';
import type { LaidOutJot } from 'src/editing/playback/timeline';
import {
  LEAD_IN_BAR_ID,
  type StructBar,
  type StructBarGeometry,
  type StructTrack,
  type StructLayer,
  type StructureStore,
} from './structure_store';
import type { PaletteStore } from 'src/editing/palette/palette_store';
import type { LayoutStore } from 'src/editing/viewport/layout_store';

const STRUCTURAL = { equals: comparer.structural } as const;

/** Per-lane row data the mixer's instrument row reads. */
export type LaneBars = {
  bars: readonly StructBar[];
  layerBeats: number;
  leadInBarsBeats: number;
  barBeatStart: readonly number[];
  startBeats: readonly number[];
  laneColor: string;
  instrumentName: string | undefined;
};

export class StructuralPresenter implements LaidOutJot {
  drumOffsetBeats = 0;
  drumOffsetBeatsBaseline = 0;

  constructor(
    private readonly structureStore: StructureStore,
    private readonly paletteStore: PaletteStore,
    private readonly layoutStore: LayoutStore,
    private readonly getJot: () => MutableJot | undefined,
    private readonly viewConfig: ViewConfig,
    registry: JotDerivedRegistry
  ) {
    makeObservable(this, {
      drumOffsetBeats: observable,
      drumOffsetBeatsBaseline: observable,
      setDrumOffset: action,
      setDrumOffsetBaseline: action,
      musicalLayers: computed,
      layers: computed,
      tempoSource: computed.struct,
      pxPerBeat: computed,
      lanes: computed.struct,
      viewLayerId: computed,
      viewGeometry: computed.struct,
      viewGeometryById: computed,
      layerBeats: computed,
      hasContent: computed,
      primaryLayer: computed,
    });
    // Install this domain's cross-domain derived fields on the document, so
    // consumers read `jot.lanes` / `jot.tempoSource` etc. without importing
    // this presenter. The getters below are the implementations. (Viewport
    // state like `pxPerBeat` stays on the presenter, it's not document data.)
    registry.lanes.define(() => this.lanes);
    registry.musicalLayers.define(() => this.musicalLayers);
    registry.barsForLane.define((lane) => this.barsForLane(lane));
    registry.tempoSource.define(() => this.tempoSource);
    registry.barDrift.define(() => this.barDrift);
    registry.instrumentFor.define((lane) => this.instrumentFor(lane));
    registry.ownerLayerFor.define((lane) => this.ownerLayerFor(lane));
    registry.renderedLayers.define(() => this.layers);
  }

  /**
   * The tempo slice the timeline maths read, projected live off the reactive
   * document: each reactive `TempoEvent` (anchored by stable `barId`)
   * re-keyed to the `barIndex` the pure tempo helpers expect (its position
   * among the non-anacrusis bars, matching DSL `TempoEvent.barIndex`), plus
   * the reactive initial `bpm`. This is what replaces the old frozen DSL
   * `source` snapshot in the runtime tempo path, so a tempo edit reflows the
   * header / playhead immediately. `computed.struct` so an unrelated edit
   * (notes, etc.) that leaves tempo untouched doesn't invalidate it.
   */
  get tempoSource(): TempoJot {
    const jot = this.getJot();
    if (!jot) return { tempoEvents: [], globalMetadata: {} };
    const barIndexById = new Map<string, number>();
    let idx = 0;
    for (const bar of jot.bars) {
      if (bar.anacrusis) continue;
      barIndexById.set(bar.id, idx++);
    }
    const tempoEvents: TempoEvent[] = [];
    for (const ev of jot.tempoEvents.values()) {
      const barIndex = barIndexById.get(ev.barId);
      if (barIndex === undefined) continue;
      tempoEvents.push({ barIndex, beat: ev.beat, bpm: ev.bpm });
    }
    // `barIndexById` counts ALL non-anacrusis bars (lead-in included), so the
    // drums-enter bar (where the initial-tempo event lives) sits at
    // barIndex == leadBars. Pass that so `initialBpm` recognises it as the
    // song-start event rather than reading the 120 default.
    return { tempoEvents, globalMetadata: { leadBars: jot.leadBars ?? 0 } };
  }

  /** Per-bar performance drift seconds, indexed by `layers[0].bars` (lead-in
   *  bars = 0), decoded from the reactive doc's `barDriftJson`. Empty for a
   *  metronomic recording or a hand-authored jot. Feeds the waveform stretch
   *  (`buildChunkLayout`) and the playback `DriftMap` (`buildTimeline`). */
  get barDrift(): readonly number[] {
    const json = this.getJot()?.barDriftJson;
    if (!json) return [];
    try {
      const parsed: unknown = JSON.parse(json);
      return Array.isArray(parsed) ? (parsed as number[]) : [];
    } catch {
      return [];
    }
  }

  /** Producer grid density (1/N-of-a-whole-note), read live off the reactive
   *  document; falls back to the default when none is set. */
  get gridDivision(): number {
    return this.getJot()?.gridDivision ?? DEFAULT_GRID_DIVISION;
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
   *  exported/scheduled content. View code reads {@link barsForLane} instead. */
  get musicalLayers(): StructLayer[] {
    const base = this.structureStore.layers;
    const eff = this.effectiveDrumOffsetBeats;
    return eff === 0 ? base : applyOffsetToStructLayers(base, eff);
  }

  /** The structure as the SCORE renders it: {@link musicalLayers} plus a
   *  view-only "virtual" lead-in bar (see {@link viewGeometry}). The full
   *  multi-lane form, kept for the timeline / minimap / lyrics consumers; the
   *  per-row render path reads {@link barsForLane}. */
  get layers(): StructLayer[] {
    const jot = this.getJot();
    const preRollSec = Math.max(0, -(jot?.songLeadIn ?? 0));
    const bpm = initialBpm(this.tempoSource);
    return this.musicalLayers.map((layer) => withVirtualLeadIn(layer, preRollSec, bpm));
  }

  get pxPerBeat(): number {
    return ((this.viewConfig.barWidth as number) * this.layoutStore.densityFactor) / 4;
  }

  /** Ordered lane list across all layers (lanes that carry a note).
   *  `computed.struct`, so an in-lane note edit never churns the mixer's row
   *  list. */
  get lanes(): string[] {
    const out: string[] = [];
    for (const layer of this.structureStore.layerOrder) {
      for (const lane of this.structureStore.lanesForLayer(layer.id)) {
        if (!out.includes(lane)) out.push(lane);
      }
    }
    return out;
  }

  /** Id of the layer the per-row render path reads (the first `||` layer). */
  get viewLayerId(): string {
    return this.structureStore.layerOrder[0]?.id ?? '';
  }

  /** Id of the `||` layer that owns `lane` (carries a note on it), for placing
   *  inserted / moved notes in the row the user clicked. `undefined` for a
   *  single-layer jot or a lane no layer carries yet. */
  ownerLayerFor(lane: string): string | undefined {
    return this.structureStore.ownerLayerFor(lane);
  }

  /** Geometry spine the score renders against: the layer-0 bar geometry plus a
   *  view-only "virtual" lead-in bar when the song has no real lead-in of its
   *  own, so the first note never clips at the left edge and a song with no
   *  pre-roll still has count-in room. Sized to at least one full bar (or the
   *  audio pre-roll when that's longer). `computed.struct`, so a note edit
   *  (which never changes bar geometry) leaves it, and everything keyed off it,
   *  untouched. */
  get viewGeometry(): StructBarGeometry[] {
    const layerId = this.viewLayerId;
    const base = this.structureStore.geometryFor(layerId);
    if (base.length === 0) return base;
    // A real lead-in (explicit `leadBars`) already gives the first note room.
    if (base.some((b) => b.index < 0)) return base;
    const jot = this.getJot();
    const preRollSec = Math.max(0, -(jot?.songLeadIn ?? 0));
    const bpm = initialBpm(this.tempoSource);
    const firstReal = base.find((b) => b.index === 1) ?? base[0];
    const oneBarBeats = (firstReal.tsCount * 4) / firstReal.tsUnit;
    const preRollBeats = (preRollSec * bpm) / 60;
    const virtual: StructBarGeometry = {
      id: LEAD_IN_BAR_ID,
      index: -1,
      beats: Math.max(preRollBeats, oneBarBeats),
      tsCount: firstReal.tsCount,
      tsUnit: firstReal.tsUnit,
      anacrusis: false,
    };
    return [virtual, ...base];
  }

  /** {@link viewGeometry} indexed by bar id, for the per-bar lane composer. */
  get viewGeometryById(): Map<string, StructBarGeometry> {
    const map = new Map<string, StructBarGeometry>();
    for (const geo of this.viewGeometry) map.set(geo.id, geo);
    return map;
  }

  get layerBeats(): number {
    let total = 0;
    for (const b of this.viewGeometry) total += b.beats;
    return total;
  }

  /** Whether a song is loaded (a stable boolean for view null-checks that must
   *  not re-render on every note edit, unlike reading {@link primaryLayer}). */
  get hasContent(): boolean {
    return this.viewGeometry.length > 0;
  }

  get primaryLayer(): StructLayer | undefined {
    return this.layers[0];
  }

  /** Instrument for a lane from the global mapping, read live off the reactive
   *  `instruments` map and adapted back to the DSL `Instrument` shape (the
   *  reactive entity stores a flat `midiNote`; consumers expect `midi.note`).
   *  Public so the mixer-order sort, the playback scheduler and the row labels
   *  share one reactive-backed lookup instead of the old frozen mapping. */
  instrumentFor(lane: string): Instrument {
    const inst = this.getJot()?.instruments.get(lane);
    if (!inst) return { kind: 'custom' };
    return {
      kind: inst.kind,
      name: inst.name,
      limb: inst.limb,
      midi: inst.midiNote !== undefined ? { note: inst.midiNote } : undefined,
    };
  }

  /**
   * One lane-scoped bar: the shared geometry plus ONLY this lane's track (and
   * the bar's span chrome). Structurally gated, so its identity is stable
   * unless this lane's notes (or the bar geometry / spans) actually change.
   * That stability is what isolates a note edit to the single bar+lane it
   * touched: a sibling lane's `laneBarFor` doesn't even re-run.
   */
  private laneBarFor = computedFn((barId: string, lane: string): StructBar => {
    const geo = this.viewGeometryById.get(barId);
    const base = geo ?? {
      id: barId,
      index: 0,
      beats: 0,
      tsCount: 4,
      tsUnit: 4,
      anacrusis: false,
    };
    if (barId === LEAD_IN_BAR_ID) {
      return { ...base, tracks: {}, patternSpans: [], tupletSpans: [], groupSpans: [] };
    }
    // Notes and bracket chrome both come from every layer that places this
    // lane / bar (so a hands/feet split shows its kick row, and a tuplet in a
    // non-first layer still draws its bracket).
    const track = this.structureStore.mergedTrackFor(barId, lane);
    const spans = this.structureStore.mergedSpansFor(barId);
    return {
      ...base,
      tracks: { [lane]: track },
      patternSpans: spans.patternSpans,
      tupletSpans: spans.tupletSpans,
      groupSpans: spans.groupSpans,
    };
  }, STRUCTURAL);

  /**
   * Per-lane derived data for one instrument row: the lane-scoped bars plus
   * the cumulative bar-start offsets and label colour/name. Memoised per lane
   * on the jot; with no beat-grid offset it composes the granular per-(bar,
   * lane) computeds, so a note edit re-runs only the affected lane's entry and
   * leaves every other row's value cached (no re-render). A non-zero beat-grid
   * offset (the Beat-offset slider, a deliberate whole-song reflow) falls back
   * to the full shifted structure.
   */
  /**
   * Like {@link laneBarFor} but scoped to ONE layer: only that layer's notes /
   * spans on the lane (via the per-`(bar, layer)` `trackFor` / `spansFor`),
   * NOT merged across layers. The per-track render path (a track = one layer +
   * one lane) reads this so two layers' same-lane tracks show their own notes.
   */
  private laneBarForLayer = computedFn((barId: string, layerId: string, lane: string): StructBar => {
    const geo = this.viewGeometryById.get(barId);
    const base = geo ?? { id: barId, index: 0, beats: 0, tsCount: 4, tsUnit: 4, anacrusis: false };
    if (barId === LEAD_IN_BAR_ID) {
      return { ...base, tracks: {}, patternSpans: [], tupletSpans: [], groupSpans: [] };
    }
    const key = this.structureStore.keyFor(barId, layerId);
    const track = this.structureStore.trackFor(key, lane);
    const spans = this.structureStore.spansFor(key);
    return {
      ...base,
      tracks: { [lane]: track },
      patternSpans: spans.patternSpans,
      tupletSpans: spans.tupletSpans,
      groupSpans: spans.groupSpans,
    };
  }, STRUCTURAL);

  /**
   * Per-track derived row data: like {@link barsForLane} but for ONE track
   * (a specific `layerId` + `lane`), so the same lane in two layers renders two
   * independent rows. With no beat-grid offset it composes the granular
   * per-(bar, layer) computeds; a non-zero offset falls back to that layer's
   * shifted structure.
   */
  barsForTrack = computedFn((layerId: string, lane: string): LaneBars => {
    let bars: readonly StructBar[];
    if (this.effectiveDrumOffsetBeats === 0) {
      bars = this.viewGeometry.map((geo) => this.laneBarForLayer(geo.id, layerId, lane));
    } else {
      const layer = this.layers.find((l) => l.id === layerId);
      bars = (layer?.bars ?? []).map((b) => ({
        ...b,
        tracks: b.tracks[lane] ? { [lane]: b.tracks[lane] } : {},
      }));
    }

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
  });

  barsForLane = computedFn((lane: string): LaneBars => {
    const bars: readonly StructBar[] =
      this.effectiveDrumOffsetBeats === 0
        ? this.viewGeometry.map((geo) => this.laneBarFor(geo.id, lane))
        : (this.layers[0]?.bars ?? []);

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
    // Colour + instrument are jot-wide functions of the lane (palette slot +
    // the instrument mapping), no longer per-track.
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
  });
}

// ---------- View-only virtual lead-in ----------

/**
 * Prepend a view-only "virtual" lead-in bar to a layer when it has no real
 * lead-in bar of its own (no negative-indexed bar), so the first note isn't
 * clipped at the score's left edge. Sized to at least one full bar; when the
 * audio pre-roll (`songLeadIn`) already exceeds a bar, the bar covers the
 * whole pre-roll instead. The bar is empty and indexed -1, exactly like a
 * real lead-in bar, so the renderer / waveform / timeline treat it uniformly;
 * it carries {@link LEAD_IN_BAR_ID} so musical paths (`musicalLayers`) and the
 * tempo builder can tell it apart.
 */
function withVirtualLeadIn(layer: StructLayer, preRollSec: number, bpm: number): StructLayer {
  if (layer.bars.length === 0) return layer;
  // A real lead-in (explicit `leadBars`) already gives the first note room.
  if (layer.bars.some((b) => b.index < 0)) return layer;
  const firstReal = layer.bars.find((b) => b.index === 1) ?? layer.bars[0];
  const oneBarBeats = (firstReal.tsCount * 4) / firstReal.tsUnit;
  const preRollBeats = (preRollSec * bpm) / 60;
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
    groupSpans: [],
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
    groupSpans: [],
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
