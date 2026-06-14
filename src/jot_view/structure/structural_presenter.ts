/**
 * Structural/layout view-model: maps the reactive document's `Struct*`
 * output (from {@link StructureStore}) onto the legacy `Structural*` /
 * `Resolved*` shapes the renderer consumes, applies the interactive
 * drum beat-grid offset, and runs the cheap pixel pass.
 *
 * This is the bulk of what `RenderedJot` used to do inline; pulled out
 * into a presenter so it can be unit-tested against mocked stores and so
 * `RenderedJot` is just a thin composition root over the domain stores +
 * this presenter. Owns the offset state (the only mutable bit) and reads
 * everything else from the three data stores it composes.
 */
import { action, computed, makeObservable, observable } from 'mobx';
import { computedFn } from 'mobx-utils';
import { Bar, Instrument, Jot, Modifier, Note, Sticking, Voice, Volume } from 'src/dsl/dsl';
import { isDyadic } from 'src/jot/element_metrics';
import type {
  JotStructure,
  ResolvedBar,
  ResolvedJot,
  ResolvedNote,
  ResolvedTrack,
  ResolvedVoice,
  StructuralBar,
  StructuralNote,
  StructuralTrack,
  StructuralVoice,
  PatternSpan,
  TupletSpan,
} from 'src/jot/resolved_jot';
import { Pixels, px, ViewConfig } from 'src/jot/view_config';
import type { Metadata } from 'src/dsl/dsl';
import {
  type StructBar,
  type StructNote,
  type StructVoice,
  type StructureStore,
} from './structure_store';
import type { PaletteStore } from 'src/jot_view/palette/palette_store';
import type { LayoutStore } from 'src/jot_view/viewport/layout_store';

export class StructuralPresenter {
  drumOffsetBeats = 0;
  drumOffsetBeatsBaseline = 0;

  constructor(
    private readonly structureStore: StructureStore,
    private readonly paletteStore: PaletteStore,
    private readonly layoutStore: LayoutStore,
    private readonly source: Jot,
    private readonly viewConfig: ViewConfig
  ) {
    makeObservable(this, {
      drumOffsetBeats: observable,
      drumOffsetBeatsBaseline: observable,
      setDrumOffset: action,
      setDrumOffsetBaseline: action,
      structure: computed,
      resolved: computed,
      pxPerBeat: computed,
      voiceBeats: computed,
      primaryStructuralVoice: computed,
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

  /** The reactive-derived structure mapped onto the legacy shapes, before
   *  the drum offset. Not annotated as computed: it's cheap and reads the
   *  already-cached store computeds. */
  private get baseStructure(): JotStructure {
    const voices = this.structureStore.voices.map((sv) => this.toStructuralVoice(sv));
    return {
      title: this.source.title,
      voices,
      globalMetadata: this.source.globalMetadata,
      densityFactor: this.layoutStore.densityFactor,
    };
  }

  get structure(): JotStructure {
    const base = this.baseStructure;
    const eff = this.effectiveDrumOffsetBeats;
    return eff === 0 ? base : applyDrumOffsetStructure(base, eff);
  }

  get resolved(): ResolvedJot {
    const base = this.pixelPass(this.baseStructure);
    const eff = this.effectiveDrumOffsetBeats;
    return eff === 0 ? base : applyDrumOffset(base, eff);
  }

  get pxPerBeat(): number {
    return ((this.viewConfig.barWidth as number) * this.layoutStore.densityFactor) / 4;
  }

  get voiceBeats(): number {
    const bars = this.structure.voices[0]?.bars;
    if (!bars) return 0;
    let total = 0;
    for (const b of bars) total += b.beats;
    return total;
  }

  get primaryStructuralVoice(): StructuralVoice | undefined {
    return this.structure.voices[0];
  }

  barsForPitch = computedFn(
    (
      pitch: string
    ): {
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
    }
  );

  // ----- reactive Struct* → legacy Structural* mapping -----

  private toStructuralVoice(sv: StructVoice): StructuralVoice {
    return {
      source: { bars: [] } as Voice,
      bars: sv.bars.map((sb) => this.toStructuralBar(sb)),
      pitches: sv.pitches,
    };
  }

  private toStructuralBar(sb: StructBar): StructuralBar {
    const tracks: Record<string, StructuralTrack> = {};
    for (const pitch of Object.keys(sb.tracks)) {
      tracks[pitch] = {
        pitch,
        instrument: this.instrumentFor(pitch),
        color: this.paletteStore.colorForPitch(pitch),
        notes: sb.tracks[pitch].notes.map(toStructuralNote),
      };
    }
    return {
      source: { elements: [] } as Bar,
      time: { count: sb.tsCount, unit: sb.tsUnit },
      beats: sb.beats,
      tracks,
      patternSpans: sb.patternSpans.map((s) => ({ ...s })),
      tupletSpans: sb.tupletSpans.map((s) => ({ ...s })),
      index: sb.index,
    };
  }

  private instrumentFor(pitch: string): Instrument {
    return this.source.globalMetadata.instrumentMapping?.[pitch] ?? { kind: 'custom' };
  }

  private pixelPass(structure: JotStructure): ResolvedJot {
    const pxPerBeat = ((this.viewConfig.barWidth as number) * structure.densityFactor) / 4;
    const padLeft = px(this.viewConfig.barNotePaddingBeats * pxPerBeat);
    const voices = structure.voices.map((sv) => pixelVoice(sv, pxPerBeat, padLeft));
    const width = px(Math.max(0, ...voices.map((v) => v.width as number)));
    return {
      title: structure.title,
      voices,
      width,
      globalMetadata: structure.globalMetadata,
    };
  }
}

// ---------- Struct → Structural note + synthesised source ----------

function toStructuralNote(n: StructNote): StructuralNote {
  return {
    id: n.id,
    source: synthNote(n),
    pitch: n.pitch,
    modifiers: new Set(n.modifiers as Modifier[]),
    sticking: n.sticking as Sticking | undefined,
    roll: n.roll,
    beat: n.beat,
    straight: n.straight,
    duration: n.duration,
  };
}

/** Synthesise a DSL `Note` for the structural note's `source`, carrying
 *  just what playback / midi export read (pitch, modifiers, sticking,
 *  roll, sub-slot offset, MIDI note + symbolic volume). */
function synthNote(n: StructNote): Note {
  const midiFields = omitUndef({ note: n.midiNote, velocity: n.velocity, tick: n.midiTick });
  const midi = Object.keys(midiFields).length > 0 ? midiFields : undefined;
  const metadata =
    midi !== undefined || n.vol !== undefined
      ? (omitUndef({ midi, vol: n.vol as Volume | undefined }) as Metadata)
      : undefined;
  return omitUndef({
    kind: 'note',
    pitch: n.pitch,
    modifiers: n.modifiers as Modifier[],
    sticking: n.sticking as Sticking | undefined,
    roll: n.roll ? true : undefined,
    offset: n.offsetMs,
    metadata,
  }) as Note;
}

function omitUndef<T extends Record<string, unknown>>(o: T): Partial<T> {
  const out: Record<string, unknown> = {};
  for (const k of Object.keys(o)) if (o[k] !== undefined) out[k] = o[k];
  return out as Partial<T>;
}

// ---------- Cheap pixel pass ----------

function pixelVoice(sv: StructuralVoice, pxPerBeat: number, padLeft: Pixels): ResolvedVoice {
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
          ...sn,
          x: px((padLeft as number) + sn.beat * pxPerBeat),
          width: px(sn.duration * pxPerBeat),
        };
      }
      tracks[pitch] = { pitch: st.pitch, instrument: st.instrument, color: st.color, notes };
    }
    const patternSpans: PatternSpan[] = sb.patternSpans.map((s) => ({
      ...s,
      x: px((padLeft as number) + s.startBeat * pxPerBeat),
      width: px((s.endBeat - s.startBeat) * pxPerBeat),
    }));
    const tupletSpans: TupletSpan[] = sb.tupletSpans.map((s) => ({
      ...s,
      x: px((padLeft as number) + s.startBeat * pxPerBeat),
      width: px((s.endBeat - s.startBeat) * pxPerBeat),
    }));
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

// ---------- Drum beat-grid offset ----------

function applyDrumOffset(resolved: ResolvedJot, offsetBeats: number): ResolvedJot {
  return { ...resolved, voices: resolved.voices.map((v) => shiftVoice(v, offsetBeats)) };
}

function applyDrumOffsetStructure(structure: JotStructure, offsetBeats: number): JotStructure {
  return {
    ...structure,
    voices: structure.voices.map((v) => shiftStructuralVoice(v, offsetBeats)),
  };
}

function shiftStructuralVoice(voice: StructuralVoice, offsetBeats: number): StructuralVoice {
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
          destTrack = { pitch, instrument: srcTrack.instrument, color: srcTrack.color, notes: [] };
          destBar.tracks[pitch] = destTrack;
        }
        destTrack.notes.push({ ...note, beat: within, straight: isDyadic(within) });
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

  const barStart: number[] = new Array(bars.length);
  let acc = 0;
  for (let i = 0; i < bars.length; i++) {
    barStart[i] = acc;
    acc += bars[i].beats;
  }
  const total = acc;
  if (total <= 0) return voice;

  const ref = bars.find((b) => b.beats > 0);
  if (!ref) return voice;
  const pxPerBeat = (ref.width as number) / ref.beats;
  const padLeft = voice.notePadPx as number;

  const newBars: ResolvedBar[] = bars.map((b) => ({
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
          destTrack = { pitch, instrument: srcTrack.instrument, color: srcTrack.color, notes: [] };
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

  for (const bar of newBars) {
    for (const pitch of Object.keys(bar.tracks)) {
      bar.tracks[pitch].notes.sort((a, b) => a.beat - b.beat);
    }
  }

  return { ...voice, bars: newBars };
}
