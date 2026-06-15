/**
 * Structural view-model over the reactive document: exposes the store-native
 * `Struct*` voices (from {@link StructureStore}) with the interactive drum
 * beat-grid offset applied, plus the layout scale (`pxPerBeat`) and the
 * per-pitch row data the mixer reads. Colours come from {@link PaletteStore}
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
import { ViewConfig } from 'src/jot_view/viewport/view_config';
import type { LaidOutJot } from 'src/jot_view/playback/timeline';
import {
  type StructBar,
  type StructTrack,
  type StructVoice,
  type StructureStore,
} from './structure_store';
import type { PaletteStore } from 'src/jot_view/palette/palette_store';
import type { LayoutStore } from 'src/jot_view/viewport/layout_store';

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
      voices: computed,
      pxPerBeat: computed,
      voiceBeats: computed,
      primaryVoice: computed,
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

  /** The store-native `Struct*` voices, drum-offset applied. The canonical
   *  beat-addressed structure consumers read. */
  get voices(): StructVoice[] {
    const base = this.structureStore.voices;
    const eff = this.effectiveDrumOffsetBeats;
    return eff === 0 ? base : applyOffsetToStructVoices(base, eff);
  }

  get pxPerBeat(): number {
    return ((this.viewConfig.barWidth as number) * this.layoutStore.densityFactor) / 4;
  }

  get voiceBeats(): number {
    const bars = this.voices[0]?.bars;
    if (!bars) return 0;
    let total = 0;
    for (const b of bars) total += b.beats;
    return total;
  }

  get primaryVoice(): StructVoice | undefined {
    return this.voices[0];
  }

  /** Instrument for a pitch from the global mapping. */
  private instrumentFor(pitch: string): Instrument {
    return this.source.globalMetadata.instrumentMapping?.[pitch] ?? { kind: 'custom' };
  }

  barsForPitch = computedFn(
    (
      pitch: string
    ): {
      bars: readonly StructBar[];
      voiceBeats: number;
      leadInBarsBeats: number;
      barBeatStart: readonly number[];
      startBeats: readonly number[];
      pitchColor: string;
      instrumentName: string | undefined;
    } => {
      const voice = this.voices[0];
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
      // Colour + instrument are jot-wide functions of the pitch (palette
      // slot + the instrument mapping), no longer per-track.
      const pitchColor = this.paletteStore.colorForPitch(pitch);
      const instrumentName = this.instrumentFor(pitch).name;
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
}

// ---------- Drum beat-grid offset ----------

function applyOffsetToStructVoices(voices: StructVoice[], offsetBeats: number): StructVoice[] {
  return voices.map((v) => shiftStructVoice(v, offsetBeats));
}

/** Slide every note across the fixed bar grid by `offsetBeats`, re-homing
 *  notes that cross a barline. Pattern/tuplet spans are cleared (their
 *  geometry no longer matches the shifted notes). */
function shiftStructVoice(voice: StructVoice, offsetBeats: number): StructVoice {
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
    for (const pitch of Object.keys(srcBar.tracks)) {
      const srcTrack = srcBar.tracks[pitch];
      for (const note of srcTrack.notes) {
        const newAbs = barStart[i] + note.beat + offsetBeats;
        if (newAbs < 0 || newAbs >= total) continue;
        const j = locate(newAbs);
        if (j < 0) continue;
        const within = newAbs - barStart[j];
        const destBar = newBars[j];
        let destTrack: StructTrack = destBar.tracks[pitch];
        if (!destTrack) {
          destTrack = { pitch, notes: [] };
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
