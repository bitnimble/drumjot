import { computed, makeObservable } from 'mobx';
import { isDyadic } from 'src/jot/pattern_expansion';
import type { Jot, Note, Voice } from 'src/schema/schema';

/**
 * Beat-addressed score structure derived from the reactive Jot, the
 * grouping/indexing layer that used to live in `JotStructure`
 * (`structureForJot`). Because the reactive model is already flat
 * (notes carry `barId`/`beat`/`pitch`/`voiceId`), this is pure grouping +
 * ordering: no pattern expansion or element-tree walking. Everything is a
 * MobX computed off the observable jot, so an edit reflows it.
 *
 * Pixels, palette colours, tempo and the drum-offset transform are NOT
 * here, they live in their own domain stores (viewport / mixer / tempo /
 * playback) and read this structure as their input.
 */

export type StructNote = {
  id: string;
  pitch: string;
  /** Quarter-note beats from the owning bar's downbeat. */
  beat: number;
  duration: number;
  modifiers: readonly string[];
  sticking?: string;
  roll: boolean;
  /** Onset lands on the binary (dyadic) grid, i.e. not a tuplet/swing
   *  position. `isDyadic(beat)`; the renderer flags non-straight notes. */
  straight: boolean;
  voiceId?: string;
  patternId?: string;
};

export type StructTrack = { pitch: string; notes: StructNote[] };

export type StructPatternSpan = {
  name: string;
  startBeat: number;
  endBeat: number;
  pitches: ReadonlySet<string>;
  /** Stable 0-based colour slot, assigned per pattern name in first-seen
   *  order across the whole jot so every usage shares a colour. */
  colorIndex: number;
};

export type StructBar = {
  id: string;
  /** Renderer numbering: anacrusis = 0, pre-drum lead-in = negative,
   *  first drum bar = 1. */
  index: number;
  /** Bar length in quarter-note beats. */
  beats: number;
  tsCount: number;
  tsUnit: number;
  anacrusis: boolean;
  tracks: Record<string, StructTrack>;
  patternSpans: StructPatternSpan[];
};

export type StructVoice = {
  id: string;
  name?: string;
  bars: StructBar[];
  /** Lane order: mapped pitches first (instrument-declaration order),
   *  then any unmapped pitches in first-seen order. */
  pitches: string[];
};

/** Id of the implicit voice notes fall into when no `||` voice is declared. */
export const PRIMARY_VOICE = 'primary';

export class StructureStore {
  constructor(private readonly getJot: () => Jot | undefined) {
    // Explicit `makeObservable` (not auto): `getJot` must stay a plain
    // closure, if it were wrapped as an action it would run untracked and
    // the computed below wouldn't depend on the jot it reads.
    makeObservable(this, { voices: computed });
  }

  get voices(): StructVoice[] {
    const jot = this.getJot();
    if (!jot) return [];

    // One pass: bucket notes by owning bar.
    const notesByBar = new Map<string, Note[]>();
    for (const note of jot.notes.values()) {
      const arr = notesByBar.get(note.barId);
      if (arr) arr.push(note);
      else notesByBar.set(note.barId, [note]);
    }

    const declared = [...jot.voices.values()];
    const single = declared.length === 0;
    const voices: Voice[] = single ? [{ id: PRIMARY_VOICE }] : declared;

    const mappedOrder = [...jot.instruments.keys()];
    const leadBars = jot.leadBars ?? 0;
    const barList = [...jot.bars];

    // Pattern name -> colour slot, shared across the whole jot.
    const colorByName = new Map<string, number>();
    const patternNameFor = (pid: string | undefined): string | undefined =>
      pid === undefined ? undefined : jot.patternInstances.get(pid)?.patternName;

    return voices.map((voice) => {
      const bars: StructBar[] = [];
      let gridPos = 0;
      for (const bar of barList) {
        const anacrusis = bar.anacrusis === true;
        const index = anacrusis
          ? 0
          : gridPos < leadBars
            ? gridPos - leadBars
            : gridPos - leadBars + 1;
        if (!anacrusis) gridPos++;

        const inBar = notesByBar.get(bar.id) ?? [];
        const mine = single ? inBar : inBar.filter((n) => n.voiceId === voice.id);

        const tracks: Record<string, StructTrack> = {};
        for (const n of mine) {
          let track = tracks[n.pitch];
          if (!track) {
            track = { pitch: n.pitch, notes: [] };
            tracks[n.pitch] = track;
          }
          track.notes.push(toStructNote(n));
        }
        for (const pitch of Object.keys(tracks)) {
          tracks[pitch].notes.sort((a, b) => a.beat - b.beat);
        }

        const beats = anacrusis ? anacrusisBeats(mine) : (bar.tsCount * 4) / bar.tsUnit;
        const patternSpans = buildPatternSpans(mine, patternNameFor, colorByName);

        bars.push({
          id: bar.id,
          index,
          beats,
          tsCount: bar.tsCount,
          tsUnit: bar.tsUnit,
          anacrusis,
          tracks,
          patternSpans,
        });
      }
      return { id: voice.id, name: voice.name, bars, pitches: orderPitches(bars, mappedOrder) };
    });
  }
}

function toStructNote(n: Note): StructNote {
  return {
    id: n.id,
    pitch: n.pitch,
    beat: n.beat,
    duration: n.duration,
    modifiers: n.modifiers,
    sticking: n.sticking,
    roll: n.roll === true,
    straight: isDyadic(n.beat),
    voiceId: n.voiceId,
    patternId: n.patternId,
  };
}

/** Anacrusis length: sized to its content (the latest onset's end). */
function anacrusisBeats(notes: readonly Note[]): number {
  let end = 0;
  for (const n of notes) end = Math.max(end, n.beat + n.duration);
  return end;
}

function buildPatternSpans(
  notes: readonly Note[],
  patternNameFor: (pid: string | undefined) => string | undefined,
  colorByName: Map<string, number>
): StructPatternSpan[] {
  const byInstance = new Map<string, Note[]>();
  for (const n of notes) {
    if (n.patternId === undefined) continue;
    const arr = byInstance.get(n.patternId);
    if (arr) arr.push(n);
    else byInstance.set(n.patternId, [n]);
  }
  const spans: StructPatternSpan[] = [];
  for (const group of byInstance.values()) {
    const name = patternNameFor(group[0].patternId);
    if (name === undefined) continue;
    let colorIndex = colorByName.get(name);
    if (colorIndex === undefined) {
      colorIndex = colorByName.size;
      colorByName.set(name, colorIndex);
    }
    let startBeat = Infinity;
    let endBeat = -Infinity;
    const pitches = new Set<string>();
    for (const n of group) {
      startBeat = Math.min(startBeat, n.beat);
      endBeat = Math.max(endBeat, n.beat);
      pitches.add(n.pitch);
    }
    spans.push({ name, startBeat, endBeat, pitches, colorIndex });
  }
  return spans;
}

function orderPitches(bars: readonly StructBar[], mappedOrder: readonly string[]): string[] {
  const seen: string[] = [];
  for (const bar of bars) {
    for (const pitch of Object.keys(bar.tracks)) {
      if (!seen.includes(pitch)) seen.push(pitch);
    }
  }
  const out: string[] = [];
  for (const p of mappedOrder) {
    if (seen.includes(p) && !out.includes(p)) out.push(p);
  }
  for (const p of seen) {
    if (!out.includes(p)) out.push(p);
  }
  return out;
}
