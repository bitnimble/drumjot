import { computed, makeObservable } from 'mobx';
import { isDyadic } from 'src/jot/pattern_expansion';
import type { Element, Jot, PatternDef } from 'src/schema/schema';

/**
 * Beat-addressed score structure derived from the reactive Jot's element
 * tree, the grouping/indexing layer that used to live in `JotStructure`.
 *
 * The reactive model is a hierarchy of `note | group | pattern` elements
 * with coordinates RELATIVE to their container; this store walks it,
 * converting to bar-absolute beats (a group of `duration` D whose children
 * span an internal length L scales them by D/L, that's the tuplet ratio),
 * and flattens it into per-bar, per-pitch tracks plus tuplet + pattern
 * spans. All MobX computeds off the observable jot, so an edit reflows it.
 *
 * Pixels, palette colours, tempo and the drum-offset transform are NOT
 * here, they live in their own domain stores and read this structure.
 */

const EPS = 1e-9;

export type StructNote = {
  id: string;
  pitch: string;
  /** Quarter-note beats from the owning bar's downbeat (absolute). */
  beat: number;
  duration: number;
  modifiers: readonly string[];
  sticking?: string;
  roll: boolean;
  /** Onset lands on the binary (dyadic) grid. `isDyadic(beat)`. */
  straight: boolean;
};

export type StructTrack = { pitch: string; notes: StructNote[] };

export type StructPatternSpan = {
  name: string;
  startBeat: number;
  endBeat: number;
  pitches: ReadonlySet<string>;
  colorIndex: number;
};

/** A group whose children don't fill its own duration, drawn as a bracket
 *  with `count` above it. */
export type StructTupletSpan = {
  count: number;
  startBeat: number;
  endBeat: number;
};

export type StructBar = {
  id: string;
  index: number;
  beats: number;
  tsCount: number;
  tsUnit: number;
  anacrusis: boolean;
  tracks: Record<string, StructTrack>;
  patternSpans: StructPatternSpan[];
  tupletSpans: StructTupletSpan[];
};

export type StructVoice = {
  id: string;
  name?: string;
  bars: StructBar[];
  pitches: string[];
};

export const PRIMARY_VOICE = 'primary';

type BarContents = {
  notes: StructNote[];
  tuplets: StructTupletSpan[];
  patternSpans: StructPatternSpan[];
};

export class StructureStore {
  constructor(private readonly getJot: () => Jot | undefined) {
    makeObservable(this, { voices: computed });
  }

  get voices(): StructVoice[] {
    const jot = this.getJot();
    if (!jot) return [];

    const tops = [...jot.elements.values()] as Element[];
    const declared = [...jot.voices.values()];
    const single = declared.length === 0;
    const voiceList = single ? [{ id: PRIMARY_VOICE, name: undefined }] : declared;

    const mappedOrder = [...jot.instruments.keys()];
    const leadBars = jot.leadBars ?? 0;
    const barList = [...jot.bars];

    // Pattern name -> colour slot, shared across the whole jot.
    const colorByName = new Map<string, number>();

    return voiceList.map((voice) => {
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

        const mine = tops.filter(
          (el) => el.barId === bar.id && (single || el.voiceId === voice.id)
        );
        const out: BarContents = { notes: [], tuplets: [], patternSpans: [] };
        for (const el of mine) {
          // Top-level coordinates are already bar-relative.
          flattenInto(el, el.beat, el.duration, jot, colorByName, out);
        }

        const tracks: Record<string, StructTrack> = {};
        for (const note of out.notes) {
          let track = tracks[note.pitch];
          if (!track) {
            track = { pitch: note.pitch, notes: [] };
            tracks[note.pitch] = track;
          }
          track.notes.push(note);
        }
        for (const pitch of Object.keys(tracks)) {
          tracks[pitch].notes.sort((a, b) => a.beat - b.beat);
        }

        const beats = anacrusis ? anacrusisBeats(mine) : (bar.tsCount * 4) / bar.tsUnit;
        bars.push({
          id: bar.id,
          index,
          beats,
          tsCount: bar.tsCount,
          tsUnit: bar.tsUnit,
          anacrusis,
          tracks,
          patternSpans: out.patternSpans,
          tupletSpans: out.tuplets,
        });
      }
      return { id: voice.id, name: voice.name, bars, pitches: orderPitches(bars, mappedOrder) };
    });
  }
}

/**
 * Recursively flatten an element positioned at bar-absolute `[absBeat,
 * absBeat+absDur)` into notes (with absolute beats), tuplet spans (groups
 * that don't fill their duration) and pattern spans (pattern usages).
 */
function flattenInto(
  el: Element,
  absBeat: number,
  absDur: number,
  jot: Jot,
  colorByName: Map<string, number>,
  out: BarContents
): void {
  if (el.kind === 'note') {
    out.notes.push({
      id: el.id,
      pitch: el.pitch,
      beat: absBeat,
      duration: absDur,
      modifiers: el.modifiers,
      sticking: el.sticking,
      roll: el.roll === true,
      straight: isDyadic(absBeat),
    });
    return;
  }

  if (el.kind === 'group') {
    const children = [...el.children.values()] as Element[];
    const internalLen = naturalSpan(children);
    if (internalLen > EPS && Math.abs(internalLen - el.duration) > EPS) {
      out.tuplets.push({ count: children.length, startBeat: absBeat, endBeat: absBeat + absDur });
    }
    const scale = internalLen > EPS ? absDur / internalLen : 1;
    for (const child of children) {
      flattenInto(child, absBeat + child.beat * scale, child.duration * scale, jot, colorByName, out);
    }
    return;
  }

  // pattern: instantiate the referenced definition, scaled to this usage.
  const def = jot.patterns.get(el.patternId) as PatternDef | undefined;
  if (!def) return;
  const pitches = new Set<string>();
  const body = [...def.body.values()] as Element[];
  collectPitches(body, jot, pitches);
  let colorIndex = colorByName.get(def.name);
  if (colorIndex === undefined) {
    colorIndex = colorByName.size;
    colorByName.set(def.name, colorIndex);
  }
  out.patternSpans.push({
    name: def.name,
    startBeat: absBeat,
    endBeat: absBeat + absDur,
    pitches,
    colorIndex,
  });
  const internalLen = naturalSpan(body);
  const scale = internalLen > EPS ? absDur / internalLen : 1;
  for (const child of body) {
    flattenInto(child, absBeat + child.beat * scale, child.duration * scale, jot, colorByName, out);
  }
}

/** Natural internal length of a container's children (latest onset end). */
function naturalSpan(children: readonly Element[]): number {
  let end = 0;
  for (const child of children) end = Math.max(end, child.beat + child.duration);
  return end;
}

/** Anacrusis length: sized to its top-level content. */
function anacrusisBeats(elements: readonly Element[]): number {
  let end = 0;
  for (const el of elements) end = Math.max(end, el.beat + el.duration);
  return end;
}

/** Pitches a subtree plays, for a pattern span's lane set. */
function collectPitches(elements: readonly Element[], jot: Jot, into: Set<string>): void {
  for (const el of elements) {
    if (el.kind === 'note') into.add(el.pitch);
    else if (el.kind === 'group') collectPitches([...el.children.values()] as Element[], jot, into);
    else {
      const def = jot.patterns.get(el.patternId) as PatternDef | undefined;
      if (def) collectPitches([...def.body.values()] as Element[], jot, into);
    }
  }
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
