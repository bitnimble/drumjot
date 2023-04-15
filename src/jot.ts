import { makeAutoObservable } from 'mobx';
import { computedFn } from 'mobx-utils';

// use an explicit pixels type to make sure we don't mix up pixels vs note value widths/positions
export type Pixels = number & { __pixels: never };

export const enum Value {
  SIXTEENTH,
  EIGHTH,
  QUARTER,
  HALF,
  WHOLE,
}

export type TimeSignature = {
  count: number;
  unit: Value;
};

export type NoteState = {
  accent: boolean;
  rest: boolean;
  valueRhythm?: number; // A divisor of the `value`, e.g. a triplet would be `3`.
  value: Value;
};

export type LoopState<T extends string> = {
  time: TimeSignature;
  tracks: Record<T, NoteState[]>;
  repeats: number;
};

export type JotState<T extends string> = {
  title: string;
  trackNames: T[];
  loops: LoopState<T>[];
};

const _mapNoteValue: Record<Value, number> = {
  [Value.SIXTEENTH]: 0.25,
  [Value.EIGHTH]: 0.5,
  [Value.QUARTER]: 1,
  [Value.HALF]: 2,
  [Value.WHOLE]: 4,
};
/** @returns the number of quarter notes that the specified value is */
export const mapNoteValue = (v: Value) => _mapNoteValue[v];

const atom = 8; // pixels
export class ViewConfig {
  /**
   * [         quarterValueWidth      ]
   * [ |-----| |-----| |-----| |-----|]
   * [o       o       o       o       ]
   */
  quarterValueGap = (atom * 10) as Pixels;
  noteWidth = (atom * 2) as Pixels;
  trackHeight = (atom * 6) as Pixels;
  noteColors = ['#9400D3', '#4B0082', '#0000FF', '#00FF00', '#FFFF00', '#FF7F00', '#FF0000'];

  constructor() {
    makeAutoObservable(this);
  }
}

export class RenderedJot<T extends string> {
  private viewConfig = new ViewConfig();

  constructor(private state: JotState<T>, viewConfigDefaults?: ViewConfig) {
    makeAutoObservable(this);
    if (viewConfigDefaults) {
      this.viewConfig = viewConfigDefaults;
    }
  }

  get trackNames(): T[] {
    return this.state.trackNames;
  }

  get loops() {
    let currentWidth = 0 as Pixels;
    return this.state.loops.map((l) => {
      // TODO: separate loop/bar layout computation from bar contents, so that content updates do not
      // trigger a relayout
      const loop = this.groupLoopByBar(l, currentWidth);
      currentWidth = (currentWidth + loop.width * loop.repeats) as Pixels;
      return loop;
    });
  }

  get trackColors() {
    return Object.fromEntries(
      this.state.trackNames.map((name, i) => [name, this.viewConfig.noteColors[i]])
    ) as Record<T, string>;
  }

  private groupLoopByBar = computedFn((l: LoopState<T>, offset: Pixels) => {
    const { time, tracks } = l;

    const barLength = mapNoteValue(time.unit) * time.count; // bar length, in quarter notes
    const barWidthPx = (barLength * this.viewConfig.quarterValueGap + 1) as Pixels; // add 1px for the left border
    let maxBars = 0;
    const groupedTracks = mapObject(tracks, (t, notes) => {
      if (!this.state.trackNames.includes(t)) {
        throw new Error(`Track ${t} was found in a loop, but not present in trackNames`);
      }
      // Segment note list into bars, based on the time signature
      const bars: Bar[] = [];

      // TODO: separate loop/bar layout computation from bar contents, so that content updates do not
      // trigger a relayout
      let barX = 0 as Pixels;
      let currentLength = 0;
      let currentBarNotes: Note[] = [];
      let noteX = 0 as Pixels;
      for (let i = 0; i < notes.length; i++) {
        const note = notes[i];
        const noteLength = mapNoteValue(note.value);
        if (currentLength + noteLength === barLength) {
          // This note completed the bar perfectly; push a new bar
          currentBarNotes.push({
            ...note,
            x: noteX,
          });
          bars.push({
            x: barX,
            notes: currentBarNotes,
          });
          currentBarNotes = [];
          currentLength = 0;
          barX = (barX + barWidthPx) as Pixels;
          noteX = 0 as Pixels;
        } else if (currentLength + noteLength > barLength) {
          // Split the note into the remaining amount, plus any rests of the appropriate value.
          // The part before the split goes into the current bar...
          const before: Note[] = lengthToNotes(barLength - currentLength).map((v, i) => {
            // TODO: handle valueRhythm
            const subnote =
              i === 0
                ? { ...note, value: v, x: noteX }
                : { accent: false, rest: true, value: v, x: noteX };
            noteX = (noteX + mapNoteValue(v) * this.viewConfig.quarterValueGap) as Pixels;
            return subnote;
          });
          currentBarNotes.push(...before);
          bars.push({
            x: barX,
            notes: currentBarNotes,
          });

          // ...and the overflow goes into the next bar as rests.
          noteX = 0 as Pixels;
          const after: Note[] = lengthToNotes(currentLength + noteLength - barLength).map((v) => {
            const subnote = {
              accent: false,
              rest: true,
              value: v,
              x: noteX,
            };
            noteX = (noteX + mapNoteValue(v) * this.viewConfig.quarterValueGap) as Pixels;
            return subnote;
          });
          currentBarNotes = [...after];
          currentLength = currentLength + noteLength - barLength;
          barX = (barX + barWidthPx) as Pixels;
        } else {
          // Keep adding to the current bar
          currentBarNotes.push({
            ...note,
            x: noteX,
          });
          currentLength += noteLength;
          noteX = (noteX + noteLength * this.viewConfig.quarterValueGap) as Pixels;
        }
      }

      // If there's remainder notes, fill it with rests and then push the bar.
      if (currentBarNotes.length || (currentLength === 0 && notes.length === 0)) {
        const fill = lengthToNotes(barLength - currentLength).map((v) => {
          const subnote = {
            accent: false,
            rest: true,
            value: v,
            x: noteX,
          };
          noteX = (noteX + mapNoteValue(v) * this.viewConfig.quarterValueGap) as Pixels;
          return subnote;
        });
        currentBarNotes.push(...fill);
        bars.push({
          x: barX,
          notes: currentBarNotes,
        });
      }

      if (bars.length > maxBars) {
        maxBars = bars.length;
      }
      return { color: this.trackColors[t], bars, height: this.viewConfig.trackHeight };
    });

    // Align loop length across the tracks

    let noteX = 0 as Pixels;
    const emptyBarNotes = lengthToNotes(barLength).map((v) => {
      const subnote = {
        accent: false,
        rest: true,
        value: v,
        x: noteX,
      };
      noteX = (noteX + mapNoteValue(v) * this.viewConfig.quarterValueGap) as Pixels;
      return subnote;
    });
    const alignedTracks = mapObject(groupedTracks, (t, track) => {
      const remainder = maxBars - track.bars.length;
      let barX = (track.bars[track.bars.length - 1].x + barWidthPx) as Pixels;
      for (let i = 0; i < remainder; i++) {
        track.bars.push({
          x: barX,
          notes: structuredClone(emptyBarNotes), // we want actual new instances of NoteState each time
        });
        barX = (barX + barWidthPx) as Pixels;
      }

      return track;
    });

    return {
      time,
      x: offset,
      width: (maxBars * barWidthPx) as Pixels,
      barWidth: barWidthPx,
      tracks: alignedTracks,
      repeats: l.repeats,
    };
  });
}

export type Loop<T extends string> = RenderedJot<T>['loops'][number];
export type Track = Loop<string>['tracks'][number];
export type Bar = {
  x: Pixels;
  notes: Note[];
};
export type Note = NoteState & {
  x: Pixels;
};

/**
 * Calculates the minimal number of notes to satisfy the specified note length (in quarter notes)
 */
const lengthToNotes = (length: number) => {
  // We only support up to 16th notes. If we're not a multiple of that then we can't continue.
  // TODO: support triplets / polyrhythms etc.
  if (!Number.isInteger(length / 0.25)) {
    throw new Error(`could not find note composition for length ${length}`);
  }
  // recursively subtract the largest possible notes.
  const notes = [];
  let currentValue = length;
  const values = [Value.WHOLE, Value.HALF, Value.QUARTER, Value.EIGHTH, Value.SIXTEENTH];
  while (true) {
    if (currentValue === 0) {
      break;
    }

    let noteToSubtract: Value | undefined;
    // Find largest subtractable note
    for (const value of values) {
      if (currentValue - mapNoteValue(value) >= 0) {
        noteToSubtract = value;
        break;
      }
    }

    if (noteToSubtract != null) {
      notes.push(noteToSubtract);
      currentValue -= mapNoteValue(noteToSubtract);
    } else {
      throw new Error(
        `could not find subtractable note for original length ${length}, current length ${currentValue}`
      );
    }
  }
  return notes;
};

const mapObject = <K extends string | number, V, V2>(o: Record<K, V>, map: (k: K, v: V) => V2) => {
  const o2: Record<K, V2> = {} as any;
  for (const [k, v] of Object.entries(o)) {
    o2[k as K] = map(k as K, v as V);
  }
  return o2;
};
