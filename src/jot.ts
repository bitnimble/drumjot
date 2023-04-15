import { createTransformer } from 'mobx-utils';


export const enum Value {
  SIXTEENTH,
  EIGHTH,
  QUARTER,
  HALF,
  WHOLE
}

export type TimeSignature = {
  count: number,
  unit: Value,
}

export type NoteState = {
  accent: boolean,
  rest: boolean,
  valueRhythm?: number, // A divisor of the `value`, e.g. a triplet would be `3`.
  value: Value,
}

export type LoopState<T extends string> = {
  time: TimeSignature,
  tracks: Record<T, NoteState[]>,
  repeats: number,
}

export type JotState<T extends string> = {
  title: string,
  tracks: T[],
  loops: LoopState<T>[],
}

const _mapNoteValue: Record<Value, number> = {
  [Value.SIXTEENTH]: 0.25,
  [Value.EIGHTH]: 0.5,
  [Value.QUARTER]: 1,
  [Value.HALF]: 2,
  [Value.WHOLE]: 4,
};
export const mapNoteValue = (v: Value) => _mapNoteValue[v];

export class Jot<T extends string> {
  constructor(private state: JotState<T>) {
  }

  get tracks(): T[] {
    return this.state.tracks;
  }

  get loops() {
    return this.state.loops.map(l => this.groupLoopByBar(l));
  }

  private groupLoopByBar = createTransformer((l: LoopState<T>) => {
    const { time, tracks } = l;

    return {
      time,
      tracks: mapObject(tracks, (t, notes) => {
        // Segment note list into bars, based on the time signature
        const bars: NoteState[][] = [];
        const barLength = mapNoteValue(time.unit) * time.count; // bar length, in quarter notes
        let currentLength = 0;
        let currentBar: NoteState[] = [];
        for (let i = 0; i < notes.length; i++) {
          const note = notes[i];
          const noteLength = mapNoteValue(note.value);
          if (currentLength + noteLength === barLength) {
            currentBar.push(note);
            bars.push(currentBar);
            currentBar = [];
            currentLength = 0;
          } else if (currentLength + noteLength > barLength) {
            // Split the note into the remaining amount, plus any rests of the appropriate value.
            // The part before the split goes into the current bar...
            const before: NoteState[] = lengthToNotes(barLength - currentLength).map((v, i) => {
              // TODO: handle valueRhythm
              if (i === 0) {
                return { ...note, value: v };
              }
              return { accent: false, rest: true, value: v };
            });
            currentBar.push(...before);
            bars.push(currentBar);

            // ...and the overflow goes into the next bar as rests.
            const after: NoteState[] = lengthToNotes(currentLength + noteLength - barLength).map((v) => ({
              accent: false,
              rest: true,
              value: v,
            }));
            currentBar = [...after];
            currentLength = currentLength + noteLength - barLength;
          } else {
            currentBar.push(note);
            currentLength += noteLength;
          }
        }

        // If there's a leftover bar, fill it with rests and then append.
        if (currentBar.length || (currentLength === 0 && notes.length === 0)) {
          const fill = lengthToNotes(barLength - currentLength).map((v) => ({
            accent: false,
            rest: true,
            value: v,
          }));
          currentBar.push(...fill);
          bars.push(currentBar);
        }

        return { bars };
      }),
      repeats: l.repeats,
    }
  });
}

export type Loop<T extends string> = Jot<T>['loops'][number];
export type Track = Loop<string>['tracks'][number];
export type Bar = Track['bars'][number];

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
}
