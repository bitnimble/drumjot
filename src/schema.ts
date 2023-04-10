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

export type Note = {
  accent: boolean,
  rest: boolean,
  valueRhythm?: number, // A divisor of the `value`, e.g. a triplet would be `3`.
  value: Value,
}

export type Loop<T extends string> = {
  time: TimeSignature,
  tracks: Record<T, Note[]>,
}

export type Jot<T extends string> = {
  title: string,
  tracks: T[],
  loops: Loop<T>[],
}
// helper to auto-infer the track types
export const jot = <T extends string>(jot: Jot<T>) => jot;

const eighth: Note = { accent: false, rest: false, value: Value.EIGHTH };
const quarter: Note = { accent: false, rest: false, value: Value.QUARTER };
const half: Note = { accent: false, rest: false, value: Value.HALF };

const eighthRest: Note = { accent: false, rest: true, value: Value.EIGHTH };
const quarterRest: Note = { accent: false, rest: true, value: Value.QUARTER };
const halfRest: Note = { accent: false, rest: true, value: Value.HALF };

export const rockJot = jot({
  title: 'Simple rock loop',
  tracks: ['hihat', 'snare', 'kick'],
  loops: [
    {
      time: { count: 4, unit: Value.QUARTER },
      tracks: {
        hihat: [eighth, eighth, eighth, eighth, eighth, eighth, eighth, eighth, eighth, eighth, eighth, eighth, eighth, eighth, eighth, eighth],
        snare: [quarterRest, quarter, quarterRest, quarter, quarterRest, quarter, quarterRest, quarter],
        kick: [quarter, quarter, quarter, quarter, quarter, quarter, quarter, quarter],
      }
    },
  ],
});
