import { JotState, NoteState, Value } from 'src/jot';


const eighth: NoteState = { accent: false, rest: false, value: Value.EIGHTH };
const quarter: NoteState = { accent: false, rest: false, value: Value.QUARTER };
const half: NoteState = { accent: false, rest: false, value: Value.HALF };

const eighthRest: NoteState = { accent: false, rest: true, value: Value.EIGHTH };
const quarterRest: NoteState = { accent: false, rest: true, value: Value.QUARTER };
const halfRest: NoteState = { accent: false, rest: true, value: Value.HALF };


// helper to auto-infer the track types
export const jot = <T extends string>(jot: JotState<T>) => jot;

export const rockJot = jot({
  title: 'Simple rock loop',
  trackNames: ['hihat', 'snare', 'kick'],
  loops: [
    {
      time: { count: 4, unit: Value.QUARTER },
      tracks: {
        hihat: [eighth, eighth, eighth, eighth, eighth, eighth, eighth, eighth, eighth, eighth, eighth, eighth, eighth, eighth, eighth, eighth],
        snare: [quarterRest, quarter, quarterRest, quarter, quarterRest, quarter, quarterRest, quarter],
        kick: [quarter, quarter, quarter, quarter, quarter, quarter, quarter, quarter],
      },
      repeats: 2,
    },
    {
      time: { count: 4, unit: Value.QUARTER },
      tracks: {
        hihat: [eighth, eighth, eighth, eighth, eighth, eighth, eighth, eighth],
        snare: [],
        kick: [],
      },
      repeats: 2,
    },
  ],
});
