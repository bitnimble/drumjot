import { makeAutoObservable } from 'mobx';
import { observer } from 'mobx-react';
import React from 'react';
import { Jot, Loop, Note, TimeSignature, Value } from 'src/schema';
import styles from './jot_view.css';

export function createJotView() {
  const store = new JotStore();
  const View = observer(() =>
    store.currentJot ? <JotView jot={store.currentJot} /> : <span>No jot loaded</span>
  );

  return { store, View };
}

export class JotStore {
  currentJot: Jot<string> | undefined;

  constructor() {
    makeAutoObservable(this);
  }
}

const JotView = (props: { jot: Jot<string> }) => {
  const { jot } = props;
  return (
    <div className={styles.jotContainer}>
      <div className={styles.tracks}>
        {jot.loops.map((l, i) => (
          <LoopView key={i} loop={l} tracks={jot.tracks} />
        ))}
      </div>
    </div>
  );
};

type LoopViewProps<T extends string> = { loop: Loop<T>; tracks: T[] };
const LoopView = <T extends string>(props: LoopViewProps<T>) => {
  const { loop, tracks } = props;
  return (
    <div className={styles.loop}>
      {tracks.map((t, i) => (
        <div key={t} className={styles.trackContainer}>
          {loop.tracks[t] ? (
            <TrackView color={colors[i]} track={loop.tracks[t]} time={loop.time} />
          ) : null}
        </div>
      ))}
    </div>
  );
};

const mapNoteValue: Record<Value, number> = {
  [Value.SIXTEENTH]: 0.25,
  [Value.EIGHTH]: 0.5,
  [Value.QUARTER]: 1,
  [Value.HALF]: 2,
  [Value.WHOLE]: 4,
};
const TrackView = (props: { color: string; track: Note[]; time: TimeSignature }) => {
  const { color, track, time } = props;

  // Segment note list into bars, based on the time signature
  const bars: Note[][] = [];
  const barLength = mapNoteValue[time.unit] * time.count; // bar length, in quarter notes
  let currentLength = 0;
  let currentBar: Note[] = [];
  for (let i = 0; i < track.length; i++) {
    const note = track[i];
    const noteLength = mapNoteValue[note.value];
    if (currentLength + noteLength === barLength) {
      currentBar.push(note);
      bars.push(currentBar);
      currentBar = [];
      currentLength = 0;
    } else if (currentLength + noteLength > barLength) {
      // Split the note into the remaining amount, plus any rests of the appropriate value.
      // The part before the split goes into the current bar...
      const before: Note[] = lengthToNotes(barLength - currentLength).map((v, i) => {
        // TODO: handle valueRhythm
        if (i === 0) {
          return { ...note, value: v };
        }
        return { accent: false, rest: true, value: v };
      });
      currentBar.push(...before);
      bars.push(currentBar);

      // ...and the overflow goes into the next bar as rests.
      const after: Note[] = lengthToNotes(currentLength + noteLength - barLength).map((v) => ({
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
  if (currentBar.length) {
    const fill = lengthToNotes(barLength - currentLength).map((v) => ({
      accent: false,
      rest: true,
      value: v,
    }));
    currentBar.push(...fill);
    bars.push(currentBar);
  }

  return (
    <div className={styles.track}>
      {bars.map((b, i) => {
        // Generate grid-template-columns column widths
        const columnFrs = b.map((n) => {
          // Use a quarter note as a "1fr" column baseline
          const fr = mapNoteValue[n.value];
          return `${fr / (n.valueRhythm || 1)}fr`;
        });
        return (
          <div
            key={i}
            className={styles.bar}
            style={{
              gridTemplateColumns: columnFrs.join(' '),
            }}
          >
            {b.map((n, i) => (
              <div key={i} className={styles.noteContainer}>
                {n.rest ? null : <NoteView color={color} note={n} />}
              </div>
            ))}
          </div>
        );
      })}
    </div>
  );
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
      if (currentValue - mapNoteValue[value] > 0) {
        noteToSubtract = value;
        break;
      }
    }

    if (noteToSubtract != null) {
      notes.push(noteToSubtract);
      currentValue -= mapNoteValue[noteToSubtract];
    } else {
      throw new Error(
        `could not find subtractable note for original length ${length}, current length ${currentValue}`
      );
    }
  }
  return notes;
};

const NoteView = (props: { color: string; note: Note }) => {
  const { color, note } = props;
  return <div className={styles.note} style={{ backgroundColor: color }}></div>;
};

const colors = [
  //
  '#9400D3',
  '#4B0082',
  '#0000FF',
  '#00FF00',
  '#FFFF00',
  '#FF7F00',
  '#FF0000',
];
