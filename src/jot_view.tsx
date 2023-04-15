import classNames from 'classnames';
import { makeAutoObservable } from 'mobx';
import { observer } from 'mobx-react';
import React from 'react';
import { Jot, Loop, mapNoteValue, NoteState, TimeSignature, Track } from 'src/jot';
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
      {Array(loop.repeats)
        .fill(0)
        .map((_, repetition) => (
          <div
            key={repetition}
            className={classNames(styles.repetition, repetition > 0 && styles.isRepeat)}
          >
            {tracks.map((t, i) => (
              <div key={t} className={styles.trackContainer}>
                {loop.tracks[t] ? (
                  <TrackView color={colors[i]} track={loop.tracks[t]} time={loop.time} />
                ) : null}
              </div>
            ))}
          </div>
        ))}
    </div>
  );
};

const TrackView = (props: { color: string; track: Track; time: TimeSignature }) => {
  const { color, track } = props;

  return (
    <div className={styles.track}>
      {track.bars.map((b, i) => {
        // Generate grid-template-columns column widths
        const columnFrs = b.map((n) => {
          // Use a quarter note as a "1fr" column baseline
          const fr = mapNoteValue(n.value);
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

const NoteView = (props: { color: string; note: NoteState }) => {
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
