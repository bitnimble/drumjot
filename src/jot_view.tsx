import classNames from 'classnames';
import { makeAutoObservable, observable } from 'mobx';
import { observer } from 'mobx-react';
import React from 'react';
import {
  RenderedJot,
  Loop,
  mapNoteValue,
  NoteState,
  TimeSignature,
  Track,
  ViewConfig,
  Pixels,
  Note,
} from 'src/jot';
import styles from './jot_view.css';

export function createJotView() {
  const store = new JotViewStore();
  const View = observer(() =>
    store.currentJot ? <JotView jot={store.currentJot} /> : <span>No jot loaded</span>
  );

  return { store, View };
}

export class JotViewStore {
  currentJot: RenderedJot<string> | undefined;

  constructor() {
    makeAutoObservable(this);
  }
}

const JotView = (props: { jot: RenderedJot<string> }) => {
  const { jot } = props;
  return (
    <div className={styles.jotContainer}>
      <div className={styles.tracks}>
        {jot.loops.map((l, i) => (
          <LoopView key={i} loop={l} trackNames={jot.trackNames} />
        ))}
      </div>
    </div>
  );
};

type LoopViewProps<T extends string> = { loop: Loop<T>; trackNames: T[] };
const LoopView = <T extends string>(props: LoopViewProps<T>) => {
  const { loop, trackNames } = props;
  return (
    <div
      className={styles.loop}
      style={{ left: `${loop.x}px`, width: `${loop.width * loop.repeats}px` }}
    >
      {Array(loop.repeats)
        .fill(0)
        .map((_, repetition) => (
          <div
            key={repetition}
            className={classNames(styles.repetition, repetition > 0 && styles.isRepeat)}
            style={{ width: `${loop.width}px` }}
          >
            {trackNames.map((trackName) => {
              const track = loop.tracks[trackName];
              return (
                <div key={trackName} className={styles.trackContainer}>
                  {track ? (
                    <TrackView barWidth={loop.barWidth} track={track} time={loop.time} />
                  ) : null}
                </div>
              );
            })}
          </div>
        ))}
    </div>
  );
};

const TrackView = (props: { barWidth: Pixels; track: Track; time: TimeSignature }) => {
  const { barWidth, track } = props;

  return (
    <div className={styles.track} style={{ height: `${track.height}px` }}>
      {track.bars.map((bar, i) => {
        return (
          <div
            key={i}
            className={styles.bar}
            style={{ left: `${bar.x}px`, width: `${barWidth}px` }}
          >
            {bar.notes.map((n, i) => (
              <div key={i} className={styles.noteContainer}>
                {n.rest ? null : (
                  <NoteView color={track.color} note={n} y={(track.height / 2) as Pixels} />
                )}
              </div>
            ))}
          </div>
        );
      })}
    </div>
  );
};

const NoteView = (props: { color: string; note: Note; y: Pixels }) => {
  const { color, note, y } = props;
  return (
    <div
      className={styles.note}
      style={{ backgroundColor: color, left: `${note.x}px`, top: `${y}px` }}
    ></div>
  );
};
