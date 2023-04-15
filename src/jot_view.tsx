import classNames from 'classnames';
import { makeAutoObservable } from 'mobx';
import { observer } from 'mobx-react';
import React from 'react';
import { Box, Point } from 'src/geom';
import { Loop, Note, Pixels, RenderedJot, TimeSignature, Track } from 'src/jot';
import { SelectionStore } from 'src/selection';
import styles from './jot_view.css';

export function createJotView() {
  const store = new JotViewStore();
  const selection = new SelectionStore(store);

  const onMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.button !== 0) {
      return;
    }
    e.preventDefault();
    selection.beginSelection(new Point(e.clientX, e.clientY));
  };
  const onMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    selection.moveSelection(new Point(e.clientX, e.clientY));
  };

  const View = observer(() =>
    store.currentJot ? (
      <JotView
        jot={store.currentJot}
        marquee={selection.marquee}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={selection.clearSelection}
      />
    ) : (
      <span>No jot loaded</span>
    )
  );

  return { store, View };
}

export class JotViewStore {
  currentJot: RenderedJot<string> | undefined;

  constructor() {
    makeAutoObservable(this);
  }
}

const JotView = observer(
  (props: {
    jot: RenderedJot<string>;
    marquee: Box | undefined;
    onMouseDown: (e: React.MouseEvent<HTMLDivElement>) => void;
    onMouseMove: (e: React.MouseEvent<HTMLDivElement>) => void;
    onMouseUp: () => void;
  }) => {
    const { jot, marquee, onMouseDown, onMouseMove, onMouseUp } = props;
    return (
      <div
        className={styles.jotContainer}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
      >
        <div className={styles.tracks}>
          {jot.loops.map((l, i) => (
            <LoopView key={i} loop={l} trackNames={jot.trackNames} />
          ))}
        </div>
        {marquee && (
          <div
            className={styles.marquee}
            style={{
              top: `${marquee.y}px`,
              left: `${marquee.x}px`,
              width: `${marquee.width}px`,
              height: `${marquee.height}px`,
            }}
          />
        )}
      </div>
    );
  }
);

type LoopViewProps<T extends string> = { loop: Loop<T>; trackNames: T[] };
const _LoopView = <T extends string>(props: LoopViewProps<T>) => {
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
const LoopView = observer(_LoopView);

const TrackView = observer((props: { barWidth: Pixels; track: Track; time: TimeSignature }) => {
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
});

const NoteView = observer((props: { color: string; note: Note; y: Pixels }) => {
  const { color, note, y } = props;
  return (
    <div
      className={styles.note}
      style={{ backgroundColor: color, left: `${note.x}px`, top: `${y}px` }}
    ></div>
  );
});
