import classNames from 'classnames';
import { makeAutoObservable } from 'mobx';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { Box, Point } from 'src/geom';
import {
  Pixels,
  RenderedJot,
  ResolvedBar,
  ResolvedJot,
  ResolvedNote,
  ResolvedVoice,
  ViewConfig,
  px,
} from 'src/jot';
import { SelectionStore } from 'src/selection';
import styles from './jot_view.module.css';

export class JotViewStore {
  currentJot: RenderedJot | undefined;

  constructor() {
    makeAutoObservable(this);
  }

  setJot(jot: RenderedJot | undefined) {
    this.currentJot = jot;
  }
}

type CreateJotViewResult = {
  store: JotViewStore;
  View: React.FC;
};

export function createJotView(): CreateJotViewResult {
  const store = new JotViewStore();
  const selection = new SelectionStore(store);

  const onMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    selection.beginSelection(new Point(e.clientX, e.clientY));
  };
  const onMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    selection.moveSelection(new Point(e.clientX, e.clientY));
  };

  const View: React.FC = observer(() => {
    const jot = store.currentJot;
    if (!jot) return <div className={styles.empty}>No jot loaded</div>;
    return (
      <JotView
        jot={jot}
        marquee={selection.marquee}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={selection.endSelection}
      />
    );
  });

  return { store, View };
}

type JotViewProps = {
  jot: RenderedJot;
  marquee: Box | undefined;
  onMouseDown: (e: React.MouseEvent<HTMLDivElement>) => void;
  onMouseMove: (e: React.MouseEvent<HTMLDivElement>) => void;
  onMouseUp: () => void;
};

const JotView = observer((props: JotViewProps) => {
  const { jot, marquee, onMouseDown, onMouseMove, onMouseUp } = props;
  const resolved = jot.resolved;
  const config = jot.config;

  return (
    <div
      className={styles.jotContainer}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
    >
      <h2 className={styles.title}>{resolved.title || 'Untitled jot'}</h2>
      <p className={styles.subtitle}>{formatSubtitle(resolved)}</p>
      <Legend jot={resolved} />
      <div className={styles.voices}>
        {resolved.voices.map((voice, i) => (
          <VoiceView key={i} voice={voice} config={config} index={i} totalVoices={resolved.voices.length} />
        ))}
      </div>
      {marquee && (
        <div
          className={styles.marquee}
          style={{
            top: marquee.y,
            left: marquee.x,
            width: marquee.width,
            height: marquee.height,
          }}
        />
      )}
    </div>
  );
});

function formatSubtitle(jot: ResolvedJot): string {
  const parts: string[] = [];
  const { bpm, time, vol } = jot.globalMetadata;
  if (typeof bpm === 'number') parts.push(`${bpm} bpm`);
  else if (bpm) parts.push(`${bpm.start ?? '?'}-${bpm.end} bpm`);
  if (time) parts.push(`${time.count}/${time.unit}`);
  if (typeof vol === 'string') parts.push(vol);
  else if (vol) parts.push(`${vol.start ?? '?'} -> ${vol.end}`);
  return parts.join('  -  ');
}

const Legend = observer(({ jot }: { jot: ResolvedJot }) => {
  // Aggregate unique pitches across all voices, in first-seen order.
  const seen = new Map<string, { color: string; name?: string }>();
  for (const voice of jot.voices) {
    for (const bar of voice.bars) {
      for (const pitch of Object.keys(bar.tracks)) {
        if (!seen.has(pitch)) {
          const track = bar.tracks[pitch];
          seen.set(pitch, { color: track.color, name: track.mapping.name });
        }
      }
    }
  }
  if (seen.size === 0) return null;
  return (
    <div className={styles.legend}>
      {Array.from(seen.entries()).map(([pitch, info]) => (
        <span key={pitch} className={styles.legendChip}>
          <span className={styles.legendSwatch} style={{ background: info.color }} />
          <strong>{pitch}</strong>
          {info.name ? <span>{info.name}</span> : null}
        </span>
      ))}
    </div>
  );
});

const VoiceView = observer(
  ({
    voice,
    config,
    index,
    totalVoices,
  }: {
    voice: ResolvedVoice;
    config: ViewConfig;
    index: number;
    totalVoices: number;
  }) => {
    const pitches = voice.pitches;
    const staffHeight = px(pitches.length * config.trackHeight);

    return (
      <div className={styles.voice}>
        {totalVoices > 1 && (
          <div className={styles.voiceLabel}>Voice {index + 1}</div>
        )}
        <div className={styles.voiceStaff} style={{ height: staffHeight, width: voice.width }}>
          {voice.bars.map((bar, i) => (
            <BarView
              key={i}
              bar={bar}
              pitches={pitches}
              config={config}
              isAnacrusis={bar.index === 0}
            />
          ))}
        </div>
      </div>
    );
  }
);

const BarView = observer(
  ({
    bar,
    pitches,
    config,
    isAnacrusis,
  }: {
    bar: ResolvedBar;
    pitches: string[];
    config: ViewConfig;
    isAnacrusis: boolean;
  }) => {
    return (
      <div
        className={classNames(styles.bar, isAnacrusis && styles.barAnacrusis)}
        style={{ width: bar.width, height: pitches.length * config.trackHeight }}
        title={`Bar ${bar.index} - ${bar.time.count}/${bar.time.unit}`}
      >
        {pitches.map((pitch) => {
          const track = bar.tracks[pitch];
          return (
            <div
              key={pitch}
              className={styles.lane}
              style={{ height: config.trackHeight }}
            >
              <span className={styles.laneLabel}>{pitch}</span>
              {track?.notes.map((note, i) => (
                <NoteView
                  key={i}
                  note={note}
                  color={track.color}
                  config={config}
                />
              ))}
            </div>
          );
        })}
      </div>
    );
  }
);

const NoteView = observer(
  ({ note, color, config }: { note: ResolvedNote; color: string; config: ViewConfig }) => {
    const isAccent = note.modifiers.has('a');
    const isGhost = note.modifiers.has('g');
    const isFlam = note.modifiers.has('fl');
    const isDrag = note.modifiers.has('dr');
    const isCross = note.modifiers.has('x');
    const badge = pickBadge(note);

    return (
      <div
        className={classNames(
          styles.note,
          isAccent && styles.accent,
          isGhost && styles.ghost,
          note.roll && styles.roll
        )}
        style={{
          left: note.x,
          top: config.trackHeight / 2,
          width: config.noteDiameter,
          height: config.noteDiameter,
          background: isCross ? '#fff' : color,
          color,
          borderStyle: isCross ? 'solid' : undefined,
          border: isCross ? `2px solid ${color}` : undefined,
        }}
        title={describeNote(note)}
      >
        {isFlam && <FlamGrace color={color} config={config} />}
        {isDrag && <DragGrace color={color} config={config} />}
        {badge && <span className={styles.modifierBadge}>{badge}</span>}
        {note.sticking && <span className={styles.stickingBadge}>{note.sticking.toUpperCase()}</span>}
      </div>
    );
  }
);

function FlamGrace({ color, config }: { color: string; config: ViewConfig }) {
  const size = (config.noteDiameter as number) * 0.55;
  return (
    <span
      style={{
        position: 'absolute',
        left: -size - 2,
        top: '50%',
        transform: 'translateY(-50%)',
        width: size,
        height: size,
        background: color,
        borderRadius: '50%',
        opacity: 0.7,
      }}
    />
  );
}

function DragGrace({ color, config }: { color: string; config: ViewConfig }) {
  const size = (config.noteDiameter as number) * 0.45;
  return (
    <>
      {[0, 1].map((i) => (
        <span
          key={i}
          style={{
            position: 'absolute',
            left: -((size + 2) * (i + 1)),
            top: '50%',
            transform: 'translateY(-50%)',
            width: size,
            height: size,
            background: color,
            borderRadius: '50%',
            opacity: 0.6,
          }}
        />
      ))}
    </>
  );
}

function pickBadge(note: ResolvedNote): string | undefined {
  const m = note.modifiers;
  if (m.has('c')) return 'C';
  if (m.has('o')) return 'O';
  if (m.has('h')) return 'H';
  if (m.has('f')) return 'F';
  if (m.has('s')) return 'S';
  if (m.has('r')) return 'R';
  if (m.has('z')) return 'Z';
  if (m.has('k')) return 'K';
  if (m.has('m')) return 'M';
  if (m.has('l')) return 'L';
  if (m.has('rf')) return 'Ruff';
  return undefined;
}

function describeNote(note: ResolvedNote): string {
  const mods = Array.from(note.modifiers);
  const parts: string[] = [note.pitch];
  if (mods.length) parts.push(`:${mods.join(':')}`);
  if (note.sticking) parts.push(`@${note.sticking}`);
  if (note.roll) parts.push('~');
  return parts.join('');
}
