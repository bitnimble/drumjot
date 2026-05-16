import classNames from 'classnames';
import { makeAutoObservable, runInAction } from 'mobx';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { Instrument, Modifier, Sticking } from 'src/dsl';
import { ExampleJot } from 'src/fakes';
import { Box, Point } from 'src/geom';
import {
  PatternSpan,
  Pixels,
  RenderedJot,
  ResolvedBar,
  ResolvedJot,
  ResolvedNote,
  ResolvedVoice,
  ViewConfig,
  px,
} from 'src/jot';
import { parse, ParseError } from 'src/parser';
import { SelectionStore } from 'src/selection';
import { RefinementLog, transcriber } from 'src/transcriber';
import styles from './jot_view.module.css';

export type TranscribeStatus =
  | { phase: 'idle' }
  | { phase: 'uploading'; filename: string }
  | { phase: 'error'; message: string }
  | {
      phase: 'success';
      filename: string;
      tempo: number;
      hasTempoChanges: boolean;
      hasTimeSigChanges: boolean;
      barCount: number;
      refinement?: RefinementLog | null;
      debugDir?: string | null;
    };

export type TranscribeOptions = {
  refine: boolean;
  selfConsistencySamples: number;
  debug: boolean;
};

export class JotViewStore {
  currentJot: RenderedJot | undefined;
  examples: readonly ExampleJot[] = [];
  currentExampleId: string | undefined = undefined;
  transcribeStatus: TranscribeStatus = { phase: 'idle' };
  /** UI-controlled options for the next transcribe call. */
  transcribeOptions: TranscribeOptions = {
    refine: true,
    selfConsistencySamples: 1,
    debug: false,
  };

  constructor() {
    makeAutoObservable(this);
  }

  setJot(jot: RenderedJot | undefined) {
    this.currentJot = jot;
    // External setJot calls invalidate the example pointer.
    this.currentExampleId = undefined;
  }

  setExamples(examples: readonly ExampleJot[]) {
    this.examples = examples;
  }

  loadExample(id: string) {
    const example = this.examples.find((e) => e.id === id);
    if (!example) return;
    this.currentJot = new RenderedJot(example.jot);
    this.currentExampleId = id;
  }

  setRefine(enabled: boolean) {
    this.transcribeOptions.refine = enabled;
  }

  setSelfConsistencySamples(n: number) {
    this.transcribeOptions.selfConsistencySamples = Math.max(1, Math.min(5, n));
  }

  setDebug(enabled: boolean) {
    this.transcribeOptions.debug = enabled;
  }

  /**
   * Upload an audio file to the transcriber service, parse the returned
   * Drumjot DSL, and load the resulting Jot. Updates `transcribeStatus`
   * so the toolbar can show progress / errors.
   */
  async transcribeAudio(file: File): Promise<void> {
    runInAction(() => {
      this.transcribeStatus = { phase: 'uploading', filename: file.name };
    });
    try {
      const response = await transcriber.transcribe(file, {
        refine: this.transcribeOptions.refine,
        selfConsistencySamples: this.transcribeOptions.selfConsistencySamples,
        debug: this.transcribeOptions.debug,
      });
      const jot = parse(response.jot_dsl);
      runInAction(() => {
        this.currentJot = new RenderedJot(jot);
        this.currentExampleId = undefined;
        this.transcribeStatus = {
          phase: 'success',
          filename: file.name,
          tempo: response.metadata.initial_tempo,
          hasTempoChanges: response.metadata.has_tempo_changes,
          hasTimeSigChanges: response.metadata.has_time_sig_changes,
          barCount: response.metadata.bars.length,
          refinement: response.refinement ?? null,
          debugDir: response.debug_dir ?? null,
        };
      });
    } catch (err) {
      const message =
        err instanceof ParseError
          ? `Transcriber returned invalid DSL: ${err.message}`
          : err instanceof Error
            ? err.message
            : String(err);
      runInAction(() => {
        this.transcribeStatus = { phase: 'error', message };
      });
    }
  }

  clearTranscribeStatus() {
    this.transcribeStatus = { phase: 'idle' };
  }
}

type CreateJotViewOptions = {
  examples?: readonly ExampleJot[];
};

type CreateJotViewResult = {
  store: JotViewStore;
  View: React.FC;
};

export function createJotView(options: CreateJotViewOptions = {}): CreateJotViewResult {
  const store = new JotViewStore();
  if (options.examples) store.setExamples(options.examples);
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
    return (
      <div className={styles.appContainer}>
        <Toolbar
          examples={store.examples}
          currentId={store.currentExampleId}
          onSelect={(id) => store.loadExample(id)}
          transcribeStatus={store.transcribeStatus}
          transcribeOptions={store.transcribeOptions}
          onTranscribe={(file) => store.transcribeAudio(file)}
          onClearTranscribeStatus={() => store.clearTranscribeStatus()}
          onSetRefine={(v) => store.setRefine(v)}
          onSetSelfConsistency={(n) => store.setSelfConsistencySamples(n)}
          onSetDebug={(v) => store.setDebug(v)}
        />
        {jot ? (
          <JotView
            jot={jot}
            marquee={selection.marquee}
            highlightedPattern={selection.selectedPattern}
            onPatternClick={(name) => selection.togglePattern(name)}
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={selection.endSelection}
          />
        ) : (
          <div className={styles.empty}>No jot loaded</div>
        )}
      </div>
    );
  });

  return { store, View };
}

const Toolbar = observer(
  ({
    examples,
    currentId,
    onSelect,
    transcribeStatus,
    transcribeOptions,
    onTranscribe,
    onClearTranscribeStatus,
    onSetRefine,
    onSetSelfConsistency,
    onSetDebug,
  }: {
    examples: readonly ExampleJot[];
    currentId: string | undefined;
    onSelect: (id: string) => void;
    transcribeStatus: TranscribeStatus;
    transcribeOptions: TranscribeOptions;
    onTranscribe: (file: File) => void;
    onClearTranscribeStatus: () => void;
    onSetRefine: (enabled: boolean) => void;
    onSetSelfConsistency: (n: number) => void;
    onSetDebug: (enabled: boolean) => void;
  }) => {
    const fileInputRef = React.useRef<HTMLInputElement>(null);
    const uploading = transcribeStatus.phase === 'uploading';

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onTranscribe(file);
      // Reset so picking the same file twice in a row still fires onChange.
      e.target.value = '';
    };

    return (
      <div className={styles.toolbar}>
        {examples.length > 0 && (
          <>
            <label htmlFor="drumjot-example-select" className={styles.toolbarLabel}>
              Example
            </label>
            <select
              id="drumjot-example-select"
              className={styles.exampleSelect}
              value={currentId ?? ''}
              onChange={(e) => onSelect(e.target.value)}
            >
              {currentId === undefined && (
                <option value="" disabled>
                  Select an example...
                </option>
              )}
              {examples.map((ex) => (
                <option key={ex.id} value={ex.id}>
                  {ex.label}
                </option>
              ))}
            </select>
            <span className={styles.toolbarDivider} aria-hidden="true" />
          </>
        )}
        <button
          type="button"
          className={styles.transcribeButton}
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
          title="Upload an audio file (wav, flac, mp3, aac/m4a, opus, ogg); the transcriber service will return a Jot."
        >
          {uploading ? 'Transcribing...' : 'Transcribe audio'}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          // `audio/*` covers anything the browser tags as audio. Explicit
          // extensions catch files whose MIME the OS hasn't filled in,
          // which is common for .opus / .oga / .flac on Windows.
          accept={[
            'audio/*',
            // Lossless / uncompressed
            '.wav', '.flac', 'audio/wav', 'audio/x-wav', 'audio/flac', 'audio/x-flac',
            // Lossy
            '.mp3', 'audio/mpeg', 'audio/mp3',
            '.aac', '.m4a', '.mp4', 'audio/aac', 'audio/x-aac', 'audio/mp4', 'audio/x-m4a',
            '.opus', '.ogg', '.oga', 'audio/opus', 'audio/ogg',
          ].join(',')}
          className={styles.hiddenInput}
          onChange={handleFileChange}
        />
        <label
          className={styles.toolbarCheckbox}
          title="Run the multi-level convergence loop after the initial transcription. Adds ~30-60s but typically lifts accuracy by 5-10 F1 points."
        >
          <input
            type="checkbox"
            checked={transcribeOptions.refine}
            disabled={uploading}
            onChange={(e) => onSetRefine(e.target.checked)}
          />
          Refine accuracy
        </label>
        <label
          className={styles.toolbarCheckbox}
          title="Generate K candidate initial transcriptions at different temperatures and pick the highest-scoring one."
        >
          <span>Samples</span>
          <select
            className={styles.samplesSelect}
            value={transcribeOptions.selfConsistencySamples}
            disabled={uploading}
            onChange={(e) => onSetSelfConsistency(Number(e.target.value))}
          >
            <option value={1}>1</option>
            <option value={3}>3</option>
            <option value={5}>5</option>
          </select>
        </label>
        <label
          className={styles.toolbarCheckbox}
          title="Persist intermediate audio (drum stems, per-instrument stems), beat tracking, onsets, and LLM input/output to the transcriber's debug directory so you can listen back and inspect issues."
        >
          <input
            type="checkbox"
            checked={transcribeOptions.debug}
            disabled={uploading}
            onChange={(e) => onSetDebug(e.target.checked)}
          />
          Save debug files
        </label>
        <TranscribeStatusPill status={transcribeStatus} onClear={onClearTranscribeStatus} />
      </div>
    );
  }
);

const TranscribeStatusPill = observer(
  ({ status, onClear }: { status: TranscribeStatus; onClear: () => void }) => {
    if (status.phase === 'idle') return null;
    if (status.phase === 'uploading') {
      return (
        <span className={classNames(styles.statusPill, styles.statusPillBusy)}>
          Transcribing {status.filename}...
        </span>
      );
    }
    if (status.phase === 'error') {
      return (
        <span
          className={classNames(styles.statusPill, styles.statusPillError)}
          onClick={onClear}
          role="button"
          title={status.message}
        >
          Error: {truncate(status.message, 60)} (click to dismiss)
        </span>
      );
    }
    const refinement = status.refinement;
    let detail = `@ ${status.tempo.toFixed(0)} bpm, ${status.barCount} bars`;
    if (status.hasTempoChanges) detail += ', tempo changes';
    if (status.hasTimeSigChanges) detail += ', time-sig changes';
    if (refinement) {
      const accepted = refinement.iterations.filter((i) => i.accepted).length;
      const delta = refinement.final_score - refinement.initial_score;
      const sign = delta >= 0 ? '+' : '';
      detail += `, F1 ${refinement.initial_score.toFixed(2)} \u2192 ${refinement.final_score.toFixed(2)} (${sign}${delta.toFixed(2)}, ${accepted} revisions)`;
    }
    if (status.debugDir) {
      detail += `, debug @ ${status.debugDir}`;
    }
    const titleLines: string[] = [];
    if (refinement) {
      titleLines.push(
        `Refined ${refinement.iterations.length} iterations in ${refinement.elapsed_seconds.toFixed(1)}s.`,
      );
    }
    if (status.debugDir) {
      titleLines.push(
        `Debug artifacts saved to ${status.debugDir} (under ./debug/ on the host with the default docker-compose mount).`,
      );
    }
    return (
      <span
        className={classNames(styles.statusPill, styles.statusPillSuccess)}
        onClick={onClear}
        role="button"
        title={titleLines.length > 0 ? titleLines.join('\n') : undefined}
      >
        Loaded {status.filename} {detail} (click to dismiss)
      </span>
    );
  }
);

function truncate(s: string, n: number): string {
  return s.length <= n ? s : `${s.slice(0, n - 1)}\u2026`;
}

type JotViewProps = {
  jot: RenderedJot;
  marquee: Box | undefined;
  highlightedPattern: string | undefined;
  onPatternClick: (name: string) => void;
  onMouseDown: (e: React.MouseEvent<HTMLDivElement>) => void;
  onMouseMove: (e: React.MouseEvent<HTMLDivElement>) => void;
  onMouseUp: () => void;
};

const JotView = observer((props: JotViewProps) => {
  const { jot, marquee, highlightedPattern, onPatternClick, onMouseDown, onMouseMove, onMouseUp } =
    props;
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
          <VoiceView
            key={i}
            voice={voice}
            config={config}
            index={i}
            totalVoices={resolved.voices.length}
            highlightedPattern={highlightedPattern}
            onPatternClick={onPatternClick}
          />
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
          seen.set(pitch, { color: track.color, name: track.instrument.name });
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
    highlightedPattern,
    onPatternClick,
  }: {
    voice: ResolvedVoice;
    config: ViewConfig;
    index: number;
    totalVoices: number;
    highlightedPattern: string | undefined;
    onPatternClick: (name: string) => void;
  }) => {
    const pitches = voice.pitches;
    const staffHeight = px(pitches.length * config.trackHeight);

    // Pick the first resolved instrument per pitch so we can label the gutter.
    const instrumentByPitch: Record<string, string> = {};
    for (const bar of voice.bars) {
      for (const pitch of pitches) {
        if (instrumentByPitch[pitch]) continue;
        const track = bar.tracks[pitch];
        if (track?.instrument.name) instrumentByPitch[pitch] = track.instrument.name;
      }
    }

    const label = voice.source.name ?? `Voice ${index + 1}`;
    return (
      <div className={styles.voice}>
        {totalVoices > 1 && <div className={styles.voiceLabel}>{label}</div>}
        <div className={styles.voiceStaff} style={{ height: staffHeight }}>
          <div className={styles.laneGutter} style={{ height: staffHeight }}>
            {pitches.map((pitch) => (
              <div
                key={pitch}
                className={styles.laneGutterCell}
                style={{ height: config.trackHeight }}
                title={instrumentByPitch[pitch] ?? `Pitch ${pitch}`}
              >
                {pitch}
              </div>
            ))}
          </div>
          <div className={styles.barsRow} style={{ width: voice.width }}>
            {voice.bars.map((bar, i) => (
              <BarView
                key={i}
                bar={bar}
                pitches={pitches}
                config={config}
                isAnacrusis={bar.index === 0}
                highlightedPattern={highlightedPattern}
                onPatternClick={onPatternClick}
              />
            ))}
          </div>
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
    highlightedPattern,
    onPatternClick,
  }: {
    bar: ResolvedBar;
    pitches: string[];
    config: ViewConfig;
    isAnacrusis: boolean;
    highlightedPattern: string | undefined;
    onPatternClick: (name: string) => void;
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
              {track?.notes.map((note, i) => (
                <NoteView
                  key={i}
                  note={note}
                  color={track.color}
                  config={config}
                  instrument={track.instrument}
                />
              ))}
            </div>
          );
        })}
        {bar.patternSpans.map((span, i) => (
          <PatternBracket
            key={i}
            span={span}
            highlighted={highlightedPattern === span.name}
            onClick={onPatternClick}
          />
        ))}
      </div>
    );
  }
);

const PatternBracket = observer(
  ({
    span,
    highlighted,
    onClick,
  }: {
    span: PatternSpan;
    highlighted: boolean;
    onClick: (name: string) => void;
  }) => {
    return (
      <div
        className={classNames(
          styles.patternBracket,
          span.isDefinition && styles.patternBracketDefinition,
          highlighted && styles.patternBracketHighlight
        )}
        style={{ left: span.x, width: span.width }}
      >
        <button
          type="button"
          className={classNames(
            styles.patternLabel,
            span.isDefinition && styles.patternLabelDefinition,
            highlighted && styles.patternLabelHighlight
          )}
          onClick={(e) => {
            e.stopPropagation();
            onClick(span.name);
          }}
          onMouseDown={(e) => e.stopPropagation()}
          title={
            span.isDefinition
              ? `Pattern definition: ${span.name} (click to highlight all usages)`
              : `Pattern usage: ${span.name} (click to highlight the definition)`
          }
        >
          {span.name}
        </button>
      </div>
    );
  }
);

const NoteView = observer(
  ({
    note,
    color,
    config,
    instrument,
  }: {
    note: ResolvedNote;
    color: string;
    config: ViewConfig;
    instrument: Instrument;
  }) => {
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
        title={describeNote(note, instrument)}
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

/**
 * Human-readable tooltip text for a note. Combines the resolved instrument
 * name with friendly modifier / sticking / roll labels.
 *
 * Examples:
 *   `s:a`       -> "Snare (accented)"
 *   `s:fl@l`    -> "Snare (flam, left hand)"
 *   `h:c`       -> "Hi-Hat (closed)"
 *   `c~_8:o`    -> "Crash (open, roll)"
 */
function describeNote(note: ResolvedNote, instrument: Instrument): string {
  const name = instrument.name ?? `Pitch ${note.pitch}`;
  const qualifiers: string[] = [];
  for (const mod of note.modifiers) {
    qualifiers.push(MODIFIER_LABELS[mod as Modifier] ?? mod);
  }
  if (note.roll) qualifiers.push('roll');
  if (note.sticking) qualifiers.push(STICKING_LABELS[note.sticking]);
  return qualifiers.length > 0 ? `${name} (${qualifiers.join(', ')})` : name;
}

const MODIFIER_LABELS: Partial<Record<Modifier, string>> = {
  a: 'accented',
  g: 'ghost',
  c: 'closed',
  h: 'half-open',
  o: 'open',
  f: 'foot',
  s: 'splash',
  r: 'rim shot',
  x: 'cross-stick',
  z: 'buzz',
  k: 'choke',
  m: 'mute',
  l: 'let ring',
  fl: 'flam',
  dr: 'drag',
  rf: 'ruff',
};

const STICKING_LABELS: Record<Sticking, string> = {
  r: 'right hand',
  l: 'left hand',
  rf: 'right foot',
  lf: 'left foot',
};
