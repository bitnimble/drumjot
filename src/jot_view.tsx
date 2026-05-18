import classNames from 'classnames';
import { autorun, makeAutoObservable, runInAction } from 'mobx';
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
import { isAudibleUnder, jotPlayer, PlayerState, timeToX } from 'src/playback';
import { SelectionStore } from 'src/selection';
import { RefinementLog, titleFromFilename, transcriber } from 'src/transcriber';
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
  lint: boolean;
  bestOfK: number;
  debug: boolean;
};

/**
 * Pixels-per-bar at zoom = 1. Same numeric value as `ViewConfig.barWidth`'s
 * own default so existing layouts are unchanged for users who never touch
 * the slider.
 */
const BASE_BAR_WIDTH = 640;
const MIN_ZOOM = 0.3;
const MAX_ZOOM = 3.0;

export class JotViewStore {
  currentJot: RenderedJot | undefined;
  examples: readonly ExampleJot[] = [];
  currentExampleId: string | undefined = undefined;
  transcribeStatus: TranscribeStatus = { phase: 'idle' };
  /** UI-controlled options for the next transcribe call. */
  transcribeOptions: TranscribeOptions = {
    refine: true,
    lint: true,
    bestOfK: 1,
    debug: false,
  };
  /**
   * Shared layout config threaded into every new `RenderedJot` we
   * construct, so the zoom slider mutates a single config object and
   * the layout reflows reactively (ViewConfig is MobX-observable;
   * RenderedJot's `layoutJot` is a computedFn that reads `barWidth`).
   */
  viewConfig: ViewConfig = new ViewConfig();
  /** Horizontal zoom multiplier; 1.0 = `BASE_BAR_WIDTH` pixels per bar. */
  zoom: number = 1;
  /** DSL pitches the user has muted via the row-gutter M button. */
  mutedPitches: Set<string> = new Set();
  /**
   * DSL pitches the user has soloed. When non-empty, ONLY these rows
   * are audible; this and `mutedPitches` are pushed to the player via
   * an autorun so toggles take effect live during playback.
   */
  soloedPitches: Set<string> = new Set();
  /**
   * Controller for the in-flight `/transcribe` request, if any. The
   * "Stop" toolbar button calls `.abort()` here; the request's
   * AbortSignal is passed into `transcriber.transcribe` which forwards
   * it to `fetch` so the request is genuinely cancelled at the
   * network layer rather than just discarding the response.
   */
  private transcribeController: AbortController | undefined;

  constructor() {
    makeAutoObservable(this);
    // Push mute / solo state to the player whenever it changes. While
    // playback is in flight, the player cancels and reschedules events
    // so the toggle takes effect immediately (including bringing
    // previously-muted rows back mid-song). When idle, the filter is
    // just stored for the next play().
    autorun(() => {
      jotPlayer.setFilter({
        mutedPitches: new Set(this.mutedPitches),
        soloedPitches: new Set(this.soloedPitches),
      });
    });
  }

  toggleMute(pitch: string) {
    if (this.mutedPitches.has(pitch)) this.mutedPitches.delete(pitch);
    else this.mutedPitches.add(pitch);
  }

  toggleSolo(pitch: string) {
    if (this.soloedPitches.has(pitch)) this.soloedPitches.delete(pitch);
    else this.soloedPitches.add(pitch);
  }

  isPitchAudible(pitch: string): boolean {
    return isAudibleUnder(pitch, {
      mutedPitches: this.mutedPitches,
      soloedPitches: this.soloedPitches,
    });
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
    this.currentJot = new RenderedJot(example.jot, this.viewConfig);
    this.currentExampleId = id;
  }

  setRefine(enabled: boolean) {
    this.transcribeOptions.refine = enabled;
  }

  setLint(enabled: boolean) {
    this.transcribeOptions.lint = enabled;
  }

  setBestOfK(n: number) {
    this.transcribeOptions.bestOfK = Math.max(1, Math.min(5, n));
  }

  setDebug(enabled: boolean) {
    this.transcribeOptions.debug = enabled;
  }

  setZoom(z: number) {
    const clamped = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z));
    this.zoom = clamped;
    this.viewConfig.barWidth = px(BASE_BAR_WIDTH * clamped);
  }

  /**
   * Upload an audio file to the transcriber service, parse the returned
   * Drumjot DSL, and load the resulting Jot. Updates `transcribeStatus`
   * so the toolbar can show progress / errors.
   *
   * A single in-flight transcription is tracked via `transcribeController`.
   * Calling `cancelTranscribe()` aborts the underlying `fetch` request and
   * surfaces a cancelled state on the toolbar; starting a new
   * transcription while one is in flight will abort the previous one
   * first (defensive - the UI disables the button during upload, but the
   * console-level `loadDsl` API doesn't).
   */
  async transcribeAudio(file: File): Promise<void> {
    if (this.transcribeController) {
      this.transcribeController.abort();
    }
    const controller = new AbortController();
    this.transcribeController = controller;
    runInAction(() => {
      this.transcribeStatus = { phase: 'uploading', filename: file.name };
    });
    try {
      const response = await transcriber.transcribe(file, {
        refine: this.transcribeOptions.refine,
        lint: this.transcribeOptions.lint,
        bestOfK: this.transcribeOptions.bestOfK,
        debug: this.transcribeOptions.debug,
        signal: controller.signal,
      });
      const jot = parse(response.jot_dsl);
      // The transcriber service no longer injects a title (the regex
      // pass over the DSL lived in Python and was the only string-level
      // mutation we did to LLM output). Set it here from the upload
      // filename so the rendered jot shows a useful heading.
      const derivedTitle = titleFromFilename(file.name);
      if (derivedTitle) jot.title = derivedTitle;
      runInAction(() => {
        this.currentJot = new RenderedJot(jot, this.viewConfig);
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
      // AbortError surfaces as DOMException with name='AbortError' (and
      // wraps as TypeError in some runtimes when the fetch was already
      // aborted at start). Treat the user-initiated cancellation
      // distinctly from real errors so we don't show a scary red pill.
      const isAbort =
        controller.signal.aborted ||
        (err instanceof DOMException && err.name === 'AbortError');
      if (isAbort) {
        runInAction(() => {
          this.transcribeStatus = { phase: 'idle' };
        });
      } else {
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
    } finally {
      if (this.transcribeController === controller) {
        this.transcribeController = undefined;
      }
    }
  }

  /**
   * Abort the in-flight transcription, if any. No-op when nothing is
   * running. The next `transcribeAudio` call resumes normally.
   */
  cancelTranscribe() {
    if (!this.transcribeController) return;
    this.transcribeController.abort();
    this.transcribeController = undefined;
  }

  clearTranscribeStatus() {
    this.transcribeStatus = { phase: 'idle' };
  }

  /**
   * Read a Drumjot DSL file from the user's machine and load it as the
   * current jot. Parse failures are surfaced through `transcribeStatus`
   * (same error pill the transcribe flow uses) so the user sees what
   * went wrong rather than getting silent dismissal.
   */
  async loadJotFile(file: File): Promise<void> {
    let text: string;
    try {
      text = await file.text();
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      runInAction(() => {
        this.transcribeStatus = { phase: 'error', message: `Could not read ${file.name}: ${message}` };
      });
      return;
    }
    try {
      const jot = parse(text);
      if (!jot.title) {
        const derivedTitle = titleFromFilename(file.name);
        if (derivedTitle) jot.title = derivedTitle;
      }
      runInAction(() => {
        this.currentJot = new RenderedJot(jot, this.viewConfig);
        this.currentExampleId = undefined;
        this.transcribeStatus = { phase: 'idle' };
      });
    } catch (err) {
      const message =
        err instanceof ParseError
          ? `Could not parse ${file.name}: ${err.message}`
          : err instanceof Error
            ? err.message
            : String(err);
      runInAction(() => {
        this.transcribeStatus = { phase: 'error', message };
      });
    }
  }

  async playCurrent(): Promise<void> {
    const jot = this.currentJot;
    if (!jot) return;
    // Pass the laid-out RenderedJot (not its source) so the player's
    // timeline reads live bar widths — the playhead then tracks correctly
    // across zoom changes.
    await jotPlayer.play(jot);
  }

  stopPlayback(): void {
    jotPlayer.stop();
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
          onLoadJot={(file) => store.loadJotFile(file)}
          onCancelTranscribe={() => store.cancelTranscribe()}
          onClearTranscribeStatus={() => store.clearTranscribeStatus()}
          onSetRefine={(v) => store.setRefine(v)}
          onSetLint={(v) => store.setLint(v)}
          onSetBestOfK={(n) => store.setBestOfK(n)}
          onSetDebug={(v) => store.setDebug(v)}
          zoom={store.zoom}
          onSetZoom={(z) => store.setZoom(z)}
          hasJot={!!jot}
          playerState={jotPlayer.state}
          playerError={jotPlayer.errorMessage}
          playbackSpeed={jotPlayer.playbackSpeed}
          onPlay={() => store.playCurrent()}
          onStopPlayback={() => store.stopPlayback()}
          onSetPlaybackSpeed={(s) => jotPlayer.setPlaybackSpeed(s)}
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
            voiceControls={{
              mutedPitches: store.mutedPitches,
              soloedPitches: store.soloedPitches,
              isPitchAudible: (pitch) => store.isPitchAudible(pitch),
              onToggleMute: (pitch) => store.toggleMute(pitch),
              onToggleSolo: (pitch) => store.toggleSolo(pitch),
            }}
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
    onLoadJot,
    onCancelTranscribe,
    onClearTranscribeStatus,
    onSetRefine,
    onSetLint,
    onSetBestOfK,
    onSetDebug,
    zoom,
    onSetZoom,
    hasJot,
    playerState,
    playerError,
    playbackSpeed,
    onPlay,
    onStopPlayback,
    onSetPlaybackSpeed,
  }: {
    examples: readonly ExampleJot[];
    currentId: string | undefined;
    onSelect: (id: string) => void;
    transcribeStatus: TranscribeStatus;
    transcribeOptions: TranscribeOptions;
    onTranscribe: (file: File) => void;
    onLoadJot: (file: File) => void;
    onCancelTranscribe: () => void;
    onClearTranscribeStatus: () => void;
    onSetRefine: (enabled: boolean) => void;
    onSetLint: (enabled: boolean) => void;
    onSetBestOfK: (n: number) => void;
    onSetDebug: (enabled: boolean) => void;
    zoom: number;
    onSetZoom: (z: number) => void;
    hasJot: boolean;
    playerState: PlayerState;
    playerError: string | undefined;
    playbackSpeed: number;
    onPlay: () => void;
    onStopPlayback: () => void;
    onSetPlaybackSpeed: (speed: number) => void;
  }) => {
    const fileInputRef = React.useRef<HTMLInputElement>(null);
    const jotInputRef = React.useRef<HTMLInputElement>(null);
    const uploading = transcribeStatus.phase === 'uploading';

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onTranscribe(file);
      // Reset so picking the same file twice in a row still fires onChange.
      e.target.value = '';
    };

    const handleJotFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onLoadJot(file);
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
        <PlaybackControls
          hasJot={hasJot}
          playerState={playerState}
          playerError={playerError}
          playbackSpeed={playbackSpeed}
          onPlay={onPlay}
          onStop={onStopPlayback}
          onSetPlaybackSpeed={onSetPlaybackSpeed}
        />
        <label
          className={styles.toolbarCheckbox}
          title="Compress or expand the score horizontally. Has no effect on audio playback, only on how the notation is laid out."
        >
          <span>Zoom</span>
          <input
            type="range"
            min={0.3}
            max={3.0}
            step={0.05}
            value={zoom}
            onChange={(e) => onSetZoom(Number(e.target.value))}
            className={styles.zoomSlider}
          />
          <span className={styles.zoomValue}>{Math.round(zoom * 100)}%</span>
        </label>
        <span className={styles.toolbarDivider} aria-hidden="true" />
        <button
          type="button"
          className={styles.playButton}
          onClick={() => jotInputRef.current?.click()}
          disabled={uploading}
          title="Load a Drumjot DSL file (`.jot`) from disk and render it. Parser runs entirely client-side; no transcriber service required."
        >
          Load .jot
        </button>
        <input
          ref={jotInputRef}
          type="file"
          accept=".jot,.txt,text/plain"
          className={styles.hiddenInput}
          onChange={handleJotFileChange}
        />
        <button
          type="button"
          className={styles.transcribeButton}
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
          title="Upload an audio file; the transcriber service will return a Jot. The Python backend decodes anything ffmpeg understands."
        >
          {uploading ? 'Transcribing...' : 'Transcribe audio'}
        </button>
        {uploading && (
          <button
            type="button"
            className={styles.cancelButton}
            onClick={onCancelTranscribe}
            title="Abort the in-flight transcription request."
          >
            Stop
          </button>
        )}
        <input
          ref={fileInputRef}
          type="file"
          // The Python backend decodes audio via librosa + soundfile +
          // ffmpeg, so anything ffmpeg understands works. Trust the
          // browser's `audio/*` filter; if the OS hasn't tagged a file
          // (rare on macOS / Linux, occasionally on Windows for .opus),
          // the user can switch the picker to "All files" and pick it
          // manually.
          accept="audio/*"
          className={styles.hiddenInput}
          onChange={handleFileChange}
        />
        <label
          className={styles.toolbarCheckbox}
          title="Run the deterministic Jot linter and ask the LLM to fix any instrument-tier (invalid modifier) or performance-tier (impossible sticking, too many hands, ...) issues it flags. Cheap relative to the F1 refinement; per-segment LLM calls scoped to the affected bars only."
        >
          <input
            type="checkbox"
            checked={transcribeOptions.lint}
            disabled={uploading}
            onChange={(e) => onSetLint(e.target.checked)}
          />
          Lint
        </label>
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
            value={transcribeOptions.bestOfK}
            disabled={uploading}
            onChange={(e) => onSetBestOfK(Number(e.target.value))}
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

const PLAYBACK_SPEEDS: readonly number[] = [0.25, 0.5, 0.75, 1.0, 1.25];

const PlaybackControls = observer(
  ({
    hasJot,
    playerState,
    playerError,
    playbackSpeed,
    onPlay,
    onStop,
    onSetPlaybackSpeed,
  }: {
    hasJot: boolean;
    playerState: PlayerState;
    playerError: string | undefined;
    playbackSpeed: number;
    onPlay: () => void;
    onStop: () => void;
    onSetPlaybackSpeed: (speed: number) => void;
  }) => {
    const loading = playerState === 'loading';
    const playing = playerState === 'playing';
    const hasError = !!playerError && !loading && !playing;
    return (
      <>
        <button
          type="button"
          className={classNames(styles.playButton, hasError && styles.playButtonError)}
          onClick={onPlay}
          disabled={!hasJot || loading || playing}
          title={
            playerError
              ? `Playback error: ${playerError}`
              : 'Play the current jot through a synthesised TR-909 drum kit. First click loads the samples (~150 KB) from the smplr CDN.'
          }
        >
          {loading ? 'Loading…' : hasError ? '⚠ Play' : '▶ Play'}
        </button>
        {playing && (
          <button
            type="button"
            className={styles.cancelButton}
            onClick={onStop}
            title="Stop playback."
          >
            ■ Stop
          </button>
        )}
        <label
          className={styles.toolbarCheckbox}
          title="Tempo multiplier applied to playback. Slowing down spaces successive hits further apart without changing the drum pitch — useful for practising along to a complex fill at half speed."
        >
          <span>Speed</span>
          <select
            className={styles.samplesSelect}
            value={String(playbackSpeed)}
            onChange={(e) => onSetPlaybackSpeed(Number(e.target.value))}
          >
            {PLAYBACK_SPEEDS.map((s) => (
              <option key={s} value={String(s)}>
                {s.toFixed(2)}×
              </option>
            ))}
          </select>
        </label>
        {hasError && (
          <span
            className={classNames(styles.statusPill, styles.statusPillError)}
            title={playerError}
          >
            Playback: {truncate(playerError ?? '', 60)}
          </span>
        )}
      </>
    );
  }
);

const Playhead = observer(() => {
  const timeline = jotPlayer.timeline;
  if (jotPlayer.state !== 'playing' || timeline.bars.length === 0) return null;
  const x = timeToX(timeline, jotPlayer.currentTime);
  return <div className={styles.playhead} style={{ left: x }} />;
});

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

type VoiceControls = {
  mutedPitches: ReadonlySet<string>;
  soloedPitches: ReadonlySet<string>;
  /** True if the row would currently make sound; false = muted via M or solo exclusion. */
  isPitchAudible: (pitch: string) => boolean;
  onToggleMute: (pitch: string) => void;
  onToggleSolo: (pitch: string) => void;
};

type JotViewProps = {
  jot: RenderedJot;
  marquee: Box | undefined;
  highlightedPattern: string | undefined;
  onPatternClick: (name: string) => void;
  onMouseDown: (e: React.MouseEvent<HTMLDivElement>) => void;
  onMouseMove: (e: React.MouseEvent<HTMLDivElement>) => void;
  onMouseUp: () => void;
  voiceControls: VoiceControls;
};

const JotView = observer((props: JotViewProps) => {
  const {
    jot,
    marquee,
    highlightedPattern,
    onPatternClick,
    onMouseDown,
    onMouseMove,
    onMouseUp,
    voiceControls,
  } = props;
  const resolved = jot.resolved;
  const config = jot.config;
  const containerRef = React.useRef<HTMLDivElement>(null);

  return (
    <div
      ref={containerRef}
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
            voiceControls={voiceControls}
          />
        ))}
      </div>
      <PlayheadAutoScroller containerRef={containerRef} />
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

/**
 * Side-effect-only component: when the playhead crosses the right (or
 * left) viewport buffer, nudges the scrollable container's `scrollLeft`
 * so the playhead stays visible during playback. Rendered nothing.
 *
 * Wrapped with `observer` so MobX reactivity drives re-renders on every
 * rAF-driven `currentTime` update; the body just reads observables and
 * runs the side effect.
 */
const PlayheadAutoScroller = observer(
  ({ containerRef }: { containerRef: React.RefObject<HTMLDivElement> }) => {
    const t = jotPlayer.currentTime;
    const state = jotPlayer.state;
    const timeline = jotPlayer.timeline;

    React.useEffect(() => {
      if (state !== 'playing' || timeline.bars.length === 0) return;
      const container = containerRef.current;
      if (!container) return;
      // Anchor x via any barsRow inside the container — they all share
      // the same left edge because voices stack vertically.
      const barsRow = container.querySelector<HTMLDivElement>(`.${styles.barsRow}`);
      if (!barsRow) return;

      const containerRect = container.getBoundingClientRect();
      const barsRect = barsRow.getBoundingClientRect();
      const playheadViewportX = barsRect.left + timeToX(timeline, t);
      // Margin keeps the playhead off the very edge so the user can see
      // a beat or two of upcoming notation.
      const rightLimit = containerRect.right - 100;
      const leftLimit = containerRect.left + 60;

      if (playheadViewportX > rightLimit) {
        container.scrollLeft += playheadViewportX - rightLimit;
      } else if (playheadViewportX < leftLimit) {
        container.scrollLeft -= leftLimit - playheadViewportX;
      }
    }, [t, state, timeline, containerRef]);

    return null;
  }
);

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
    voiceControls,
  }: {
    voice: ResolvedVoice;
    config: ViewConfig;
    index: number;
    totalVoices: number;
    highlightedPattern: string | undefined;
    onPatternClick: (name: string) => void;
    voiceControls: VoiceControls;
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
              <GutterCell
                key={pitch}
                pitch={pitch}
                height={config.trackHeight}
                instrumentName={instrumentByPitch[pitch]}
                muted={voiceControls.mutedPitches.has(pitch)}
                soloed={voiceControls.soloedPitches.has(pitch)}
                audible={voiceControls.isPitchAudible(pitch)}
                onToggleMute={voiceControls.onToggleMute}
                onToggleSolo={voiceControls.onToggleSolo}
              />
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
                isPitchAudible={voiceControls.isPitchAudible}
              />
            ))}
            <Playhead />
          </div>
        </div>
      </div>
    );
  }
);

/**
 * One pitch row in the lane gutter: shows the DSL letter plus a Mute and
 * Solo button. Stops click propagation so toggling doesn't also fire the
 * page-level mouse selection logic.
 */
const GutterCell = observer(
  ({
    pitch,
    height,
    instrumentName,
    muted,
    soloed,
    audible,
    onToggleMute,
    onToggleSolo,
  }: {
    pitch: string;
    height: Pixels;
    instrumentName: string | undefined;
    muted: boolean;
    soloed: boolean;
    audible: boolean;
    onToggleMute: (pitch: string) => void;
    onToggleSolo: (pitch: string) => void;
  }) => {
    const stopBubble = (e: React.MouseEvent) => e.stopPropagation();
    return (
      <div
        className={classNames(styles.laneGutterCell, !audible && styles.laneGutterCellDim)}
        style={{ height }}
        title={instrumentName ?? `Pitch ${pitch}`}
      >
        <span className={styles.gutterPitch}>{pitch}</span>
        <button
          type="button"
          className={classNames(styles.muteButton, muted && styles.muteButtonActive)}
          onClick={(e) => {
            stopBubble(e);
            onToggleMute(pitch);
          }}
          onMouseDown={stopBubble}
          title={muted ? `Unmute ${pitch}` : `Mute ${pitch}`}
          aria-label={muted ? `Unmute ${pitch}` : `Mute ${pitch}`}
          aria-pressed={muted}
        >
          M
        </button>
        <button
          type="button"
          className={classNames(styles.soloButton, soloed && styles.soloButtonActive)}
          onClick={(e) => {
            stopBubble(e);
            onToggleSolo(pitch);
          }}
          onMouseDown={stopBubble}
          title={soloed ? `Unsolo ${pitch}` : `Solo ${pitch}`}
          aria-label={soloed ? `Unsolo ${pitch}` : `Solo ${pitch}`}
          aria-pressed={soloed}
        >
          S
        </button>
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
    isPitchAudible,
  }: {
    bar: ResolvedBar;
    pitches: string[];
    config: ViewConfig;
    isAnacrusis: boolean;
    highlightedPattern: string | undefined;
    onPatternClick: (name: string) => void;
    isPitchAudible: (pitch: string) => boolean;
  }) => {
    return (
      <div
        className={classNames(styles.bar, isAnacrusis && styles.barAnacrusis)}
        style={{ width: bar.width, height: pitches.length * config.trackHeight }}
        title={`Bar ${bar.index} - ${bar.time.count}/${bar.time.unit}`}
      >
        {pitches.map((pitch) => {
          const track = bar.tracks[pitch];
          const dim = !isPitchAudible(pitch);
          return (
            <div
              key={pitch}
              className={classNames(styles.lane, dim && styles.laneDim)}
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
          highlighted && styles.patternBracketHighlight
        )}
        style={{ left: span.x, width: span.width }}
      >
        <button
          type="button"
          className={classNames(
            styles.patternLabel,
            highlighted && styles.patternLabelHighlight
          )}
          onClick={(e) => {
            e.stopPropagation();
            onClick(span.name);
          }}
          onMouseDown={(e) => e.stopPropagation()}
          title={`Pattern usage: ${span.name} (click to highlight other usages)`}
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
    qualifiers.push(MODIFIER_LABELS[mod] ?? mod);
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
