import classNames from 'classnames';
import { makeAutoObservable, reaction, runInAction } from 'mobx';
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
  TupletSpan,
  ViewConfig,
  px,
} from 'src/jot';
import { parse, ParseError } from 'src/parser';
import { loadParadbZip } from 'src/rlrr';
import {
  buildTimeline,
  computeWaveformPeaksForJot,
  isAudibleUnder,
  isAudioTrackAudibleUnder,
  jotPlayer,
  PlayerState,
  SampleLoadProgress,
  KitInfo,
  AudioTrackId,
  AudioTrack,
  timeToX,
  xToTime,
} from 'src/playback';
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
const BASE_BAR_WIDTH = 448;
const MIN_ZOOM = 0.3;
const MAX_ZOOM = 3.0;
// Row volume faders are pure attenuation (0 = silent, 1 = unscaled).
// The kit's overall loudness is handled by the drum master gain.
const VOLUME_STEP = 0.05;
function clampVolume(v: number): number {
  if (!Number.isFinite(v)) return 1;
  return Math.max(0, Math.min(1, v));
}

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
  /** Audio-track ids the user has muted via the gutter M button. */
  mutedAudioTracks: Set<AudioTrackId> = new Set();
  /** Soloed audio-track ids — same semantics as `soloedPitches`. */
  soloedAudioTracks: Set<AudioTrackId> = new Set();
  /**
   * Per-row volume faders, 0..1 (1 = full). Sparse: a row absent from
   * the map plays at full volume. Pitch volumes scale note velocity in
   * the scheduler; audio-track volumes scale the track's GainNode.
   */
  pitchVolumes: Map<string, number> = new Map();
  audioTrackVolumes: Map<AudioTrackId, number> = new Map();
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
    //
    // This MUST be a `reaction`, not an `autorun`: `setFilter` both
    // reads (`scheduleEvents` → `isAudibleUnder(..., this.currentFilter)`)
    // and writes (`this.currentFilter = filter`) an observable on the
    // MobX-observable player while playing. An `autorun` tracks reads
    // made during the effect, so it would depend on the very observable
    // it writes — a non-converging reaction that MobX bails on, after
    // which the UI (observer components reading store state directly)
    // keeps updating but the filter stops reaching the player (e.g.
    // un-solo visually clears yet audio stays soloed). `reaction` only
    // tracks the data selector; the effect runs untracked, so the
    // player's internal reads/writes stay out of the dependency graph.
    reaction(
      () => ({
        mutedPitches: new Set(this.mutedPitches),
        soloedPitches: new Set(this.soloedPitches),
        soloActive: this.soloActive,
        volumes: new Map(this.pitchVolumes),
      }),
      (filter) => jotPlayer.setFilter(filter),
      { fireImmediately: true },
    );
    // Same shape for audio tracks — observed mutations push immediately
    // so toggling M/S on a track is sample-accurate during playback
    // (per-track GainNode flip, no source recreation). Same
    // read-and-write-the-same-observable hazard as above
    // (`setAudioTrackFilter` reads/writes `currentAudioTrackFilter`), so this is a
    // `reaction` for the same reason.
    reaction(
      () => ({
        mutedAudioTracks: new Set(this.mutedAudioTracks),
        soloedAudioTracks: new Set(this.soloedAudioTracks),
        soloActive: this.soloActive,
        volumes: new Map(this.audioTrackVolumes),
      }),
      (filter) => jotPlayer.setAudioTrackFilter(filter),
      { fireImmediately: true },
    );
  }

  /**
   * Solo is one global mode across both the pitch and audio-track domains: any
   * soloed row (drum *or* music) puts every non-soloed row — in either
   * domain — into the "solo-excluded" state. Without this, soloing a
   * drum to practise it would leave the backing music playing.
   */
  get soloActive(): boolean {
    return this.soloedPitches.size > 0 || this.soloedAudioTracks.size > 0;
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
      soloActive: this.soloActive,
      volumes: this.pitchVolumes,
    });
  }

  pitchVolume(pitch: string): number {
    return this.pitchVolumes.get(pitch) ?? 1;
  }

  setPitchVolume(pitch: string, v: number) {
    this.pitchVolumes.set(pitch, clampVolume(v));
  }

  toggleAudioTrackMute(id: AudioTrackId) {
    if (this.mutedAudioTracks.has(id)) this.mutedAudioTracks.delete(id);
    else this.mutedAudioTracks.add(id);
  }

  toggleAudioTrackSolo(id: AudioTrackId) {
    if (this.soloedAudioTracks.has(id)) this.soloedAudioTracks.delete(id);
    else this.soloedAudioTracks.add(id);
  }

  isAudioTrackAudible(id: AudioTrackId): boolean {
    return isAudioTrackAudibleUnder(id, {
      mutedAudioTracks: this.mutedAudioTracks,
      soloedAudioTracks: this.soloedAudioTracks,
      soloActive: this.soloActive,
      volumes: this.audioTrackVolumes,
    });
  }

  audioTrackVolume(id: AudioTrackId): number {
    return this.audioTrackVolumes.get(id) ?? 1;
  }

  setAudioTrackVolume(id: AudioTrackId, v: number) {
    this.audioTrackVolumes.set(id, clampVolume(v));
  }

  /**
   * Load an audio file as a new audio track and update the status pill
   * on failure. Decoding goes through the shared `AudioContext`, so the
   * call has to occur inside a user gesture (the file-picker click
   * satisfies that). Every call appends an independent track — load N
   * files to get N tracks. Returns the new track's id, or `undefined`
   * if the load failed (so callers can e.g. default it to muted).
   */
  async loadAudioTrack(file: File): Promise<AudioTrackId | undefined> {
    try {
      return await jotPlayer.loadAudioTrack(file);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      runInAction(() => {
        this.transcribeStatus = {
          phase: 'error',
          message: `Audio track load failed: ${message}`,
        };
      });
      return undefined;
    }
  }

  clearAudioTrack(id: AudioTrackId): void {
    jotPlayer.clearAudioTrack(id);
    // Drop the removed track's mute/solo/volume so it doesn't linger
    // (ids are never reused, so the entries would be dead weight) — and
    // critically so clearing the only soloed audio track doesn't leave a
    // phantom solo silencing everything else.
    this.mutedAudioTracks.delete(id);
    this.soloedAudioTracks.delete(id);
    this.audioTrackVolumes.delete(id);
  }

  setJot(jot: RenderedJot | undefined) {
    this.currentJot = jot;
    // External setJot calls invalidate the example pointer.
    this.currentExampleId = undefined;
    jotPlayer.clearCue();
  }

  setExamples(examples: readonly ExampleJot[]) {
    this.examples = examples;
  }

  loadExample(id: string) {
    const example = this.examples.find((e) => e.id === id);
    if (!example) return;
    this.currentJot = new RenderedJot(example.jot, this.viewConfig);
    this.currentExampleId = id;
    jotPlayer.clearCue();
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
        jotPlayer.clearCue();
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
        jotPlayer.clearCue();
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

  /**
   * Load a ParaDB / Paradiddle map pack (`.zip`): convert its `.rlrr`
   * chart to a Jot and auto-load its audio tracks so the pack is
   * immediately play-along ready. Audio decoding shares the
   * `AudioContext`, so this must run inside the file-picker's user
   * gesture (the same constraint as {@link loadAudioTrack}). Errors surface
   * through the shared status pill, matching {@link loadJotFile}.
   */
  async loadParadbMap(file: File): Promise<void> {
    let map: Awaited<ReturnType<typeof loadParadbZip>>;
    try {
      map = await loadParadbZip(file);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      runInAction(() => {
        this.transcribeStatus = {
          phase: 'error',
          message: `Could not load ${file.name}: ${message}`,
        };
      });
      return;
    }

    const jot = map.jot;
    if (!jot.title) {
      const derivedTitle = titleFromFilename(file.name);
      if (derivedTitle) jot.title = derivedTitle;
    }
    runInAction(() => {
      this.currentJot = new RenderedJot(jot, this.viewConfig);
      this.currentExampleId = undefined;
      jotPlayer.clearCue();
      this.transcribeStatus = { phase: 'idle' };
    });

    // Audio tracks are best-effort: a chart with the score loaded is
    // still useful even if one is absent or fails to decode.
    // loadAudioTrack already reports its own failures on the status pill.
    // Drum tracks load too but start muted — you're playing the drums,
    // so the backing music should be the only thing you hear by default.
    for (const track of map.audioTracks) {
      const id = await this.loadAudioTrack(track.file);
      if (id && track.defaultMuted) {
        runInAction(() => {
          this.mutedAudioTracks.add(id);
        });
      }
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

  /**
   * Click-to-seek. `x` is a pixel offset within the bars row — the same
   * coordinate space `bar.x` / the playhead use (origin at the left
   * edge of the bars region, after the gutter). While playing this
   * scrubs live; while idle it parks the playhead and the next Play
   * starts from there. Uses the live timeline when one exists so a
   * mid-playback scrub reads the exact bars being played.
   */
  seekToX(x: number): void {
    const jot = this.currentJot;
    if (!jot) return;
    const timeline =
      jotPlayer.timeline.bars.length > 0 ? jotPlayer.timeline : buildTimeline(jot);
    jotPlayer.seek(jot, xToTime(timeline, x));
  }

  /**
   * Single transport action shared by the spacebar shortcut and the
   * toolbar's play/pause button:
   *   idle    -> play the current jot from the start
   *   playing -> pause (freezes the clock, playhead stays put)
   *   paused  -> resume from the same spot
   * `loading` is intentionally a no-op so a double-press during the
   * one-time sample fetch can't stack two `play()` calls.
   */
  async togglePlayPause(): Promise<void> {
    switch (jotPlayer.state) {
      case 'idle':
        await this.playCurrent();
        break;
      case 'playing':
        await jotPlayer.pause();
        break;
      case 'paused':
        await jotPlayer.resume();
        break;
    }
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

    // Spacebar = play / pause / resume, from anywhere on the page. Skip
    // only when a text-entry control has focus (the user is typing) or
    // a SELECT is focused (let space/arrows drive the native picker).
    // A focused BUTTON deliberately falls through: preventDefault both
    // stops the browser's space-to-scroll and suppresses the button's
    // space-activation, so spacebar *always* toggles transport.
    React.useEffect(() => {
      const onKeyDown = (e: KeyboardEvent) => {
        if (e.code !== 'Space' && e.key !== ' ') return;
        const el = e.target as HTMLElement | null;
        const tag = el?.tagName;
        if (
          tag === 'INPUT' ||
          tag === 'TEXTAREA' ||
          tag === 'SELECT' ||
          el?.isContentEditable
        ) {
          return;
        }
        e.preventDefault();
        void store.togglePlayPause();
      };
      window.addEventListener('keydown', onKeyDown);
      return () => window.removeEventListener('keydown', onKeyDown);
    }, []);

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
          onLoadParadb={(file) => store.loadParadbMap(file)}
          onLoadAudioTrack={(file) => store.loadAudioTrack(file)}
          onCancelTranscribe={() => store.cancelTranscribe()}
          onClearTranscribeStatus={() => store.clearTranscribeStatus()}
          onSetRefine={(v) => store.setRefine(v)}
          onSetLint={(v) => store.setLint(v)}
          onSetBestOfK={(n) => store.setBestOfK(n)}
          onSetDebug={(v) => store.setDebug(v)}
          zoom={store.zoom}
          onSetZoom={(z) => store.setZoom(z)}
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
            onSeek={(x) => store.seekToX(x)}
            onZoomBy={(factor) => store.setZoom(store.zoom * factor)}
            voiceControls={{
              mutedPitches: store.mutedPitches,
              soloedPitches: store.soloedPitches,
              isPitchAudible: (pitch) => store.isPitchAudible(pitch),
              volumeFor: (pitch) => store.pitchVolume(pitch),
              onSetVolume: (pitch, v) => store.setPitchVolume(pitch, v),
              onToggleMute: (pitch) => store.toggleMute(pitch),
              onToggleSolo: (pitch) => store.toggleSolo(pitch),
            }}
            audioTrackControls={{
              mutedAudioTracks: store.mutedAudioTracks,
              soloedAudioTracks: store.soloedAudioTracks,
              isAudioTrackAudible: (id) => store.isAudioTrackAudible(id),
              volumeFor: (id) => store.audioTrackVolume(id),
              onSetVolume: (id, v) => store.setAudioTrackVolume(id, v),
              onToggleMute: (id) => store.toggleAudioTrackMute(id),
              onToggleSolo: (id) => store.toggleAudioTrackSolo(id),
              onClear: (id) => store.clearAudioTrack(id),
            }}
          />
        ) : (
          <div className={styles.empty}>No jot loaded</div>
        )}
        <PlaybackBar store={store} />
      </div>
    );
  });

  return { store, View };
}

/**
 * Native `<select>` that releases focus once a value is committed.
 *
 * The global spacebar play/pause shortcut deliberately ignores
 * keystrokes while a SELECT is focused (so arrow/space can drive the
 * open list), which means a dropdown that keeps focus after the user
 * picks a value would silently swallow the shortcut until they click
 * elsewhere. Routing every dropdown through this wrapper makes the
 * correct behaviour the default — new dropdowns can't reintroduce the
 * regression. All native `<select>` props pass straight through.
 */
const Select = ({
  onChange,
  ...rest
}: React.ComponentPropsWithoutRef<'select'>) => (
  <select
    {...rest}
    onChange={(e) => {
      onChange?.(e);
      e.currentTarget.blur();
    }}
  />
);

/**
 * A toolbar button that toggles a floating panel of related controls,
 * used to collapse the formerly-flat header into grouped menus ("Load",
 * "Transcribe"). Closes on outside click or Escape. `children` is a
 * render prop receiving a `close` callback so menu items that complete
 * an action (open a file picker, fire a request) can dismiss the panel,
 * while sticky controls (option checkboxes) can leave it open.
 */
const DropdownButton = ({
  label,
  title,
  className,
  panelClassName,
  children,
}: {
  label: React.ReactNode;
  title?: string;
  className?: string;
  panelClassName?: string;
  children: (close: () => void) => React.ReactNode;
}) => {
  const [open, setOpen] = React.useState(false);
  const ref = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onPointerDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onPointerDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  return (
    <div className={styles.dropdown} ref={ref}>
      <button
        type="button"
        className={className ?? styles.playButton}
        title={title}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        {label} <span aria-hidden="true">▾</span>
      </button>
      {open && (
        <div className={classNames(styles.dropdownPanel, panelClassName)} role="menu">
          {children(() => setOpen(false))}
        </div>
      )}
    </div>
  );
};

const Toolbar = observer(
  ({
    examples,
    currentId,
    onSelect,
    transcribeStatus,
    transcribeOptions,
    onTranscribe,
    onLoadJot,
    onLoadParadb,
    onLoadAudioTrack,
    onCancelTranscribe,
    onClearTranscribeStatus,
    onSetRefine,
    onSetLint,
    onSetBestOfK,
    onSetDebug,
    zoom,
    onSetZoom,
  }: {
    examples: readonly ExampleJot[];
    currentId: string | undefined;
    onSelect: (id: string) => void;
    transcribeStatus: TranscribeStatus;
    transcribeOptions: TranscribeOptions;
    onTranscribe: (file: File) => void;
    onLoadJot: (file: File) => void;
    onLoadParadb: (file: File) => void;
    onLoadAudioTrack: (file: File) => void;
    onCancelTranscribe: () => void;
    onClearTranscribeStatus: () => void;
    onSetRefine: (enabled: boolean) => void;
    onSetLint: (enabled: boolean) => void;
    onSetBestOfK: (n: number) => void;
    onSetDebug: (enabled: boolean) => void;
    zoom: number;
    onSetZoom: (z: number) => void;
  }) => {
    const fileInputRef = React.useRef<HTMLInputElement>(null);
    const jotInputRef = React.useRef<HTMLInputElement>(null);
    const paradbInputRef = React.useRef<HTMLInputElement>(null);
    const audioTrackInputRef = React.useRef<HTMLInputElement>(null);
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

    const handleParadbChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onLoadParadb(file);
      e.target.value = '';
    };

    const handleAudioTrackChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      // Multiple-select: load every chosen file as its own track.
      for (const file of Array.from(e.target.files ?? [])) onLoadAudioTrack(file);
      e.target.value = '';
    };

    return (
      <div className={styles.toolbar}>
        {examples.length > 0 && (
          <>
            <label htmlFor="drumjot-example-select" className={styles.toolbarLabel}>
              Example
            </label>
            <Select
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
            </Select>
            <span className={styles.toolbarDivider} aria-hidden="true" />
          </>
        )}
        <DropdownButton label="Load" title="Load a score or audio tracks from disk">
          {(close) => (
            <>
              <button
                type="button"
                className={styles.dropdownItem}
                onClick={() => {
                  jotInputRef.current?.click();
                  close();
                }}
                title="Load a Drumjot DSL file (`.jot`) from disk and render it. Parser runs entirely client-side; no transcriber service required."
              >
                Load .jot file
              </button>
              <button
                type="button"
                className={styles.dropdownItem}
                onClick={() => {
                  paradbInputRef.current?.click();
                  close();
                }}
                title="Load a ParaDB / Paradiddle map pack (`.zip`). The chart is converted to a score and its audio tracks are loaded automatically for play-along practice. Runs entirely client-side."
              >
                Load ParaDB map (.zip)
              </button>
              <button
                type="button"
                className={styles.dropdownItem}
                onClick={() => {
                  audioTrackInputRef.current?.click();
                  close();
                }}
                title="Load one or more audio files (FLAC / WAV / MP3 / ...) as backing tracks. Each plays alongside the MIDI drums and shows a waveform aligned to the score; mute/solo/volume each from its track gutter. Select multiple files to load them all at once."
              >
                Load audio track(s)
              </button>
            </>
          )}
        </DropdownButton>
        <DropdownButton
          label={uploading ? 'Transcribing…' : 'Transcribe'}
          className={styles.transcribeButton}
          panelClassName={styles.dropdownPanelWide}
          title="Transcribe an audio file to a Jot, with refinement options"
        >
          {(close) => (
            <>
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
                <Select
                  className={styles.samplesSelect}
                  value={transcribeOptions.bestOfK}
                  disabled={uploading}
                  onChange={(e) => onSetBestOfK(Number(e.target.value))}
                >
                  <option value={1}>1</option>
                  <option value={3}>3</option>
                  <option value={5}>5</option>
                </Select>
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
              <span className={styles.dropdownDivider} aria-hidden="true" />
              <button
                type="button"
                className={styles.transcribeButton}
                onClick={() => {
                  fileInputRef.current?.click();
                  close();
                }}
                disabled={uploading}
                title="Upload an audio file; the transcriber service will return a Jot. The Python backend decodes anything ffmpeg understands."
              >
                {uploading ? 'Transcribing…' : 'Transcribe audio…'}
              </button>
              {uploading && (
                <button
                  type="button"
                  className={styles.cancelButton}
                  onClick={() => {
                    onCancelTranscribe();
                    close();
                  }}
                  title="Abort the in-flight transcription request."
                >
                  Stop transcription
                </button>
              )}
            </>
          )}
        </DropdownButton>
        {/* Hidden file inputs live outside the dropdown panels so the
            refs stay mounted whether or not a menu is open. */}
        <input
          ref={jotInputRef}
          type="file"
          accept=".jot,.txt,text/plain"
          className={styles.hiddenInput}
          onChange={handleJotFileChange}
        />
        <input
          ref={paradbInputRef}
          type="file"
          accept=".zip,application/zip"
          className={styles.hiddenInput}
          onChange={handleParadbChange}
        />
        <input
          ref={audioTrackInputRef}
          type="file"
          accept="audio/*,.flac"
          multiple
          className={styles.hiddenInput}
          onChange={handleAudioTrackChange}
        />
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
        <span className={styles.toolbarDivider} aria-hidden="true" />
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
        <TranscribeStatusPill status={transcribeStatus} onClear={onClearTranscribeStatus} />
      </div>
    );
  }
);

const PLAYBACK_SPEEDS: readonly number[] = [0.25, 0.5, 0.75, 1.0, 1.25];

function samplePct(p: SampleLoadProgress): number {
  return Math.min(100, Math.round((p.loaded / p.total) * 100));
}

/** Bar fill width: real percentage when the server sent a size,
 * otherwise a fixed sliver so an unknown-total download still reads as
 * "working" rather than empty. */
function sampleProgressWidth(p: SampleLoadProgress | undefined): string {
  if (!p) return '8%';
  if (p.fromCache) return '100%';
  return p.total > 0 ? `${samplePct(p)}%` : '40%';
}

function sampleProgressLabel(p: SampleLoadProgress | undefined): string {
  if (!p) return 'Drums…';
  if (p.fromCache) return 'Drums (cached)';
  return p.total > 0 ? `Drums ${samplePct(p)}%` : 'Drums…';
}

const PlaybackControls = observer(
  ({
    hasJot,
    playerState,
    playerError,
    sampleProgress,
    playbackSpeed,
    drumKits,
    drumPreset,
    onTogglePlayPause,
    onStop,
    onSetPlaybackSpeed,
    onSetDrumPreset,
  }: {
    hasJot: boolean;
    playerState: PlayerState;
    playerError: string | undefined;
    sampleProgress: SampleLoadProgress | undefined;
    playbackSpeed: number;
    drumKits: KitInfo[];
    drumPreset: number;
    onTogglePlayPause: () => void;
    onStop: () => void;
    onSetPlaybackSpeed: (speed: number) => void;
    onSetDrumPreset: (preset: number) => void;
  }) => {
    const loading = playerState === 'loading';
    const playing = playerState === 'playing';
    const paused = playerState === 'paused';
    // Playback is "active" (Stop is meaningful, playhead is on screen)
    // while either playing or paused.
    const active = playing || paused;
    const hasError = !!playerError && !loading && !active;
    // Icon-only, like a media player. Glyphs: ▶ play/resume, ⏸ pause,
    // ■ stop, ⚠ error, ⏳ loading.
    const transportIcon = loading ? '⏳' : playing ? '⏸' : hasError ? '⚠' : '▶';
    const transportAria = loading
      ? 'Loading'
      : playing
        ? 'Pause'
        : paused
          ? 'Resume'
          : 'Play';
    return (
      <>
        {/* Empty left cell balances the right-hand aux controls so the
            transport group stays optically centred in the bar. */}
        <div className={styles.transportSpacer} aria-hidden="true" />
        <div className={styles.transportCenter}>
          <button
            type="button"
            className={classNames(
              styles.transportButton,
              hasError && styles.transportButtonError
            )}
            onClick={onTogglePlayPause}
            disabled={!hasJot || loading}
            aria-label={transportAria}
            title={
              playerError
                ? `Playback error: ${playerError}`
                : playing
                  ? 'Pause playback (spacebar). The playhead and audio freeze in place; press again to resume.'
                  : paused
                    ? 'Resume playback (spacebar).'
                    : 'Play the current jot through an acoustic General MIDI drum kit (GeneralUser GS, spacebar also toggles play/pause). The first play downloads a ~30 MB SoundFont; it is then cached in the browser for instant loads on later sessions.'
            }
          >
            {transportIcon}
          </button>
          <button
            type="button"
            className={classNames(
              styles.transportButton,
              styles.transportButtonStop,
              styles.transportStop
            )}
            onClick={onStop}
            disabled={!active}
            aria-label="Stop"
            title={
              active
                ? 'Stop playback and reset to the start.'
                : 'Stop (available once playback has started).'
            }
          >
            ■
          </button>
        </div>
        <div className={styles.transportAux}>
          {drumKits.length > 0 && (
            <label
              className={styles.toolbarCheckbox}
              title="Drum kit (a preset of the GeneralUser GS SoundFont). Switching is instant — the SoundFont is already downloaded; only the active samples change. Takes effect immediately, including mid-playback."
            >
              <span>Kit</span>
              <Select
                className={styles.samplesSelect}
                value={String(drumPreset)}
                onChange={(e) => onSetDrumPreset(Number(e.target.value))}
              >
                {drumKits.map((k) => (
                  <option key={k.preset} value={String(k.preset)}>
                    {k.name}
                  </option>
                ))}
              </Select>
            </label>
          )}
          <label
            className={styles.toolbarCheckbox}
            title="Tempo multiplier applied to playback. Slowing down spaces the drum hits further apart and time-stretches the audio tracks — pitch is preserved for both, so a half-speed practice pass stays in tune."
          >
            <span>Speed</span>
            <Select
              className={styles.samplesSelect}
              value={String(playbackSpeed)}
              onChange={(e) => onSetPlaybackSpeed(Number(e.target.value))}
            >
              {PLAYBACK_SPEEDS.map((s) => (
                <option key={s} value={String(s)}>
                  {s.toFixed(2)}×
                </option>
              ))}
            </Select>
          </label>
          {loading && (
            <span
              className={styles.sampleProgress}
              title="Downloading the GeneralUser GS SoundFont (~30 MB, one time). Cached in the browser after the first load — instant next time."
            >
              <span className={styles.sampleProgressTrack}>
                <span
                  className={styles.sampleProgressFill}
                  style={{ width: sampleProgressWidth(sampleProgress) }}
                />
              </span>
              <span>{sampleProgressLabel(sampleProgress)}</span>
            </span>
          )}
          {hasError && (
            <span
              className={classNames(styles.statusPill, styles.statusPillError)}
              title={playerError}
            >
              Playback: {truncate(playerError ?? '', 60)}
            </span>
          )}
        </div>
      </>
    );
  }
);

/**
 * Bottom transport bar. Pinned below the score so the (formerly
 * header-crowding) play / pause / stop / speed controls have their own
 * dedicated strip. `observer` + reading `jotPlayer` here keeps player
 * state re-renders scoped to this bar instead of bubbling up through
 * `View` and re-rendering the score on every transport change.
 */
const PlaybackBar = observer(({ store }: { store: JotViewStore }) => (
  <div className={styles.playbackBar}>
    <PlaybackControls
      hasJot={!!store.currentJot}
      playerState={jotPlayer.state}
      playerError={jotPlayer.errorMessage}
      sampleProgress={jotPlayer.sampleLoadProgress}
      playbackSpeed={jotPlayer.playbackSpeed}
      drumKits={jotPlayer.drumKits}
      drumPreset={jotPlayer.drumPreset}
      onTogglePlayPause={() => store.togglePlayPause()}
      onStop={() => store.stopPlayback()}
      onSetPlaybackSpeed={(s) => jotPlayer.setPlaybackSpeed(s)}
      onSetDrumPreset={(p) => jotPlayer.setDrumPreset(p)}
    />
  </div>
));

const Playhead = observer(() => {
  const timeline = jotPlayer.timeline;
  const active =
    jotPlayer.state === 'playing' ||
    jotPlayer.state === 'paused' ||
    // Idle but the user clicked to position the playhead before
    // pressing Play — show it parked at the cued spot.
    jotPlayer.cued;
  if (!active || timeline.bars.length === 0) return null;
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
  /** Current row fader value, 0..1 (1 = full). */
  volumeFor: (pitch: string) => number;
  onSetVolume: (pitch: string, v: number) => void;
  onToggleMute: (pitch: string) => void;
  onToggleSolo: (pitch: string) => void;
};

type AudioTrackControls = {
  mutedAudioTracks: ReadonlySet<AudioTrackId>;
  soloedAudioTracks: ReadonlySet<AudioTrackId>;
  isAudioTrackAudible: (id: AudioTrackId) => boolean;
  volumeFor: (id: AudioTrackId) => number;
  onSetVolume: (id: AudioTrackId, v: number) => void;
  onToggleMute: (id: AudioTrackId) => void;
  onToggleSolo: (id: AudioTrackId) => void;
  /** Drop a loaded audio track (button in the gutter clears the slot). */
  onClear: (id: AudioTrackId) => void;
};

type JotViewProps = {
  jot: RenderedJot;
  marquee: Box | undefined;
  highlightedPattern: string | undefined;
  onPatternClick: (name: string) => void;
  onMouseDown: (e: React.MouseEvent<HTMLDivElement>) => void;
  onMouseMove: (e: React.MouseEvent<HTMLDivElement>) => void;
  onMouseUp: () => void;
  /** Click-to-seek with a bars-row-local pixel x. */
  onSeek: (x: number) => void;
  /**
   * Multiply the current score zoom by `factor` (Cmd/Ctrl + wheel).
   * The store clamps to the slider's range.
   */
  onZoomBy: (factor: number) => void;
  voiceControls: VoiceControls;
  audioTrackControls: AudioTrackControls;
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
    onSeek,
    onZoomBy,
    voiceControls,
    audioTrackControls,
  } = props;
  const resolved = jot.resolved;
  const config = jot.config;
  const containerRef = React.useRef<HTMLDivElement>(null);

  // Cmd/Ctrl + wheel zooms the score (mirrors the Zoom slider). The
  // listener is registered natively with `{ passive: false }` because
  // React's synthetic `onWheel` is passive — `preventDefault` there is
  // a no-op, and we must cancel the browser's own page zoom on
  // Ctrl/Cmd + wheel. `ctrlKey` also covers the macOS trackpad pinch
  // gesture, which Chrome/Safari deliver as a synthetic Ctrl + wheel.
  const onZoomByRef = React.useRef(onZoomBy);
  onZoomByRef.current = onZoomBy;
  React.useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      // deltaMode 1 = lines (typically a notched mouse wheel); scale
      // it up so a single notch zooms a comparable amount to a
      // pixel-mode trackpad swipe. Scrolling up (deltaY < 0) zooms in.
      const unit = e.deltaMode === 1 ? 16 : 1;
      onZoomByRef.current(Math.exp(-e.deltaY * unit * 0.0015));
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

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
      <AudioTracksView jot={jot} audioTrackControls={audioTrackControls} onSeek={onSeek} />
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
            onSeek={onSeek}
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
 * Header section above the staff that hosts the loaded audio tracks.
 * Each track renders as one row with the same gutter geometry as the
 * per-pitch lanes, plus a Canvas-rendered waveform aligned to the
 * score's bar grid (audio time → jot time → pixel x via the playback
 * timeline). Mute/solo work the same way as the per-pitch controls —
 * toggles propagate to live playback through MobX.
 */
const AudioTracksView = observer(
  ({
    jot,
    audioTrackControls,
    onSeek,
  }: {
    jot: RenderedJot;
    audioTrackControls: AudioTrackControls;
    onSeek: (x: number) => void;
  }) => {
    // One row per loaded audio track, in load order (the player's track
    // map preserves insertion order). Nothing is rendered when none are
    // loaded so the gap between the legend and the staff stays flush.
    const tracks = Array.from(jotPlayer.audioTracks.values());
    if (tracks.length === 0) return null;
    return (
      <div className={styles.musicTracks}>
        {tracks.map((track) => (
          <AudioTrackRow
            key={track.id}
            id={track.id}
            track={track}
            jot={jot}
            controls={audioTrackControls}
            onSeek={onSeek}
          />
        ))}
      </div>
    );
  }
);

/** Audio-track display name: filename with its extension stripped. */
function audioTrackLabel(filename: string): string {
  return filename.replace(/\.[^./\\]+$/, '') || filename;
}
const AUDIO_TRACK_HEIGHT = 56;

const AudioTrackRow = observer(
  ({
    id,
    track,
    jot,
    controls,
    onSeek,
  }: {
    id: AudioTrackId;
    track: AudioTrack;
    jot: RenderedJot;
    controls: AudioTrackControls;
    onSeek: (x: number) => void;
  }) => {
    const voice = jot.resolved.voices[0];
    const width = (voice?.width ?? 0) as number;
    const audible = controls.isAudioTrackAudible(id);
    const muted = controls.mutedAudioTracks.has(id);
    const soloed = controls.soloedAudioTracks.has(id);
    const label = audioTrackLabel(track.filename);
    const lc = `"${track.filename}"`;
    return (
      <div className={styles.musicTrack} data-testid={`audio-track-row-${id}`}>
        <div className={styles.musicTrackGutter} style={{ height: AUDIO_TRACK_HEIGHT }}>
          <div className={classNames(styles.musicTrackLabel, !audible && styles.musicTrackLabelDim)}>
            <span className={styles.musicTrackName}>{label}</span>
            <span className={styles.musicTrackFile} title={track.filename}>
              {track.filename}
            </span>
          </div>
          <div className={styles.musicTrackButtons}>
            <RowVolumeSlider
              value={controls.volumeFor(id)}
              onChange={(v) => controls.onSetVolume(id, v)}
              label={`${label} audio track`}
            />
            {/* Clear sits first so Mute/Solo stay flush with the gutter's
                right edge — lining up with the M/S column on the
                instrument rows below (both gutters share a width). */}
            <button
              type="button"
              className={styles.musicTrackClear}
              onClick={(e) => {
                e.stopPropagation();
                controls.onClear(id);
              }}
              onMouseDown={(e) => e.stopPropagation()}
              title={`Remove the ${lc} audio track`}
              aria-label={`Remove the ${lc} audio track`}
              data-testid={`audio-track-clear-${id}`}
            >
              ×
            </button>
            <button
              type="button"
              className={classNames(styles.muteButton, muted && styles.muteButtonActive)}
              onClick={(e) => {
                e.stopPropagation();
                controls.onToggleMute(id);
              }}
              onMouseDown={(e) => e.stopPropagation()}
              title={muted ? `Unmute ${lc} audio track` : `Mute ${lc} audio track`}
              aria-label={muted ? `Unmute ${lc} audio track` : `Mute ${lc} audio track`}
              aria-pressed={muted}
              data-testid={`audio-track-mute-${id}`}
            >
              M
            </button>
            <button
              type="button"
              className={classNames(styles.soloButton, soloed && styles.soloButtonActive)}
              onClick={(e) => {
                e.stopPropagation();
                controls.onToggleSolo(id);
              }}
              onMouseDown={(e) => e.stopPropagation()}
              title={soloed ? `Unsolo ${lc} audio track` : `Solo ${lc} audio track`}
              aria-label={soloed ? `Unsolo ${lc} audio track` : `Solo ${lc} audio track`}
              aria-pressed={soloed}
              data-testid={`audio-track-solo-${id}`}
            >
              S
            </button>
          </div>
        </div>
        <div
          className={styles.musicTrackBarsRow}
          style={{ width, height: AUDIO_TRACK_HEIGHT }}
          onClick={(e) => seekFromClick(e, onSeek)}
        >
          <AudioTrackWaveformCanvas
            jot={jot}
            track={track}
            width={width}
            height={AUDIO_TRACK_HEIGHT}
            dim={!audible}
            testId={`audio-track-waveform-${id}`}
          />
          <Playhead />
        </div>
      </div>
    );
  }
);

/**
 * Canvas-rendered waveform for one audio track, aligned to the score's
 * bar timeline. Peaks are recomputed in a `useEffect` whenever the
 * (zoom-dependent) total bar width changes or the underlying track
 * swaps — same cadence the score uses to re-flow under
 * `viewConfig.barWidth`.
 */
const AudioTrackWaveformCanvas = observer(
  ({
    jot,
    track,
    width,
    height,
    dim,
    testId,
  }: {
    jot: RenderedJot;
    track: AudioTrack;
    width: number;
    height: number;
    dim: boolean;
    testId?: string;
  }) => {
    const canvasRef = React.useRef<HTMLCanvasElement>(null);

    React.useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas || width <= 0) return;
      const dpr = window.devicePixelRatio || 1;
      // Browsers cap a canvas's backing-store dimensions (and total
      // area). A long score at high zoom × dpr easily blows past that
      // and throws "Canvas exceeds max size". Clamp the backing store;
      // the element stays CSS-sized to `width`, so past the cap it just
      // renders at reduced horizontal resolution instead of crashing.
      // 16384 is the safe cross-browser per-axis limit (Safari/iOS is
      // the tightest; Chrome/Firefox allow more).
      const MAX_CANVAS_DIM = 16384;
      const backingW = Math.min(Math.max(1, Math.floor(width * dpr)), MAX_CANVAS_DIM);
      const backingH = Math.min(Math.max(1, Math.floor(height * dpr)), MAX_CANVAS_DIM);
      canvas.width = backingW;
      canvas.height = backingH;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      // Map CSS-pixel drawing coords (0..width, 0..height) onto the
      // possibly-clamped backing store. Reduces to ctx.scale(dpr, dpr)
      // when nothing was clamped.
      ctx.setTransform(backingW / width, 0, 0, backingH / height, 0, 0);

      const rawOffset = jot.resolved.globalMetadata.startOffset;
      const startOffsetSec =
        typeof rawOffset === 'number' && rawOffset > 0 ? rawOffset : 0;
      const { peaks } = computeWaveformPeaksForJot(
        jot,
        track.mono,
        track.buffer.sampleRate,
        startOffsetSec,
      );

      ctx.clearRect(0, 0, width, height);
      const mid = height / 2;
      ctx.fillStyle = dim ? '#d3c8b6' : '#5BA8E8';
      // Each pixel column is a vertical line from min*scale to max*scale.
      // A single fillRect per column is faster than building a Path2D
      // for thousands of segments and lets us keep the colour-by-column
      // option open if we ever want to tint clipped peaks differently.
      const scale = mid * 0.95;
      for (let p = 0; p < width; p++) {
        const mn = peaks[p * 2];
        const mx = peaks[p * 2 + 1];
        if (mn === 0 && mx === 0) continue;
        const y0 = mid - mx * scale;
        const y1 = mid - mn * scale;
        ctx.fillRect(p, y0, 1, Math.max(1, y1 - y0));
      }
    }, [jot, track, width, height, dim]);

    if (width <= 0) return null;
    return (
      <canvas
        ref={canvasRef}
        className={styles.musicTrackWaveform}
        style={{ width, height }}
        data-testid={testId}
      />
    );
  }
);

/**
 * Side-effect-only component: keeps the playhead pinned to the
 * horizontal centre of the viewport during playback by tracking
 * `scrollLeft` to it every frame. `scrollLeft` is auto-clamped by the
 * browser, so near the start / end of the score — where there isn't
 * enough content on one side to centre — the playhead simply rides
 * toward that edge instead of snapping. Renders nothing.
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
      // Pin the playhead to the viewport's horizontal centre. Assigning
      // an out-of-range scrollLeft is clamped by the browser, so the
      // first/last screenful (not enough content to centre) degrades
      // gracefully — the playhead rides toward that edge instead.
      const viewportCenter = containerRect.left + containerRect.width / 2;
      container.scrollLeft += playheadViewportX - viewportCenter;
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
    onSeek,
    voiceControls,
  }: {
    voice: ResolvedVoice;
    config: ViewConfig;
    index: number;
    totalVoices: number;
    highlightedPattern: string | undefined;
    onPatternClick: (name: string) => void;
    onSeek: (x: number) => void;
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
                volume={voiceControls.volumeFor(pitch)}
                onSetVolume={voiceControls.onSetVolume}
                onToggleMute={voiceControls.onToggleMute}
                onToggleSolo={voiceControls.onToggleSolo}
              />
            ))}
          </div>
          <div
            className={styles.barsRow}
            style={{ width: voice.width }}
            onClick={(e) => seekFromClick(e, onSeek)}
          >
            {voice.leadInPx > 0 && (
              <div
                className={styles.leadIn}
                style={{ width: voice.leadInPx, height: staffHeight }}
                title={`Lead-in: ${voice.leadInSec.toFixed(
                  2
                )}s of pre-roll before the first beat — keeps the drum notation aligned with a loaded audio-track waveform.`}
              >
                <span className={styles.leadInLabel}>lead-in</span>
              </div>
            )}
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
 * Compact horizontal volume fader shared by the pitch gutter and the
 * audio-track gutter. Range is 0..1 (pure attenuation). All mouse events are
 * kept from bubbling so dragging the fader doesn't start the page-level
 * marquee selection or trip the seek-on-click handler.
 */
const RowVolumeSlider = ({
  value,
  onChange,
  label,
}: {
  value: number;
  onChange: (v: number) => void;
  label: string;
}) => {
  const stop = (e: React.MouseEvent) => e.stopPropagation();
  return (
    <input
      type="range"
      className={styles.rowVolume}
      min={0}
      max={1}
      step={VOLUME_STEP}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      onClick={stop}
      onMouseDown={stop}
      onMouseUp={stop}
      title={`${label} volume: ${Math.round(value * 100)}%`}
      aria-label={`${label} volume`}
    />
  );
};

/**
 * One pitch row in the lane gutter: shows the DSL letter, a volume
 * fader, and a Mute and Solo button. Stops click propagation so the
 * controls don't also fire the page-level mouse selection logic.
 */
const GutterCell = observer(
  ({
    pitch,
    height,
    instrumentName,
    muted,
    soloed,
    audible,
    volume,
    onSetVolume,
    onToggleMute,
    onToggleSolo,
  }: {
    pitch: string;
    height: Pixels;
    instrumentName: string | undefined;
    muted: boolean;
    soloed: boolean;
    audible: boolean;
    volume: number;
    onSetVolume: (pitch: string, v: number) => void;
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
        <RowVolumeSlider
          value={volume}
          onChange={(v) => onSetVolume(pitch, v)}
          label={instrumentName ?? `Pitch ${pitch}`}
        />
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
                  // A non-straight note already inside a tuplet bracket
                  // is explained by that bracket, so only flag the
                  // strays (e.g. an off-grid note not authored as a
                  // group) individually.
                  offGrid={!note.straight && !coveredByTuplet(bar, note.beat)}
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
        {bar.tupletSpans.map((span, i) => (
          <TupletBracket key={i} span={span} />
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
          data-noseek="true"
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

/**
 * Classic engraved tuplet bracket: a thin line over the grouped notes
 * with the slot count (3 = triplet, 5 = quintuplet, ...) on it. Purely
 * decorative — no interaction, unlike the pattern bracket.
 */
const TupletBracket = observer(({ span }: { span: TupletSpan }) => (
  <div
    className={styles.tupletBracket}
    style={{ left: span.x, width: span.width }}
    title={`${span.count}-tuplet (not a straight subdivision)`}
  >
    <span className={styles.tupletNumber}>{span.count}</span>
  </div>
));

/**
 * Shared click-to-seek handler for the score bars row and the audio-track
 * waveforms. Bails on clicks that originated on a note, pattern label,
 * or anything else tagged `data-noseek` so those keep their own
 * behaviour. `e.currentTarget` is the bars-row element whose left edge
 * is x=0 in `bar.x` space, so `clientX - rect.left` is the bars-row-
 * local pixel regardless of horizontal scroll.
 */
function seekFromClick(
  e: React.MouseEvent<HTMLDivElement>,
  onSeek: (x: number) => void
): void {
  if (e.button !== 0) return;
  if ((e.target as HTMLElement).closest('[data-noseek]')) return;
  const rect = e.currentTarget.getBoundingClientRect();
  onSeek(e.clientX - rect.left);
}

/**
 * True when `beat` falls inside any tuplet bracket on this bar. The
 * upper bound is inclusive because `endBeat` is now the last slot's
 * onset (see jot.ts) — the final tuplet note sits exactly on it and is
 * still covered by the bracket.
 */
function coveredByTuplet(bar: ResolvedBar, beat: number): boolean {
  const eps = 1e-6;
  return bar.tupletSpans.some(
    (s) => beat >= s.startBeat - eps && beat <= s.endBeat + eps
  );
}

const NoteView = observer(
  ({
    note,
    color,
    config,
    instrument,
    offGrid,
  }: {
    note: ResolvedNote;
    color: string;
    config: ViewConfig;
    instrument: Instrument;
    offGrid: boolean;
  }) => {
    const isAccent = note.modifiers.has('a');
    const isGhost = note.modifiers.has('g');
    const isFlam = note.modifiers.has('fl');
    const isDrag = note.modifiers.has('dr');
    const isCross = note.modifiers.has('x');
    const badge = pickBadge(note);

    return (
      <div
        // Notes opt out of click-to-seek so clicking a note keeps its
        // own meaning (tooltip / future selection) instead of moving
        // the playhead.
        data-noseek="true"
        className={classNames(
          styles.note,
          isAccent && styles.accent,
          isGhost && styles.ghost,
          note.roll && styles.roll,
          offGrid && styles.offGrid
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
        title={
          offGrid
            ? `${describeNote(note, instrument)} — off the straight grid (triplet/tuplet)`
            : describeNote(note, instrument)
        }
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
