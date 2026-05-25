import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { ExampleJot } from 'src/fakes';
import { jotPlayer, SampleLoadProgress } from 'src/playback';
import { themeStore, ThemeMode } from 'src/theme';
import {
  BeatInput,
  TranscribeStage,
  TranscriptionSummary,
} from 'src/transcriber';
import sharedStyles from '../jot_view.module.css';
import styles from './toolbar.module.css';
import { JotViewStore, TranscribeOptions, TranscribeStatus } from './store';

/** Stage labels in pipeline order — shown verbatim in the resume stage
 *  picker. Mirrors `Stage` in `transcriber/app/pipeline/runner.py`. */
const STAGE_ORDER: readonly TranscribeStage[] = [
  'stems_all',
  'stems_per',
  'beats',
  'onsets',
  'filter',
  'transcribe',
];

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
export const Select = ({
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
 *
 * Wrapped in `observer` so observable reads inside the children
 * render-prop (e.g. `transcribeOptions.onsetBackend`) are tracked
 * against THIS component's reactive context. Without that, the parent
 * Toolbar's `observer` only sees the closure being created — it never
 * dereferences the observable properties itself — so MobX has no
 * subscriber when those properties change. The store mutation lands but
 * the controlled `<select>`'s `value` prop stays stale until some
 * unrelated re-render (zoom, playback, …) rebuilds the closure.
 */
const DropdownButton = observer(({
  label,
  title,
  className,
  panelClassName,
  onOpen,
  children,
}: {
  label: React.ReactNode;
  title?: string;
  className?: string;
  panelClassName?: string;
  /** Called once each time the panel transitions from closed to open.
   *  Used by the Transcribe dropdown to refresh its recent-runs list
   *  without forcing a page reload. */
  onOpen?: () => void;
  children: (close: () => void) => React.ReactNode;
}) => {
  const [open, setOpen] = React.useState(false);
  const ref = React.useRef<HTMLDivElement>(null);
  const onOpenRef = React.useRef(onOpen);
  onOpenRef.current = onOpen;

  React.useEffect(() => {
    if (!open) return;
    onOpenRef.current?.();
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
});

/**
 * A nested menu item inside a {@link DropdownButton} panel. Renders as a
 * regular dropdown row with a trailing ▸; clicking toggles a fly-out
 * panel anchored to its right edge. Outside-click + Escape are already
 * handled by the enclosing DropdownButton (which will tear down this
 * whole subtree), so we only need local open/close state.
 */
const SubmenuItem = ({
  label,
  children,
}: {
  label: React.ReactNode;
  children: (close: () => void) => React.ReactNode;
}) => {
  const [open, setOpen] = React.useState(false);
  return (
    <div className={styles.submenu}>
      <button
        type="button"
        className={classNames(styles.dropdownItem, styles.submenuTrigger)}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        <span>{label}</span>
        <span aria-hidden="true" className={styles.submenuArrow}>▸</span>
      </button>
      {open && (
        <div className={styles.submenuPanel} role="menu">
          {children(() => setOpen(false))}
        </div>
      )}
    </div>
  );
};

/** Format one row of the resume picker. The label needs to compress
 *  three things into the cramped space of a native `<option>`: the
 *  original upload filename, when the run was originally requested, and
 *  when its artifacts were most-recently regenerated (with the resume
 *  stage tagged on if the most-recent run was a resume). All three are
 *  diagnostic — picker rows that match by filename + recent-run time
 *  are the operator's hook back to a specific working state. */
function formatTranscriptionSummary(s: TranscriptionSummary): string {
  const filename = s.original_filename ?? s.folder;
  const requested = formatTimestamp(s.requested_at);
  const lastRun = s.last_run_at ? formatTimestamp(s.last_run_at) : null;
  let detail = `requested ${requested}`;
  if (lastRun && lastRun !== requested) {
    detail += `, last run ${lastRun}`;
    if (s.last_resume_stage) {
      detail += ` (from ${s.last_resume_stage})`;
    }
  }
  return `${filename} — ${detail}`;
}

/** Human-readable label for one pipeline stage, used in the status
 *  pill. Mirrors the StrEnum values one-for-one but with friendlier
 *  wording where the raw identifier ("stems_per") reads worse than its
 *  description ("separating drum pieces"). */
function formatStageLabel(stage: TranscribeStage): string {
  switch (stage) {
    case 'stems_all':
      return 'separating drums';
    case 'stems_per':
      return 'separating drum pieces';
    case 'beats':
      return 'tracking beats';
    case 'onsets':
      return 'detecting onsets';
    case 'filter':
      return 'filtering artifact onsets';
    case 'transcribe':
      return 'rendering MIDI';
  }
}

function formatTimestamp(iso: string): string {
  // The backend stamps timestamps as UTC ISO (Z-suffixed) — see
  // `mint_request_folder_name` + `_parse_folder_timestamp` in
  // transcriber/app/pipeline/resume.py. Parse with `Date` and emit the
  // user's local wall clock so the picker doesn't surprise operators in
  // non-UTC timezones.
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export const Toolbar = observer(
  ({
    examples,
    currentId,
    onSelect,
    transcribeStatus,
    transcribeOptions,
    onTranscribe,
    onResumeTranscribe,
    onLoadJot,
    onLoadMidi,
    onLoadParadb,
    onLoadDebugBundle,
    onLoadAudioTrack,
    onCancelTranscribe,
    onClearTranscribeStatus,
    onSetBeatInput,
    zoom,
    onSetZoom,
    hasNoteProvenance,
    showFilteredOnsets,
    onSetShowFilteredOnsets,
    recentTranscriptions,
    selectedResumeFolder,
    selectedResumeStage,
    onSetSelectedResumeFolder,
    onSetSelectedResumeStage,
    onRefreshRecentTranscriptions,
  }: {
    examples: readonly ExampleJot[];
    currentId: string | undefined;
    onSelect: (id: string) => void;
    transcribeStatus: TranscribeStatus;
    transcribeOptions: TranscribeOptions;
    onTranscribe: (file: File) => void;
    onResumeTranscribe: (folder: string, stage: TranscribeStage) => void;
    onLoadJot: (file: File) => void;
    onLoadMidi: (file: File) => void;
    onLoadParadb: (file: File) => void;
    onLoadDebugBundle: (file: File) => void;
    onLoadAudioTrack: (file: File) => void;
    onCancelTranscribe: () => void;
    onClearTranscribeStatus: () => void;
    onSetBeatInput: (input: BeatInput) => void;
    zoom: number;
    onSetZoom: (z: number) => void;
    /** True iff a filter-mode debug bundle is loaded — gates the
     * `Show filtered` checkbox so it's only present when there's
     * actually filtered-onset data to render. */
    hasNoteProvenance: boolean;
    showFilteredOnsets: boolean;
    onSetShowFilteredOnsets: (show: boolean) => void;
    recentTranscriptions: readonly TranscriptionSummary[];
    selectedResumeFolder: string | undefined;
    selectedResumeStage: TranscribeStage | undefined;
    onSetSelectedResumeFolder: (folder: string | undefined) => void;
    onSetSelectedResumeStage: (stage: TranscribeStage | undefined) => void;
    onRefreshRecentTranscriptions: () => void;
  }) => {
    const fileInputRef = React.useRef<HTMLInputElement>(null);
    const jotInputRef = React.useRef<HTMLInputElement>(null);
    const midiInputRef = React.useRef<HTMLInputElement>(null);
    const paradbInputRef = React.useRef<HTMLInputElement>(null);
    const debugBundleInputRef = React.useRef<HTMLInputElement>(null);
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

    const handleMidiFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onLoadMidi(file);
      e.target.value = '';
    };

    const handleParadbChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onLoadParadb(file);
      e.target.value = '';
    };

    const handleDebugBundleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onLoadDebugBundle(file);
      e.target.value = '';
    };

    const handleAudioTrackChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      // Multiple-select: load every chosen file as its own track.
      for (const file of Array.from(e.target.files ?? [])) onLoadAudioTrack(file);
      e.target.value = '';
    };

    return (
      <div className={styles.toolbar}>
        <DropdownButton label="Load" title="Load a score or audio tracks from disk">
          {(close) => (
            <>
              {examples.length > 0 && (
                <>
                  <SubmenuItem label="Examples">
                    {(closeSub) =>
                      examples.map((ex) => (
                        <button
                          key={ex.id}
                          type="button"
                          className={classNames(
                            styles.dropdownItem,
                            ex.id === currentId && styles.dropdownItemActive,
                          )}
                          onClick={() => {
                            onSelect(ex.id);
                            closeSub();
                            close();
                          }}
                          title={`Load the built-in example "${ex.label}".`}
                        >
                          {ex.label}
                        </button>
                      ))
                    }
                  </SubmenuItem>
                  <span className={styles.dropdownDivider} aria-hidden="true" />
                </>
              )}
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
                  midiInputRef.current?.click();
                  close();
                }}
                title="Load a Standard MIDI File (`.mid`) from disk. Drum-channel notes are quantized to a 16th grid and converted to a score. Runs entirely client-side; no transcriber service required."
              >
                Load midi
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
                  debugBundleInputRef.current?.click();
                  close();
                }}
                title="Load a transcriber debug bundle (`.zip`) — the same artifact `Transcribe audio` produces server-side. Restores the score, every per-stem audio track (MP3), and surfaces the captured logs + per-stage timings in the debug panel for inspection. Runs entirely client-side."
              >
                Load debug bundle (.zip)
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
          title="Transcribe an audio file using the filter pathway (MIDI). The debug bundle loads automatically when the run completes."
          onOpen={onRefreshRecentTranscriptions}
        >
          {(close) => {
            const selectedSummary = recentTranscriptions.find(
              (s) => s.folder === selectedResumeFolder,
            );
            const resumableSet = new Set(selectedSummary?.resumable_stages ?? []);
            const canResume =
              selectedResumeFolder !== undefined &&
              selectedResumeStage !== undefined &&
              resumableSet.has(selectedResumeStage);
            return (
              <>
                <label
                  className={sharedStyles.toolbarCheckbox}
                  title="Which audio feeds the beat tracker. `full_mix` (default) is madmom's training distribution; `drum_stem` can help on tracks with heavy non-drum syncopation."
                >
                  <span>Beat input</span>
                  <Select
                    className={sharedStyles.samplesSelect}
                    value={transcribeOptions.beatInput}
                    disabled={uploading}
                    onChange={(e) => onSetBeatInput(e.target.value as BeatInput)}
                  >
                    <option value="full_mix">full mix</option>
                    <option value="drum_stem">drum stem</option>
                  </Select>
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
                  title="Pick an audio file to transcribe via the filter pathway. The debug bundle is loaded automatically when the run finishes."
                >
                  {uploading ? 'Transcribing…' : 'Select file'}
                </button>
                <span className={styles.dropdownDivider} aria-hidden="true" />
                <span className={styles.toolbarLabel}>Resume previous</span>
                <Select
                  className={sharedStyles.samplesSelect}
                  value={selectedResumeFolder ?? ''}
                  disabled={uploading || recentTranscriptions.length === 0}
                  onChange={(e) =>
                    onSetSelectedResumeFolder(e.target.value || undefined)
                  }
                  title="Pick a previous /transcribe run by its original filename + request time. The picker reads /transcribe/list off the server; recent runs land first."
                >
                  <option value="">
                    {recentTranscriptions.length === 0
                      ? 'No previous runs available'
                      : 'Select a previous run...'}
                  </option>
                  {recentTranscriptions.map((s) => (
                    <option key={s.folder} value={s.folder}>
                      {formatTranscriptionSummary(s)}
                    </option>
                  ))}
                </Select>
                <Select
                  className={sharedStyles.samplesSelect}
                  value={selectedResumeStage ?? ''}
                  disabled={uploading || selectedResumeFolder === undefined}
                  onChange={(e) =>
                    onSetSelectedResumeStage(
                      (e.target.value || undefined) as
                        | TranscribeStage
                        | undefined,
                    )
                  }
                  title="Pick the stage to resume from. Stages whose prerequisites are missing on disk for the selected run are disabled."
                >
                  <option value="">Select stage...</option>
                  {STAGE_ORDER.map((stage) => (
                    <option
                      key={stage}
                      value={stage}
                      disabled={
                        selectedResumeFolder !== undefined &&
                        !resumableSet.has(stage)
                      }
                    >
                      {stage}
                    </option>
                  ))}
                </Select>
                <button
                  type="button"
                  className={styles.transcribeButton}
                  onClick={() => {
                    if (
                      selectedResumeFolder !== undefined &&
                      selectedResumeStage !== undefined
                    ) {
                      onResumeTranscribe(
                        selectedResumeFolder,
                        selectedResumeStage,
                      );
                      close();
                    }
                  }}
                  disabled={uploading || !canResume}
                  title="Re-run the pipeline from the chosen stage against the selected debug folder."
                >
                  Resume
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
            );
          }}
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
          ref={midiInputRef}
          type="file"
          accept=".mid,.midi,audio/midi,audio/x-midi"
          className={styles.hiddenInput}
          onChange={handleMidiFileChange}
        />
        <input
          ref={paradbInputRef}
          type="file"
          accept=".zip,application/zip"
          className={styles.hiddenInput}
          onChange={handleParadbChange}
        />
        <input
          ref={debugBundleInputRef}
          type="file"
          accept=".zip,application/zip"
          className={styles.hiddenInput}
          onChange={handleDebugBundleChange}
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
          className={sharedStyles.toolbarCheckbox}
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
            style={{ ['--value' as string]: (zoom - 0.3) / 2.7 } as React.CSSProperties}
          />
          <span className={styles.zoomValue}>{Math.round(zoom * 100)}%</span>
        </label>
        {hasNoteProvenance && (
          <label
            className={sharedStyles.toolbarCheckbox}
            title="Render the onsets the filter LLM rejected as dashed ghost overlays at their detected (bar, beat) position. Click one to see why it was filtered out. Only available when a filter-mode debug bundle is loaded."
          >
            <input
              type="checkbox"
              checked={showFilteredOnsets}
              onChange={(e) => onSetShowFilteredOnsets(e.target.checked)}
            />
            Show filtered
          </label>
        )}
        <div className={styles.toolbarSpacer} aria-hidden="true" />
        <DrumLoadingIndicator />
        <TranscribeStatusPill status={transcribeStatus} onClear={onClearTranscribeStatus} />
        <ThemeMenu />
      </div>
    );
  }
);

const THEME_MODE_LABELS: Record<ThemeMode, string> = {
  system: 'System',
  light: 'Light',
  dark: 'Dark',
};

const THEME_MODE_ORDER: readonly ThemeMode[] = ['system', 'light', 'dark'];

/**
 * Trailing toolbar dropdown for picking the color theme. `System`
 * (default) defers to the OS `prefers-color-scheme` and tracks live
 * changes; `Light`/`Dark` persist as an explicit override in
 * localStorage so subsequent visits skip the OS check entirely.
 *
 * The actual data-theme attribute is owned by {@link themeStore}; this
 * component is purely the picker.
 */
const ThemeMenu = observer(() => {
  const mode = themeStore.mode;
  return (
    <DropdownButton
      label={`Theme: ${THEME_MODE_LABELS[mode]}`}
      title="Pick a color theme. `System` follows your OS appearance setting and switches live when it changes."
    >
      {(close) => (
        <>
          {THEME_MODE_ORDER.map((m) => (
            <button
              key={m}
              type="button"
              className={classNames(
                styles.dropdownItem,
                mode === m && styles.dropdownItemActive,
              )}
              onClick={() => {
                themeStore.setMode(m);
                close();
              }}
              title={
                m === 'system'
                  ? 'Follow the OS appearance setting (prefers-color-scheme).'
                  : `Use the ${m} theme regardless of the OS setting.`
              }
            >
              {THEME_MODE_LABELS[m]}
            </button>
          ))}
        </>
      )}
    </DropdownButton>
  );
});

function samplePct(p: SampleLoadProgress): number {
  return Math.min(100, Math.round((p.loaded / p.total) * 100));
}

function formatMb(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

type SampleLoadPhase = 'connecting' | 'downloading' | 'decoding';

/** Bar fill width per phase. While decoding the bytes are all in, so we
 * pin the bar at 100%; the indeterminate "connecting" / unknown-total
 * fallbacks use a fixed sliver that reads as "working" rather than empty. */
function sampleProgressWidth(
  phase: SampleLoadPhase | undefined,
  p: SampleLoadProgress | undefined,
): string {
  if (phase === 'decoding') return '100%';
  if (phase === 'connecting' || !p) return '8%';
  if (p.fromCache) return '100%';
  return p.total > 0 ? `${samplePct(p)}%` : '40%';
}

function sampleProgressLabel(
  phase: SampleLoadPhase | undefined,
  p: SampleLoadProgress | undefined,
): string {
  if (phase === 'connecting' || !p) return 'Drums · waiting for server…';
  if (phase === 'decoding') return 'Drums · decoding samples…';
  if (p.fromCache) return 'Drums · loading from cache';
  return p.total > 0
    ? `Drums · downloading ${formatMb(p.loaded)} / ${formatMb(p.total)}`
    : `Drums · downloading ${formatMb(p.loaded)}`;
}

/**
 * Top-right drum-sample download indicator. Reads `jotPlayer` directly so
 * the toolbar around it doesn't re-render on every progress tick.
 */
const DrumLoadingIndicator = observer(() => {
  if (jotPlayer.state !== 'loading') return null;
  const progress = jotPlayer.sampleLoadProgress;
  const phase = jotPlayer.sampleLoadPhase;
  return (
    <span
      className={styles.sampleProgress}
      title="One-time download of the GeneralUser GS SoundFont (~30 MB). Cached in the browser after the first load — instant next time."
    >
      <span className={styles.sampleProgressTrack}>
        <span
          className={styles.sampleProgressFill}
          style={{ width: sampleProgressWidth(phase, progress) }}
        />
      </span>
      <span>{sampleProgressLabel(phase, progress)}</span>
    </span>
  );
});

const TranscribeStatusPill = observer(
  ({ status, onClear }: { status: TranscribeStatus; onClear: () => void }) => {
    if (status.phase === 'idle') return null;
    if (status.phase === 'uploading') {
      // Surface the live pipeline stage (and substage detail, if any)
      // alongside the filename so the operator can see what the server
      // is actually working on. Fed from the NDJSON progress stream via
      // `JotViewStore.applyProgress`; the inline spinner reinforces
      // that work is in flight while text labels swap between stages.
      const stagePart = status.stage
        ? ` · ${formatStageLabel(status.stage)}${status.substage ? ` (${status.substage})` : ''}`
        : '';
      return (
        <span
          className={classNames(sharedStyles.statusPill, sharedStyles.statusPillBusy)}
          title={status.substage ?? status.stage ?? 'starting'}
        >
          <span className={styles.statusPillSpinner} aria-hidden="true" />
          Transcribing {status.filename}{stagePart}…
        </span>
      );
    }
    if (status.phase === 'error') {
      return (
        <span
          className={classNames(sharedStyles.statusPill, sharedStyles.statusPillError)}
          onClick={onClear}
          role="button"
          title={status.message}
        >
          Error: {truncate(status.message, 60)} (click to dismiss)
        </span>
      );
    }
    let detail = `@ ${status.tempo.toFixed(0)} bpm, ${status.barCount} bars`;
    if (status.hasTempoChanges) detail += ', tempo changes';
    if (status.hasTimeSigChanges) detail += ', time-sig changes';
    if (status.debugDir) {
      detail += `, debug @ ${status.debugDir}`;
    }
    const titleLines: string[] = [];
    if (status.debugDir) {
      titleLines.push(
        `Debug artifacts saved to ${status.debugDir} (under ./debug/ on the host with the default docker-compose mount).`,
      );
    }
    return (
      <span
        className={classNames(sharedStyles.statusPill, sharedStyles.statusPillSuccess)}
        title={titleLines.length > 0 ? titleLines.join('\n') : undefined}
      >
        <span onClick={onClear} role="button">
          Loaded {status.filename} {detail} (click to dismiss)
        </span>
        {status.debugZipUrl && (
          <>
            {' '}
            <a
              href={status.debugZipUrl}
              download
              data-noseek="true"
              onClick={(e) => e.stopPropagation()}
              title="Download the debug bundle (.zip) for this run — score + per-stem MP3s + JSON manifest with stage timings + the full log stream. Drop the file into `Load > Load debug bundle` to inspect it back in this UI."
            >
              [debug.zip]
            </a>
          </>
        )}
      </span>
    );
  }
);

function truncate(s: string, n: number): string {
  return s.length <= n ? s : `${s.slice(0, n - 1)}…`;
}

/**
 * Bottom-pinned drawer that surfaces a loaded transcriber debug bundle.
 *
 * Renders nothing until {@link JotViewStore.loadDebugBundleFile} has
 * populated `lastDebugBundle`. When open, shows the per-stage timings on
 * the left and the captured log stream on the right; collapses to a thin
 * toggle bar so the user can hide it without forgetting the bundle.
 *
 * No interaction beyond expand/collapse — the underlying score + audio
 * tracks are operated through the existing toolbar / gutter controls
 * exactly as if they had been loaded by hand.
 */
export const DebugPanel = observer(({ store }: { store: JotViewStore }) => {
  const bundle = store.lastDebugBundle;
  if (!bundle) return null;
  if (!store.debugPanelOpen) {
    return (
      <div
        className={styles.debugPanelToggleBar}
        role="button"
        onClick={() => store.toggleDebugPanel()}
        title="Re-open the debug panel."
      >
        ▴ Debug bundle loaded — show panel
      </div>
    );
  }
  const stages = bundle.stage_timings ?? [];
  const logs = bundle.logs ?? [];
  const totalElapsed = bundle.elapsed_seconds;
  const startedAt = bundle.started_at;
  const onResizePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    const startY = e.clientY;
    const startHeight = store.debugPanelHeight;
    const target = e.currentTarget;
    target.setPointerCapture(e.pointerId);
    const onMove = (ev: PointerEvent) => {
      // Drag up = grow the panel (top edge moves up).
      store.setDebugPanelHeight(startHeight + (startY - ev.clientY));
    };
    const onUp = (ev: PointerEvent) => {
      target.releasePointerCapture(ev.pointerId);
      target.removeEventListener('pointermove', onMove);
      target.removeEventListener('pointerup', onUp);
      target.removeEventListener('pointercancel', onUp);
    };
    target.addEventListener('pointermove', onMove);
    target.addEventListener('pointerup', onUp);
    target.addEventListener('pointercancel', onUp);
  };
  return (
    <div className={styles.debugPanel} style={{ height: store.debugPanelHeight }}>
      <div
        className={styles.debugPanelResizeHandle}
        onPointerDown={onResizePointerDown}
        title="Drag to resize the debug panel."
      />
      <div
        className={styles.debugPanelHeader}
        onClick={() => store.toggleDebugPanel()}
      >
        <span className={styles.debugPanelTitle}>Debug bundle</span>
        <span className={styles.debugPanelStats}>
          {bundle.filename ? `${bundle.filename} · ` : ''}
          {stages.length} stage{stages.length === 1 ? '' : 's'} · {logs.length} log
          line{logs.length === 1 ? '' : 's'}
          {typeof totalElapsed === 'number' ? ` · ${totalElapsed.toFixed(2)}s total` : ''}
          {startedAt ? ` · ${startedAt}` : ''}
        </span>
      </div>
      <div className={styles.debugPanelBody}>
        <div className={styles.debugPanelColumn}>
          <h4>Stage timings</h4>
          {stages.length === 0 ? (
            <p className={styles.debugPanelEmpty}>No stage timings recorded.</p>
          ) : (
            <ul className={styles.debugStageList}>
              {stages.map((s, i) => (
                <li key={i} className={styles.debugStageRow}>
                  <span className={styles.debugStageName}>{s.stage}</span>
                  <span className={styles.debugStageElapsed}>
                    {s.elapsed_seconds.toFixed(2)}s
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className={styles.debugPanelColumn}>
          <h4>Logs</h4>
          {logs.length === 0 ? (
            <p className={styles.debugPanelEmpty}>No logs captured.</p>
          ) : (
            <ul className={styles.debugLogList}>
              {logs.map((entry, i) => (
                <li key={i} className={styles.debugLogRow}>
                  <span className={styles.debugLogTimestamp}>
                    +{entry.elapsed_seconds.toFixed(2)}s
                  </span>
                  <span
                    className={classNames(
                      styles.debugLogLevel,
                      entry.level === 'WARNING' && styles.debugLogLevelWARNING,
                      entry.level === 'ERROR' && styles.debugLogLevelERROR
                    )}
                  >
                    {entry.level}
                  </span>
                  <span className={styles.debugLogLogger}>{entry.logger}</span>
                  <span className={styles.debugLogMessage}>{entry.message}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
});
