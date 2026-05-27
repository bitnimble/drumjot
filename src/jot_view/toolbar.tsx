import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { ExampleJot } from 'src/fakes';
import {
  jotPlayer,
  PLAYBACK_SPEED_MAX,
  PLAYBACK_SPEED_MIN,
  PLAYBACK_SPEED_STEP,
  SampleLoadProgress,
} from 'src/playback';
import { themeStore, ThemeMode } from 'src/theme';
import {
  BeatInput,
  LLM_MODEL_LABELS,
  LLM_MODEL_ORDER,
  LlmModel,
  TranscribeStage,
  TranscriptionSummary,
} from 'src/transcriber';
import sharedStyles from '../jot_view.module.css';
import {
  DropdownButton,
  DropdownSection,
  dropdownStyles,
  SubmenuItem,
  ToggleMenuItem,
} from './components/dropdown';
import { NumberStepper } from './components/number_stepper';
import { Tabs } from './components/tabs';
import { formatTranscriptionSummary, RecentTranscriptionsPicker } from './recent_transcriptions';
import styles from './toolbar.module.css';
import {
  GridLineSettings,
  JotViewStore,
  LyricsAlignStatus,
  TranscribeOptions,
  TranscribeStatus,
} from './store';

/** Stage labels in pipeline order, shown verbatim in the resume stage
 *  picker. Mirrors `Stage` in `transcriber/app/pipeline/runner.py`. */
const STAGE_ORDER: readonly TranscribeStage[] = [
  'stems_all',
  'stems_per',
  'beats',
  'onsets',
  'filter',
  'quantise',
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
export const Select = ({ onChange, ...rest }: React.ComponentPropsWithoutRef<'select'>) => (
  <select
    {...rest}
    onChange={(e) => {
      onChange?.(e);
      e.currentTarget.blur();
    }}
  />
);

/**
 * Toolbar dropdown trigger label with a trailing caret indicator. The
 * shared {@link DropdownButton} no longer renders the caret itself
 * (overflow-icon callers like the mixer don't want one); toolbar
 * triggers compose it via this helper instead.
 */
const ToolbarDropdownLabel = ({ children }: { children: React.ReactNode }) => (
  <>
    {children} <span aria-hidden="true">▾</span>
  </>
);

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
    case 'quantise':
      return 'quantising onsets';
    case 'transcribe':
      return 'rendering MIDI';
  }
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
    onLoadLyricsFile,
    onOpenLyricsTextLoad,
    onOpenLyricsSearch,
    onClearLyrics,
    hasLyrics,
    onCancelTranscribe,
    lyricsAlignStatus,
    onSetBeatInput,
    onSetLlmModel,
    zoom,
    onSetZoom,
    hasNoteProvenance,
    showFilteredOnsets,
    onSetShowFilteredOnsets,
    gridLines,
    onToggleGridLine,
    uniformWaveforms,
    onSetUniformWaveforms,
    recentTranscriptions,
    recentTranscriptionsLoaded,
    recentTranscriptionsLoading,
    selectedResumeFolder,
    selectedResumeStage,
    onSetSelectedResumeFolder,
    onSetSelectedResumeStage,
    onRefreshRecentTranscriptions,
    onLoadRecentTranscription,
    transcribeMode,
    onSetTranscribeMode,
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
    /** Load a synced-lyrics file (.lrc or .txt in LRC format) from disk.
     *  Pushes the parsed lines straight into the session lyrics store. */
    onLoadLyricsFile: (file: File) => void;
    /** Open the plain-text lyrics loader modal. The modal owns its own
     *  textarea + file picker; the toolbar's only job is to open it. */
    onOpenLyricsTextLoad: () => void;
    /** Open the LRCLIB search modal. The modal pre-fills + auto-fires
     *  against the current jot's title/artist; the toolbar's only job
     *  is to surface the entry point. */
    onOpenLyricsSearch: () => void;
    /** Drop the session lyrics store. The gutter Clear button on the
     *  row has its own affordance; this is the alternate menu path. */
    onClearLyrics: () => void;
    /** True when the session lyrics store has lines loaded; used to
     *  enable a "Clear lyrics" menu item inside the Lyrics dropdown. */
    hasLyrics: boolean;
    onCancelTranscribe: () => void;
    /** Status of the in-flight Whisper lyric-alignment request. Rendered
     *  as a busy pill in the toolbar so the user sees that the backend
     *  is extracting vocals + running whisperx after they pick "Align
     *  with Whisper" from the Lyrics menu. */
    lyricsAlignStatus: LyricsAlignStatus;
    onSetBeatInput: (input: BeatInput) => void;
    onSetLlmModel: (model: LlmModel) => void;
    zoom: number;
    onSetZoom: (z: number) => void;
    /** True iff a filter-mode debug bundle is loaded — gates the
     * `Show filtered` checkbox so it's only present when there's
     * actually filtered-onset data to render. */
    hasNoteProvenance: boolean;
    showFilteredOnsets: boolean;
    onSetShowFilteredOnsets: (show: boolean) => void;
    gridLines: GridLineSettings;
    onToggleGridLine: (key: keyof GridLineSettings) => void;
    uniformWaveforms: boolean;
    onSetUniformWaveforms: (on: boolean) => void;
    recentTranscriptions: readonly TranscriptionSummary[];
    /** Whether {@link onRefreshRecentTranscriptions} has resolved at
     *  least once (success or empty). The Recent submenu uses this to
     *  decide whether to issue an initial fetch on first open. */
    recentTranscriptionsLoaded: boolean;
    /** Whether a refresh is in flight; drives the Recent submenu spinner. */
    recentTranscriptionsLoading: boolean;
    selectedResumeFolder: string | undefined;
    selectedResumeStage: TranscribeStage | undefined;
    onSetSelectedResumeFolder: (folder: string | undefined) => void;
    onSetSelectedResumeStage: (stage: TranscribeStage | undefined) => void;
    onRefreshRecentTranscriptions: () => void;
    /** Load a previous transcription's already-produced debug bundle
     *  (no pipeline re-run). Used by the Load → Recent submenu. */
    onLoadRecentTranscription: (folder: string) => void;
    /** Active flow inside the Transcribe dropdown; `new` for a fresh
     *  upload, `resume` for re-running from a previous debug folder. */
    transcribeMode: 'new' | 'resume';
    onSetTranscribeMode: (mode: 'new' | 'resume') => void;
  }) => {
    const fileInputRef = React.useRef<HTMLInputElement>(null);
    const jotInputRef = React.useRef<HTMLInputElement>(null);
    const midiInputRef = React.useRef<HTMLInputElement>(null);
    const paradbInputRef = React.useRef<HTMLInputElement>(null);
    const debugBundleInputRef = React.useRef<HTMLInputElement>(null);
    const audioTrackInputRef = React.useRef<HTMLInputElement>(null);
    const lyricsInputRef = React.useRef<HTMLInputElement>(null);
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

    const handleLyricsFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onLoadLyricsFile(file);
      e.target.value = '';
    };

    return (
      <div className={styles.toolbar}>
        <DropdownButton
          label={<ToolbarDropdownLabel>Load</ToolbarDropdownLabel>}
          className={styles.playButton}
          title="Load a score or audio tracks from disk"
        >
          {(close) => (
            <>
              <RecentTranscriptionsPicker
                variant="menu"
                triggerLabel="Recent"
                items={recentTranscriptions}
                loaded={recentTranscriptionsLoaded}
                loading={recentTranscriptionsLoading}
                onRefresh={onRefreshRecentTranscriptions}
                onPick={onLoadRecentTranscription}
                onAfterPick={close}
              />
              {examples.length > 0 && (
                <>
                  <SubmenuItem label="Examples">
                    {(closeSub) =>
                      examples.map((ex) => (
                        <button
                          key={ex.id}
                          type="button"
                          className={classNames(
                            dropdownStyles.dropdownItem,
                            ex.id === currentId && dropdownStyles.dropdownItemActive
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
                  <span className={dropdownStyles.dropdownDivider} aria-hidden="true" />
                </>
              )}
              <button
                type="button"
                className={dropdownStyles.dropdownItem}
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
                className={dropdownStyles.dropdownItem}
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
                className={dropdownStyles.dropdownItem}
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
                className={dropdownStyles.dropdownItem}
                onClick={() => {
                  debugBundleInputRef.current?.click();
                  close();
                }}
                title="Load a transcriber debug bundle (`.zip`), the same artifact `Transcribe audio` produces server-side. Restores the score, every per-stem audio track (MP3), and surfaces the captured logs + per-stage timings in the debug panel for inspection. Runs entirely client-side."
              >
                Load debug bundle (.zip)
              </button>
              <button
                type="button"
                className={dropdownStyles.dropdownItem}
                onClick={() => {
                  audioTrackInputRef.current?.click();
                  close();
                }}
                title="Load one or more audio files (FLAC / WAV / MP3 / ...) as backing tracks. Each plays alongside the MIDI drums and shows a waveform aligned to the score; mute/solo/volume each from its track gutter. Select multiple files to load them all at once."
              >
                Load audio track(s)
              </button>
              <span className={dropdownStyles.dropdownDivider} aria-hidden="true" />
              <SubmenuItem
                label="Lyrics"
                title="Load time-aligned lyrics that scroll along the score timeline. Lyrics are session-only and clear when a new song is loaded."
              >
                {(closeSub) => (
                  <>
                    <button
                      type="button"
                      className={dropdownStyles.dropdownItem}
                      onClick={() => {
                        onOpenLyricsSearch();
                        closeSub();
                        close();
                      }}
                      title="Search LRCLIB (free public DB of synced lyrics) for the current song's title and artist. If exactly one match exists it loads automatically; otherwise pick one from a list."
                      data-testid="lyrics-menu-search"
                    >
                      Search LRCLIB…
                    </button>
                    <button
                      type="button"
                      className={dropdownStyles.dropdownItem}
                      onClick={() => {
                        lyricsInputRef.current?.click();
                        closeSub();
                        close();
                      }}
                      title="Load a synced-lyrics file (.lrc) from disk."
                      data-testid="lyrics-menu-load-file"
                    >
                      Load from file…
                    </button>
                    <button
                      type="button"
                      className={dropdownStyles.dropdownItem}
                      onClick={() => {
                        onOpenLyricsTextLoad();
                        closeSub();
                        close();
                      }}
                      title="Paste or type plain-text lyrics. Re-time them against an audio track afterward."
                      data-testid="lyrics-menu-load-plaintext"
                    >
                      Load from plain text…
                    </button>
                    {hasLyrics && (
                      <>
                        <span className={dropdownStyles.dropdownDivider} aria-hidden="true" />
                        <button
                          type="button"
                          className={dropdownStyles.dropdownItem}
                          onClick={() => {
                            onClearLyrics();
                            closeSub();
                            close();
                          }}
                          title="Drop the currently-loaded lyrics."
                          data-testid="lyrics-menu-clear"
                        >
                          Clear lyrics
                        </button>
                      </>
                    )}
                  </>
                )}
              </SubmenuItem>
            </>
          )}
        </DropdownButton>
        <DropdownButton
          label={
            <ToolbarDropdownLabel>
              {uploading ? 'Transcribing…' : 'Transcribe'}
            </ToolbarDropdownLabel>
          }
          className={styles.transcribeButton}
          panelClassName={dropdownStyles.dropdownPanelWide}
          title="Transcribe an audio file using the filter pathway (MIDI). The debug bundle loads automatically when the run completes."
          onOpen={onRefreshRecentTranscriptions}
        >
          {(close) => {
            const selectedSummary = recentTranscriptions.find(
              (s) => s.folder === selectedResumeFolder
            );
            const resumableSet = new Set(selectedSummary?.resumable_stages ?? []);
            const canResume =
              selectedResumeFolder !== undefined &&
              selectedResumeStage !== undefined &&
              resumableSet.has(selectedResumeStage);
            const noRecentRuns = recentTranscriptions.length === 0;
            // Force the visible flow back to `new` whenever there's
            // nothing resumable; otherwise a stale `resume` selection
            // would render an empty form on first open of the dropdown.
            const effectiveMode = noRecentRuns ? 'new' : transcribeMode;
            return (
              <>
                {/* Shared options sit ABOVE the tab strip so they read
                    as "applies to both modes" rather than belonging to
                    the active tab's body. */}
                <label
                  className={sharedStyles.toolbarCheckbox}
                  title="Which audio feeds the beat tracker. `full_mix` (default) is madmom's training distribution; `drum_stem` can help on tracks with heavy non-drum syncopation. Applies to both New and Resume."
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
                <label
                  className={sharedStyles.toolbarCheckbox}
                  title="Anthropic model used by the three classification stages (filter; hihat_split; cymbal_split). Quantise stays on Haiku regardless. Opus is the highest-quality default; Haiku is ~15× cheaper for what is mostly pattern-matching. Applies to both New and Resume."
                >
                  <span>Model</span>
                  <Select
                    className={sharedStyles.samplesSelect}
                    value={transcribeOptions.llmModel}
                    disabled={uploading}
                    onChange={(e) => onSetLlmModel(e.target.value as LlmModel)}
                  >
                    {LLM_MODEL_ORDER.map((m) => (
                      <option key={m} value={m}>
                        {LLM_MODEL_LABELS[m]}
                      </option>
                    ))}
                  </Select>
                </label>
                <Tabs
                  ariaLabel="Transcribe mode"
                  value={effectiveMode}
                  onChange={onSetTranscribeMode}
                  options={[
                    { value: 'new', label: 'New', testId: 'transcribe-tab-new' },
                    {
                      value: 'resume',
                      label: 'Resume',
                      disabled: noRecentRuns,
                      title: noRecentRuns
                        ? 'No previous runs available to resume.'
                        : 'Resume a previous run from a chosen pipeline stage.',
                      testId: 'transcribe-tab-resume',
                    },
                  ]}
                />
                {effectiveMode === 'new' ? (
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
                ) : (
                  <>
                    <label
                      className={classNames(sharedStyles.toolbarCheckbox, styles.resumeField)}
                      title="Pick a previous /transcribe run by its original filename + request time. The picker reads /transcribe/list off the server; recent runs land first."
                    >
                      <span>Previous run</span>
                      <Select
                        className={sharedStyles.samplesSelect}
                        value={selectedResumeFolder ?? ''}
                        disabled={uploading}
                        onChange={(e) => onSetSelectedResumeFolder(e.target.value || undefined)}
                      >
                        <option value="">Select a previous run...</option>
                        {recentTranscriptions.map((s) => (
                          <option key={s.folder} value={s.folder}>
                            {formatTranscriptionSummary(s)}
                          </option>
                        ))}
                      </Select>
                    </label>
                    <label
                      className={classNames(sharedStyles.toolbarCheckbox, styles.resumeField)}
                      title="Pick the stage to resume from. Stages whose prerequisites are missing on disk for the selected run are disabled."
                    >
                      <span>From stage</span>
                      <Select
                        className={sharedStyles.samplesSelect}
                        value={selectedResumeStage ?? ''}
                        disabled={uploading || selectedResumeFolder === undefined}
                        onChange={(e) =>
                          onSetSelectedResumeStage(
                            (e.target.value || undefined) as TranscribeStage | undefined
                          )
                        }
                      >
                        <option value="">Select stage...</option>
                        {STAGE_ORDER.map((stage) => (
                          <option
                            key={stage}
                            value={stage}
                            disabled={
                              selectedResumeFolder !== undefined && !resumableSet.has(stage)
                            }
                          >
                            {stage}
                          </option>
                        ))}
                      </Select>
                    </label>
                    <button
                      type="button"
                      className={styles.transcribeButton}
                      onClick={() => {
                        if (
                          selectedResumeFolder !== undefined &&
                          selectedResumeStage !== undefined
                        ) {
                          onResumeTranscribe(selectedResumeFolder, selectedResumeStage);
                          close();
                        }
                      }}
                      disabled={uploading || !canResume}
                      title="Re-run the pipeline from the chosen stage against the selected debug folder."
                    >
                      Resume
                    </button>
                  </>
                )}
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
          ref={lyricsInputRef}
          type="file"
          accept=".lrc,.txt,text/plain"
          className={styles.hiddenInput}
          onChange={handleLyricsFileChange}
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
        <DropdownButton
          label={<ToolbarDropdownLabel>View</ToolbarDropdownLabel>}
          className={styles.playButton}
          title="Toggle on-screen reference grids, overlays, and the color theme."
        >
          {() => (
            <>
              <DropdownSection label="Overlays">
                <ToggleMenuItem
                  label="Show filtered"
                  active={showFilteredOnsets}
                  onToggle={() => onSetShowFilteredOnsets(!showFilteredOnsets)}
                  disabled={!hasNoteProvenance}
                  title={
                    hasNoteProvenance
                      ? 'Render the onsets the filter LLM rejected as dashed ghost overlays at their detected (bar, beat) position. Click one to see why it was filtered out.'
                      : 'Load a filter-mode debug bundle to enable filtered-onset overlays.'
                  }
                />
              </DropdownSection>
              <DropdownSection label="Waveforms">
                <ToggleMenuItem
                  label="Uniform amplitude"
                  active={uniformWaveforms}
                  onToggle={() => onSetUniformWaveforms(!uniformWaveforms)}
                  title="Normalise each audio track's waveform so the median non-silent peak fills most of the row, regardless of the source recording's amplitude. Silence still renders as silence. Off = accurate, on = uniform (easier to see quiet recordings)."
                />
              </DropdownSection>
              <DropdownSection label="Grid lines">
                <ToggleMenuItem
                  label="Main beat"
                  active={gridLines.mainBeat}
                  onToggle={() => onToggleGridLine('mainBeat')}
                  title="Dashed line under each notehead on the main beat (1, 2, 3, 4 in 4/4)."
                />
                <ToggleMenuItem
                  label="Sub-beat (16ths)"
                  active={gridLines.subBeat16}
                  onToggle={() => onToggleGridLine('subBeat16')}
                  title="Dotted reference lines at every 16th-note position within each beat."
                />
                <ToggleMenuItem
                  label="Sub-beat (6ths / quarter triplets)"
                  active={gridLines.subBeatQuarterTriplet}
                  onToggle={() => onToggleGridLine('subBeatQuarterTriplet')}
                  title="Dotted violet reference lines at every quarter-note triplet position (3 lines per 2 beats; 6 per bar in 4/4). Use to read quarter-note triplet phrases that 8th-triplet lines fragment too finely."
                />
                <ToggleMenuItem
                  label="Sub-beat (12ths / triplets)"
                  active={gridLines.subBeatTriplet}
                  onToggle={() => onToggleGridLine('subBeatTriplet')}
                  title="Dotted violet reference lines at every triplet (1/3 of a beat) position."
                />
                <ToggleMenuItem
                  label="Sub-beat (48ths)"
                  active={gridLines.subBeat48}
                  onToggle={() => onToggleGridLine('subBeat48')}
                  title="Very faint dotted lines at every 1/48 grid position (12 per beat). Covers both the 16th and triplet positions in one grid; useful for ultra-precise timing reference."
                />
              </DropdownSection>
              <ThemeSection />
            </>
          )}
        </DropdownButton>
        <DropdownButton
          label={<ToolbarDropdownLabel>Playback</ToolbarDropdownLabel>}
          className={styles.playButton}
          title="Drum kit (sample set) and playback speed."
        >
          {() => (
            <>
              <PlaybackKitSubmenu />
              <PlaybackSpeedItem />
              <AudioLatencyItem />
            </>
          )}
        </DropdownButton>
        <label
          className={sharedStyles.toolbarCheckbox}
          title="Compress or expand the score horizontally. Has no effect on audio playback, only on how the notation is laid out."
        >
          <span>Zoom</span>
          <input
            type="range"
            min={0.1}
            max={4.0}
            step={0.05}
            value={zoom}
            onChange={(e) => onSetZoom(Number(e.target.value))}
            className={styles.zoomSlider}
            style={{ ['--value' as string]: (zoom - 0.1) / 3.9 } as React.CSSProperties}
          />
          <span className={styles.zoomValue}>{Math.round(zoom * 100)}%</span>
        </label>
        <div className={styles.toolbarSpacer} aria-hidden="true" />
        <DrumLoadingIndicator />
        <LyricsAlignBusyPill status={lyricsAlignStatus} />
        <TranscribeBusyPill status={transcribeStatus} />
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
 * Theme picker section rendered inside the View dropdown. `System`
 * (default) defers to the OS `prefers-color-scheme` and tracks live
 * changes; `Light`/`Dark` persist as an explicit override in
 * localStorage so subsequent visits skip the OS check entirely.
 *
 * Rendered as radio-style menu items (only one tick at a time); clicks
 * leave the View panel open so the user can switch and immediately
 * compare with other view toggles. The data-theme attribute is owned by
 * {@link themeStore}; this component is purely the picker.
 */
const ThemeSection = observer(() => {
  const mode = themeStore.mode;
  return (
    <DropdownSection label="Theme">
      {THEME_MODE_ORDER.map((m) => (
        <ToggleMenuItem
          key={m}
          label={THEME_MODE_LABELS[m]}
          active={mode === m}
          onToggle={() => themeStore.setMode(m)}
          title={
            m === 'system'
              ? 'Follow the OS appearance setting (prefers-color-scheme).'
              : `Use the ${m} theme regardless of the OS setting.`
          }
        />
      ))}
    </DropdownSection>
  );
});

/**
 * Drum-kit picker inside the toolbar's Playback menu. Reads
 * `jotPlayer.drumKits` + `drumPreset` directly so re-renders are scoped
 * to this submenu, not the whole toolbar. Renders disabled until the
 * SoundFont is loaded and reports its kit list, so the menu's shape
 * stays stable across the load.
 */
const PlaybackKitSubmenu = observer(() => {
  const kits = jotPlayer.drumKits;
  const current = jotPlayer.drumPreset;
  const noKits = kits.length === 0;
  return (
    <SubmenuItem
      label="Drum kit"
      disabled={noKits}
      title={
        noKits
          ? 'Drum kit picker. Available once the GeneralUser GS SoundFont has finished loading; press Play to trigger the one-time download.'
          : 'Drum kit (a preset of the GeneralUser GS SoundFont). Switching is instant, the SoundFont is already downloaded; only the active samples change. Takes effect immediately, including mid-playback.'
      }
    >
      {() =>
        kits.map((k) => (
          <ToggleMenuItem
            key={k.preset}
            label={k.name}
            active={k.preset === current}
            onToggle={() => jotPlayer.setDrumPreset(k.preset)}
          />
        ))
      }
    </SubmenuItem>
  );
});

/**
 * Numeric input inside the toolbar's Playback menu for the tempo
 * multiplier. ± buttons step by `PLAYBACK_SPEED_STEP` (0.25), so the
 * usual practice grid (0.25 / 0.5 / 0.75 / 1.0 / ...) is one click
 * apart; arbitrary typed values still snap to the grid in the player.
 * Reads `jotPlayer.playbackSpeed` directly so speed changes only
 * re-render this row, not the whole toolbar.
 */
const PlaybackSpeedItem = observer(() => (
  <label
    className={styles.dropdownStepperRow}
    title="Tempo multiplier applied to playback. Slowing down spaces the drum hits further apart and time-stretches the audio tracks (pitch preserved), so a half-speed practice pass stays in tune."
  >
    <span>Speed</span>
    <span className={styles.dropdownStepperControl}>
      <NumberStepper
        value={jotPlayer.playbackSpeed}
        onChange={(v) => jotPlayer.setPlaybackSpeed(v)}
        step={PLAYBACK_SPEED_STEP}
        min={PLAYBACK_SPEED_MIN}
        max={PLAYBACK_SPEED_MAX}
        precision={2}
        ariaLabel="Playback speed multiplier"
      />
      <span className={styles.dropdownStepperUnit}>×</span>
    </span>
  </label>
));

/**
 * Numeric input inside the toolbar's Playback menu for the audio-vs-visual
 * sync trim, in milliseconds. Positive values delay the perceived audio
 * relative to the playhead (visual leads); negative values pull audio
 * earlier. Reads `jotPlayer.audioLatencyMs` directly so changes only
 * re-render this row, not the whole toolbar.
 */
const AudioLatencyItem = observer(() => (
  <label
    className={styles.dropdownStepperRow}
    title="Adjust the on-screen playhead's position relative to the audio playback clock. Positive values mean the audio is delayed (visual playhead leads); negative values pull the audio earlier. Use this to compensate if the playhead appears to lead or lag the sound. Has no effect on the jot, beats, notes, or audio tracks themselves."
  >
    <span>Audio latency</span>
    <span className={styles.dropdownStepperControl}>
      <NumberStepper
        value={jotPlayer.audioLatencyMs}
        onChange={(v) => jotPlayer.setAudioLatencyMs(v)}
        step={5}
        min={-500}
        max={500}
        precision={0}
        ariaLabel="Audio latency adjustment in milliseconds"
      />
      <span className={styles.dropdownStepperUnit}>ms</span>
    </span>
  </label>
));

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
  p: SampleLoadProgress | undefined
): string {
  if (phase === 'decoding') return '100%';
  if (phase === 'connecting' || !p) return '8%';
  if (p.fromCache) return '100%';
  return p.total > 0 ? `${samplePct(p)}%` : '40%';
}

function sampleProgressLabel(
  phase: SampleLoadPhase | undefined,
  p: SampleLoadProgress | undefined
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

/**
 * Busy pill for the Whisper lyric-alignment flow. While the backend is
 * extracting vocals (BS-Roformer) + running whisperx the pill shows a
 * spinner + the file being aligned. Returns to nothing on completion;
 * success is signalled by the row appearing, failure by an error toast.
 */
const LyricsAlignBusyPill = observer(({ status }: { status: LyricsAlignStatus }) => {
  if (status.phase !== 'aligning') return null;
  return (
    <span
      className={classNames(sharedStyles.statusPill, sharedStyles.statusPillBusy)}
      title={`Extracting vocals + running whisperx on ${status.detail}…`}
      data-testid="lyrics-align-busy"
    >
      <span className={styles.statusPillSpinner} aria-hidden="true" />
      Aligning lyrics: {status.detail}…
    </span>
  );
});

/**
 * Busy pill for an in-flight transcribe / resume call. Surfaces the
 * live pipeline stage (and substage detail, if any) alongside the
 * filename so the operator can see what the server is actually
 * working on; fed from the NDJSON progress stream via
 * `JotViewStore.applyProgress`. Completion (success or failure) drops
 * back to nothing; the user-visible result surfaces as a toast.
 */
const TranscribeBusyPill = observer(({ status }: { status: TranscribeStatus }) => {
  if (status.phase !== 'uploading') return null;
  const stagePart = status.stage
    ? ` · ${formatStageLabel(status.stage)}${status.substage ? ` (${status.substage})` : ''}`
    : '';
  return (
    <span
      className={classNames(sharedStyles.statusPill, sharedStyles.statusPillBusy)}
      title={status.substage ?? status.stage ?? 'starting'}
    >
      <span className={styles.statusPillSpinner} aria-hidden="true" />
      Transcribing {status.filename}
      {stagePart}…
    </span>
  );
});

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
      <div className={styles.debugPanelHeader} onClick={() => store.toggleDebugPanel()}>
        <span className={styles.debugPanelTitle}>Debug bundle</span>
        <span className={styles.debugPanelStats}>
          {bundle.filename ? `${bundle.filename} · ` : ''}
          {stages.length} stage{stages.length === 1 ? '' : 's'} · {logs.length} log line
          {logs.length === 1 ? '' : 's'}
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
                  <span className={styles.debugStageElapsed}>{s.elapsed_seconds.toFixed(2)}s</span>
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
