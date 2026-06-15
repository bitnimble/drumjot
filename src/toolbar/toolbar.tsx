import classNames from 'classnames';
import { ChevronDown } from 'lucide-react';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { ExampleJot } from 'src/fakes/fakes';
import {
  BeatInput,
  DRUM_SEPARATOR_LABELS,
  DRUM_SEPARATOR_ORDER,
  DrumSeparator,
  LLM_MODEL_LABELS,
  LLM_MODEL_ORDER,
  LlmModel,
  TranscribeStage,
  TranscriptionSummary,
} from 'src/jot_view/transcribe/transcriber';
import sharedStyles from '../jot_view/jot_view.module.css';
import {
  DropdownButton,
  DropdownSection,
  dropdownStyles,
  SubmenuItem,
  ToggleMenuItem,
} from 'src/ui/dropdown/dropdown';
import { Checkbox } from 'src/ui/checkbox/checkbox';
import { Logo } from 'src/ui/logo/logo';
import { Tabs } from 'src/ui/tabs/tabs';
import { formatTranscriptionSummary, RecentTranscriptionsPicker } from '../jot_view/transcribe/recent_transcriptions';
import styles from './toolbar.module.css';
import type { GridLineSettings } from 'src/settings/settings_store';
import type { TranscribeOptions, TranscribeStatus } from 'src/jot_view/transcribe/transcribe_store';
import { PlaybackKitSubmenu, PlaybackSpeedItem, AudioLatencyItem } from './playback_menu';
import { ZoomControl, ThemeSection } from './view_menu';
import { DrumLoadingIndicator, LyricsAlignBusyPill, TranscribeBusyPill } from './toolbar_status';

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
  <span className={styles.toolbarDropdownLabel}>
    {children}
    <ChevronDown size={14} aria-hidden="true" />
  </span>
);

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
    onScoreParadb,
    onLoadDebugBundle,
    onLoadAudioTrack,
    onLoadLyricsFile,
    onOpenLyricsTextLoad,
    onOpenLyricsSearch,
    onCancelTranscribe,
    lyricsAlignBusyPhase,
    onSetBeatInput,
    onSetDrumSeparator,
    onSetLlmModel,
    onSetQuantise,
    onSetQuantiseUseLlm,
    onSetZoom,
    hasNoteProvenance,
    showFilteredOnsets,
    onSetShowFilteredOnsets,
    gridLines,
    onToggleGridLine,
    uniformWaveforms,
    onSetUniformWaveforms,
    autoFollowOnPlay,
    onSetAutoFollowOnPlay,
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
    /** Score a ParaDB map against its own audio (dev test harness for the
     *  corpus-filtering scorer); reports a quality number, doesn't load it. */
    onScoreParadb: (file: File) => void;
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
    onCancelTranscribe: () => void;
    /** Aggregate lyrics-alignment state, for the toolbar busy pill (which
     *  doesn't display *which* row; the per-row spinner does). `queued`
     *  means waiting behind another GPU job; `aligning` means actively
     *  running. See `JotViewStore.lyricsAlignBusyPhase`. */
    lyricsAlignBusyPhase: 'idle' | 'queued' | 'aligning';
    onSetBeatInput: (input: BeatInput) => void;
    onSetDrumSeparator: (separator: DrumSeparator) => void;
    onSetLlmModel: (model: LlmModel) => void;
    onSetQuantise: (enabled: boolean) => void;
    onSetQuantiseUseLlm: (enabled: boolean) => void;
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
    /** When true, transitioning to playing re-enables auto-follow if it
     *  was disabled during the previous play session (pan, minimap,
     *  follow button mid-play). Off-states set while idle/paused survive
     *  regardless. */
    autoFollowOnPlay: boolean;
    onSetAutoFollowOnPlay: (on: boolean) => void;
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
    const scoreParadbInputRef = React.useRef<HTMLInputElement>(null);
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

    const handleScoreParadbChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onScoreParadb(file);
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
        <Logo size={28} title="Drumjot" />
        <DropdownButton
          label={<ToolbarDropdownLabel>File</ToolbarDropdownLabel>}
          className={styles.playButton}
          title="Load a score or audio tracks from disk"
        >
          {(close) => (
            <>
              <SubmenuItem
                label="Load"
                title="Load a score, audio tracks, or a previous transcription from disk."
              >
                {(closeSub) => {
                  const closeAll = () => {
                    closeSub();
                    close();
                  };
                  return (
                    <>
                      <RecentTranscriptionsPicker
                        variant="menu"
                        triggerLabel="Recent"
                        items={recentTranscriptions}
                        loaded={recentTranscriptionsLoaded}
                        loading={recentTranscriptionsLoading}
                        onRefresh={onRefreshRecentTranscriptions}
                        onPick={onLoadRecentTranscription}
                        onAfterPick={closeAll}
                      />
                      {examples.length > 0 && (
                        <>
                          <SubmenuItem label="Examples">
                            {(closeExamples) =>
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
                                    closeExamples();
                                    closeAll();
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
                          closeAll();
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
                          closeAll();
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
                          closeAll();
                        }}
                        title="Load a ParaDB / Paradiddle map pack (`.zip`). The chart is converted to a score and its audio tracks are loaded automatically for play-along practice. Runs entirely client-side."
                      >
                        Load ParaDB map (.zip)
                      </button>
                      <button
                        type="button"
                        className={dropdownStyles.dropdownItem}
                        onClick={() => {
                          scoreParadbInputRef.current?.click();
                          closeAll();
                        }}
                        title="Score a ParaDB map pack (`.zip`) against its own audio: how faithfully the chart's onsets line up with the detected drum onsets (0-100, after a global offset/tempo align). A dev test harness for the corpus-quality scorer, reports a number as a toast (full breakdown in the console); does NOT load the chart. Requires the transcriber service."
                      >
                        Score ParaDB map (.zip)
                      </button>
                      <button
                        type="button"
                        className={dropdownStyles.dropdownItem}
                        onClick={() => {
                          debugBundleInputRef.current?.click();
                          closeAll();
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
                          closeAll();
                        }}
                        title="Load one or more audio files (FLAC / WAV / MP3 / ...) as backing tracks. Each plays alongside the MIDI drums and shows a waveform aligned to the score; mute/solo/volume each from its track gutter. Select multiple files to load them all at once."
                      >
                        Load audio track(s)
                      </button>
                    </>
                  );
                }}
              </SubmenuItem>
              {/* Sibling of Load (not nested inside it): the SubmenuItem
                  registry treats every open submenu as mutually exclusive,
                  so a Lyrics submenu opened from inside Load would close
                  Load underneath it and the click would never reach a
                  loader. Hoisting Lyrics one level up dodges the conflict. */}
              <SubmenuItem
                label="Lyrics"
                title="Load time-aligned lyrics that scroll along the score timeline. Lyrics are session-only and clear when a new song is loaded."
              >
                {(closeLyrics) => {
                  const closeAll = () => {
                    closeLyrics();
                    close();
                  };
                  return (
                    <>
                      <button
                        type="button"
                        className={dropdownStyles.dropdownItem}
                        onClick={() => {
                          onOpenLyricsSearch();
                          closeAll();
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
                          closeAll();
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
                          closeAll();
                        }}
                        title="Paste or type plain-text lyrics. Re-time them against an audio track afterward."
                        data-testid="lyrics-menu-load-plaintext"
                      >
                        Load from plain text…
                      </button>
                    </>
                  );
                }}
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
                  title="Stage-2 drum-piece separator. MDX23C (default) is cleaner; LarsNet is ~20-40× faster but bleedier and its weights are CC BY-NC (non-commercial). Applies to both New and Resume."
                >
                  <span>Drum separator</span>
                  <Select
                    className={sharedStyles.samplesSelect}
                    value={transcribeOptions.drumSeparator}
                    disabled={uploading}
                    onChange={(e) => onSetDrumSeparator(e.target.value as DrumSeparator)}
                  >
                    {DRUM_SEPARATOR_ORDER.map((s) => (
                      <option key={s} value={s}>
                        {DRUM_SEPARATOR_LABELS[s]}
                      </option>
                    ))}
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
                <label
                  className={sharedStyles.toolbarCheckbox}
                  title="Run the optional quantise stage. Off skips snapping entirely; every onset keeps its raw detected time, the MIDI emitter writes it as a near-grid tick + sub-slot offset, and the UI / playback honour the offset so nothing re-snaps on load. Applies to both New and Resume."
                >
                  <span>Quantise</span>
                  <Checkbox
                    checked={transcribeOptions.quantise}
                    disabled={uploading}
                    onChange={(e) => onSetQuantise(e.target.checked)}
                  />
                </label>
                <label
                  className={classNames(sharedStyles.toolbarCheckbox, styles.subCheckbox)}
                  title="Run the LLM residual pass inside the quantise stage. Off skips that pass entirely; geometric + envelope + grid still run. No-op when Quantise is off."
                >
                  <span>Include LLM adjustment</span>
                  <Checkbox
                    checked={transcribeOptions.quantise && transcribeOptions.quantiseUseLlm}
                    disabled={uploading || !transcribeOptions.quantise}
                    onChange={(e) => onSetQuantiseUseLlm(e.target.checked)}
                  />
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
          ref={scoreParadbInputRef}
          type="file"
          accept=".zip,application/zip"
          className={styles.hiddenInput}
          onChange={handleScoreParadbChange}
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
              <ZoomControl onSetZoom={onSetZoom} />
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
              <ToggleMenuItem
                label="Auto-enable follow on play"
                active={autoFollowOnPlay}
                onToggle={() => onSetAutoFollowOnPlay(!autoFollowOnPlay)}
                title="When on, pressing Play (or resuming) re-enables Auto-follow if it was disabled mid-playback (pan, minimap drag, follow-button toggle while playing). Turning Auto-follow off while paused or stopped is treated as deliberate and survives the next play. Off = current Auto-follow state is always preserved across plays."
              />
            </>
          )}
        </DropdownButton>
        <div className={styles.toolbarSpacer} aria-hidden="true" />
        <DrumLoadingIndicator />
        <LyricsAlignBusyPill phase={lyricsAlignBusyPhase} />
        <TranscribeBusyPill status={transcribeStatus} />
      </div>
    );
  }
);
