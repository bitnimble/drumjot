import classNames from 'classnames';
import { ChevronDown } from 'lucide-react';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { ExampleJot } from 'src/fakes/fakes';
import { TranscriptionSummary } from 'src/editing/transcribe/transcriber';
import {
  DropdownButton,
  DropdownSection,
  dropdownStyles,
  SubmenuItem,
  ToggleMenuItem,
} from 'src/ui/dropdown/dropdown';
import { Logo } from 'src/ui/logo/logo';
import { RecentTranscriptionsPicker } from '../editing/transcribe/recent_transcriptions';
import styles from './toolbar.module.css';
import type { GridLineSettings } from 'src/settings/settings_store';
import { PlaybackKitSubmenu, PlaybackSpeedItem, AudioLatencyItem } from './playback_menu';
import { ZoomControl, ThemeSection } from './view_menu';
import { EditMenu } from './edit_menu';
import { DrumLoadingIndicator, LyricsAlignBusyPill, TranscribeBusyPill } from './toolbar_status';

/**
 * Toolbar dropdown trigger label with a trailing caret indicator. The
 * shared {@link DropdownButton} no longer renders the caret itself
 * (overflow-icon callers like the mixer don't want one); toolbar
 * triggers compose it via this helper instead.
 */
export const ToolbarDropdownLabel = ({ children }: { children: React.ReactNode }) => (
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
    onNewJot,
    onSaveJot,
    onLoadJot,
    onLoadMidi,
    onLoadZip,
    onScoreParadb,
    onLoadAudioTrack,
    onLoadLyricsFile,
    onOpenLyricsTextLoad,
    onOpenLyricsSearch,
    onOpenSettings,
    lyricsAlignBusyPhase,
    onSetZoom,
    hasNoteProvenance,
    showFilteredOnsets,
    onSetShowFilteredOnsets,
    gridLines,
    onToggleGridLine,
    uniformWaveforms,
    onSetUniformWaveforms,
    waveformGridLines,
    onSetWaveformGridLines,
    mergeLayers,
    onSetMergeLayers,
    autoFollowOnPlay,
    onSetAutoFollowOnPlay,
    recentTranscriptions,
    recentTranscriptionsLoaded,
    recentTranscriptionsLoading,
    onRefreshRecentTranscriptions,
    onOpenRecentTranscription,
  }: {
    examples: readonly ExampleJot[];
    currentId: string | undefined;
    onSelect: (id: string) => void;
    /** Start a fresh, empty jot (default kit lanes, one empty bar, no audio).
     *  Prompts before discarding unsaved changes is the caller's job. */
    onNewJot: () => void;
    /** Save the current session as a mutable `.jot` file (lossless superset:
     *  the edited document + editor metadata). Browser download. */
    onSaveJot: () => void;
    onLoadJot: (file: File) => void;
    onLoadMidi: (file: File) => void;
    /** Load a `.zip` and auto-detect its type (ParaDB map, transcriber
     *  debug bundle, or a zipped `.jot`), routing through the same
     *  drag-and-drop auto-load flow (with the replace-confirm gate). */
    onLoadZip: (file: File) => void;
    /** Score a ParaDB map against its own audio (dev test harness for the
     *  corpus-filtering scorer); reports a quality number, doesn't load it. */
    onScoreParadb: (file: File) => void;
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
    /** Open the Settings dialog (About/licenses everywhere; the Capabilities +
     *  Hardware tabs are desktop-only). */
    onOpenSettings: () => void;
    /** Aggregate lyrics-alignment state, for the toolbar busy pill (which
     *  doesn't display *which* row; the per-row spinner does). `queued`
     *  means waiting behind another GPU job; `aligning` means actively
     *  running. See `JotEditorStore.lyricsAlignBusyPhase`. */
    lyricsAlignBusyPhase: 'idle' | 'queued' | 'aligning';
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
    waveformGridLines: boolean;
    onSetWaveformGridLines: (on: boolean) => void;
    mergeLayers: boolean;
    onSetMergeLayers: (on: boolean) => void;
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
    onRefreshRecentTranscriptions: () => void;
    /** Open the transcribe dialog (replace mode) for a previous run, so the
     *  user can pick a resume stage + options before re-running it. */
    onOpenRecentTranscription: (folder: string) => void;
  }) => {
    const jotInputRef = React.useRef<HTMLInputElement>(null);
    const midiInputRef = React.useRef<HTMLInputElement>(null);
    const zipInputRef = React.useRef<HTMLInputElement>(null);
    const scoreParadbInputRef = React.useRef<HTMLInputElement>(null);
    const audioTrackInputRef = React.useRef<HTMLInputElement>(null);
    const lyricsInputRef = React.useRef<HTMLInputElement>(null);

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

    const handleZipChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onLoadZip(file);
      e.target.value = '';
    };

    const handleScoreParadbChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onScoreParadb(file);
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
              <button
                type="button"
                className={dropdownStyles.dropdownItem}
                onClick={() => {
                  onNewJot();
                  close();
                }}
                title="Start a fresh, empty jot: the standard drum-kit lanes (crash, ride, hi-hat, snare, kick) with no notes and no audio tracks, ready to chart from scratch. Prompts before discarding unsaved changes."
                data-testid="file-menu-new"
              >
                New jot
              </button>
              <button
                type="button"
                className={dropdownStyles.dropdownItem}
                onClick={() => {
                  onSaveJot();
                  close();
                }}
                title="Save the current session to a `.jot` file: the full editable document plus the editor metadata (mixer faders, display settings, palette) and the loaded audio tracks the DSL text format can't carry. This is the lossless save that preserves your edits; downloads to your machine."
                data-testid="file-menu-save"
              >
                Save .jot file
              </button>
              <span className={dropdownStyles.dropdownDivider} aria-hidden="true" />
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
                        title="Load a `.jot` file from disk: either a saved session (the lossless mutable format, with your edits + mixer/display/palette settings) or a hand-authored Drumjot DSL file. The format is detected automatically. Runs entirely client-side; no transcriber service required."
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
                          zipInputRef.current?.click();
                          closeAll();
                        }}
                        title="Load a `.zip` and auto-detect its type: a ParaDB / Paradiddle map pack, a transcriber debug bundle, or a zipped `.jot`. The matching loader runs automatically (you're asked to confirm before replacing the open score). Runs entirely client-side."
                        data-testid="file-menu-load-zip"
                      >
                        Load zip
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
              {/* Recent transcriptions: picking one opens the transcribe
                  dialog (replace mode) to choose a resume stage + options,
                  then re-runs it as a wholesale jot replacement. */}
              <RecentTranscriptionsPicker
                variant="menu"
                triggerLabel="Recent"
                items={recentTranscriptions}
                loaded={recentTranscriptionsLoaded}
                loading={recentTranscriptionsLoading}
                onRefresh={onRefreshRecentTranscriptions}
                onPick={onOpenRecentTranscription}
                onAfterPick={close}
              />
              {/* Universal: the About / licenses tab shows everywhere; the
                  Capabilities + Hardware tabs are Tauri-only (hidden in web). */}
              <span className={dropdownStyles.dropdownDivider} aria-hidden="true" />
              <button
                type="button"
                className={dropdownStyles.dropdownItem}
                onClick={() => {
                  onOpenSettings();
                  close();
                }}
                title="Settings, downloadable capabilities, hardware info, and licenses."
                data-testid="file-menu-settings"
              >
                Settings…
              </button>
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
          ref={midiInputRef}
          type="file"
          accept=".mid,.midi,audio/midi,audio/x-midi"
          className={styles.hiddenInput}
          onChange={handleMidiFileChange}
        />
        <input
          ref={zipInputRef}
          type="file"
          accept=".zip,application/zip"
          className={styles.hiddenInput}
          onChange={handleZipChange}
          data-testid="load-zip-input"
        />
        <input
          ref={scoreParadbInputRef}
          type="file"
          accept=".zip,application/zip"
          className={styles.hiddenInput}
          onChange={handleScoreParadbChange}
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
              <DropdownSection label="Layers">
                <ToggleMenuItem
                  label="Visually merge layers"
                  active={mergeLayers}
                  onToggle={() => onSetMergeLayers(!mergeLayers)}
                  title="Collapse tracks of the same lane across every || layer into a single row (the flat per-lane view), dropping the layer bands. View-only: notes keep their layer, so edits still route per-note and a new note lands on the firstmost layer carrying the lane."
                />
              </DropdownSection>
              <DropdownSection label="Waveforms">
                <ToggleMenuItem
                  label="Uniform amplitude"
                  active={uniformWaveforms}
                  onToggle={() => onSetUniformWaveforms(!uniformWaveforms)}
                  title="Normalise each audio track's waveform so the median non-silent peak fills most of the row, regardless of the source recording's amplitude. Silence still renders as silence. Off = accurate, on = uniform (easier to see quiet recordings)."
                />
                <ToggleMenuItem
                  label="Bar & beat lines"
                  active={waveformGridLines}
                  onToggle={() => onSetWaveformGridLines(!waveformGridLines)}
                  title="Draw bar lines and the beat grid over each audio-track waveform, aligned with the score above. Which sub-beat lines show follows the Grid lines section below, so a vertical line traces cleanly from the score down through every waveform."
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
        <EditMenu />
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
        <TranscribeBusyPill />
      </div>
    );
  }
);
