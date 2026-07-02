import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import React from 'react';
import {
  JotEditorStoreContext,
  JotEditorPresenterContext,
} from 'src/editing/jot_editor_contexts';
import { LyricsPresenterContext } from 'src/editing/lyrics/lyrics_contexts';
import {
  TranscribeStoreContext,
  TranscribePresenterContext,
} from 'src/editing/transcribe/transcribe_contexts';
import { RecentTranscriptionsPicker } from 'src/editing/transcribe/recent_transcriptions';
import { DropdownButton, dropdownStyles, SubmenuItem } from 'src/ui/dropdown/dropdown';
import { ToolbarDropdownLabel } from './toolbar';
import styles from './toolbar.module.css';

/**
 * The "File" toolbar dropdown: examples, load/save, lyrics, recent
 * transcriptions, and Settings. Self-contained `observer` that reads the
 * jot-editor / lyrics / transcribe stores + presenters off context (the Toolbar
 * renders inside their providers), so the app shell no longer threads a dozen
 * loader callbacks through the Toolbar.
 *
 * The three genuinely view-level actions stay as props because their state is
 * React-local, not store data: `onNewJot` gates a confirm dialog on unsaved
 * edits, `onOpenSettings` toggles the Settings modal, and `onLoadZip` routes
 * through the window drag-and-drop auto-load flow (with its replace-confirm
 * gate). The hidden file inputs live here (outside the dropdown panels) so
 * their refs stay mounted whether or not a menu is open.
 */
export const FileMenu = observer(
  ({
    onNewJot,
    onOpenSettings,
    onLoadZip,
  }: {
    /** Start a fresh, empty jot (default kit lanes, one empty bar, no audio).
     *  Prompts before discarding unsaved changes is the caller's job. */
    onNewJot: () => void;
    /** Open the Settings dialog (About/licenses everywhere; the Capabilities +
     *  Hardware tabs are desktop-only). */
    onOpenSettings: () => void;
    /** Load a `.zip` and auto-detect its type (ParaDB map, transcriber debug
     *  bundle, or a zipped `.jot`), routing through the same drag-and-drop
     *  auto-load flow (with the replace-confirm gate). */
    onLoadZip: (file: File) => void;
  }) => {
    const store = React.useContext(JotEditorStoreContext);
    const presenter = React.useContext(JotEditorPresenterContext);
    const lyricsPresenter = React.useContext(LyricsPresenterContext);
    const transcribe = React.useContext(TranscribeStoreContext);
    const transcribePresenter = React.useContext(TranscribePresenterContext);

    const jotInputRef = React.useRef<HTMLInputElement>(null);
    const midiInputRef = React.useRef<HTMLInputElement>(null);
    const zipInputRef = React.useRef<HTMLInputElement>(null);
    const scoreParadbInputRef = React.useRef<HTMLInputElement>(null);
    const audioTrackInputRef = React.useRef<HTMLInputElement>(null);
    const lyricsInputRef = React.useRef<HTMLInputElement>(null);

    if (!store || !presenter) return null;
    const examples = store.examples;
    const currentId = store.currentExampleId;

    const handleJotFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) presenter.loadJotFile(file);
      e.target.value = '';
    };

    const handleMidiFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) presenter.loadMidiFile(file);
      e.target.value = '';
    };

    const handleZipChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onLoadZip(file);
      e.target.value = '';
    };

    const handleScoreParadbChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) presenter.scoreParadbMap(file);
      e.target.value = '';
    };

    const handleAudioTrackChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      // Multiple-select: load every chosen file as its own track.
      for (const file of Array.from(e.target.files ?? [])) presenter.loadAudioTrack(file);
      e.target.value = '';
    };

    const handleLyricsFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) presenter.loadLyricsFile(file);
      e.target.value = '';
    };

    return (
      <>
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
                  presenter.saveMutableFile();
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
                                    presenter.loadExample(ex.id);
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
                          lyricsPresenter?.setLyricsSearchOpen(true);
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
                          lyricsPresenter?.setLyricsTextOpen(true);
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
                items={transcribe?.recentTranscriptions ?? []}
                loaded={transcribe?.recentTranscriptionsLoaded ?? false}
                loading={transcribe?.recentTranscriptionsLoading ?? false}
                onRefresh={() => transcribePresenter?.refreshRecentTranscriptions()}
                onPick={(folder) => transcribePresenter?.openReplaceDialog(folder)}
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
      </>
    );
  }
);
