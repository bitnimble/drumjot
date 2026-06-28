import classNames from 'classnames';
import { Info, Search } from 'lucide-react';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { parseLrc } from 'src/lyrics/lrc';
import { ciTrimEq, LrclibMatch, searchLrclib } from 'src/lyrics/lrclib';
import { jotPlayer } from 'src/editing/playback/player';
import { Checkbox } from 'src/ui/checkbox/checkbox';
import { Modal, ModalHeader } from 'src/ui/modal/modal';
import { Spinner } from 'src/ui/spinner/spinner';
import styles from './lyrics_search_modal.module.css';
import { LyricsPresenter } from './lyrics_presenter';

/**
 * LRCLIB search modal. Opens with `initialTitle` / `initialArtist`
 * pre-filled from the current jot's metadata; auto-fires a search on
 * mount whenever the title field is non-empty so the user lands on
 * results (or a "no synced lyrics" message) without an extra click.
 *
 * Result rows are always rendered for the user to pick (even a single
 * result; no auto-load). Picking a row reveals a footer with a word-
 * level alignment checkbox + Load button:
 *
 *  - word-level OFF: line-level LRCLIB timing applied as-is.
 *  - word-level ON: LRCLIB lines load immediately for instant feedback,
 *                   then a background CTC forced-alignment pass
 *                   upgrades them with per-word timings. Requires a
 *                   loaded audio track (the checkbox is disabled
 *                   otherwise).
 *
 * Each search supersedes the previous one via a `requestIdRef`; a late
 * stale response is silently discarded so the visible state never
 * regresses to an older search's results.
 */
type Phase =
  | { kind: 'idle' }
  | { kind: 'searching' }
  | { kind: 'no-results' }
  | { kind: 'error'; message: string }
  | { kind: 'results'; matches: LrclibMatch[] };

const WORD_LEVEL_INFO =
  'Word-level alignment uses the song audio to time each individual word for karaoke-style highlighting. Usually takes under a minute.';
const WORD_LEVEL_NEEDS_AUDIO = 'Load an audio track first to enable this.';

export const LyricsSearchModal = observer(
  ({
    open,
    initialTitle,
    initialArtist,
    onClose,
    presenter,
  }: {
    open: boolean;
    initialTitle: string;
    initialArtist: string;
    onClose: () => void;
    presenter: LyricsPresenter;
  }) => {
    const [title, setTitle] = React.useState(initialTitle);
    const [artist, setArtist] = React.useState(initialArtist);
    const [phase, setPhase] = React.useState<Phase>({ kind: 'idle' });
    const [selectedId, setSelectedId] = React.useState<string | undefined>(undefined);
    const [wordLevel, setWordLevel] = React.useState(true);
    const requestIdRef = React.useRef(0);

    // When the modal opens, re-seed the fields with the latest jot
    // metadata; an earlier search's text shouldn't persist across
    // open/close cycles (the jot may have changed).
    React.useEffect(() => {
      if (!open) return;
      setTitle(initialTitle);
      setArtist(initialArtist);
      setPhase({ kind: 'idle' });
      setSelectedId(undefined);
      setWordLevel(true);
      requestIdRef.current += 1;
    }, [open, initialTitle, initialArtist]);

    const runSearch = React.useCallback(async (searchTitle: string, searchArtist: string) => {
      const requestId = ++requestIdRef.current;
      setPhase({ kind: 'searching' });
      setSelectedId(undefined);
      let matches: LrclibMatch[];
      try {
        matches = await searchLrclib({
          trackName: searchTitle,
          artistName: searchArtist,
        });
      } catch (err) {
        if (requestIdRef.current !== requestId) return;
        const message = err instanceof Error ? err.message : String(err);
        setPhase({ kind: 'error', message });
        return;
      }
      if (requestIdRef.current !== requestId) return;

      if (matches.length === 0) {
        setPhase({ kind: 'no-results' });
        return;
      }
      // Pre-select the best match (exact title+artist if available;
      // else the single result; else nothing) so the footer pops open
      // immediately and the user can just hit Load. Multi-result with
      // no exact match leaves the user to pick deliberately.
      const exact = matches.filter(
        (m) => ciTrimEq(m.trackName, searchTitle) && ciTrimEq(m.artistName, searchArtist)
      );
      let preselect: LrclibMatch | undefined;
      if (matches.length === 1) preselect = matches[0];
      else if (exact.length === 1) preselect = exact[0];
      setSelectedId(preselect ? matchKey(preselect) : undefined);
      // Selecting a result implies the user is about to load it, so
      // default the alignment toggle on. (Re-checked manually below
      // in onPickResult for the explicit-click path.)
      if (preselect) setWordLevel(true);
      setPhase({ kind: 'results', matches });
    }, []);

    // Auto-fire on open when title is set; honour the latest field
    // values rather than the initial props in case the user reopens the
    // modal after editing them.
    const autoFiredRef = React.useRef(false);
    React.useEffect(() => {
      if (!open) {
        autoFiredRef.current = false;
        return;
      }
      if (autoFiredRef.current) return;
      autoFiredRef.current = true;
      if (initialTitle.trim().length > 0) {
        void runSearch(initialTitle, initialArtist);
      }
    }, [open, initialTitle, initialArtist, runSearch]);

    if (!open) return null;

    const onSubmit = (e: React.FormEvent) => {
      e.preventDefault();
      void runSearch(title, artist);
    };

    const matches = phase.kind === 'results' ? phase.matches : [];
    const selectedMatch = matches.find((m) => matchKey(m) === selectedId);
    const hasAudioTracks = jotPlayer.audioTracks.size > 0;
    const effectiveWordLevel = wordLevel && hasAudioTracks;

    const onPickResult = (match: LrclibMatch) => {
      setSelectedId(matchKey(match));
      setWordLevel(true);
    };

    const onLoad = () => {
      if (!selectedMatch) return;
      const lines = parseLrc(selectedMatch.syncedLyrics ?? '');
      if (lines.length === 0) {
        setPhase({
          kind: 'error',
          message: 'LRCLIB returned a match but it has no parseable synced lyrics.',
        });
        return;
      }
      presenter.applyLrclibResult(
        lines,
        { trackName: selectedMatch.trackName, artistName: selectedMatch.artistName },
        { wordLevel: effectiveWordLevel }
      );
      onClose();
    };

    return (
      <Modal
        open={open}
        onClose={onClose}
        ariaLabel="Search lyrics on LRCLIB"
        width={560}
        maxHeight
        testId="lyrics-search-modal"
      >
        <ModalHeader
          title="Search lyrics on LRCLIB"
          onClose={onClose}
          closeLabel="Close lyrics search"
        />
        <form className={styles.modalForm} onSubmit={onSubmit}>
            <input
              type="text"
              className={styles.field}
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Song name (e.g. Black Dog)"
              aria-label="Song name"
              autoFocus
              data-testid="lyrics-search-title"
            />
            <input
              type="text"
              className={styles.field}
              value={artist}
              onChange={(e) => setArtist(e.target.value)}
              placeholder="Artist (e.g. Led Zeppelin)"
              aria-label="Artist"
              data-testid="lyrics-search-artist"
            />
            <button
              type="submit"
              className={styles.searchButton}
              /* Only disable on empty title. While searching, leave the
                 button live so its accent background + white spinner
                 stay readable instead of falling into the muted
                 `:disabled` palette where the currentColor-based spinner
                 colours blend into the grey background. Re-clicking
                 mid-search just supersedes the in-flight request via
                 `requestIdRef` above, so live-during-search is safe. */
              disabled={title.trim().length === 0}
              aria-busy={phase.kind === 'searching'}
              aria-label={phase.kind === 'searching' ? 'Searching' : 'Search'}
              title={phase.kind === 'searching' ? 'Searching…' : 'Search'}
              data-testid="lyrics-search-submit"
            >
              {phase.kind === 'searching' ? (
                <Spinner size={16} tone="current" />
              ) : (
                <Search size={16} aria-hidden="true" />
              )}
            </button>
          </form>
        <PhaseView phase={phase} selectedId={selectedId} onPickResult={onPickResult} />
        <LoadFooter
          wordLevel={wordLevel}
          onSetWordLevel={setWordLevel}
          hasAudioTracks={hasAudioTracks}
          onLoad={onLoad}
          disabled={!selectedMatch}
        />
      </Modal>
    );
  }
);

/** Stable key for a match. LRCLIB returns numeric IDs but defensively
 *  fall back to title|artist|album when zero/missing. */
function matchKey(m: LrclibMatch): string {
  return m.id ? String(m.id) : `${m.trackName}|${m.artistName}|${m.albumName}`;
}

const PhaseView = ({
  phase,
  selectedId,
  onPickResult,
}: {
  phase: Phase;
  selectedId: string | undefined;
  onPickResult: (match: LrclibMatch) => void;
}) => {
  // Always render the `.results` container so the modal keeps a stable
  // height across phase transitions (idle → searching → results/no-
  // results/error) instead of jumping as the section appears.
  if (phase.kind === 'results') {
    return (
      <div className={styles.results} data-testid="lyrics-search-results">
        <ul className={styles.resultsList}>
          {phase.matches.map((m) => {
            const key = matchKey(m);
            const selected = key === selectedId;
            return (
              <li key={key}>
                <button
                  type="button"
                  className={classNames(styles.resultItem, selected && styles.resultItemSelected)}
                  aria-pressed={selected}
                  onClick={() => onPickResult(m)}
                  data-testid={`lyrics-search-result-${m.id || m.trackName}`}
                >
                  <span className={styles.resultPrimary}>
                    <strong>{m.trackName}</strong> by {m.artistName}
                  </span>
                  <span className={styles.resultSecondary}>
                    <AlbumLabel albumName={m.albumName} />
                    {typeof m.duration === 'number' ? ` · ${formatDuration(m.duration)}` : ''}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      </div>
    );
  }
  let message: React.ReactNode = null;
  let messageTitle: string | undefined;
  let placeholderClass: string | undefined;
  if (phase.kind === 'searching') {
    message = 'Searching…';
  } else if (phase.kind === 'no-results') {
    message = 'No synced lyrics found.';
  } else if (phase.kind === 'error') {
    message = `Search failed: ${phase.message}`;
    messageTitle = phase.message;
    placeholderClass = styles.resultsPlaceholderError;
  }
  return (
    <div className={styles.results} data-testid="lyrics-search-results">
      <div
        className={classNames(styles.resultsPlaceholder, placeholderClass)}
        title={messageTitle}
      >
        {message}
      </div>
    </div>
  );
};

const LoadFooter = ({
  wordLevel,
  onSetWordLevel,
  hasAudioTracks,
  onLoad,
  disabled,
}: {
  wordLevel: boolean;
  onSetWordLevel: (v: boolean) => void;
  hasAudioTracks: boolean;
  onLoad: () => void;
  disabled: boolean;
}) => {
  const checkboxDisabled = !hasAudioTracks;
  const checkboxTitle = checkboxDisabled ? WORD_LEVEL_NEEDS_AUDIO : WORD_LEVEL_INFO;
  return (
    <footer className={styles.loadFooter} data-testid="lyrics-search-load-footer">
      <span
        className={styles.infoIcon}
        title={WORD_LEVEL_INFO}
        aria-label="Word-level alignment info"
        role="img"
      >
        <Info size={14} aria-hidden="true" />
      </span>
      <label
        className={classNames(
          styles.wordLevelLabel,
          checkboxDisabled && styles.wordLevelLabelDisabled
        )}
        title={checkboxTitle}
      >
        <Checkbox
          checked={wordLevel && !checkboxDisabled}
          disabled={checkboxDisabled}
          onChange={(e) => onSetWordLevel(e.target.checked)}
          data-testid="lyrics-search-word-level"
        />
        Word-level alignment
      </label>
      <button
        type="button"
        className={styles.loadButton}
        onClick={onLoad}
        disabled={disabled}
        data-testid="lyrics-search-load"
      >
        Load
      </button>
    </footer>
  );
};

/** Album line for a result row. LRCLIB sometimes returns a blank album
 *  or the literal string "null"; render a subtle placeholder for those
 *  rather than "Album: null". */
const AlbumLabel = ({ albumName }: { albumName: string | null | undefined }) => {
  const trimmed = (albumName ?? '').trim();
  if (trimmed === '' || trimmed.toLowerCase() === 'null') {
    return <em className={styles.resultAlbumEmpty}>(no album)</em>;
  }
  return <>{`Album: ${trimmed}`}</>;
};

function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const min = Math.floor(total / 60);
  const sec = total % 60;
  return `${min}:${String(sec).padStart(2, '0')}`;
}
