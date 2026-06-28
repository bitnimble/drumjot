import classNames from 'classnames';
import { Info } from 'lucide-react';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { jotPlayer } from 'src/editing/playback/player';
import { Checkbox } from 'src/ui/checkbox/checkbox';
import { Modal, ModalBody, ModalFooter, ModalHeader, modalStyles } from 'src/ui/modal/modal';
import styles from './lyrics_text_modal.module.css';
import { LyricsPresenter } from './lyrics_presenter';

/**
 * Plain-text lyrics loader. Paste or type lyrics into a textarea, or
 * pull them in from a `.txt` file. On Load the text is split into
 * lines, section markers like `[Chorus]` are stripped, and the
 * remaining lines are spread evenly across the song's duration (see
 * `JotEditorStore.applyPlainTextLyrics`) so they're immediately visible
 * on the bars row. The word-level checkbox kicks off the same CTC
 * forced-alignment as the LRCLIB modal so untimed pastes can jump
 * straight to karaoke-style highlighting when an audio track is
 * loaded.
 */
const WORD_LEVEL_INFO =
  'Word-level alignment uses the song audio to time each individual word for karaoke-style highlighting. Usually takes under a minute.';
const WORD_LEVEL_NEEDS_AUDIO = 'Load an audio track first to enable this.';

export const LyricsTextLoadModal = observer(
  ({
    open,
    onClose,
    presenter,
  }: {
    open: boolean;
    onClose: () => void;
    presenter: LyricsPresenter;
  }) => {
    const [text, setText] = React.useState('');
    const [error, setError] = React.useState<string | undefined>(undefined);
    const [wordLevel, setWordLevel] = React.useState(true);
    const fileInputRef = React.useRef<HTMLInputElement>(null);

    // Reset whenever the modal reopens; stale text from the previous
    // session shouldn't survive a close. Word-level defaults back to
    // ON each open (matches the LRCLIB modal's "default on" behaviour
    // once a result is picked).
    React.useEffect(() => {
      if (!open) return;
      setText('');
      setError(undefined);
      setWordLevel(true);
    }, [open]);

    if (!open) return null;

    const trimmed = text.trim();
    const canLoad = trimmed.length > 0;
    const hasAudioTracks = jotPlayer.audioTracks.size > 0;
    const effectiveWordLevel = wordLevel && hasAudioTracks;

    const onLoad = () => {
      const count = presenter.applyPlainTextLyrics(text, {
        wordLevel: effectiveWordLevel,
      });
      if (count === 0) {
        setError(
          'No usable lyric lines after stripping blanks and section markers like [Chorus].',
        );
        return;
      }
      onClose();
    };

    const onPickFile = () => {
      fileInputRef.current?.click();
    };

    const onFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      e.target.value = '';
      if (!file) return;
      try {
        const contents = await file.text();
        setText(contents);
        setError(undefined);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setError(`Could not read ${file.name}: ${message}`);
      }
    };

    return (
      <Modal
        open={open}
        onClose={onClose}
        ariaLabel="Load lyrics from plain text"
        width={640}
        maxHeight
        testId="lyrics-text-modal"
      >
        <ModalHeader
          title="Load lyrics from plain text"
          onClose={onClose}
          closeLabel="Close plain-text lyrics loader"
          closeTestId="lyrics-text-close"
        />
        <ModalBody>
          <textarea
              className={styles.textarea}
              value={text}
              onChange={(e) => {
                setText(e.target.value);
                if (error) setError(undefined);
              }}
              placeholder={
                'Paste or type lyrics here, one line per row.\n\nSection markers like [Chorus] or [Verse 1] are stripped automatically.'
              }
              aria-label="Lyrics text"
              autoFocus
              data-testid="lyrics-text-textarea"
            />
            {error !== undefined && (
              <div
                className={styles.errorMessage}
                role="alert"
                data-testid="lyrics-text-error"
              >
                {error}
              </div>
            )}
          </ModalBody>
          <ModalFooter>
            <button
              type="button"
              className={modalStyles.secondaryButton}
              onClick={onPickFile}
              data-testid="lyrics-text-load-file"
            >
              Load from file…
            </button>
            <span className={modalStyles.footerSpacer} />
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
                !hasAudioTracks && styles.wordLevelLabelDisabled,
              )}
              title={hasAudioTracks ? WORD_LEVEL_INFO : WORD_LEVEL_NEEDS_AUDIO}
            >
              <Checkbox
                checked={effectiveWordLevel}
                disabled={!hasAudioTracks}
                onChange={(e) => setWordLevel(e.target.checked)}
                data-testid="lyrics-text-word-level"
              />
              Word-level alignment
            </label>
            <button
              type="button"
              className={modalStyles.primaryButton}
              onClick={onLoad}
              disabled={!canLoad}
              data-testid="lyrics-text-submit"
            >
              Load
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".txt,text/plain"
              className={styles.hiddenInput}
              onChange={onFileChange}
            />
          </ModalFooter>
        </Modal>
    );
  },
);
