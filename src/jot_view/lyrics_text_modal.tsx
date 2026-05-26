import { observer } from 'mobx-react-lite';
import React from 'react';
import styles from './lyrics_text_modal.module.css';
import { JotViewStore } from './store';

/**
 * Plain-text lyrics loader. Paste or type lyrics into a textarea, or
 * pull them in from a `.txt` file. On Load the text is split into
 * lines, section markers like `[Chorus]` are stripped, and each
 * remaining line is committed at `startSec: 0` so the user can re-time
 * them via "Re-time loaded lyrics from vocals…".
 */
export const LyricsTextLoadModal = observer(
  ({
    open,
    onClose,
    store,
  }: {
    open: boolean;
    onClose: () => void;
    store: JotViewStore;
  }) => {
    const [text, setText] = React.useState('');
    const [error, setError] = React.useState<string | undefined>(undefined);
    const fileInputRef = React.useRef<HTMLInputElement>(null);

    // Reset whenever the modal reopens; stale text from the previous
    // session shouldn't survive a close.
    React.useEffect(() => {
      if (!open) return;
      setText('');
      setError(undefined);
    }, [open]);

    if (!open) return null;

    const trimmed = text.trim();
    const canLoad = trimmed.length > 0;

    const onLoad = () => {
      const count = store.applyPlainTextLyrics(text);
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
      <div
        className={styles.modalBackdrop}
        role="dialog"
        aria-modal="true"
        aria-label="Load lyrics from plain text"
        onMouseDown={(e) => {
          if (e.target === e.currentTarget) onClose();
        }}
        data-testid="lyrics-text-modal"
      >
        <div className={styles.modalPanel}>
          <header className={styles.modalHeader}>
            <h3 className={styles.modalTitle}>Load lyrics from plain text</h3>
            <button
              type="button"
              className={styles.modalClose}
              onClick={onClose}
              aria-label="Close plain-text lyrics loader"
              data-testid="lyrics-text-close"
            >
              ×
            </button>
          </header>
          <div className={styles.modalBody}>
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
          </div>
          <footer className={styles.modalFooter}>
            <button
              type="button"
              className={styles.secondaryButton}
              onClick={onPickFile}
              data-testid="lyrics-text-load-file"
            >
              Load from file…
            </button>
            <button
              type="button"
              className={styles.primaryButton}
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
          </footer>
        </div>
      </div>
    );
  },
);
