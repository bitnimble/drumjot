import { serializeEnhancedLrc } from 'src/lyrics/enhanced_lrc';
import { LYRICS_OFFSET_MAX_SEC, LYRICS_OFFSET_MIN_SEC, LYRICS_OFFSET_STEP_SEC, LyricsTrackId, lyricsStore } from 'src/lyrics/store';
import { downloadTextFile } from 'src/utils/download';
import { DropdownButton, dropdownStyles } from 'src/ui/dropdown/dropdown';
import { NumberStepper } from 'src/ui/number_stepper/number_stepper';
import mixerStyles from '../mixer/mixer.module.css';
import styles from './lyrics_track_view.module.css';

/** Build a filesystem-friendly `.lrc` name from a track's source label.
 *  Drops a leading `Source · ` prefix, strips an existing `.lrc`
 *  extension, and replaces filename-hostile characters. */
function lyricsExportFilename(sourceLabel: string): string {
  let base = sourceLabel.replace(/^[^·]*·\s*/, '').trim() || sourceLabel;
  base = base.replace(/\.lrc$/i, '');
  base = base
    .replace(/[^\p{L}\p{N}\-_. ]+/gu, '_')
    .replace(/\s+/g, ' ')
    .trim();
  return `${base || 'lyrics'}.lrc`;
}

/** Serialize a lyrics track to an enhanced-LRC file (with word-level
 *  durations + the offset nudge) and trigger a download. No-op if the
 *  track raced a removal. */
function exportLyricsTrack(id: LyricsTrackId): void {
  const track = lyricsStore.get(id);
  if (!track) return;
  const text = serializeEnhancedLrc(track.lines, { offsetSec: track.offsetSec });
  downloadTextFile(lyricsExportFilename(track.sourceLabel), text);
}

/** Per-row overflow menu on lyrics tracks. Hosts the time-offset stepper
 *  (replacing the inline gutter control), the enhanced-LRC export, and
 *  the "Remove lyrics" action; same trigger position as the audio-track
 *  row's overflow so the chrome reads identically across the mixer. */
export const LyricsOverflowMenu = ({
  id,
  offsetSec,
  onSetOffset,
  onRemove,
}: {
  id: LyricsTrackId;
  offsetSec: number;
  onSetOffset: (sec: number) => void;
  onRemove: () => void;
}) => (
  <DropdownButton
    label="⋯"
    className={mixerStyles.overflowTrigger}
    title="More actions for this lyrics track"
  >
    {(close) => (
      <>
        <label
          className={styles.offsetStepperRow}
          title="Lyrics offset (seconds). Positive values delay the lyric chips relative to the audio."
        >
          <span>Offset</span>
          <span className={styles.offsetStepperControl}>
            <NumberStepper
              value={offsetSec}
              onChange={onSetOffset}
              step={LYRICS_OFFSET_STEP_SEC}
              min={LYRICS_OFFSET_MIN_SEC}
              max={LYRICS_OFFSET_MAX_SEC}
              ariaLabel="Lyrics time offset (seconds)"
              title="Lyrics offset (seconds)"
              testId={`lyrics-offset-input-${id}`}
            />
            <span className={styles.offsetStepperUnit}>s</span>
          </span>
        </label>
        <span className={dropdownStyles.dropdownDivider} aria-hidden="true" />
        <button
          type="button"
          className={dropdownStyles.dropdownItem}
          role="menuitem"
          onClick={() => {
            exportLyricsTrack(id);
            close();
          }}
          data-testid={`lyrics-export-${id}`}
          title="Download this track as an enhanced LRC file (round-trips word timings)"
        >
          Export enhanced LRC
        </button>
        <button
          type="button"
          className={dropdownStyles.dropdownItem}
          role="menuitem"
          onClick={() => {
            onRemove();
            close();
          }}
          data-testid="lyrics-clear"
          title="Remove this lyrics track from the mixer"
        >
          Remove track
        </button>
      </>
    )}
  </DropdownButton>
);
