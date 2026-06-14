import { observer } from 'mobx-react-lite';
import { AudioTrack, AudioTrackId, AudioTrackRole } from 'src/jot_view/playback/audio_tracks';
import { InstrumentTrack, PICKER_PALETTE } from 'src/tracks/tracks';
import { ColorPickerMenuRow } from '../components/color_picker_menu_row';
import { DropdownButton, dropdownStyles } from '../components/dropdown';
import styles from './mixer.module.css';

/** Per-menu-item availability for the {@link AudioTrackOverflowMenu}.
 *  `enabled` drives the disabled prop; `reason` is the tooltip shown
 *  on the disabled item so the user sees why the action is blocked. */
type AudioTrackMenuItemState = {
  enabled: boolean;
  reason: string;
};

/** Compute whether the "Split into drums + backing" item is actionable
 *  for an audio track of the given role. Stage 1 (`stems_all`) only
 *  makes sense on a recording that may contain non-drum content; running
 *  it on an already-isolated drum stem, a drumless backing, or a single
 *  drum piece all produce garbage or noop. `unknown` defaults to enabled
 *  (ad-hoc loads where we couldn't classify; let the user try). */
export function splitFromMixState(role: AudioTrackRole | undefined): AudioTrackMenuItemState {
  switch (role ?? 'unknown') {
    case 'full-mix':
      return { enabled: true, reason: 'Isolate drums and a drumless backing from this recording.' };
    case 'unknown':
      return {
        enabled: true,
        reason: 'Try isolating drums and a drumless backing from this recording.',
      };
    case 'drums':
      return { enabled: false, reason: 'Already drums-only.' };
    case 'no-drums':
      return { enabled: false, reason: 'No drums to split.' };
    case 'drum-piece':
      return { enabled: false, reason: 'Already a single drum piece.' };
  }
}

/** Compute whether the "Split into kick / snare / hi-hat / cymbals" item
 *  is actionable for the given role. Stage 2 (`stems_per`) requires an
 *  already-isolated drum stem; the model was trained on isolated drums
 *  only and produces garbage when fed a full mix. */
export function splitDrumPiecesState(role: AudioTrackRole | undefined): AudioTrackMenuItemState {
  switch (role ?? 'unknown') {
    case 'drums':
      return { enabled: true, reason: 'Split this drum recording into per-instrument pieces.' };
    case 'unknown':
      return {
        enabled: true,
        reason: 'Try splitting this recording into per-instrument drum pieces.',
      };
    case 'full-mix':
      return { enabled: false, reason: 'Isolate drums first.' };
    case 'no-drums':
      return { enabled: false, reason: 'No drums to split.' };
    case 'drum-piece':
      return { enabled: false, reason: 'Already a single drum piece.' };
  }
}

/** Per-row overflow menu on audio tracks. Hosts the two separation
 *  operations (stage 1, stage 2) with enable state derived from the
 *  track's {@link AudioTrackRole}, the waveform-colour picker, plus
 *  the "Remove track" action. The trigger always renders since Remove
 *  is always available. */
export const AudioTrackOverflowMenu = observer(({
  track,
  trackLabel,
  onSplitFromMix,
  onSplitDrumPieces,
  onClear,
}: {
  track: AudioTrack;
  trackLabel: string;
  onSplitFromMix: (id: AudioTrackId) => void;
  onSplitDrumPieces: (id: AudioTrackId) => void;
  onClear: (id: AudioTrackId) => void;
}) => {
  const mixState = splitFromMixState(track.role);
  const piecesState = splitDrumPiecesState(track.role);
  return (
    <DropdownButton
      label="⋯"
      className={styles.overflowTrigger}
      title={`More actions for ${trackLabel}`}
    >
      {(close) => (
        <>
          <AudioTrackMenuItem
            label="Split into drums + backing"
            state={mixState}
            onClick={() => {
              onSplitFromMix(track.id);
              close();
            }}
            testId={`audio-track-split-mix-${track.id}`}
          />
          <AudioTrackMenuItem
            label="Split into kick / snare / hi-hat / cymbals"
            state={piecesState}
            onClick={() => {
              onSplitDrumPieces(track.id);
              close();
            }}
            testId={`audio-track-split-pieces-${track.id}`}
          />
          <span className={dropdownStyles.dropdownDivider} aria-hidden="true" />
          <ColorPickerMenuRow
            label="Colour"
            value={normaliseColorForPicker(track.color)}
            palette={PICKER_PALETTE}
            hasOverride={track.hasOverride}
            onChange={(hex) => {
              track.color = hex;
            }}
            onReset={() => track.clearColor()}
            ariaLabel={`Waveform colour for ${trackLabel}`}
          />
          <span className={dropdownStyles.dropdownDivider} aria-hidden="true" />
          <button
            type="button"
            className={dropdownStyles.dropdownItem}
            role="menuitem"
            onClick={() => {
              onClear(track.id);
              close();
            }}
            data-testid={`audio-track-clear-${track.id}`}
            title={`Remove the ${trackLabel} audio track`}
          >
            Remove track
          </button>
        </>
      )}
    </DropdownButton>
  );
});

/** Per-instrument-row overflow menu. Currently hosts only the note-
 *  colour picker; the chrome stays consistent with
 *  {@link AudioTrackOverflowMenu} so future additions (per-pitch
 *  velocity scale, label rename, etc.) drop in without an entirely
 *  new affordance. */
export const InstrumentRowOverflowMenu = observer(({
  instrumentTrack,
  trackLabel,
}: {
  instrumentTrack: InstrumentTrack;
  trackLabel: string;
}) => {
  return (
    <DropdownButton
      label="⋯"
      className={styles.overflowTrigger}
      title={`More actions for ${trackLabel}`}
    >
      {() => (
        <ColorPickerMenuRow
          label="Colour"
          value={normaliseColorForPicker(instrumentTrack.color)}
          palette={PICKER_PALETTE}
          hasOverride={instrumentTrack.hasOverride}
          onChange={(hex) => {
            instrumentTrack.color = hex;
          }}
          onReset={() => instrumentTrack.clearColor()}
          ariaLabel={`Note colour for ${trackLabel}`}
        />
      )}
    </DropdownButton>
  );
});

/** The colour-picker popover's HSL wheel takes an `#rrggbb` string and
 *  ignores anything else. Instrument tracks may fall through to a
 *  `var(...)` CSS expression when no palette default is available;
 *  convert anything that isn't a 7-char hex into a neutral grey so the
 *  picker opens at a sensible starting colour. */
function normaliseColorForPicker(color: string): string {
  return /^#[0-9a-fA-F]{6}$/.test(color) ? color : '#7e7e7e';
}

const AudioTrackMenuItem = ({
  label,
  state,
  onClick,
  testId,
}: {
  label: string;
  state: AudioTrackMenuItemState;
  onClick: () => void;
  testId?: string;
}) => (
  <button
    type="button"
    className={dropdownStyles.dropdownItem}
    role="menuitem"
    disabled={!state.enabled}
    title={state.reason}
    onClick={onClick}
    data-testid={testId}
  >
    {label}
  </button>
);
