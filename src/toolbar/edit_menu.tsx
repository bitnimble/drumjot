import { observer } from 'mobx-react-lite';
import React from 'react';
import {
  ActionMenuItem,
  DropdownButton,
  DropdownSection,
  ToggleMenuItem,
} from 'src/ui/dropdown/dropdown';
import {
  EditingStoreContext,
  EditingPresenterContext,
} from 'src/editing/editing_contexts';
import {
  HistoryStoreContext,
  HistoryPresenterContext,
} from 'src/editing/history/history_contexts';
import {
  PlaybackStoreContext,
  PlaybackPresenterContext,
} from 'src/editing/playback/playback_contexts';
import { jotPlayer } from 'src/editing/playback/player';
import { DEFAULT_GRID_DIVISION, gridDivisionFor } from 'src/grid/grid';
import { shortcutForCommand } from 'src/editing/keyboard/keymap';
import { NumberStepper } from 'src/ui/number_stepper/number_stepper';
import { ToolbarDropdownLabel } from './toolbar';
import styles from './toolbar.module.css';

/**
 * One alignment row inside the Edit menu's Alignment section: a label, a
 * {@link NumberStepper}, and a unit. The drum-beat and audio offsets used
 * to live in the playback bar (`playback.tsx::OffsetControl`); they moved
 * here to shrink that bar, but keep identical semantics, the stepper owns
 * the buffered-text editing so the value commits live without snapping
 * mid-keystroke.
 */
const AlignmentMenuItem = ({
  label,
  unit,
  value,
  step,
  min,
  precision = 2,
  title,
  ariaLabel,
  testId,
  onChange,
}: {
  label: string;
  unit: string;
  value: number;
  step: number;
  min?: number;
  precision?: number;
  title: string;
  ariaLabel: string;
  testId?: string;
  onChange: (v: number) => void;
}) => (
  <div className={styles.dropdownStepperRow} title={title}>
    <span>{label}</span>
    <span className={styles.dropdownStepperControl}>
      <NumberStepper
        value={value}
        onChange={onChange}
        step={step}
        min={min}
        precision={precision}
        ariaLabel={ariaLabel}
        testId={testId}
      />
      <span className={styles.dropdownStepperUnit}>{unit}</span>
    </span>
  </div>
);

/**
 * The "Edit" toolbar dropdown: undo/redo + note-editing options. Self-contained
 * `observer` that reads the editing + history stores/presenters off context
 * (the Toolbar renders inside their providers), so it needs no prop plumbing
 * through the app shell.
 *
 * Undo/Redo mirror Loro's UndoManager availability ({@link HistoryStore}); each
 * row is disabled but stays visible when its stack is empty, and shows its
 * keyboard shortcut pulled from the keymap registry (so a rebind reflects here
 * automatically rather than being hardcoded).
 *
 * Snapping targets the grid at the resolution of whichever grid-line families
 * are currently enabled (View → Grid lines), the union of their lines, and
 * applies to both inserting and moving notes.
 *
 * Alignment hosts the drum-beat and audio-track offset steppers (formerly in
 * the playback bar), read off {@link PlaybackStoreContext} and nudged through
 * {@link PlaybackPresenterContext}. The Beat row shows when a jot is loaded;
 * the Audio row when any backing audio track exists; the whole section is
 * omitted when neither holds (or the playback contexts are absent in tests).
 */
export const EditMenu = observer(() => {
  const editing = React.useContext(EditingStoreContext);
  const presenter = React.useContext(EditingPresenterContext);
  const history = React.useContext(HistoryStoreContext);
  const historyPresenter = React.useContext(HistoryPresenterContext);
  const playback = React.useContext(PlaybackStoreContext);
  const playbackPresenter = React.useContext(PlaybackPresenterContext);
  if (!editing || !presenter) return null;
  const snapping = editing.snappingEnabled;

  // Alignment section data, read off the playback store. A loaded jot
  // gates the Beat row (a drum-grid offset only makes sense with a jot
  // loaded); the audio offset row appears once any backing audio track
  // exists. `gridDivision` converts the stored quarter-note beats to the
  // 1/N-note units the Beat stepper shows.
  const jot = playback?.jotEditorStore.jot;
  const hasAudioTracks = jotPlayer.audioTracks.size > 0;
  const gridDivision = jot ? gridDivisionFor(jot) : DEFAULT_GRID_DIVISION;
  const showAlignment = !!playback && !!playbackPresenter && (!!jot || hasAudioTracks);
  return (
    <DropdownButton
      label={<ToolbarDropdownLabel>Edit</ToolbarDropdownLabel>}
      className={styles.playButton}
      title="Undo / redo and note-editing options."
    >
      {(close) => (
        <>
          <DropdownSection label="History">
            <ActionMenuItem
              label="Undo"
              disabled={!history?.canUndo}
              shortcut={shortcutForCommand('undo')}
              onClick={() => {
                historyPresenter?.undo();
                close();
              }}
              testId="edit-menu-undo"
              title="Undo the last edit."
            />
            <ActionMenuItem
              label="Redo"
              disabled={!history?.canRedo}
              shortcut={shortcutForCommand('redo')}
              onClick={() => {
                historyPresenter?.redo();
                close();
              }}
              testId="edit-menu-redo"
              title="Redo the last undone edit."
            />
          </DropdownSection>
          <DropdownSection label="Snapping">
            <ToggleMenuItem
              label="Enable snapping"
              active={snapping}
              onToggle={() => presenter.setSnapping(!snapping)}
              testId="edit-menu-snapping"
              title="Snap inserted and moved notes to the grid, at the resolution of the currently-enabled grid-line families (View → Grid lines; the union of their lines). Off = free placement."
            />
          </DropdownSection>
          {showAlignment && (
            <DropdownSection label="Alignment">
              {jot && (
                <AlignmentMenuItem
                  label="Beat"
                  unit={`/${gridDivision}`}
                  value={playback.drumOffsetBeats * (gridDivision / 4)}
                  step={1}
                  precision={0}
                  title={`Slide every drum note across the bars by this many 1/${gridDivision}-note units to realign a consistently mis-detected groove (${gridDivision / 4} = one quarter-note beat). Positive = later, negative = earlier. Reflows the score and reschedules playback live. Notes pushed off either end of the score are dropped.`}
                  ariaLabel={`Drum beat offset in 1/${gridDivision} units`}
                  testId="edit-menu-beat-offset"
                  onChange={(units) => playbackPresenter.setDrumOffset(units / (gridDivision / 4))}
                />
              )}
              {hasAudioTracks && (
                <AlignmentMenuItem
                  label="Audio"
                  unit="s"
                  value={-playback.songLeadInSec}
                  step={0.01}
                  min={0}
                  title="Drum-to-audio-track offset (the recording's lead-in), in seconds. Raising it slides the backing audio ahead of the drums; lowering it pulls them together. Takes effect instantly, including mid-playback, so you can nudge it until the drums lock to the track."
                  ariaLabel="Drum to audio track offset in seconds"
                  testId="edit-menu-audio-offset"
                  onChange={(sec) => playbackPresenter.setSongLeadIn(-sec)}
                />
              )}
            </DropdownSection>
          )}
        </>
      )}
    </DropdownButton>
  );
});
