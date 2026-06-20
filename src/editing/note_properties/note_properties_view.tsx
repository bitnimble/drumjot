import classNames from 'classnames';
import { ChevronDown } from 'lucide-react';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { Checkbox } from 'src/ui/checkbox/checkbox';
import { DropdownButton } from 'src/ui/dropdown/dropdown';
import { RadioGroup } from 'src/ui/radio_group/radio_group';
import { Stepper } from 'src/ui/stepper/stepper';
import { BarBeatField } from 'src/ui/bar_beat_field/bar_beat_field';
import type { Sticking } from 'src/schema/schema';
import { NotePropertiesStoreContext, NotePropertiesPresenterContext } from './note_properties_contexts';
import { MIXED, type Mixed } from './note_properties_store';
import { STICKING_FIELDS } from './note_properties_fields';
import styles from './note_properties_view.module.css';

/** A field value that may be shared, mixed, or absent -> a control value
 *  (mixed/absent become `null`, which controls render as `--`). */
function num(v: number | Mixed | undefined): number | null {
  return typeof v === 'number' ? v : null;
}

/**
 * Editable properties of the current note selection: lane, position (bar +
 * beat), volume, micro-timing, roll, articulation modifiers, and sticking.
 * Edits apply to the whole selection; fields whose members disagree show `--`
 * or an indeterminate control. Reads the store, calls the presenter for writes.
 */
export const NotePropertiesView = observer(function NotePropertiesView() {
  const store = React.useContext(NotePropertiesStoreContext);
  const presenter = React.useContext(NotePropertiesPresenterContext);
  if (!store || !presenter) return null;

  if (store.count === 0) {
    return (
      <p className={styles.empty} data-testid="note-properties-empty">
        Select a note to see its properties.
      </p>
    );
  }

  const lane = store.lane;
  const laneValue = typeof lane === 'string' ? lane : '';

  return (
    <div className={styles.props} data-testid="note-properties">
      <div className={styles.field}>
        <span className={styles.label}>Lane</span>
        <select
          className={styles.select}
          value={laneValue}
          onChange={(e) => presenter.setLane(e.target.value)}
          data-testid="np-lane"
        >
          {lane === MIXED && (
            <option value="" disabled>
              Multiple
            </option>
          )}
          {store.availableLanes.map((l) => (
            <option key={l.lane} value={l.lane}>
              {l.name}
            </option>
          ))}
        </select>
      </div>

      <div className={styles.field}>
        <span className={styles.label}>Volume</span>
        <div className={styles.inline}>
          <Stepper
            value={num(store.volumeUi)}
            onStep={(d) => presenter.stepVolume(d)}
            onSet={(v) => presenter.setVolume(v)}
            ariaLabel="Volume"
            testId="np-volume"
          />
          <span className={styles.unit} data-testid="np-volume-label">
            {store.volumeLabel ?? ''}
          </span>
        </div>
      </div>

      <div className={styles.row}>
        <div className={styles.field}>
          <BarBeatField
            bar={num(store.bar)}
            beat={num(store.beat)}
            onStepBar={(d) => presenter.stepBar(d)}
            onStepBeat={(d) => presenter.stepBeat(d)}
            onSetBar={(v) => presenter.setBar(v)}
            onSetBeat={(v) => presenter.setBeat(v)}
            testId="np-barbeat"
          />
        </div>

        <div className={styles.field}>
          <span className={styles.label}>Micro timing</span>
          <div className={styles.inline}>
            <Stepper
              value={num(store.microTiming)}
              onStep={(d) => presenter.stepMicroTiming(d)}
              onSet={(v) => presenter.setMicroTiming(v)}
              ariaLabel="Micro timing"
              testId="np-microtiming"
            />
            <span className={styles.unit}>ms</span>
          </div>
        </div>
      </div>

      <div className={styles.field}>
        <span className={styles.label}>Articulation</span>
        <DropdownButton
          title="Articulation"
          className={styles.articulationTrigger}
          panelClassName={styles.articulationPanel}
          label={
            <>
              <span
                className={classNames(
                  styles.articulationSummary,
                  store.articulationSummary === '' && styles.placeholder
                )}
                data-testid="np-articulation-summary"
              >
                {store.articulationSummary || 'None'}
              </span>
              <ChevronDown size={14} aria-hidden="true" className={styles.articulationChevron} />
            </>
          }
        >
          {() => (
            <>
              <label className={styles.checkRow}>
                <Checkbox
                  checked={store.roll === true}
                  indeterminate={store.roll === MIXED}
                  onChange={() => presenter.toggleRoll()}
                  data-testid="np-roll"
                />
                <span>Roll</span>
              </label>
              <div className={styles.modifierGrid}>
                {store.modifierRows.map((row) => (
                  <label
                    key={row.mod}
                    className={styles.checkRow}
                    data-disabled={!row.enabled || undefined}
                  >
                    <Checkbox
                      checked={row.state === true}
                      indeterminate={row.state === MIXED}
                      disabled={!row.enabled}
                      onChange={() => presenter.toggleModifier(row.mod)}
                      data-testid={`np-modifier-${row.mod}`}
                    />
                    <span>{row.label}</span>
                  </label>
                ))}
              </div>
            </>
          )}
        </DropdownButton>
      </div>

      <div className={styles.field}>
        <span className={styles.label}>Sticking</span>
        <RadioGroup<Sticking | 'none'>
          options={STICKING_FIELDS.map((s) => ({
            value: s.value,
            label: s.label,
            testId: `np-sticking-${s.value}`,
          }))}
          selected={store.stickingValues}
          onSelect={(v) => presenter.setSticking(v)}
          ariaLabel="Sticking"
        />
      </div>
    </div>
  );
});
