import classNames from 'classnames';
import React from 'react';
import styles from './radio_group.module.css';

export type RadioOption<T extends string> = {
  value: T;
  label: React.ReactNode;
  disabled?: boolean;
  title?: string;
  testId?: string;
};

type RadioGroupProps<T extends string> = {
  options: ReadonlyArray<RadioOption<T>>;
  /**
   * Highlighted values. Usually exactly one; a multi-selection whose members
   * disagree highlights ALL the values present, so the spread is visible.
   * Clicking any option commits that single value to everything.
   */
  selected: ReadonlySet<T>;
  onSelect: (value: T) => void;
  ariaLabel: string;
  disabled?: boolean;
};

/**
 * A segmented single-choice control rendered as a button group. Accepts a SET
 * of active values (not one) so a mixed multi-selection can light up every
 * value it spans; picking one collapses the group to that choice.
 */
export function RadioGroup<T extends string>({
  options,
  selected,
  onSelect,
  ariaLabel,
  disabled = false,
}: RadioGroupProps<T>) {
  return (
    <div className={styles.group} role="radiogroup" aria-label={ariaLabel}>
      {options.map((opt) => {
        const active = selected.has(opt.value);
        return (
          <button
            key={opt.value}
            type="button"
            role="radio"
            aria-checked={active}
            className={classNames(styles.option, active && styles.optionActive)}
            disabled={disabled || opt.disabled}
            title={opt.title}
            onClick={() => onSelect(opt.value)}
            data-testid={opt.testId}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
