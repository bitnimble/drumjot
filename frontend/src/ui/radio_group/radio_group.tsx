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
 *
 * Implements the WAI-ARIA radio-group keyboard pattern: the group is a single
 * Tab stop (roving `tabIndex`, only the checked option, or the first enabled
 * one when none is checked, is `0`); Arrow keys move selection + focus between
 * enabled options (wrapping), Home/End jump to the first/last enabled option.
 * Arrow selection commits through the same `onSelect` as a click.
 */
export function RadioGroup<T extends string>({
  options,
  selected,
  onSelect,
  ariaLabel,
  disabled = false,
}: RadioGroupProps<T>) {
  const refs = React.useRef<Array<HTMLButtonElement | null>>([]);
  refs.current.length = options.length;

  const isEnabled = (opt: RadioOption<T>): boolean => !disabled && !opt.disabled;
  const enabledIndices = options
    .map((opt, i) => (isEnabled(opt) ? i : -1))
    .filter((i) => i >= 0);

  // The single Tab stop: the first enabled checked option, else the first
  // enabled option. A fully-disabled group has none and stays untabbable.
  const tabStopIndex = (() => {
    const checked = enabledIndices.find((i) => selected.has(options[i].value));
    return checked ?? enabledIndices[0] ?? -1;
  })();

  const selectAt = (idx: number) => {
    const opt = options[idx];
    if (!opt || !isEnabled(opt)) return;
    onSelect(opt.value);
    refs.current[idx]?.focus();
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLButtonElement>, currentIdx: number) => {
    if (enabledIndices.length === 0) return;
    const pos = enabledIndices.indexOf(currentIdx);
    if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
      e.preventDefault();
      selectAt(enabledIndices[(pos + 1) % enabledIndices.length]);
    } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
      e.preventDefault();
      selectAt(enabledIndices[(pos - 1 + enabledIndices.length) % enabledIndices.length]);
    } else if (e.key === 'Home') {
      e.preventDefault();
      selectAt(enabledIndices[0]);
    } else if (e.key === 'End') {
      e.preventDefault();
      selectAt(enabledIndices[enabledIndices.length - 1]);
    }
  };

  return (
    <div className={styles.group} role="radiogroup" aria-label={ariaLabel}>
      {options.map((opt, i) => {
        const active = selected.has(opt.value);
        return (
          <button
            key={opt.value}
            ref={(el) => {
              refs.current[i] = el;
            }}
            type="button"
            role="radio"
            aria-checked={active}
            tabIndex={i === tabStopIndex ? 0 : -1}
            className={classNames(styles.option, active && styles.optionActive)}
            disabled={disabled || opt.disabled}
            title={opt.title}
            onClick={() => onSelect(opt.value)}
            onKeyDown={(e) => onKeyDown(e, i)}
            data-testid={opt.testId}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
