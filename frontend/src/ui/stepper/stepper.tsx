import classNames from 'classnames';
import React from 'react';
import { useBufferedStepper } from './use_buffered_stepper';
import styles from './stepper.module.css';

type StepperProps = {
  /** Current value, or `null` to show {@link placeholder} (a multi-selection
   *  whose members disagree). */
  value: number | null;
  /**
   * Step up (`+1`) or down (`-1`). Only the DIRECTION is reported: the owner
   * applies the delta, which for a multi-selection nudges every member by its
   * own step rather than snapping them to a shared value.
   */
  onStep: (dir: 1 | -1) => void;
  /** Commit a typed absolute value (Enter / blur). Omit for a step-only control
   *  (the input becomes read-only). */
  onSet?: (value: number) => void;
  placeholder?: string;
  /** Decimals shown for the committed value. */
  precision?: number;
  ariaLabel: string;
  disabled?: boolean;
  testId?: string;
};

/**
 * Compact numeric up/down: `[−] [value] [+]` in one pill border. Unlike
 * {@link NumberStepper}, the buttons emit a step DIRECTION (not an absolute
 * value) so a caller editing several items at once can nudge each
 * independently, and `value` may be `null` to show a mixed-state placeholder.
 */
export const Stepper = ({
  value,
  onStep,
  onSet,
  placeholder = '--',
  precision = 0,
  ariaLabel,
  disabled = false,
  testId,
}: StepperProps) => {
  const buffered = useBufferedStepper({
    commit: (raw) => {
      const trimmed = raw.trim();
      const n = Number(trimmed);
      if (trimmed !== '' && Number.isFinite(n)) onSet?.(n);
    },
    deriveDisplay: (text) =>
      text !== null ? text : value === null ? '' : value.toFixed(precision),
    step: onStep,
    handleArrowKeys: true,
    commitOnChange: false,
  });

  return (
    <div className={styles.stepper} data-disabled={disabled || undefined} data-testid={testId}>
      <button
        type="button"
        className={classNames(styles.button, styles.minus)}
        aria-label={`${ariaLabel}: decrease`}
        disabled={disabled}
        // Keep focus in the input so a typed value isn't lost to a button click.
        onMouseDown={(e) => e.preventDefault()}
        onClick={() => onStep(-1)}
      >
        −
      </button>
      <input
        className={styles.input}
        value={buffered.display}
        placeholder={placeholder}
        inputMode="decimal"
        aria-label={ariaLabel}
        disabled={disabled}
        readOnly={!onSet}
        onChange={(e) => onSet && buffered.onChange(e.target.value)}
        onFocus={(e) => e.currentTarget.select()}
        onBlur={buffered.onBlur}
        onKeyDown={buffered.onKeyDown}
        data-testid={testId ? `${testId}-input` : undefined}
      />
      <button
        type="button"
        className={classNames(styles.button, styles.plus)}
        aria-label={`${ariaLabel}: increase`}
        disabled={disabled}
        onMouseDown={(e) => e.preventDefault()}
        onClick={() => onStep(1)}
      >
        +
      </button>
    </div>
  );
};
