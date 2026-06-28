import classNames from 'classnames';
import React from 'react';
import styles from './progress_bar.module.css';

/**
 * Determinate horizontal progress bar. `value` is a 0..1 fraction (clamped);
 * the fill animates to it. Carries `role="progressbar"` with the matching
 * aria-value* state. Pass `className` for the track geometry (width).
 */
export const ProgressBar = ({
  value,
  ariaLabel,
  className,
  testId,
}: {
  value: number;
  ariaLabel?: string;
  className?: string;
  testId?: string;
}) => {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <span
      className={classNames(styles.track, className)}
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(pct)}
      aria-label={ariaLabel}
      data-testid={testId}
    >
      <span className={styles.fill} style={{ width: `${pct}%` }} />
    </span>
  );
};
