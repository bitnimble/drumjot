import classNames from 'classnames';
import React from 'react';
import styles from './spinner.module.css';

const TONE_CLASS = {
  accent: styles.toneAccent,
  muted: styles.toneMuted,
  current: styles.toneCurrent,
} as const;

export type SpinnerTone = keyof typeof TONE_CLASS;

/**
 * Rotating-border in-flight indicator. The keyframes + circle geometry are
 * shared; `size` / `thickness` come through as inline styles and `tone`
 * picks the ring colours, so a call site is just props (no per-spinner CSS).
 * Decorative by default (`aria-hidden`); pass `label` to make it an
 * accessible `role="status"` (when the spinner itself is the busy signal,
 * not a sibling of one). `title` adds a hover tooltip.
 */
export const Spinner = ({
  size = 16,
  thickness = 1.5,
  tone = 'accent',
  label,
  title,
  className,
  testId,
}: {
  size?: number;
  thickness?: number;
  tone?: SpinnerTone;
  /** Accessible name; promotes the spinner to `role="status"`. */
  label?: string;
  title?: string;
  className?: string;
  testId?: string;
}) => (
  <span
    className={classNames(styles.spinner, TONE_CLASS[tone], className)}
    {...(label ? { role: 'status', 'aria-label': label } : { 'aria-hidden': true })}
    title={title}
    data-testid={testId}
    style={{ width: size, height: size, borderWidth: thickness }}
  />
);
