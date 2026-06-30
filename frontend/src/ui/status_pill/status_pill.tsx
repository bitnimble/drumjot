import classNames from 'classnames';
import React from 'react';
import styles from './status_pill.module.css';

const TONE_CLASS = {
  busy: styles.busy,
  error: styles.error,
  success: styles.success,
} as const;

export type StatusPillTone = keyof typeof TONE_CLASS;

/**
 * Compact rounded status pill for an in-flight / error / success signal in
 * the toolbar or transport. `tone` swaps the colour triplet; the body
 * (`children`) is the label, optionally preceded by a {@link Spinner}.
 */
export const StatusPill = ({
  tone = 'busy',
  title,
  className,
  testId,
  children,
}: {
  tone?: StatusPillTone;
  title?: string;
  className?: string;
  testId?: string;
  children: React.ReactNode;
}) => (
  <span
    className={classNames(styles.pill, TONE_CLASS[tone], className)}
    title={title}
    data-testid={testId}
  >
    {children}
  </span>
);
