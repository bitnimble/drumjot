import classNames from 'classnames';
import { X } from 'lucide-react';
import React from 'react';
import styles from './icon_button.module.css';

type IconButtonProps = {
  active?: boolean;
  activeClassName?: string;
  className?: string;
  children: React.ReactNode;
} & Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, 'className' | 'type'>;

/**
 * Compact 18x18 icon button shared by mixer-row controls. Always stops
 * mouse-event propagation so clicking the button doesn't also fire the
 * surrounding row's seek-on-click handler. Specialised wrappers below
 * (Mute/Solo/Clear) bind the right classes and toggle semantics.
 */
export const IconButton = ({
  active,
  activeClassName,
  className,
  children,
  onClick,
  onMouseDown,
  ...rest
}: IconButtonProps) => (
  <button
    type="button"
    className={classNames(styles.iconButton, className, active && activeClassName)}
    onClick={(e) => {
      e.stopPropagation();
      onClick?.(e);
    }}
    onMouseDown={(e) => {
      e.stopPropagation();
      onMouseDown?.(e);
    }}
    {...rest}
  >
    {children}
  </button>
);

type ToggleProps = {
  active: boolean;
  onToggle: () => void;
  /** Title/aria-label when `active` is false (e.g. "Mute kick"). */
  offTitle: string;
  /** Title/aria-label when `active` is true (e.g. "Unmute kick"). */
  onTitle: string;
  testId?: string;
};

export const MuteButton = ({ active, onToggle, offTitle, onTitle, testId }: ToggleProps) => {
  const title = active ? onTitle : offTitle;
  return (
    <IconButton
      active={active}
      activeClassName={styles.muteActive}
      onClick={onToggle}
      title={title}
      aria-label={title}
      aria-pressed={active}
      data-testid={testId}
    >
      M
    </IconButton>
  );
};

export const SoloButton = ({ active, onToggle, offTitle, onTitle, testId }: ToggleProps) => {
  const title = active ? onTitle : offTitle;
  return (
    <IconButton
      active={active}
      activeClassName={styles.soloActive}
      onClick={onToggle}
      title={title}
      aria-label={title}
      aria-pressed={active}
      data-testid={testId}
    >
      S
    </IconButton>
  );
};

type ClearButtonProps = {
  onClear: () => void;
  /** Title + aria-label, e.g. "Remove the kick audio track". */
  label: string;
  testId?: string;
};

export const ClearButton = ({ onClear, label, testId }: ClearButtonProps) => (
  <IconButton
    className={styles.clear}
    onClick={onClear}
    title={label}
    aria-label={label}
    data-testid={testId}
  >
    <X size={12} aria-hidden="true" />
  </IconButton>
);
