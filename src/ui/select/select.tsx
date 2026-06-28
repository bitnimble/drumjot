import classNames from 'classnames';
import React from 'react';
import styles from './select.module.css';

/**
 * Native `<select>` with the shared form-field chrome. Releases focus once a
 * value is committed so the global spacebar play/pause shortcut isn't
 * swallowed while focus lingers on a just-used dropdown. Forwards every
 * native select prop; pass `className` for call-site geometry (width / flex).
 */
export const Select = ({
  className,
  onChange,
  ...rest
}: React.ComponentPropsWithoutRef<'select'>) => (
  <select
    {...rest}
    className={classNames(styles.select, className)}
    onChange={(e) => {
      onChange?.(e);
      e.currentTarget.blur();
    }}
  />
);
