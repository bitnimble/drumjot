import classNames from 'classnames';
import React from 'react';
import styles from './checkbox.module.css';

type CheckboxProps = Omit<React.InputHTMLAttributes<HTMLInputElement>, 'type'> & {
  /** Tri-state middle: draws the indeterminate dash. Native `indeterminate` is
   *  a DOM property (no HTML attribute), so it's applied through a ref. A
   *  multi-selection with differing values renders this. */
  indeterminate?: boolean;
};

export const Checkbox = ({ className, indeterminate = false, ...rest }: CheckboxProps) => {
  const ref = React.useRef<HTMLInputElement>(null);
  React.useEffect(() => {
    if (ref.current) ref.current.indeterminate = indeterminate;
  }, [indeterminate]);
  return (
    <input
      {...rest}
      ref={ref}
      type="checkbox"
      className={classNames(styles.checkbox, className)}
    />
  );
};
