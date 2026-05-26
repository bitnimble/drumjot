import classNames from 'classnames';
import React from 'react';
import styles from './checkbox.module.css';

type CheckboxProps = Omit<React.InputHTMLAttributes<HTMLInputElement>, 'type'>;

export const Checkbox = ({ className, ...rest }: CheckboxProps) => (
  <input {...rest} type="checkbox" className={classNames(styles.checkbox, className)} />
);
