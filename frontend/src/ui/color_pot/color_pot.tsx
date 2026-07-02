import classNames from 'classnames';
import React from 'react';
import styles from './color_pot.module.css';

/**
 * A small fixed-size colour swatch button. One canonical shape and size
 * shared everywhere a colour is shown as a clickable chip: the row
 * inside an overflow menu (opens the picker popover), each preset in
 * that popover's palette grid, and any future swatch-style affordance.
 *
 * `selected` paints the accent-coloured ring so the popover can flag
 * the current colour against its palette row. Screen readers see the
 * same state via `aria-selected` when it's an `option` (inside the
 * picker listbox) or `aria-pressed` otherwise (a plain toggle button).
 */
export const ColorPot = React.forwardRef<
  HTMLButtonElement,
  {
    color: string;
    selected?: boolean;
    onClick?: (e: React.MouseEvent<HTMLButtonElement>) => void;
    title?: string;
    ariaLabel?: string;
    ariaHasPopup?: 'dialog' | 'menu';
    ariaExpanded?: boolean;
    /** Optional role override (e.g. `option` when used inside a
     *  `listbox`). Defaults to a plain button. */
    role?: string;
  }
>(({ color, selected, onClick, title, ariaLabel, ariaHasPopup, ariaExpanded, role }, ref) => {
  const isOption = role === 'option';
  return (
    <button
      ref={ref}
      type="button"
      role={role}
      className={classNames(styles.pot, selected && styles.selected)}
      style={{ background: color }}
      title={title}
      aria-label={ariaLabel}
      aria-selected={isOption ? selected : undefined}
      aria-pressed={isOption ? undefined : selected}
      aria-haspopup={ariaHasPopup}
      aria-expanded={ariaExpanded}
      onClick={onClick}
    />
  );
});
