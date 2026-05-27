import React from 'react';
import styles from './color_picker_menu_item.module.css';

/**
 * Colour-picker menu row, intended to be dropped inside a
 * `DropdownButton` panel. Lays out as `[label] [swatch] [Reset]` so it
 * reads as a single setting line alongside other in-menu rows (the
 * lyrics overflow's offset stepper, etc.). The swatch is a native
 * `<input type="color">` whose tile is restyled to fit the menu's
 * warm-paper chrome; clicking it opens the OS colour picker.
 *
 * State is owned by the caller. `value` is the colour currently in
 * effect (override OR derived default if no override is set);
 * `onChange` is called on every picker commit; `onReset` clears the
 * override; and `isOverridden` controls whether the Reset button is
 * enabled. The caller owns the override / default distinction because
 * the comparison shape (hex string match, palette-index check, etc.)
 * varies by call site.
 */
export const ColorPickerMenuItem = ({
  label,
  value,
  isOverridden,
  onChange,
  onReset,
  ariaLabel,
}: {
  label: React.ReactNode;
  value: string;
  isOverridden: boolean;
  onChange: (hex: string) => void;
  onReset: () => void;
  ariaLabel: string;
}) => {
  return (
    <label
      className={styles.row}
      onClick={(e) => e.stopPropagation()}
      onMouseDown={(e) => e.stopPropagation()}
    >
      <span>{label}</span>
      <span className={styles.controls}>
        <input
          type="color"
          className={styles.swatch}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          aria-label={ariaLabel}
        />
        <button
          type="button"
          className={styles.reset}
          disabled={!isOverridden}
          onClick={onReset}
          title={
            isOverridden
              ? 'Clear the custom colour and revert to the default'
              : 'No custom colour set'
          }
        >
          Reset
        </button>
      </span>
    </label>
  );
};
