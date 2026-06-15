import React from 'react';
import { ColorPicker } from 'src/ui/color_picker/color_picker';
import { ColorPot } from 'src/ui/color_pot/color_pot';
import styles from './color_picker_menu_row.module.css';

/**
 * One row inside a `DropdownButton` panel that exposes a colour picker.
 * Renders as `[Label] [colour pot]`; clicking the pot opens the
 * {@link ColorPicker} popover anchored to the pot's bottom-left.
 *
 * State is owned by the caller; `value` is the currently-effective
 * colour, `onChange` fires on every commit, `onReset` clears the
 * override, and `hasOverride` drives the picker's Reset button state.
 *
 * Stops mousedown propagation so a row drag in the mixer doesn't fire
 * when the user clicks the pot.
 */
export const ColorPickerMenuRow = ({
  label,
  value,
  palette,
  hasOverride,
  onChange,
  onReset,
  ariaLabel,
}: {
  label: React.ReactNode;
  value: string;
  palette: readonly string[];
  hasOverride: boolean;
  onChange: (hex: string) => void;
  onReset: () => void;
  ariaLabel: string;
}) => {
  const [open, setOpen] = React.useState(false);
  const [anchor, setAnchor] = React.useState<{ top: number; left: number } | null>(
    null,
  );
  const potRef = React.useRef<HTMLButtonElement | null>(null);

  const reposition = React.useCallback(() => {
    const rect = potRef.current?.getBoundingClientRect();
    if (!rect) return;
    // Slot the popover under the pot's bottom edge with a small gap so
    // it doesn't appear to be glued onto the swatch.
    setAnchor({ top: rect.bottom + 6, left: rect.left });
  }, []);

  React.useEffect(() => {
    if (!open) return;
    reposition();
    window.addEventListener('scroll', reposition, true);
    window.addEventListener('resize', reposition);
    return () => {
      window.removeEventListener('scroll', reposition, true);
      window.removeEventListener('resize', reposition);
    };
  }, [open, reposition]);

  return (
    <div
      className={styles.row}
      onClick={(e) => e.stopPropagation()}
      onMouseDown={(e) => e.stopPropagation()}
    >
      <span>{label}</span>
      <ColorPot
        ref={potRef}
        color={value}
        ariaLabel={ariaLabel}
        ariaHasPopup="dialog"
        ariaExpanded={open}
        onClick={() => {
          setOpen((o) => !o);
        }}
      />
      <ColorPicker
        open={open}
        anchor={anchor}
        value={value}
        palette={palette}
        hasOverride={hasOverride}
        onChange={onChange}
        onReset={() => {
          onReset();
        }}
        onClose={() => setOpen(false)}
      />
    </div>
  );
};
