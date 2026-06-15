import React from 'react';
import { createPortal } from 'react-dom';
import { HslColorPicker, HslColor } from 'react-colorful';
import { ColorPot } from 'src/ui/color_pot/color_pot';
import styles from './color_picker.module.css';

/**
 * Reusable colour-picker popover. Layout, top to bottom:
 *
 *   1. Palette row: clickable swatches for the project's default lane
 *      colours. Pre-selecting a palette colour is the fast path for
 *      users who just want a known-good hue.
 *   2. HSL wheel + saturation/lightness square (via `react-colorful`'s
 *      {@link HslColorPicker}). Live-updates on drag.
 *   3. Footer: hex input (read-only display of the current value) and a
 *      Reset button that clears the override via {@link onReset}.
 *
 * State is fully controlled by the caller; `value` is the colour
 * currently in effect (computed from override OR fallback), `onChange`
 * fires for every commit (palette click, wheel drag, hex edit), and
 * `onReset` is invoked to drop the override. Renders through a portal
 * positioned at `anchor` so the popover floats above the host dropdown
 * panel; closes when the user clicks outside or presses Escape.
 */
export const ColorPicker = ({
  open,
  anchor,
  value,
  palette,
  hasOverride,
  onChange,
  onReset,
  onClose,
}: {
  open: boolean;
  /** Top-left in viewport coordinates where the popover should appear. */
  anchor: { top: number; left: number } | null;
  /** Current `#rrggbb` colour to seed the picker with. */
  value: string;
  /** Pre-determined swatch row shown above the wheel. */
  palette: readonly string[];
  /** Whether the caller-owned state currently holds a user override.
   *  Drives the Reset button's enabled state. */
  hasOverride: boolean;
  /** Fired whenever the user picks a new colour (palette / wheel / hex). */
  onChange: (hex: string) => void;
  /** Fired when the user clicks the Reset button. */
  onReset: () => void;
  /** Fired when the popover should close (outside click / Escape). */
  onClose: () => void;
}) => {
  const panelRef = React.useRef<HTMLDivElement>(null);

  // Track the HSL value in component state during drag so the wheel
  // doesn't snap-back on round-trip through hex normalisation. The
  // controlled `value` prop is treated as the authority for the
  // initial seed; subsequent prop changes (e.g. from a palette click)
  // re-seed.
  const [hsl, setHsl] = React.useState<HslColor>(() => hexToHsl(value));
  const lastSeededValue = React.useRef(value);
  React.useEffect(() => {
    if (value !== lastSeededValue.current) {
      lastSeededValue.current = value;
      setHsl(hexToHsl(value));
    }
  }, [value]);

  // Outside-click + Escape. The host dropdown also has its own outside-
  // click handler; both ignore clicks inside this panel because the
  // panel sits in a portal that neither's wrapper contains.
  React.useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (panelRef.current?.contains(target)) return;
      onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('mousedown', onPointerDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onPointerDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open, onClose]);

  if (!open || !anchor) return null;
  return createPortal(
    <div
      ref={panelRef}
      className={styles.panel}
      role="dialog"
      aria-label="Colour picker"
      style={{ position: 'fixed', top: anchor.top, left: anchor.left }}
      // Keep clicks inside the popover from kicking off marquee
      // selection / row drag in the mixer below.
      onMouseDown={(e) => e.stopPropagation()}
      onClick={(e) => e.stopPropagation()}
    >
      <div
        className={styles.swatches}
        role="listbox"
        aria-label="Preset colours"
      >
        {palette.map((hex) => (
          <ColorPot
            key={hex}
            color={hex}
            selected={colorsEqual(hex, value)}
            role="option"
            title={hex}
            ariaLabel={`Preset colour ${hex}`}
            onClick={() => {
              onChange(hex);
              setHsl(hexToHsl(hex));
            }}
          />
        ))}
      </div>
      <HslColorPicker
        color={hsl}
        onChange={(next) => {
          setHsl(next);
          onChange(hslToHex(next));
        }}
        className={styles.wheel}
      />
      <div className={styles.footer}>
        <input
          type="text"
          className={styles.hexInput}
          value={value}
          aria-label="Hex colour value"
          onChange={(e) => {
            const raw = e.target.value.trim();
            if (/^#[0-9a-fA-F]{6}$/.test(raw)) {
              onChange(raw);
              setHsl(hexToHsl(raw));
            }
          }}
        />
        <button
          type="button"
          className={styles.reset}
          disabled={!hasOverride}
          onClick={onReset}
          title={
            hasOverride
              ? 'Clear the custom colour and revert to the default'
              : 'No custom colour set'
          }
        >
          Reset
        </button>
      </div>
    </div>,
    document.body,
  );
};

/** Loose equality on hex strings, case-insensitive and #-tolerant. */
function colorsEqual(a: string, b: string): boolean {
  return a.toLowerCase() === b.toLowerCase();
}

/** Convert `#rrggbb` to an HSL triple in the {@link HslColor} shape
 *  `react-colorful` expects (h ∈ [0,360), s/l ∈ [0,100]). Returns
 *  neutral grey on a malformed input so the wheel still has a sane
 *  starting position. */
function hexToHsl(hex: string): HslColor {
  if (!/^#[0-9a-fA-F]{6}$/.test(hex)) return { h: 0, s: 0, l: 50 };
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  let h = 0;
  let s = 0;
  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r:
        h = (g - b) / d + (g < b ? 6 : 0);
        break;
      case g:
        h = (b - r) / d + 2;
        break;
      case b:
        h = (r - g) / d + 4;
        break;
    }
    h *= 60;
  }
  return { h: Math.round(h), s: Math.round(s * 100), l: Math.round(l * 100) };
}

/** Inverse of {@link hexToHsl}: HSL triple → `#rrggbb`. */
function hslToHex({ h, s, l }: HslColor): string {
  const sN = s / 100;
  const lN = l / 100;
  const c = (1 - Math.abs(2 * lN - 1)) * sN;
  const hp = (h % 360) / 60;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  let r1 = 0;
  let g1 = 0;
  let b1 = 0;
  if (hp < 1) {
    r1 = c;
    g1 = x;
  } else if (hp < 2) {
    r1 = x;
    g1 = c;
  } else if (hp < 3) {
    g1 = c;
    b1 = x;
  } else if (hp < 4) {
    g1 = x;
    b1 = c;
  } else if (hp < 5) {
    r1 = x;
    b1 = c;
  } else {
    r1 = c;
    b1 = x;
  }
  const m = lN - c / 2;
  const r = Math.round((r1 + m) * 255);
  const g = Math.round((g1 + m) * 255);
  const b = Math.round((b1 + m) * 255);
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}
