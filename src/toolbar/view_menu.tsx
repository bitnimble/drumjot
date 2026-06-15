import { observer } from 'mobx-react-lite';
import React from 'react';
import { themeStore, ThemeMode } from 'src/settings/theme';
import { DropdownSection, ToggleMenuItem } from 'src/ui/dropdown/dropdown';
import { ViewportStoreContext } from '../jot_view/viewport/viewport_contexts';
import styles from './toolbar.module.css';

/**
 * Score-zoom slider, isolated as its own `observer` so the live
 * `store.zoom` read lives here, not in the app-root `View` / `Toolbar`.
 * Zoom ticks at up to display-refresh cadence during a wheel/pinch
 * gesture; if `View` read `store.zoom` (to thread it through as a
 * Toolbar prop) the whole shell + toolbar reconciled every tick. Reading
 * it in this leaf instead means only this single control reacts, and it
 * only mounts while the View dropdown is open anyway. Pairs with the
 * stable-prop isolation that keeps `JotView` off the zoom path entirely.
 */
export const ZoomControl = observer(({ onSetZoom }: { onSetZoom: (z: number) => void }) => {
  const viewport = React.useContext(ViewportStoreContext);
  const zoom = viewport?.zoom ?? 1;
  return (
    <label
      className={styles.dropdownStepperRow}
      title="Compress or expand the score horizontally. Has no effect on audio playback, only on how the notation is laid out."
    >
      <span>Zoom</span>
      <span className={styles.dropdownStepperControl}>
        <input
          type="range"
          min={0.1}
          max={4.0}
          step={0.05}
          value={zoom}
          onChange={(e) => onSetZoom(Number(e.target.value))}
          className={styles.zoomSlider}
          style={{ ['--value' as string]: (zoom - 0.1) / 3.9 } as React.CSSProperties}
          aria-label="Score zoom"
        />
        <span className={styles.zoomValue}>{Math.round(zoom * 100)}%</span>
      </span>
    </label>
  );
});

const THEME_MODE_LABELS: Record<ThemeMode, string> = {
  system: 'System',
  light: 'Light',
  dark: 'Dark',
};

const THEME_MODE_ORDER: readonly ThemeMode[] = ['system', 'light', 'dark'];

/**
 * Theme picker section rendered inside the View dropdown. `System`
 * (default) defers to the OS `prefers-color-scheme` and tracks live
 * changes; `Light`/`Dark` persist as an explicit override in
 * localStorage so subsequent visits skip the OS check entirely.
 *
 * Rendered as radio-style menu items (only one tick at a time); clicks
 * leave the View panel open so the user can switch and immediately
 * compare with other view toggles. The data-theme attribute is owned by
 * {@link themeStore}; this component is purely the picker.
 */
export const ThemeSection = observer(() => {
  const mode = themeStore.mode;
  return (
    <DropdownSection label="Theme">
      {THEME_MODE_ORDER.map((m) => (
        <ToggleMenuItem
          key={m}
          label={THEME_MODE_LABELS[m]}
          active={mode === m}
          onToggle={() => themeStore.setMode(m)}
          title={
            m === 'system'
              ? 'Follow the OS appearance setting (prefers-color-scheme).'
              : `Use the ${m} theme regardless of the OS setting.`
          }
        />
      ))}
    </DropdownSection>
  );
});
