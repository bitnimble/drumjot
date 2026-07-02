import { observer } from 'mobx-react-lite';
import React from 'react';
import {
  ProvenanceStoreContext,
  ProvenancePresenterContext,
} from 'src/editing/provenance/provenance_contexts';
import {
  SettingsStoreContext,
  SettingsPresenterContext,
} from 'src/settings/settings_contexts';
import { themeStore, ThemeMode } from 'src/settings/theme';
import { DropdownButton, DropdownSection, ToggleMenuItem } from 'src/ui/dropdown/dropdown';
import { Slider } from 'src/ui/slider/slider';
import { ViewportStoreContext } from '../editing/viewport/viewport_contexts';
import { ToolbarDropdownLabel } from './toolbar';
import styles from './toolbar.module.css';

/**
 * Score-zoom slider, isolated as its own `observer` so the live
 * `store.zoom` read lives here, not in the app-root `View` / `Toolbar`.
 * Zoom ticks at up to display-refresh cadence during a wheel/pinch
 * gesture; if `View` read `store.zoom` (to thread it through as a
 * Toolbar prop) the whole shell + toolbar reconciled every tick. Reading
 * it in this leaf instead means only this single control reacts, and it
 * only mounts while the View dropdown is open anyway. Pairs with the
 * stable-prop isolation that keeps `JotEditor` off the zoom path entirely.
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
        <Slider
          className={styles.zoomSlider}
          min={0.1}
          max={4.0}
          step={0.05}
          value={zoom}
          onChange={onSetZoom}
          ariaLabel="Score zoom"
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
          role="menuitemradio"
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

/**
 * The "View" toolbar dropdown: score zoom, reference-grid / overlay toggles,
 * and the theme picker. Self-contained `observer` that reads the display
 * settings + provenance stores/presenters off context. `onSetZoom` stays a
 * prop: it centers the zoom on the viewport via a DOM-anchored closure the app
 * shell owns (the store holds no such view-glue), see {@link ZoomControl}.
 */
export const ViewMenu = observer(({ onSetZoom }: { onSetZoom: (z: number) => void }) => {
  const settings = React.useContext(SettingsStoreContext);
  const settingsPresenter = React.useContext(SettingsPresenterContext);
  const provenance = React.useContext(ProvenanceStoreContext);
  const provenancePresenter = React.useContext(ProvenancePresenterContext);
  if (!settings || !settingsPresenter) return null;
  const hasNoteProvenance = provenance?.noteProvenance !== undefined;
  const showFilteredOnsets = provenance?.showFilteredOnsets ?? false;
  const gridLines = settings.gridLines;
  return (
    <DropdownButton
      label={<ToolbarDropdownLabel>View</ToolbarDropdownLabel>}
      className={styles.playButton}
      title="Toggle on-screen reference grids, overlays, and the color theme."
    >
      {() => (
        <>
          <ZoomControl onSetZoom={onSetZoom} />
          <DropdownSection label="Overlays">
            <ToggleMenuItem
              label="Show filtered"
              active={showFilteredOnsets}
              onToggle={() => provenancePresenter?.setShowFilteredOnsets(!showFilteredOnsets)}
              disabled={!hasNoteProvenance}
              title={
                hasNoteProvenance
                  ? 'Render the onsets the filter LLM rejected as dashed ghost overlays at their detected (bar, beat) position. Click one to see why it was filtered out.'
                  : 'Load a filter-mode debug bundle to enable filtered-onset overlays.'
              }
            />
          </DropdownSection>
          <DropdownSection label="Layers">
            <ToggleMenuItem
              label="Visually merge layers"
              active={settings.mergeLayers}
              onToggle={() => settingsPresenter.setMergeLayers(!settings.mergeLayers)}
              title="Collapse tracks of the same lane across every || layer into a single row (the flat per-lane view), dropping the layer bands. View-only: notes keep their layer, so edits still route per-note and a new note lands on the firstmost layer carrying the lane."
            />
          </DropdownSection>
          <DropdownSection label="Waveforms">
            <ToggleMenuItem
              label="Uniform amplitude"
              active={settings.uniformWaveforms}
              onToggle={() => settingsPresenter.setUniformWaveforms(!settings.uniformWaveforms)}
              title="Normalise each audio track's waveform so the median non-silent peak fills most of the row, regardless of the source recording's amplitude. Silence still renders as silence. Off = accurate, on = uniform (easier to see quiet recordings)."
            />
            <ToggleMenuItem
              label="Bar & beat lines"
              active={settings.waveformGridLines}
              onToggle={() => settingsPresenter.setWaveformGridLines(!settings.waveformGridLines)}
              title="Draw bar lines and the beat grid over each audio-track waveform, aligned with the score above. Which sub-beat lines show follows the Grid lines section below, so a vertical line traces cleanly from the score down through every waveform."
            />
          </DropdownSection>
          <DropdownSection label="Grid lines">
            <ToggleMenuItem
              label="Main beat"
              active={gridLines.mainBeat}
              onToggle={() => settingsPresenter.toggleGridLine('mainBeat')}
              title="Dashed line under each notehead on the main beat (1, 2, 3, 4 in 4/4)."
            />
            <ToggleMenuItem
              label="Sub-beat (16ths)"
              active={gridLines.subBeat16}
              onToggle={() => settingsPresenter.toggleGridLine('subBeat16')}
              title="Dotted reference lines at every 16th-note position within each beat."
            />
            <ToggleMenuItem
              label="Sub-beat (6ths / quarter triplets)"
              active={gridLines.subBeatQuarterTriplet}
              onToggle={() => settingsPresenter.toggleGridLine('subBeatQuarterTriplet')}
              title="Dotted violet reference lines at every quarter-note triplet position (3 lines per 2 beats; 6 per bar in 4/4). Use to read quarter-note triplet phrases that 8th-triplet lines fragment too finely."
            />
            <ToggleMenuItem
              label="Sub-beat (12ths / triplets)"
              active={gridLines.subBeatTriplet}
              onToggle={() => settingsPresenter.toggleGridLine('subBeatTriplet')}
              title="Dotted violet reference lines at every triplet (1/3 of a beat) position."
            />
            <ToggleMenuItem
              label="Sub-beat (48ths)"
              active={gridLines.subBeat48}
              onToggle={() => settingsPresenter.toggleGridLine('subBeat48')}
              title="Very faint dotted lines at every 1/48 grid position (12 per beat). Covers both the 16th and triplet positions in one grid; useful for ultra-precise timing reference."
            />
          </DropdownSection>
          <ThemeSection />
        </>
      )}
    </DropdownButton>
  );
});
