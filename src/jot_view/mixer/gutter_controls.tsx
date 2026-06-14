import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { jotPlayer } from 'src/jot_view/playback/player';
import { GutterResizeHandle } from '../components/gutter_resize_handle';
import { MuteButton, SoloButton } from '../components/icon_button';
import styles from './mixer.module.css';
import { VOLUME_STEP } from '../store';

/**
 * A per-section master fader that sits in the sticky lane gutter,
 * directly above the section it controls (the loaded audio tracks, or
 * the drum/instrument staff). Gutter-aligned (same sticky column width
 * — `--gutter-width` — as the per-row M/S/volume controls below it) so
 * it reads as the "header" for that column. Reads/writes the global
 * observable `jotPlayer`; all pointer events are kept from bubbling so
 * dragging the fader doesn't start the page marquee or trip
 * seek-on-click.
 */
export const GutterMasterRow = observer(
  ({
    label,
    title,
    value,
    onChange,
    muted,
    soloed,
    audible,
    onToggleMute,
    onToggleSolo,
    testId,
    onResizeGutterStart,
  }: {
    label: string;
    title: string;
    value: number;
    onChange: (v: number) => void;
    muted: boolean;
    soloed: boolean;
    /** True when the section's bus would currently make sound (master
     * mute / cross-domain solo can drop it). Dims the row to match the
     * per-row label-dim treatment when the section is silent. */
    audible: boolean;
    onToggleMute: () => void;
    onToggleSolo: () => void;
    testId?: string;
    onResizeGutterStart: (e: React.PointerEvent<HTMLDivElement>) => void;
  }) => {
    const stop = (e: React.MouseEvent) => e.stopPropagation();
    const pct = Math.round(value * 100);
    return (
      <div className={styles.gutterMasterRow}>
        <div className={styles.gutterMasterGutter} title={title} data-testid={testId}>
          <span
            className={classNames(styles.gutterMasterLabel, !audible && styles.musicTrackLabelDim)}
          >
            {label}
          </span>
          <div className={styles.gutterMasterControls}>
            <input
              type="range"
              className={styles.gutterMasterSlider}
              min={0}
              max={1}
              step={VOLUME_STEP}
              value={value}
              onChange={(e) => onChange(parseFloat(e.target.value))}
              onClick={stop}
              onMouseDown={stop}
              onMouseUp={stop}
              aria-label={`${label} volume`}
              title={`${label} volume: ${pct}%`}
              style={{ ['--value' as string]: value } as React.CSSProperties}
            />
            <span className={styles.gutterMasterValue}>{pct}%</span>
            <MuteButton
              active={muted}
              onToggle={onToggleMute}
              offTitle={`Mute ${label}`}
              onTitle={`Unmute ${label}`}
              testId={testId ? `${testId}-mute` : undefined}
            />
            <SoloButton
              active={soloed}
              onToggle={onToggleSolo}
              offTitle={`Solo ${label}`}
              onTitle={`Unsolo ${label}`}
              testId={testId ? `${testId}-solo` : undefined}
            />
          </div>
          <GutterResizeHandle onResizeStart={onResizeGutterStart} />
        </div>
      </div>
    );
  }
);

/**
 * Compact horizontal volume fader shared by the pitch gutter and the
 * audio-track gutter. Range is 0..1 (pure attenuation). All mouse events are
 * kept from bubbling so dragging the fader doesn't start the page-level
 * marquee selection or trip the seek-on-click handler.
 */
export const RowVolumeSlider = ({
  value,
  onChange,
  label,
}: {
  value: number;
  onChange: (v: number) => void;
  label: string;
}) => {
  const stop = (e: React.MouseEvent) => e.stopPropagation();
  return (
    <input
      type="range"
      className={styles.rowVolume}
      min={0}
      max={1}
      step={VOLUME_STEP}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      onClick={stop}
      onMouseDown={stop}
      onMouseUp={stop}
      title={`${label} volume: ${Math.round(value * 100)}%`}
      aria-label={`${label} volume`}
      style={{ ['--value' as string]: value } as React.CSSProperties}
    />
  );
};
