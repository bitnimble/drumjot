import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import React from 'react';
import {
  JotTimeline,
  jotPlayer,
  KitInfo,
  PlayerState,
  timeToX,
} from 'src/playback';
import sharedStyles from '../jot_view.module.css';
import styles from './playback.module.css';
import { Select } from './toolbar';
import { JotViewStore, VOLUME_STEP } from './store';

const PLAYBACK_SPEEDS: readonly number[] = [0.25, 0.5, 0.75, 1.0, 1.25];

function truncate(s: string, n: number): string {
  return s.length <= n ? s : `${s.slice(0, n - 1)}…`;
}

/**
 * Numeric up/down for a playback offset. Editing commits live — every
 * keystroke and spinner click pushes the new value through `onChange`,
 * which the caller applies immediately (including mid-playback). A local
 * text buffer lets the user clear/retype the field freely; it re-syncs to
 * the incoming value whenever the input isn't focused (e.g. when loading a
 * new jot reseeds the offset).
 *
 * Used for two distinct offsets: the audio-track offset (seconds, the
 * recording's lead-in) and the drum beat-grid offset (beats, realigning a
 * mis-detected groove).
 */
const OffsetControl = ({
  label,
  unit,
  value,
  step,
  min,
  title,
  ariaLabel,
  onChange,
}: {
  label: string;
  unit: string;
  value: number;
  step: number;
  min?: number;
  title: string;
  ariaLabel: string;
  onChange: (v: number) => void;
}) => {
  const [text, setText] = React.useState(value.toFixed(2));
  const [editing, setEditing] = React.useState(false);
  React.useEffect(() => {
    if (!editing) setText(value.toFixed(2));
  }, [value, editing]);
  const commit = (raw: string) => {
    const n = parseFloat(raw);
    if (Number.isFinite(n)) onChange(n);
  };
  return (
    <label className={sharedStyles.toolbarCheckbox} title={title}>
      <span>{label}</span>
      <input
        type="number"
        className={sharedStyles.offsetInput}
        min={min}
        step={step}
        value={text}
        onFocus={() => setEditing(true)}
        onBlur={(e) => {
          setEditing(false);
          commit(e.target.value);
        }}
        onChange={(e) => {
          setText(e.target.value);
          commit(e.target.value);
        }}
        aria-label={ariaLabel}
      />
      <span>{unit}</span>
    </label>
  );
};

const PlaybackControls = observer(
  ({
    hasJot,
    playerState,
    playerError,
    playbackSpeed,
    drumKits,
    drumPreset,
    hasAudioTracks,
    audioOffsetSec,
    drumOffsetBeats,
    onTogglePlayPause,
    onStop,
    onSetPlaybackSpeed,
    onSetDrumPreset,
    onSetAudioOffset,
    onSetDrumOffset,
  }: {
    hasJot: boolean;
    playerState: PlayerState;
    playerError: string | undefined;
    playbackSpeed: number;
    drumKits: KitInfo[];
    drumPreset: number;
    hasAudioTracks: boolean;
    audioOffsetSec: number;
    drumOffsetBeats: number;
    onTogglePlayPause: () => void;
    onStop: () => void;
    onSetPlaybackSpeed: (speed: number) => void;
    onSetDrumPreset: (preset: number) => void;
    onSetAudioOffset: (sec: number) => void;
    onSetDrumOffset: (beats: number) => void;
  }) => {
    const loading = playerState === 'loading';
    const playing = playerState === 'playing';
    const paused = playerState === 'paused';
    // Playback is "active" (Stop is meaningful, playhead is on screen)
    // while either playing or paused.
    const active = playing || paused;
    const hasError = !!playerError && !loading && !active;
    // Icon-only, like a media player. Glyphs: ▶ play/resume, ⏸ pause,
    // ■ stop, ⚠ error, ⏳ loading.
    const transportIcon = loading ? '⏳' : playing ? '⏸' : hasError ? '⚠' : '▶';
    const transportAria = loading
      ? 'Loading'
      : playing
        ? 'Pause'
        : paused
          ? 'Resume'
          : 'Play';
    return (
      <>
        {/* Empty left cell balances the right-hand aux controls so the
            transport group stays optically centred in the bar. */}
        <div className={styles.transportSpacer} aria-hidden="true" />
        <div className={styles.transportCenter}>
          <button
            type="button"
            className={classNames(
              styles.transportButton,
              hasError && styles.transportButtonError
            )}
            onClick={onTogglePlayPause}
            disabled={!hasJot || loading}
            aria-label={transportAria}
            title={
              playerError
                ? `Playback error: ${playerError}`
                : playing
                  ? 'Pause playback (spacebar). The playhead and audio freeze in place; press again to resume.'
                  : paused
                    ? 'Resume playback (spacebar).'
                    : 'Play the current jot through an acoustic General MIDI drum kit (GeneralUser GS, spacebar also toggles play/pause). The first play downloads a ~30 MB SoundFont; it is then cached in the browser for instant loads on later sessions.'
            }
          >
            {transportIcon}
          </button>
          <button
            type="button"
            className={classNames(
              styles.transportButton,
              styles.transportButtonStop,
              styles.transportStop
            )}
            onClick={onStop}
            disabled={!active}
            aria-label="Stop"
            title={
              active
                ? 'Stop playback and reset to the start.'
                : 'Stop (available once playback has started).'
            }
          >
            ■
          </button>
        </div>
        <div className={styles.transportAux}>
          <MasterVolumes />
          {drumKits.length > 0 && (
            <label
              className={sharedStyles.toolbarCheckbox}
              title="Drum kit (a preset of the GeneralUser GS SoundFont). Switching is instant — the SoundFont is already downloaded; only the active samples change. Takes effect immediately, including mid-playback."
            >
              <span>Kit</span>
              <Select
                className={sharedStyles.samplesSelect}
                value={String(drumPreset)}
                onChange={(e) => onSetDrumPreset(Number(e.target.value))}
              >
                {drumKits.map((k) => (
                  <option key={k.preset} value={String(k.preset)}>
                    {k.name}
                  </option>
                ))}
              </Select>
            </label>
          )}
          <label
            className={sharedStyles.toolbarCheckbox}
            title="Tempo multiplier applied to playback. Slowing down spaces the drum hits further apart and time-stretches the audio tracks — pitch is preserved for both, so a half-speed practice pass stays in tune."
          >
            <span>Speed</span>
            <Select
              className={sharedStyles.samplesSelect}
              value={String(playbackSpeed)}
              onChange={(e) => onSetPlaybackSpeed(Number(e.target.value))}
            >
              {PLAYBACK_SPEEDS.map((s) => (
                <option key={s} value={String(s)}>
                  {s.toFixed(2)}×
                </option>
              ))}
            </Select>
          </label>
          {hasJot && (
            <OffsetControl
              label="Beat"
              unit="beats"
              value={drumOffsetBeats}
              step={0.25}
              title="Slide every drum note across the bars by this many beats to realign a consistently mis-detected groove (e.g. a kick transcribed 1.5 beats late in every bar). Positive = later, negative = earlier. Reflows the score and reschedules playback live. Notes pushed off either end of the score are dropped."
              ariaLabel="Drum beat offset in beats"
              onChange={onSetDrumOffset}
            />
          )}
          {hasAudioTracks && (
            <OffsetControl
              label="Audio"
              unit="s"
              value={audioOffsetSec}
              step={0.01}
              min={0}
              title="Drum-to-audio-track offset (the recording's lead-in), in seconds. Raising it slides the backing audio ahead of the drums; lowering it pulls them together. Takes effect instantly, including mid-playback, so you can nudge it until the drums lock to the track."
              ariaLabel="Drum to audio track offset in seconds"
              onChange={onSetAudioOffset}
            />
          )}
          {hasError && (
            <span
              className={classNames(sharedStyles.statusPill, sharedStyles.statusPillError)}
              title={playerError}
            >
              Playback: {truncate(playerError ?? '', 60)}
            </span>
          )}
        </div>
      </>
    );
  }
);

/**
 * Bottom transport bar. Pinned below the score so the (formerly
 * header-crowding) play / pause / stop / speed controls have their own
 * dedicated strip. `observer` + reading `jotPlayer` here keeps player
 * state re-renders scoped to this bar instead of bubbling up through
 * `View` and re-rendering the score on every transport change.
 */
export const PlaybackBar = observer(({ store }: { store: JotViewStore }) => (
  <div className={styles.playbackBar}>
    <PlaybackControls
      hasJot={!!store.currentJot}
      playerState={jotPlayer.state}
      playerError={jotPlayer.errorMessage}
      playbackSpeed={jotPlayer.playbackSpeed}
      drumKits={jotPlayer.drumKits}
      drumPreset={jotPlayer.drumPreset}
      hasAudioTracks={jotPlayer.audioTracks.size > 0}
      audioOffsetSec={jotPlayer.drumsT0Sec}
      drumOffsetBeats={store.drumOffsetBeats}
      onTogglePlayPause={() => store.togglePlayPause()}
      onStop={() => store.stopPlayback()}
      onSetPlaybackSpeed={(s) => jotPlayer.setPlaybackSpeed(s)}
      onSetDrumPreset={(p) => jotPlayer.setDrumPreset(p)}
      onSetAudioOffset={(sec) => jotPlayer.setDrumsT0Sec(sec)}
      onSetDrumOffset={(beats) => store.setDrumOffset(beats)}
    />
  </div>
));

export const Playhead = observer(
  ({
    showLabel = false,
    onSeek,
  }: {
    showLabel?: boolean;
    onSeek: (x: number) => void;
  }) => {
    const timeline = jotPlayer.timeline;
    const active =
      jotPlayer.state === 'playing' ||
      jotPlayer.state === 'paused' ||
      // Idle but the user clicked to position the playhead before
      // pressing Play — show it parked at the cued spot.
      jotPlayer.cued;
    if (!active || timeline.bars.length === 0) return null;
    const x = timeToX(timeline, jotPlayer.currentTime);

    // Drag-to-scrub on the line itself or its label. stopPropagation
    // blocks the page-level marquee start; data-noseek prevents the
    // bars-row onClick from firing on mouseup of the drag.
    const onMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
      if (e.button !== 0) return;
      e.stopPropagation();
      e.preventDefault();
      const parent = e.currentTarget.parentElement;
      if (!parent) return;
      const rect = parent.getBoundingClientRect();
      onSeek(e.clientX - rect.left);
      const onMove = (ev: MouseEvent) => {
        onSeek(ev.clientX - rect.left);
      };
      const onUp = () => {
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('mouseup', onUp);
      };
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', onUp);
    };

    return (
      <div
        className={styles.playhead}
        style={{ left: x }}
        onMouseDown={onMouseDown}
        data-noseek
      >
        {showLabel && (
          <div className={styles.playheadLabel}>
            <div>{formatPlayheadTime(jotPlayer.currentTime)}</div>
            {(() => {
              const pos = playheadBarBeat(timeline, jotPlayer.currentTime);
              return pos ? (
                <div className={styles.playheadLabelBarBeat}>{pos}</div>
              ) : null;
            })()}
          </div>
        )}
      </div>
    );
  }
);

function formatPlayheadTime(seconds: number): string {
  const negative = seconds < 0;
  const abs = Math.abs(seconds);
  const totalSec = Math.floor(abs);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  const cs = Math.floor((abs - totalSec) * 100);
  return `${negative ? '-' : ''}${min}:${String(sec).padStart(2, '0')}.${String(cs).padStart(2, '0')}`;
}

/**
 * Convert the playhead's jot-time position to `Bar N, X.XXb` for the
 * second line of the label. Walks the timeline's per-bar timings to
 * find the bar containing `jotTime`, then computes beat-in-bar in the
 * bar's time-signature beats (1-indexed at the downbeat). Returns
 * `null` when no bar can be resolved (empty timeline / no rendered
 * voice).
 */
function playheadBarBeat(timeline: JotTimeline, jotTime: number): string | null {
  const renderedBars = timeline.rendered?.structure.voices[0]?.bars ?? [];
  if (renderedBars.length === 0 || timeline.bars.length === 0) return null;
  for (let i = 0; i < timeline.bars.length; i++) {
    const t = timeline.bars[i]!;
    if (jotTime < t.startSec + t.durationSec) {
      const rb = renderedBars[i];
      if (!rb || t.durationSec <= 0) return null;
      const beatInBar = 1 + ((jotTime - t.startSec) / t.durationSec) * rb.time.count;
      return `Bar ${rb.index}, ${beatInBar.toFixed(2)}b`;
    }
  }
  // Past the end of the last bar — pin to its final beat so the label
  // doesn't blank out when scrubbing slightly past the score's tail.
  const last = renderedBars[renderedBars.length - 1];
  if (!last) return null;
  return `Bar ${last.index}, ${(last.time.count + 1).toFixed(2)}b`;
}

/**
 * One labelled master fader in the transport bar. Pure attenuation
 * (0..1); the percent readout doubles as a "back to default" affordance
 * since 100% is unity.
 */
const MasterVolumeSlider = ({
  label,
  title,
  value,
  onChange,
}: {
  label: string;
  title: string;
  value: number;
  onChange: (v: number) => void;
}) => (
  <label className={styles.masterVolume} title={title}>
    <span>{label}</span>
    <input
      type="range"
      min={0}
      max={1}
      step={VOLUME_STEP}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      aria-label={`${label} master volume`}
      style={{ ['--value' as string]: value } as React.CSSProperties}
    />
    <span className={styles.masterVolumeValue}>{Math.round(value * 100)}%</span>
  </label>
);

/**
 * The page-wide master fader, read straight off the observable
 * `jotPlayer` (no prop drilling — it's app-wide, not per-jot). Takes
 * effect instantly, including mid-playback, and persists across plays.
 * It's the last gain stage so it scales the drums and every audio track
 * together; the per-section masters live in their gutters (see
 * `GutterMasterRow` in the mixer module).
 */
const MasterVolumes = observer(() => (
  <div className={styles.masterVolumes}>
    <MasterVolumeSlider
      label="Master"
      title="Page-wide master volume — scales the drums and every audio track together. The last fader before output."
      value={jotPlayer.masterVolume}
      onChange={(v) => jotPlayer.setMasterVolume(v)}
    />
  </div>
));
