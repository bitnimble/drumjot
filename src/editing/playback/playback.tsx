import classNames from 'classnames';
import { AlertTriangle, Loader, Pause, Play, Square } from 'lucide-react';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { jotPlayer, PlayerState } from 'src/editing/playback/player';
import { Slider } from 'src/ui/slider/slider';
import sharedStyles from '../jot_editor.module.css';
import { FollowPlayheadContext } from './playback_contexts';
import styles from './playback.module.css';
import { VOLUME_STEP } from '../mixer/mixer_store';
import { JotEditorStore } from '../jot_editor_store';
import { PlaybackPresenter } from './playback_presenter';

function truncate(s: string, n: number): string {
  return s.length <= n ? s : `${s.slice(0, n - 1)}…`;
}

const PlaybackControls = observer(
  ({
    hasJot,
    playerState,
    playerError,
    onTogglePlayPause,
    onStop,
  }: {
    hasJot: boolean;
    playerState: PlayerState;
    playerError: string | undefined;
    onTogglePlayPause: () => void;
    onStop: () => void;
  }) => {
    const loading = playerState === 'loading';
    const playing = playerState === 'playing';
    const paused = playerState === 'paused';
    // Playback is "active" (Stop is meaningful, playhead is on screen)
    // while either playing or paused.
    const active = playing || paused;
    const hasError = !!playerError && !loading && !active;
    // Icon-only, like a media player. lucide icons: Play (play/resume),
    // Pause, Square (stop), AlertTriangle (error), Loader (loading; the
    // .transportButtonLoading class spins it).
    const TransportIcon = loading
      ? Loader
      : playing
        ? Pause
        : hasError
          ? AlertTriangle
          : Play;
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
          <FollowToggle />
          <button
            type="button"
            className={classNames(
              styles.transportButton,
              hasError && styles.transportButtonError,
              loading && styles.transportButtonLoading
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
            <TransportIcon size={15} aria-hidden="true" />
          </button>
          <button
            type="button"
            className={classNames(styles.transportButton, styles.transportButtonStop)}
            onClick={onStop}
            disabled={!active}
            aria-label="Stop"
            title={
              active
                ? 'Stop playback and reset to the start.'
                : 'Stop (available once playback has started).'
            }
          >
            <Square size={15} fill="currentColor" aria-hidden="true" />
          </button>
        </div>
        <div className={styles.transportAux}>
          <MasterVolumes />
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
 * header-crowding) play / pause / stop controls have their own
 * dedicated strip. `observer` + reading `jotPlayer` here keeps player
 * state re-renders scoped to this bar instead of bubbling up through
 * `View` and re-rendering the score on every transport change.
 */
export const PlaybackBar = observer(
  ({
    jotEditorStore,
    presenter,
  }: {
    jotEditorStore: JotEditorStore;
    presenter: PlaybackPresenter;
  }) => (
  <div className={styles.playbackBar}>
    <PlaybackControls
      hasJot={!!jotEditorStore.jot}
      playerState={jotPlayer.state}
      playerError={jotPlayer.errorMessage}
      onTogglePlayPause={() => presenter.togglePlayPause()}
      onStop={() => presenter.stopPlayback()}
    />
  </div>
));

/**
 * Bottom-bar toggle that flips `FollowPlayheadContext.follow`. On (the
 * default) keeps the score scrolled to the centred playhead during
 * playback; off lets the user pan to other sections while playing.
 * Flat-orange filled when on so its identity matches the playhead
 * marker it controls; outlined when off so the bar's neutral chrome
 * makes the disengaged state read at a glance.
 */
const FollowToggle = () => {
  const { follow, toggle } = React.useContext(FollowPlayheadContext);
  return (
    <button
      type="button"
      className={classNames(styles.followToggle, !follow && styles.followToggleOff)}
      onClick={toggle}
      aria-pressed={follow}
      aria-label={follow ? 'Disable playhead follow' : 'Enable playhead follow'}
      title={
        follow
          ? 'Follow is on: the score auto-scrolls to keep the playhead centred during playback. Click to turn off and pan freely while playing.'
          : 'Follow is off: the score stays put during playback. Click to re-engage auto-scroll.'
      }
    >
      <svg
        width="14"
        height="14"
        viewBox="0 0 12 12"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        {/* Arrow shaft + head pointing right at a vertical bar (the
            playhead). Reads as "follow this line". */}
        <line x1="1.5" y1="6" x2="8" y2="6" />
        <polyline points="5.5 3.5 8 6 5.5 8.5" />
        <line x1="10" y1="2.5" x2="10" y2="9.5" />
      </svg>
    </button>
  );
};

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
    <Slider
      step={VOLUME_STEP}
      value={value}
      onChange={onChange}
      ariaLabel={`${label} master volume`}
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
