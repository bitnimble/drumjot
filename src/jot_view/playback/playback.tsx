import classNames from 'classnames';
import { AlertTriangle, Loader, Pause, Play, Square } from 'lucide-react';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { DEFAULT_GRID_DIVISION, gridDivisionFor } from 'src/grid';
import { JotTimeline, jotPlayer, PlayerState } from 'src/playback';
import sharedStyles from '../../jot_view.module.css';
import { NumberStepper } from '../components/number_stepper';
import { FollowPlayheadContext } from '../contexts';
import styles from './playback.module.css';
import { VOLUME_STEP } from '../store';
import { DocumentStore } from '../document/document_store';
import { PlaybackStore } from './playback_store';
import { PlaybackPresenter } from './playback_presenter';

function truncate(s: string, n: number): string {
  return s.length <= n ? s : `${s.slice(0, n - 1)}…`;
}

/**
 * Numeric up/down for a playback offset, wrapped with a label and unit.
 * Used for two distinct offsets: the audio-track offset (seconds, the
 * recording's lead-in) and the drum beat-grid offset (beats, realigning
 * a mis-detected groove). The stepper itself owns the buffered-text
 * editing semantics so the value commits live (every keystroke and +/−
 * click) without snapping mid-keystroke.
 */
const OffsetControl = ({
  label,
  unit,
  value,
  step,
  min,
  precision = 2,
  title,
  ariaLabel,
  onChange,
}: {
  label: string;
  unit: string;
  value: number;
  step: number;
  min?: number;
  precision?: number;
  title: string;
  ariaLabel: string;
  onChange: (v: number) => void;
}) => (
  <label className={sharedStyles.toolbarCheckbox} title={title}>
    <span>{label}</span>
    <NumberStepper
      value={value}
      onChange={onChange}
      step={step}
      min={min}
      precision={precision}
      ariaLabel={ariaLabel}
    />
    <span>{unit}</span>
  </label>
);

const PlaybackControls = observer(
  ({
    hasJot,
    playerState,
    playerError,
    hasAudioTracks,
    audioOffsetSec,
    drumOffsetBeats,
    gridDivision,
    onTogglePlayPause,
    onStop,
    onSetAudioOffset,
    onSetDrumOffset,
  }: {
    hasJot: boolean;
    playerState: PlayerState;
    playerError: string | undefined;
    hasAudioTracks: boolean;
    audioOffsetSec: number;
    drumOffsetBeats: number;
    /** Grid density (1/N-of-whole-note) of the current jot. */
    gridDivision: number;
    onTogglePlayPause: () => void;
    onStop: () => void;
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
            <TransportIcon size={22} aria-hidden="true" />
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
            <Square size={22} fill="currentColor" aria-hidden="true" />
          </button>
        </div>
        <div className={styles.transportAux}>
          <MasterVolumes />
          {(hasJot || hasAudioTracks) && (
            <div className={styles.offsetStack}>
              {hasJot && (
                <OffsetControl
                  label="Beat"
                  unit={`/${gridDivision}`}
                  value={drumOffsetBeats * (gridDivision / 4)}
                  step={1}
                  precision={0}
                  title={`Slide every drum note across the bars by this many 1/${gridDivision}-note units to realign a consistently mis-detected groove (${gridDivision / 4} = one quarter-note beat). Positive = later, negative = earlier. Reflows the score and reschedules playback live. Notes pushed off either end of the score are dropped.`}
                  ariaLabel={`Drum beat offset in 1/${gridDivision} units`}
                  onChange={(units) => onSetDrumOffset(units / (gridDivision / 4))}
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
            </div>
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
 * header-crowding) play / pause / stop controls have their own
 * dedicated strip. `observer` + reading `jotPlayer` here keeps player
 * state re-renders scoped to this bar instead of bubbling up through
 * `View` and re-rendering the score on every transport change.
 */
export const PlaybackBar = observer(
  ({
    documentStore,
    playback,
    presenter,
  }: {
    documentStore: DocumentStore;
    playback: PlaybackStore;
    presenter: PlaybackPresenter;
  }) => (
  <div className={styles.playbackBar}>
    <PlaybackControls
      hasJot={!!documentStore.currentJot}
      playerState={jotPlayer.state}
      playerError={jotPlayer.errorMessage}
      hasAudioTracks={jotPlayer.audioTracks.size > 0}
      audioOffsetSec={jotPlayer.drumsT0Sec}
      drumOffsetBeats={playback.drumOffsetBeats}
      gridDivision={
        documentStore.currentJot ? gridDivisionFor(documentStore.currentJot) : DEFAULT_GRID_DIVISION
      }
      onTogglePlayPause={() => presenter.togglePlayPause()}
      onStop={() => presenter.stopPlayback()}
      onSetAudioOffset={(sec) => jotPlayer.setDrumsT0Sec(sec)}
      onSetDrumOffset={(beats) => presenter.setDrumOffset(beats)}
    />
  </div>
));

/**
 * Travelling playback marker. Rendered in three places (timeline header,
 * each audio-track row, each instrument row) so the playhead line is
 * visible across the whole vertical stack.
 *
 * Position is NOT driven from this component; it's read from the
 * `--playhead-x` CSS custom property that `PlayheadPosVar` writes once
 * per frame on each `[data-playhead="1"]` element (see jot_view.tsx).
 * The var is registered `inherits: false`, so the per-tick `setProperty`
 * has to be made per-playhead; `PlayheadPosVar` iterates a cached
 * `DomTargetCache.playheads` Set (maintained via a single
 * `MutationObserver` on the JotView root) instead of `querySelectorAll`-ing
 * every frame. The shell only re-renders when `state` / `cued` / `timeline`
 * change (i.e. transport events, not per-frame playback) so a debug bundle
 * with many tracks doesn't pay N × (reconciliation + new style object +
 * new closure) every frame the way it used to.
 *
 * Drag-to-scrub still works because the mousedown handler measures
 * against the parent bars-row's bounding rect, which is unaffected by
 * this element's own transform.
 */
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
      // pressing Play; show it parked at the cued spot.
      jotPlayer.cued;
    if (!active || timeline.bars.length === 0) return null;

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
        onMouseDown={onMouseDown}
        data-noseek
        data-playhead="1"
      >
        {showLabel && <PlayheadLabel timeline={timeline} />}
      </div>
    );
  }
);

/**
 * Time / bar-beat readout shown on top of the timeline-header playhead.
 * Only ONE instance ever mounts (the per-row playheads pass
 * `showLabel={false}`), so reading `jotPlayer.currentTime` here gives
 * us a per-frame re-render of a tiny tree (two text nodes) and nothing
 * else, no shell rerenders, no per-row label rerenders.
 */
const PlayheadLabel = observer(({ timeline }: { timeline: JotTimeline }) => {
  const t = jotPlayer.currentTime;
  const pos = playheadBarBeat(timeline, t);
  return (
    <div className={styles.playheadLabel}>
      <div>{formatPlayheadTime(t)}</div>
      {pos && <div className={styles.playheadLabelBarBeat}>{pos}</div>}
    </div>
  );
});

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
      // Truncate (not round) so the tail of a bar never rounds up to the
      // next bar's downbeat, e.g. 4/4 must go 4.99 → 1.00, never 5.00.
      const beatDisplay = (Math.floor(beatInBar * 100) / 100).toFixed(2);
      return `Bar ${rb.index}, ${beatDisplay}b`;
    }
  }
  // Past the end of the last bar; pin to its final beat so the label
  // doesn't blank out when scrubbing slightly past the score's tail.
  const last = renderedBars[renderedBars.length - 1];
  if (!last) return null;
  return `Bar ${last.index}, ${(last.time.count + 0.99).toFixed(2)}b`;
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
