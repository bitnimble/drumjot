import { observer } from 'mobx-react-lite';
import React from 'react';
import { jotPlayer, PLAYBACK_SPEED_MAX, PLAYBACK_SPEED_MIN, PLAYBACK_SPEED_STEP } from 'src/editing/playback/player';
import {
  PlaybackStoreContext,
  PlaybackPresenterContext,
} from 'src/editing/playback/playback_contexts';
import { DropdownButton, SubmenuItem, ToggleMenuItem } from 'src/ui/dropdown/dropdown';
import { NumberStepper } from 'src/ui/number_stepper/number_stepper';
import { ToolbarDropdownLabel } from './toolbar';
import styles from './toolbar.module.css';

/**
 * Drum-kit picker inside the toolbar's Playback menu. Reads
 * `jotPlayer.drumKits` + `drumPreset` directly so re-renders are scoped
 * to this submenu, not the whole toolbar. Renders disabled until the
 * SoundFont is loaded and reports its kit list, so the menu's shape
 * stays stable across the load.
 */
export const PlaybackKitSubmenu = observer(() => {
  const kits = jotPlayer.drumKits;
  const current = jotPlayer.drumPreset;
  const noKits = kits.length === 0;
  return (
    <SubmenuItem
      label="Drum kit"
      disabled={noKits}
      title={
        noKits
          ? 'Drum kit picker. Available once the GeneralUser GS SoundFont has finished loading; press Play to trigger the one-time download.'
          : 'Drum kit (a preset of the GeneralUser GS SoundFont). Switching is instant, the SoundFont is already downloaded; only the active samples change. Takes effect immediately, including mid-playback.'
      }
    >
      {() =>
        kits.map((k) => (
          <ToggleMenuItem
            key={k.preset}
            label={k.name}
            active={k.preset === current}
            onToggle={() => jotPlayer.setDrumPreset(k.preset)}
          />
        ))
      }
    </SubmenuItem>
  );
});

/**
 * Numeric input inside the toolbar's Playback menu for the tempo
 * multiplier. ± buttons step by `PLAYBACK_SPEED_STEP` (0.25), so the
 * usual practice grid (0.25 / 0.5 / 0.75 / 1.0 / ...) is one click
 * apart; arbitrary typed values still snap to the grid in the player.
 * Reads `jotPlayer.playbackSpeed` directly so speed changes only
 * re-render this row, not the whole toolbar.
 */
export const PlaybackSpeedItem = observer(() => (
  <label
    className={styles.dropdownStepperRow}
    title="Tempo multiplier applied to playback. Slowing down spaces the drum hits further apart and time-stretches the audio tracks (pitch preserved), so a half-speed practice pass stays in tune."
  >
    <span>Speed</span>
    <span className={styles.dropdownStepperControl}>
      <NumberStepper
        value={jotPlayer.playbackSpeed}
        onChange={(v) => jotPlayer.setPlaybackSpeed(v)}
        step={PLAYBACK_SPEED_STEP}
        min={PLAYBACK_SPEED_MIN}
        max={PLAYBACK_SPEED_MAX}
        precision={2}
        ariaLabel="Playback speed multiplier"
      />
      <span className={styles.dropdownStepperUnit}>×</span>
    </span>
  </label>
));

/**
 * Numeric input inside the toolbar's Playback menu for the audio-vs-visual
 * sync trim, in milliseconds. Positive values delay the perceived audio
 * relative to the playhead (visual leads); negative values pull audio
 * earlier. Reads `jotPlayer.audioLatencyMs` directly so changes only
 * re-render this row, not the whole toolbar.
 */
export const AudioLatencyItem = observer(() => (
  <label
    className={styles.dropdownStepperRow}
    title="Adjust the on-screen playhead's position relative to the audio playback clock. Positive values mean the audio is delayed (visual playhead leads); negative values pull the audio earlier. Use this to compensate if the playhead appears to lead or lag the sound. Has no effect on the jot, beats, notes, or audio tracks themselves."
  >
    <span>Audio latency</span>
    <span className={styles.dropdownStepperControl}>
      <NumberStepper
        value={jotPlayer.audioLatencyMs}
        onChange={(v) => jotPlayer.setAudioLatencyMs(v)}
        step={5}
        min={-500}
        max={500}
        precision={0}
        ariaLabel="Audio latency adjustment in milliseconds"
      />
      <span className={styles.dropdownStepperUnit}>ms</span>
    </span>
  </label>
));

/**
 * The "Playback" toolbar dropdown: drum kit, speed, audio-latency trim, and the
 * auto-follow-on-play toggle. Self-contained `observer`; the kit/speed/latency
 * rows read `jotPlayer` directly, the toggle reads the playback store/presenter
 * off context (the Toolbar renders inside their providers).
 */
export const PlaybackMenu = observer(() => {
  const playback = React.useContext(PlaybackStoreContext);
  const playbackPresenter = React.useContext(PlaybackPresenterContext);
  return (
    <DropdownButton
      label={<ToolbarDropdownLabel>Playback</ToolbarDropdownLabel>}
      className={styles.playButton}
      title="Drum kit (sample set) and playback speed."
    >
      {() => (
        <>
          <PlaybackKitSubmenu />
          <PlaybackSpeedItem />
          <AudioLatencyItem />
          <ToggleMenuItem
            label="Auto-enable follow on play"
            active={playback?.autoFollowOnPlay ?? false}
            onToggle={() =>
              playbackPresenter?.setAutoFollowOnPlay(!(playback?.autoFollowOnPlay ?? false))
            }
            title="When on, pressing Play (or resuming) re-enables Auto-follow if it was disabled mid-playback (pan, minimap drag, follow-button toggle while playing). Turning Auto-follow off while paused or stopped is treated as deliberate and survives the next play. Off = current Auto-follow state is always preserved across plays."
          />
        </>
      )}
    </DropdownButton>
  );
});
