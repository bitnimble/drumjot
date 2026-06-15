import { observer } from 'mobx-react-lite';
import { jotPlayer, PLAYBACK_SPEED_MAX, PLAYBACK_SPEED_MIN, PLAYBACK_SPEED_STEP } from 'src/editing/playback/player';
import { SubmenuItem, ToggleMenuItem } from 'src/ui/dropdown/dropdown';
import { NumberStepper } from 'src/ui/number_stepper/number_stepper';
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
