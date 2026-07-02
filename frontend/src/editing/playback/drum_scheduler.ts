/**
 * Schedules drum-kit notes onto the AudioContext clock for one play cycle.
 *
 * Owns the per-note stop callbacks and the running "last scheduled note"
 * time; the transport ({@link JotPlayer}) feeds it the event list, play
 * cursor, drift map, speed, and mute/solo filter and interleaves the
 * results with its timeline / tail-timer bookkeeping.
 *
 * NOT MobX-observable: it holds imperative scheduling state only, read
 * from the transport's imperative flow, never a render/reaction.
 */
import { PlaybackEvent } from './events';
import { DriftMap } from './drift_map';
import { GeneralUserGsKit } from './gm_kit';
import { isAudibleUnder, PlayerFilter } from './player_filter';

type Drums = ReturnType<typeof GeneralUserGsKit>;
type StopFn = (time?: number) => void;

// Per-row loudness trim applied on top of the user's volume fader,
// keyed by DSL lane letter ('k' = kick, 'h' = hi-hat, …). The GM
// SoundFont's hats are hot and the kick is weak relative to a real
// kit / backing track, so we duck the hats and lift the kick by
// default. Rows not listed play at their native velocity (1.0).
// Scaling velocity (not a GainNode) keeps accents/ghosts' relative
// dynamics intact and matches how the user volume fader already works.
const DEFAULT_PITCH_GAIN: Record<string, number> = {
  h: 0.6,
  k: 1.5,
};
// Playback velocity floor. smplr's per-note gain is quadratic in velocity
// (`vel² / 16129`; see `midiVelToGain`); so a notated `p` ghost at MIDI
// velocity 33 plays at gain ~0.068 (-23 dB) and on hats, which the
// DEFAULT_PITCH_GAIN trim scales down further, drops to ~-32 dB. That's
// fine in isolation but inaudible against a backing track; which is exactly
// when the user is practising and most wants to hear every hit. Floor the
// velocity passed to the kit so even the quietest written dynamic is
// reliably audible; smplr gain at velocity 50 is ~0.155 (-16 dB); still
// clearly below an unaccented mf (vel 64) hit so accent/ghost contrast
// survives. Floor applies *before* the per-row volume slider so manual
// attenuation still scales the row down to silent.
const MIN_PLAYBACK_VELOCITY = 50;
// Minimum effective per-row volume for any non-zero slider position. The
// raw fader [0, 1] is remapped to {0} ∪ [MIDI_VOLUME_FLOOR, 1] so the
// smallest audible setting still sits at a useful level against a
// backing track, below this, GM layers vanish into the mix. 0 still
// silences the row.
const MIDI_VOLUME_FLOOR = 0.4;

export class DrumScheduler {
  /**
   * Per-note stop callbacks returned by `drums.start()`. `drums.stop()`
   * on its own only halts notes that have already begun sounding, notes
   * scheduled for future audio-context times keep firing until they
   * reach their start, so we have to invoke each scheduled note's stop
   * function explicitly to make Stop actually stop playback.
   */
  private scheduledStops: StopFn[] = [];
  // Audio-context time of the last drum event scheduled by the most
  // recent `scheduleEvents` call. Tracked separately from the transport's
  // `tailAudioTime` (which already takes the max with audio-track endings)
  // so callers that don't reschedule drums (`repositionAudioForOffset` is
  // the only one today) can recompute the tail when only the audio side moves.
  private lastScheduledDrumTimeValue: number = 0;

  /** @param drums resolves the current (possibly still-unloaded) drum kit. */
  constructor(private readonly getDrums: () => Drums | undefined) {}

  get lastScheduledDrumTime(): number {
    return this.lastScheduledDrumTimeValue;
  }

  /** Halt the kit's live notes (flushes smplr's whole pending queue). */
  stopDrums(): void {
    this.getDrums()?.stop();
  }

  /**
   * Schedule every event whose source time is >= `fromOffset` (in jot
   * seconds) to play at `audioStartTime + (event.time - fromOffset) /
   * speed` on the audio context. The speed division spaces successive
   * hits further apart in real time at sub-1x speeds without touching
   * sample lane, drums still sound like drums at half speed, they just
   * play more slowly.
   *
   * Events filtered out by `filter` are skipped.
   *
   * Returns the latest audio context time at which a note was scheduled
   * (or `audioStartTime` if nothing scheduled) so the caller can compute
   * when to drop back to idle.
   *
   * @param isPlaying whether the transport is already `playing` (governs
   *   whether an all-notes-dropped schedule is a valid silent state or a
   *   genuine failure that should throw).
   */
  scheduleEvents(
    events: readonly PlaybackEvent[],
    fromOffset: number,
    audioStartTime: number,
    driftMap: DriftMap,
    speed: number,
    filter: PlayerFilter,
    isPlaying: boolean,
  ): number {
    const drums = this.getDrums();
    if (!drums) return audioStartTime;

    let lastTime = audioStartTime;
    let scheduled = 0;
    let mutedFiltered = 0;
    // Events whose `time` falls before the play cursor are silently
    // skipped (they're already in the past for this play call). Track
    // them separately so the "no audible notes scheduled" guard below
    // doesn't conflate them with "audible notes the kit failed to
    // schedule"; that wrong attribution turned a clean soloed-audio
    // playback (where the cymbal lane has a couple of cued events
    // sitting at the playhead's exact start time and skipping by ≤µs
    // float drift) into a hard error abort.
    let silentlySkipped = 0;
    // Schedule against MEDIA time so notes line up with the recording through
    // any per-bar drift. No drift → `jotToMedia` is `jot - songLeadIn` and the
    // delta below collapses to `ev.time - fromOffset` (the old formula).
    const map = driftMap;
    const fromMedia = map.jotToMedia(fromOffset);

    for (const ev of events) {
      if (ev.time < fromOffset) {
        silentlySkipped++;
        continue;
      }
      const evTrack = `${ev.layerId}/${ev.lane}`;
      if (!isAudibleUnder(evTrack, filter)) {
        mutedFiltered++;
        continue;
      }
      // The GeneralUser GS kit is keyed by GM percussion MIDI note
      // number (36 = kick, 38 = snare, …), exactly what `jotToEvents`
      // emits, so trigger the note directly; the SF2 zones map each
      // note to its own sample, no kit-group resolution needed.
      //
      // Per-row volume scales the note's velocity. smplr maps velocity
      // to gain, so a 0.5 fader roughly halves the row's loudness while
      // accents/ghosts (already baked into ev.velocity) keep their
      // relative dynamics. isAudibleUnder already rejected vol <= 0.
      // The DEFAULT_PITCH_GAIN trim stacks on top so hats/kick sit
      // right out of the box even before the user touches a fader.
      const rawVol = filter.volumes.get(evTrack) ?? 1;
      const vol = rawVol <= 0 ? 0 : MIDI_VOLUME_FLOOR + rawVol * (1 - MIDI_VOLUME_FLOOR);
      const defaultGain = DEFAULT_PITCH_GAIN[ev.lane] ?? 1;
      const floored = Math.max(MIN_PLAYBACK_VELOCITY, Math.round(ev.velocity * defaultGain));
      const velocity = Math.max(1, Math.min(127, Math.round(floored * vol)));
      const t = audioStartTime + (map.jotToMedia(ev.time) - fromMedia) / speed;
      const stopFn = drums.start({ note: ev.midiNote, time: t, velocity });
      this.scheduledStops.push(stopFn);
      scheduled++;
      if (t > lastTime) lastTime = t;
    }
    this.lastScheduledDrumTimeValue = lastTime;

    console.log(
      `[jotPlayer] scheduled ${scheduled}/${events.length} events ` +
        `(filtered by mute/solo: ${mutedFiltered}, ` +
        `skipped pre-cursor: ${silentlySkipped})`
    );

    // "Audible" here = passed both the pre-cursor time check AND the
    // mute/solo filter. Notes that were silently skipped for being
    // before `fromOffset` aren't candidates this call ever tried to
    // schedule, so they don't count toward "the kit failed us".
    const audible = events.length - mutedFiltered - silentlySkipped;
    if (scheduled === 0 && audible > 0 && !isPlaying) {
      // Nothing scheduled but notes survived BOTH the time check and
      // the mute/solo filter; a genuine, otherwise-invisible failure
      // (e.g. the kit loaded with no usable zones). Notes dropped
      // purely by an active mute/solo (audible=0) are instead a valid
      // silent-start state, handled by the caller exactly like a live
      // reschedule.
      throw new Error(
        `None of ${audible} audible notes could be ` +
          `scheduled on the GeneralUser GS kit. See console for the breakdown.`
      );
    }
    return lastTime;
  }

  cancelScheduledStops(): void {
    for (const fn of this.scheduledStops) {
      try {
        fn();
      } catch (err) {
        // A stop fn for a note that already finished may throw; ignore.
        console.debug('[jotPlayer] stopFn threw:', err);
      }
    }
    this.scheduledStops = [];
  }
}
