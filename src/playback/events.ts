/**
 * Drumjot Jot -> flat playback event list.
 *
 * We reuse the canonical {@link toMidi} converter and re-parse the bytes with
 * `midi-file`'s `parseMidi` so playback exercises the exact same serialisation
 * path that downstream MIDI consumers see — there's only one source of truth
 * for tempo handling, time-signature changes, velocity resolution, and the
 * pitch->MIDI-note mapping.
 *
 * deltaTime is in ticks; we convert to absolute seconds while walking the
 * track, taking each `setTempo` meta event into account (per-bar BPM in the
 * Jot lowers into bar-boundary `setTempo` events on the MIDI side).
 */
import { parseMidi } from 'midi-file';
import { Jot } from 'src/dsl';
import { TICKS_PER_BEAT, toMidi } from 'src/midi';

export type PlaybackEvent = {
  /** Absolute time from the start of the jot, in seconds. */
  time: number;
  /** GM percussion note number. */
  midiNote: number;
  /** MIDI velocity (1-127). */
  velocity: number;
};

const DEFAULT_MICROS_PER_BEAT = 500_000; // 120 BPM

export function jotToEvents(jot: Jot): PlaybackEvent[] {
  const bytes = toMidi(jot);
  const parsed = parseMidi(bytes);
  const ppq = parsed.header.ticksPerBeat ?? TICKS_PER_BEAT;
  const events: PlaybackEvent[] = [];

  for (const track of parsed.tracks) {
    let microsPerBeat = DEFAULT_MICROS_PER_BEAT;
    let seconds = 0;
    for (const ev of track) {
      seconds += (ev.deltaTime / ppq) * (microsPerBeat / 1_000_000);
      if (ev.type === 'setTempo') {
        microsPerBeat = ev.microsecondsPerBeat;
      } else if (ev.type === 'noteOn' && ev.velocity > 0) {
        events.push({ time: seconds, midiNote: ev.noteNumber, velocity: ev.velocity });
      }
    }
  }

  events.sort((a, b) => a.time - b.time);
  return events;
}
