/**
 * RenderedJot -> flat playback event list, keyed by DSL pitch.
 *
 * We walk the laid-out jot directly (rather than round-tripping through
 * `toMidi` + `parseMidi`) for two reasons:
 *
 *  1. The DSL pitch letter ('k', 's', 'h', ...) survives end-to-end. The
 *     scheduler uses it to filter by mute/solo row; MIDI bytes don't
 *     carry that information.
 *
 *  2. The tempo timeline lives on `jot.tempoEvents`, which supports
 *     mid-bar tempo changes natively. The MIDI export carries the same
 *     information but the scheduler reads `tempoEvents` directly via the
 *     shared per-bar tempo helper so the playhead, audio waveform and
 *     scheduled drums all share one clock.
 */
import { Volume } from 'src/dsl';
import { RenderedJot, ResolvedNote, ResolvedTrack } from 'src/jot';
import {
  ACCENT_BOOST,
  DEFAULT_VELOCITY,
  GHOST_REDUCTION,
  VOLUME_TO_VELOCITY,
} from 'src/dynamics';
import { defaultMidiNote } from 'src/midi/gm';
import { beatToSecWithinBar, buildBarTempos } from 'src/tempo';

export type PlaybackEvent = {
  /** Absolute time from the start of the jot, in seconds. */
  time: number;
  /** GM percussion note number (post resolution of per-note overrides). */
  midiNote: number;
  /** MIDI velocity (1-127). */
  velocity: number;
  /** DSL pitch letter the note was written under; used by mute/solo. */
  pitch: string;
};

// Velocity defaults (DEFAULT_VELOCITY / ACCENT_BOOST / GHOST_REDUCTION /
// VOLUME_TO_VELOCITY) come from the shared `src/dynamics.ts` so playback
// loudness matches exactly what gets written to an exported `.mid` /
// `.rlrr` file.

// Flam = a grace stroke shortly before the main hit on the same drum.
// Acoustic flams sit ~25-35 ms apart, but two SF2 voices that close on the
// same key get swallowed by voice-stealing / overlapping attack envelopes
// and read as one strike. 50 ms is the standard "wide flam" — clearly two
// hits and survives smplr retriggering.
const FLAM_GRACE_OFFSET_SEC = 0.03;
// Grace stroke loudness, relative to the main hit. Flam grace notes are
// slightly softer than the primary, but not as quiet as a ghost — keep
// it well above the SF2's near-silent low-velocity layers. Scaling
// (rather than a fixed value) keeps an accented flam's grace proportional.
const FLAM_GRACE_VELOCITY_RATIO = 0.9;

export function jotToEvents(rendered: RenderedJot): PlaybackEvent[] {
  const resolved = rendered.resolved;
  const events: PlaybackEvent[] = [];

  // Bar 1 (= first non-lead-in bar) sits at jot time 0 by convention,
  // matching `buildTimeline`'s anchor. Pre-drum bars get a negative
  // bar-offset so an empty lead-in produces no events but a lead-in bar
  // that DOES contain notes (rare; today only relevant if the upstream
  // generator stamps drums into a pre-drum bar) fires its events at
  // negative jot time; at media `jot + drumsT0Sec >= 0` in audio time,
  // which is still valid. Lead-in count comes from the structure
  // (counting leading negative-indexed bars); `structureForVoice`
  // materialises both the explicit-leadBars and the chrome-only
  // (`drumsT0Sec` without `leadBars`) source shapes into the same
  // negative-indexed-bar form, so the scheduler reads only the
  // structure.
  for (const voice of resolved.voices) {
    let leadBars = 0;
    for (const b of voice.bars) {
      if (b.index >= 0) break;
      leadBars++;
    }
    // Per-bar tempo segments from `jot.tempoEvents` give the exact
    // intra-bar tempo curve. Each note's time is `barOffset +
    // beatToSecWithinBar(barTempos, note.beat)` so a note that sits
    // after a mid-bar tempo change picks up the post-change rate.
    const tempos = buildBarTempos(rendered.source, voice.bars);
    let leadOffsetSec = 0;
    for (let i = 0; i < leadBars; i++) leadOffsetSec += tempos[i].durationSec;

    let barOffsetSec = -leadOffsetSec;
    for (let i = 0; i < voice.bars.length; i++) {
      const bar = voice.bars[i];
      const barTempos = tempos[i];
      for (const pitch of voice.pitches) {
        const track = bar.tracks[pitch];
        if (!track) continue;
        for (const note of track.notes) {
          const midiNote = resolveMidiNote(note, track);
          if (midiNote === undefined) continue;
          const velocity = resolveVelocity(note);
          // Sub-slot timing offset (ms) nudges the scheduled time directly;
          // it's already in real seconds so no tempo conversion is needed.
          const offsetSec = (note.source.offset ?? 0) / 1000;
          const time = barOffsetSec + beatToSecWithinBar(barTempos, note.beat) + offsetSec;
          events.push({ time, midiNote, velocity, pitch });
          if (note.modifiers.has('fl')) {
            // The grace stroke clamps to -leadOffsetSec so it can't run
            // past the start of the pre-drum window (a flam on bar 1
            // beat 1 with no lead-in clamps to 0 = bar 1 downbeat). Above
            // that floor it sounds the configured offset earlier than the
            // main hit; at the floor it collapses with the main hit
            // rather than disappearing entirely.
            const graceFloor = -leadOffsetSec;
            const graceTime = Math.max(graceFloor, time - FLAM_GRACE_OFFSET_SEC);
            const graceVel = Math.max(1, Math.round(velocity * FLAM_GRACE_VELOCITY_RATIO));
            events.push({ time: graceTime, midiNote, velocity: graceVel, pitch });
          }
        }
      }
      barOffsetSec += barTempos.durationSec;
    }
  }

  events.sort((a, b) => a.time - b.time);
  return events;
}

function resolveMidiNote(note: ResolvedNote, track: ResolvedTrack): number | undefined {
  const meta = note.source.metadata as { midi?: { note?: number } } | undefined;
  if (meta?.midi?.note !== undefined) return meta.midi.note;
  if (track.instrument.midi?.note !== undefined) return track.instrument.midi.note;
  return defaultMidiNote(note.pitch, note.modifiers);
}

function resolveVelocity(note: ResolvedNote): number {
  const meta = note.source.metadata as
    | { midi?: { velocity?: number }; vol?: Volume | { start?: Volume; end: Volume } }
    | undefined;
  if (typeof meta?.midi?.velocity === 'number') return clamp(Math.round(meta.midi.velocity));

  let baseline = DEFAULT_VELOCITY;
  const vol = meta?.vol;
  if (typeof vol === 'string') {
    baseline = VOLUME_TO_VELOCITY[vol] ?? baseline;
  } else if (vol && typeof vol === 'object') {
    // Match to_midi: use start (or end) of a transition; no interpolation.
    const v = vol.start ?? vol.end;
    if (v) baseline = VOLUME_TO_VELOCITY[v] ?? baseline;
  }

  if (note.modifiers.has('a')) baseline += ACCENT_BOOST;
  if (note.modifiers.has('g')) baseline -= GHOST_REDUCTION;

  return clamp(Math.round(baseline));
}

function clamp(v: number): number {
  if (!Number.isFinite(v)) return 1;
  if (v < 1) return 1;
  if (v > 127) return 127;
  return v;
}
