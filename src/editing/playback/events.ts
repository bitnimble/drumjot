/**
 * StructuralPresenter -> flat playback event list, keyed by DSL lane.
 *
 * We walk the laid-out jot directly (rather than round-tripping through
 * `toMidi` + `parseMidi`) for two reasons:
 *
 *  1. The DSL lane letter ('k', 's', 'h', ...) survives end-to-end. The
 *     scheduler uses it to filter by mute/solo row; MIDI bytes don't
 *     carry that information.
 *
 *  2. The tempo timeline lives on `jot.tempoEvents`, which supports
 *     mid-bar tempo changes natively. The MIDI export carries the same
 *     information but the scheduler reads `tempoEvents` directly via the
 *     shared per-bar tempo helper so the playhead, audio waveform and
 *     scheduled drums all share one clock.
 */
import { Instrument, Modifier, Volume } from 'src/schema/dsl/dsl';
import type { StructuralPresenter } from 'src/editing/structure/structural_presenter';
import type { StructNote } from 'src/editing/structure/structure_store';
import {
  ACCENT_BOOST,
  DEFAULT_VELOCITY,
  GHOST_REDUCTION,
  VOLUME_TO_VELOCITY,
} from 'src/dynamics/dynamics';
import { defaultMidiNote } from 'src/midi/gm';
import { beatToSecWithinBar, buildBarTempos } from 'src/schema/dsl/tempo';

export type PlaybackEvent = {
  /** Absolute time from the start of the jot, in seconds. */
  time: number;
  /** GM percussion note number (post resolution of per-note overrides). */
  midiNote: number;
  /** MIDI velocity (1-127). */
  velocity: number;
  /** DSL lane letter the note was written under; used by mute/solo. */
  lane: string;
};

// Velocity defaults (DEFAULT_VELOCITY / ACCENT_BOOST / GHOST_REDUCTION /
// VOLUME_TO_VELOCITY) come from the shared `src/dynamics.ts` so playback
// loudness matches exactly what gets written to an exported `.mid` /
// `.rlrr` file.

// Flam = a grace stroke shortly before the main hit on the same drum.
// Acoustic flams sit ~25-35 ms apart, but two SF2 layers that close on the
// same key get swallowed by layer-stealing / overlapping attack envelopes
// and read as one strike. 50 ms is the standard "wide flam" — clearly two
// hits and survives smplr retriggering.
const FLAM_GRACE_OFFSET_SEC = 0.03;
// Grace stroke loudness, relative to the main hit. Flam grace notes are
// slightly softer than the primary, but not as quiet as a ghost — keep
// it well above the SF2's near-silent low-velocity layers. Scaling
// (rather than a fixed value) keeps an accented flam's grace proportional.
const FLAM_GRACE_VELOCITY_RATIO = 0.9;

export function jotToEvents(structural: StructuralPresenter): PlaybackEvent[] {
  // Musical structure only: the view-only virtual lead-in must never schedule
  // (or shift) drum events.
  const layers = structural.musicalLayers;
  const events: PlaybackEvent[] = [];
  const instrumentFor = (lane: string): Instrument =>
    structural.source.globalMetadata.instrumentMapping?.[lane] ?? { kind: 'custom' };

  // Bar 1 (= first non-lead-in bar) sits at jot time 0 by convention,
  // matching `buildTimeline`'s anchor. Pre-drum bars get a negative
  // bar-offset so an empty lead-in produces no events but a lead-in bar
  // that DOES contain notes (rare; today only relevant if the upstream
  // generator stamps drums into a pre-drum bar) fires its events at
  // negative jot time; at media `jot - songLeadIn >= 0` in audio time,
  // which is still valid. Lead-in count comes from the structure
  // (counting leading negative-indexed bars); `structureForLayer`
  // materialises both the explicit-leadBars and the chrome-only
  // (`songLeadIn` without `leadBars`) source shapes into the same
  // negative-indexed-bar form, so the scheduler reads only the
  // structure.
  for (const layer of layers) {
    let leadBars = 0;
    for (const b of layer.bars) {
      if (b.index >= 0) break;
      leadBars++;
    }
    // Per-bar tempo segments from `jot.tempoEvents` give the exact
    // intra-bar tempo curve. Each note's time is `barOffset +
    // beatToSecWithinBar(barTempos, note.beat)` so a note that sits
    // after a mid-bar tempo change picks up the post-change rate.
    const tempos = buildBarTempos(structural.source, layer.bars);
    let leadOffsetSec = 0;
    for (let i = 0; i < leadBars; i++) leadOffsetSec += tempos[i].durationSec;

    let barOffsetSec = -leadOffsetSec;
    for (let i = 0; i < layer.bars.length; i++) {
      const bar = layer.bars[i];
      const barTempos = tempos[i];
      for (const lane of layer.lanes) {
        const track = bar.tracks[lane];
        if (!track) continue;
        const instrument = instrumentFor(lane);
        for (const note of track.notes) {
          const midiNote = resolveMidiNote(note, instrument);
          if (midiNote === undefined) continue;
          const velocity = resolveVelocity(note);
          // Sub-slot timing offset (ms) nudges the scheduled time directly;
          // it's already in real seconds so no tempo conversion is needed.
          const offsetSec = (note.offsetMs ?? 0) / 1000;
          const time = barOffsetSec + beatToSecWithinBar(barTempos, note.beat) + offsetSec;
          events.push({ time, midiNote, velocity, lane });
          if (note.modifiers.includes('fl')) {
            // The grace stroke clamps to -leadOffsetSec so it can't run
            // past the start of the pre-drum window (a flam on bar 1
            // beat 1 with no lead-in clamps to 0 = bar 1 downbeat). Above
            // that floor it sounds the configured offset earlier than the
            // main hit; at the floor it collapses with the main hit
            // rather than disappearing entirely.
            const graceFloor = -leadOffsetSec;
            const graceTime = Math.max(graceFloor, time - FLAM_GRACE_OFFSET_SEC);
            const graceVel = Math.max(1, Math.round(velocity * FLAM_GRACE_VELOCITY_RATIO));
            events.push({ time: graceTime, midiNote, velocity: graceVel, lane });
          }
        }
      }
      barOffsetSec += barTempos.durationSec;
    }
  }

  events.sort((a, b) => a.time - b.time);
  return events;
}

function resolveMidiNote(note: StructNote, instrument: Instrument): number | undefined {
  if (note.midiNote !== undefined) return note.midiNote;
  if (instrument.midi?.note !== undefined) return instrument.midi.note;
  return defaultMidiNote(note.lane, new Set(note.modifiers as Modifier[]));
}

function resolveVelocity(note: StructNote): number {
  if (typeof note.velocity === 'number') return clamp(Math.round(note.velocity));

  let baseline = DEFAULT_VELOCITY;
  const vol = note.vol as Volume | undefined;
  if (typeof vol === 'string') {
    baseline = VOLUME_TO_VELOCITY[vol] ?? baseline;
  }

  if (note.modifiers.includes('a')) baseline += ACCENT_BOOST;
  if (note.modifiers.includes('g')) baseline -= GHOST_REDUCTION;

  return clamp(Math.round(baseline));
}

function clamp(v: number): number {
  if (!Number.isFinite(v)) return 1;
  if (v < 1) return 1;
  if (v > 127) return 127;
  return v;
}
