/**
 * RenderedJot -> flat playback event list, keyed by DSL pitch.
 *
 * We walk the laid-out jot directly (rather than round-tripping through
 * `toMidi` + `parseMidi` like an earlier revision did) for two reasons:
 *
 *  1. The DSL pitch letter ('k', 's', 'h', ...) survives end-to-end. The
 *     scheduler uses it to filter by mute/solo row — MIDI bytes don't
 *     carry that information, so the previous round-trip stripped it.
 *
 *  2. `toMidi` only emits a `setTempo` at tick 0, so the MIDI bytes
 *     can't carry per-bar `{{ bpm }}` overrides. Walking the rendered
 *     bars lets us accumulate the effective tempo bar by bar (same logic
 *     as `buildTimeline`), which is what keeps a variable-tempo chart —
 *     e.g. a ParaDB import whose `bpmEvents` became per-bar overrides —
 *     locked to its backing recording instead of slewing apart.
 *
 * The MIDI-byte path is still the source of truth for downstream MIDI
 * consumers — `toMidi` is unchanged.
 */
import { Volume } from 'src/dsl';
import { RenderedJot, ResolvedNote, ResolvedTrack } from 'src/jot';
import { defaultMidiNote } from 'src/midi/gm';
import { resolveBpm } from './timeline';

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

// Velocity defaults mirror `src/midi/to_midi.ts`'s `DEFAULTS` so playback
// loudness matches what gets written to an exported `.mid` file.
const DEFAULT_VELOCITY = 80;
const ACCENT_BOOST = 24;
const GHOST_REDUCTION = 32;
const VOLUME_TO_VELOCITY: Record<Volume, number> = {
  pp: 16,
  p: 33,
  mp: 49,
  mf: 64,
  f: 80,
  ff: 96,
};

export function jotToEvents(rendered: RenderedJot): PlaybackEvent[] {
  const resolved = rendered.resolved;
  const globalBpm = resolveBpm(resolved.globalMetadata.bpm, 120);
  const events: PlaybackEvent[] = [];

  for (const voice of resolved.voices) {
    let barOffsetSec = 0;
    // Per-bar `{{ bpm }}` overrides are sticky; carry the effective
    // tempo across bars exactly as `buildTimeline` does so the playhead
    // and the scheduled audio stay on the same clock.
    let currentBpm = globalBpm;
    for (const bar of voice.bars) {
      const override = bar.source.metadata?.bpm;
      if (override !== undefined) currentBpm = resolveBpm(override, currentBpm);
      const barDurationSec = bar.beats * (60 / currentBpm);
      for (const pitch of voice.pitches) {
        const track = bar.tracks[pitch];
        if (!track) continue;
        for (const note of track.notes) {
          const midiNote = resolveMidiNote(note, track);
          if (midiNote === undefined) continue;
          const velocity = resolveVelocity(note);
          const time = barOffsetSec + (note.beat / bar.beats) * barDurationSec;
          events.push({ time, midiNote, velocity, pitch });
        }
      }
      barOffsetSec += barDurationSec;
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
