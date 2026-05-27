/**
 * Drumjot -> MIDI conversion.
 *
 * Design decisions / assumptions (each tagged [B#] inline below):
 *
 *  [B1] We always emit a single Format-1 track on the drum channel (default
 *       10). All voices in the source Jot are merged onto that track because
 *       `||` global simultaneity in the DSL is about timing, not about
 *       distinct MIDI tracks.
 *
 *  [B2] PPQN is fixed at 480 ticks per beat, a common high-resolution choice
 *       that avoids rounding noise for sixteenth-note and triplet grids.
 *
 *  [B3] Notes are emitted with a fixed 1-tick gate (noteOff immediately after
 *       noteOn). Drum samples are typically triggered by note-on only, so a
 *       longer gate carries no useful information and only confuses some
 *       hosts.
 *
 *  [B4] Per-note `metadata.midi.note` and `metadata.midi.velocity` win when
 *       present (this is the channel the reader uses to preserve fidelity).
 *       Otherwise we fall back to `instrumentMapping[pitch].midi` and, for
 *       unknown pitches, to a heuristic in `defaultMidiNote`.
 *
 *  [B5] When no velocity is supplied we map `vol` buckets to GM-ish values
 *       and apply `:a` / `:g` adjustments. Volume transitions
 *       (`VolTransition`) are NOT interpolated; we use only the bucket's
 *       baseline (`vol.start` if present, else `vol.end`).
 *
 *  [B6] Roll / buzz (`~`) is rendered as a single note-on. We do not yet
 *       emit a tremolo or expand it into multiple strikes; doing so well
 *       depends on tempo/genre and is left for a future pass.
 *
 *  [B7] Initial BPM (`globalMetadata.bpm`) and initial time signature meta
 *       events are emitted at tick 0. Subsequent tempo changes come from
 *       `jot.tempoEvents` and are emitted as `setTempo` at the precise
 *       tick (bar offset + beat * TICKS_PER_BEAT), so mid-bar tempo
 *       changes survive the export. Per-bar time-signature overrides are
 *       still honoured from `bar.metadata.time`.
 *
 *  [B8] Modifier-only effects that have no MIDI analogue (e.g. `:x`
 *       cross-stick on a snare) are realised by remapping the note number
 *       (e.g. cross-stick -> 37). The mapping is exactly the inverse of the
 *       read path so a round trip preserves the visible note.
 */
import { writeMidi, MidiEvent } from 'midi-file';
import { Jot, Volume } from 'src/dsl';
import { RenderedJot, ResolvedNote, ResolvedTrack } from 'src/jot';
import { resolveBpm } from 'src/tempo';
import { defaultMidiNote } from './gm';

export type ToMidiOptions = {
  drumChannel?: number;
  defaultVelocity?: number;
  accentBoost?: number;
  ghostReduction?: number;
};

const DEFAULTS: Required<ToMidiOptions> = {
  drumChannel: 10,
  defaultVelocity: 80,
  // `accentBoost` must put `mf + accent` at or above `from_midi`'s
  // accent threshold (100), otherwise a fresh-authored `:a` note
  // round-trips into a plain note. Was 24 (→ 88, below threshold);
  // 36 puts the default-volume accent at 100 exactly.
  accentBoost: 36,
  ghostReduction: 32,
};

/** [B2] */
export const TICKS_PER_BEAT = 480;

/** Convert a Drumjot `Jot` into a MIDI byte buffer (Standard MIDI File). */
export function toMidi(jot: Jot, options: ToMidiOptions = {}): Uint8Array {
  const opts = { ...DEFAULTS, ...options };
  const rendered = new RenderedJot(jot);
  const resolved = rendered.resolved;

  type AbsEvent = {
    tick: number;
    kind: 'noteOn' | 'noteOff';
    note: number;
    velocity: number;
  };
  const abs: AbsEvent[] = [];
  const channel = opts.drumChannel - 1;

  type TsChange = { tick: number; count: number; unit: number };
  const tsChanges: TsChange[] = [];

  type TempoChange = { tick: number; bpm: number };
  const tempoChanges: TempoChange[] = [];
  const globalBpm = resolveBpm(jot.globalMetadata.bpm, 120);

  // Compute each voice-0 bar's absolute tick offset; we'll resolve
  // `jot.tempoEvents` anchors against this. Bars are uniform in length
  // across voices (same time-signature sequence) so voice 0 is canonical.
  const voice0 = resolved.voices[0];
  const barTickStart: number[] = [];
  if (voice0) {
    let cursor = 0;
    for (const bar of voice0.bars) {
      barTickStart.push(cursor);
      cursor += Math.max(1, Math.round(bar.beats * TICKS_PER_BEAT));
    }
  }

  // [B7] Tempo events: walk `jot.tempoEvents` and emit setTempo at the
  // exact tick of each anchor. Dedup no-ops (parser already dedupes,
  // but defensive). Skipped events whose barIndex is out of range
  // collapse onto the last valid bar.
  {
    let currentBpm = globalBpm;
    for (const ev of jot.tempoEvents ?? []) {
      const bpm = resolveBpm(ev.bpm, currentBpm);
      if (bpm === currentBpm) continue;
      const idx = Math.min(Math.max(0, ev.barIndex), barTickStart.length - 1);
      const barStart = barTickStart[idx] ?? 0;
      const tick = barStart + Math.round(ev.beat * TICKS_PER_BEAT);
      tempoChanges.push({ tick, bpm });
      currentBpm = bpm;
    }
  }

  // [B1] Merge all voices onto one MIDI stream.
  // tsChanges are collected on the first voice only; bar tick offsets
  // are identical across voices (same time-signature sequence and bar
  // count), and emitting the same meta event from every voice would
  // duplicate it on the merged track.
  let firstVoice = true;
  for (const voice of resolved.voices) {
    let barOffset = 0;
    let prevTime = jot.globalMetadata.time ?? { count: 4, unit: 4 };
    for (let bi = 0; bi < voice.bars.length; bi++) {
      const bar = voice.bars[bi];
      const barTicks = Math.max(1, Math.round(bar.beats * TICKS_PER_BEAT));

      if (firstVoice && bi > 0 && (prevTime.count !== bar.time.count || prevTime.unit !== bar.time.unit)) {
        tsChanges.push({
          tick: barOffset,
          count: bar.time.count,
          unit: bar.time.unit,
        });
      }
      prevTime = bar.time;

      for (const pitch of voice.pitches) {
        const track = bar.tracks[pitch];
        if (!track) continue;
        for (const note of track.notes) {
          const midiNote = resolveMidiNote(note, track);
          if (midiNote === undefined) continue;
          const velocity = clampVelocity(resolveVelocity(note, opts));
          const tick = barOffset + Math.round((note.beat / bar.beats) * barTicks);
          abs.push({ tick, kind: 'noteOn', note: midiNote, velocity });
          // [B3] One-tick gate.
          abs.push({ tick: tick + 1, kind: 'noteOff', note: midiNote, velocity: 0 });
        }
      }
      barOffset += barTicks;
    }
    firstVoice = false;
  }

  // Sort: by tick, noteOff before noteOn when ticks collide so any same-tick
  // retrigger comes out cleanly.
  abs.sort((a, b) => {
    if (a.tick !== b.tick) return a.tick - b.tick;
    if (a.kind === b.kind) return a.note - b.note;
    return a.kind === 'noteOff' ? -1 : 1;
  });

  // Build the MIDI track event list with deltaTime values.
  const trackEvents: MidiEvent[] = [];

  // [B7] Initial tempo & time signature.
  trackEvents.push({
    deltaTime: 0,
    meta: true,
    type: 'setTempo',
    microsecondsPerBeat: Math.max(1, Math.round(60_000_000 / globalBpm)),
  });

  const initTs = jot.globalMetadata.time ?? { count: 4, unit: 4 };
  trackEvents.push({
    deltaTime: 0,
    meta: true,
    type: 'timeSignature',
    numerator: initTs.count,
    denominator: initTs.unit,
    metronome: 24,
    thirtyseconds: 8,
  });

  if (jot.title) {
    trackEvents.push({
      deltaTime: 0,
      meta: true,
      type: 'trackName',
      text: jot.title,
    });
  }

  // Merge time-sig + tempo change events into the timeline along with note events.
  type Pending =
    | { tick: number; kind: 'noteOn' | 'noteOff'; note: number; velocity: number }
    | { tick: number; kind: 'timeSig'; count: number; unit: number }
    | { tick: number; kind: 'tempo'; bpm: number };

  const merged: Pending[] = [
    ...abs.map<Pending>((e) => ({ ...e })),
    ...tsChanges.map<Pending>((t) => ({
      tick: t.tick,
      kind: 'timeSig',
      count: t.count,
      unit: t.unit,
    })),
    ...tempoChanges.map<Pending>((t) => ({
      tick: t.tick,
      kind: 'tempo',
      bpm: t.bpm,
    })),
  ];
  merged.sort((a, b) => {
    if (a.tick !== b.tick) return a.tick - b.tick;
    // Meta events first at a tick boundary; tempo before time-sig is
    // arbitrary (both are meta) but stable so output is deterministic.
    const metaRank = (k: Pending['kind']) =>
      k === 'tempo' ? 0 : k === 'timeSig' ? 1 : k === 'noteOff' ? 2 : 3;
    return metaRank(a.kind) - metaRank(b.kind);
  });

  let lastTick = 0;
  for (const ev of merged) {
    const dt = ev.tick - lastTick;
    if (ev.kind === 'tempo') {
      trackEvents.push({
        deltaTime: dt,
        meta: true,
        type: 'setTempo',
        microsecondsPerBeat: Math.max(1, Math.round(60_000_000 / ev.bpm)),
      });
    } else if (ev.kind === 'timeSig') {
      trackEvents.push({
        deltaTime: dt,
        meta: true,
        type: 'timeSignature',
        numerator: ev.count,
        denominator: ev.unit,
        metronome: 24,
        thirtyseconds: 8,
      });
    } else if (ev.kind === 'noteOn') {
      trackEvents.push({
        deltaTime: dt,
        type: 'noteOn',
        noteNumber: ev.note,
        velocity: ev.velocity,
        channel,
      });
    } else {
      trackEvents.push({
        deltaTime: dt,
        type: 'noteOff',
        noteNumber: ev.note,
        velocity: ev.velocity,
        channel,
      });
    }
    lastTick = ev.tick;
  }

  trackEvents.push({ deltaTime: 0, meta: true, type: 'endOfTrack' });

  const bytes = writeMidi({
    header: { format: 1, numTracks: 1, ticksPerBeat: TICKS_PER_BEAT },
    tracks: [trackEvents],
  });
  return new Uint8Array(bytes);
}

// ---------- Helpers ----------

function resolveMidiNote(note: ResolvedNote, track: ResolvedTrack): number | undefined {
  // [B4] Per-note override wins.
  const meta = note.source.metadata as { midi?: { note?: number } } | undefined;
  if (meta?.midi?.note !== undefined) return meta.midi.note;
  if (track.instrument.midi?.note !== undefined) return track.instrument.midi.note;
  return defaultMidiNote(note.pitch, note.modifiers);
}

const VOLUME_TO_VELOCITY: Record<Volume, number> = {
  pp: 16,
  p: 33,
  mp: 49,
  mf: 64,
  f: 80,
  ff: 96,
};

function resolveVelocity(note: ResolvedNote, opts: Required<ToMidiOptions>): number {
  const meta = note.source.metadata as
    | { midi?: { velocity?: number }; vol?: Volume | { start?: Volume; end: Volume } }
    | undefined;
  if (typeof meta?.midi?.velocity === 'number') return meta.midi.velocity;

  let baseline = opts.defaultVelocity;
  const vol = meta?.vol;
  if (typeof vol === 'string') {
    baseline = VOLUME_TO_VELOCITY[vol] ?? baseline;
  } else if (vol && typeof vol === 'object') {
    // [B5] Use start (or end) of a volume transition; no interpolation.
    const v = vol.start ?? vol.end;
    if (v) baseline = VOLUME_TO_VELOCITY[v] ?? baseline;
  }

  if (note.modifiers.has('a')) baseline += opts.accentBoost;
  if (note.modifiers.has('g')) baseline -= opts.ghostReduction;

  return baseline;
}

function clampVelocity(v: number): number {
  if (!Number.isFinite(v)) return 1;
  if (v < 1) return 1;
  if (v > 127) return 127;
  return Math.round(v);
}
