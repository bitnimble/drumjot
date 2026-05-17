/**
 * MIDI -> Drumjot conversion.
 *
 * Design decisions / assumptions (each tagged with [A#] inline below):
 *
 *  [A1] Only the drum channel (default GM channel 10, i.e. internal index 9)
 *       is read. Notes on other channels are ignored because the DSL is
 *       drums-only.
 *
 *  [A2] Notes are quantized to a fixed grid (default sixteenth notes). Notes
 *       that fall off-grid snap to the nearest grid slot. Multiple notes
 *       landing on the same slot collapse into a `simul` element.
 *
 *  [A3] Note durations are discarded. Drums are modelled as one-shot strikes;
 *       the DSL has no concept of sustain (besides the `:l` modifier which we
 *       do not attempt to infer from MIDI).
 *
 *  [A4] Only the first `setTempo` is honoured; tempo changes mid-track are
 *       dropped. Reading tempo automations is out of scope for v1.
 *
 *  [A5] Time-signature changes ARE honoured. A new bar is started whenever a
 *       `timeSignature` meta event arrives, and the bar carries inline
 *       metadata to record the change.
 *
 *  [A6] Velocity is preserved exactly via a custom per-note metadata key
 *       `metadata.midi = { note, velocity }`. The writer prefers this over
 *       the volume bucket so that round-trips through DSL <-> MIDI are
 *       lossless for note pitch and dynamic.
 *
 *  [A7] Velocity is ALSO mapped to display modifiers: very loud notes
 *       (>= 100) gain `:a` and very soft notes (< 40) gain `:g`. This is
 *       purely cosmetic for the renderer and is not reversed on write
 *       (write uses the preserved raw velocity).
 *
 *  [A8] Note-off events (or noteOn with velocity 0, which MIDI treats as
 *       note-off) are ignored entirely.
 *
 *  [A9] The first complete bar starts at tick 0. There is no anacrusis
 *       inference; if you need a pickup, encode it in the DSL by hand.
 *
 *  [A10] Multiple MIDI tracks are merged. We sum delta times within each
 *        track to absolute ticks then concatenate. This is safe because the
 *        DSL only renders the drum channel anyway.
 */
import { parseMidi, MidiEvent } from 'midi-file';
import {
  Bar,
  Element,
  Instrument,
  Jot,
  Metadata,
  Modifier,
  Note,
  Simultaneity,
  TimeSignature,
} from 'src/dsl';
import { defaultKindForPitch } from 'src/instruments';
import { GM_PERCUSSION, allocatePitchesForMidi } from './gm';

export type FromMidiOptions = {
  /** 1-based MIDI channel for drums. Defaults to GM convention (10). */
  drumChannel?: number;
  /** Subdivisions per quarter note used for quantization (16 = sixteenth). */
  gridDivision?: number;
  /** Velocity at and above which the note also gains a `:a` accent modifier. */
  accentThreshold?: number;
  /** Velocity below which the note also gains a `:g` ghost modifier. */
  ghostThreshold?: number;
};

const DEFAULTS: Required<FromMidiOptions> = {
  drumChannel: 10,
  gridDivision: 16,
  accentThreshold: 100,
  ghostThreshold: 40,
};

type AbsEvent = { tick: number; ev: MidiEvent };

/** Convert a MIDI byte buffer into a Drumjot `Jot`. */
export function fromMidi(
  buffer: Uint8Array | ArrayBuffer | ArrayLike<number>,
  options: FromMidiOptions = {}
): Jot {
  const opts = { ...DEFAULTS, ...options };
  const bytes = toByteArray(buffer);
  const midi = parseMidi(bytes);

  const ticksPerBeat = midi.header.ticksPerBeat;
  if (!ticksPerBeat || ticksPerBeat <= 0) {
    // SMPTE-timed files use framesPerSecond/ticksPerFrame instead of ticksPerBeat.
    // We don't support that mode; throw rather than guess.
    throw new Error(
      'SMPTE-timed MIDI files are not supported (ticksPerBeat missing or zero)'
    );
  }

  // [A10] Merge tracks with absolute ticks.
  const events: AbsEvent[] = [];
  for (const track of midi.tracks) {
    let t = 0;
    for (const ev of track) {
      t += ev.deltaTime;
      events.push({ tick: t, ev });
    }
  }
  events.sort((a, b) => a.tick - b.tick);

  // [A1, A4, A5, A8] First pass: tempo, time-signature changes, drum noteOns.
  let bpm = 120;
  let foundTempo = false;
  const timeSigChanges: Array<{ tick: number; time: TimeSignature }> = [];
  const drumNotes: Array<{ tick: number; note: number; velocity: number }> = [];
  const drumChannelIdx = opts.drumChannel - 1;

  for (const { tick, ev } of events) {
    if ((ev as { meta?: true }).meta) {
      if (ev.type === 'setTempo' && !foundTempo) {
        bpm = Math.max(1, Math.round(60_000_000 / ev.microsecondsPerBeat));
        foundTempo = true;
      } else if (ev.type === 'timeSignature') {
        timeSigChanges.push({
          tick,
          time: { count: ev.numerator, unit: ev.denominator },
        });
      }
    } else if (
      ev.type === 'noteOn' &&
      ev.channel === drumChannelIdx &&
      ev.velocity > 0 // [A8]
    ) {
      drumNotes.push({ tick, note: ev.noteNumber, velocity: ev.velocity });
    }
  }

  // Ensure we have a time signature anchor at tick 0.
  if (timeSigChanges.length === 0 || timeSigChanges[0].tick > 0) {
    timeSigChanges.unshift({ tick: 0, time: { count: 4, unit: 4 } });
  }

  // [A2] Quantize tick positions to grid slots.
  // gridTicks = ticksPerBeat / (gridDivision / 4); for the default 16th-note
  // grid this is `ticksPerBeat / 4`.
  const gridTicks = (ticksPerBeat * 4) / opts.gridDivision;
  if (!Number.isFinite(gridTicks) || gridTicks <= 0) {
    throw new Error(`Invalid gridDivision ${opts.gridDivision}`);
  }

  // [A5] Compute bar boundaries from time-signature changes.
  const barSpans = computeBarSpans(timeSigChanges, drumNotes, ticksPerBeat);

  // Per-song allocation of `midi -> pitch` so unknown drums get unique
  // fallback letters that don't collide with GM_PERCUSSION or with each
  // other. See `allocatePitchesForMidi`.
  const pitchByMidi = allocatePitchesForMidi(drumNotes.map((d) => d.note));

  // Index drum notes into (bar, slot) buckets.
  type Slot = {
    notes: Array<{
      pitch: string;
      modifiers: Modifier[];
      midi: number;
      velocity: number;
    }>;
  };
  const slotsByBar: Map<number, Map<number, Slot>> = new Map();

  for (const dn of drumNotes) {
    const snapped = Math.round(dn.tick / gridTicks) * gridTicks;
    const barIdx = locateBar(barSpans, snapped);
    if (barIdx < 0) continue;
    const bar = barSpans[barIdx];
    const slotIdx = Math.round((snapped - bar.startTick) / gridTicks);

    const entry = GM_PERCUSSION[dn.note];
    const pitch = pitchByMidi.get(dn.note) ?? entry?.pitch ?? 'z';
    const modifiers = entry?.modifiers ? [...entry.modifiers] : [];

    let bucket = slotsByBar.get(barIdx);
    if (!bucket) {
      bucket = new Map();
      slotsByBar.set(barIdx, bucket);
    }
    let slot = bucket.get(slotIdx);
    if (!slot) {
      slot = { notes: [] };
      bucket.set(slotIdx, slot);
    }
    slot.notes.push({
      pitch,
      modifiers,
      midi: dn.note,
      velocity: dn.velocity,
    });
  }

  // Build bar elements: for each grid slot, either a Rest, a Note, or a Simul.
  const bars: Bar[] = [];
  let lastEmittedTime: TimeSignature | null = null;
  for (let bi = 0; bi < barSpans.length; bi++) {
    const span = barSpans[bi];
    const slotsInBar = Math.max(
      1,
      Math.round(((span.endTick - span.startTick) / gridTicks))
    );
    const bucket = slotsByBar.get(bi);
    const elements: Element[] = [];
    for (let si = 0; si < slotsInBar; si++) {
      const slot = bucket?.get(si);
      if (!slot || slot.notes.length === 0) {
        elements.push({ kind: 'rest' });
        continue;
      }
      const notes: Note[] = slot.notes.map((s) =>
        buildNote(s.pitch, s.modifiers, s.midi, s.velocity, opts)
      );
      if (notes.length === 1) {
        elements.push(notes[0]);
      } else {
        const simul: Simultaneity = { kind: 'simul', elements: notes };
        elements.push(simul);
      }
    }

    // [A5] Attach inline time-signature metadata on bars where the sig changes.
    const bar: Bar = { elements };
    if (
      !lastEmittedTime ||
      lastEmittedTime.count !== span.time.count ||
      lastEmittedTime.unit !== span.time.unit
    ) {
      // Only emit on bars after the first; the global time is already on the jot.
      if (bi > 0) bar.metadata = { time: span.time };
      lastEmittedTime = span.time;
    }
    bars.push(bar);
  }

  // Strip trailing all-rest bars - common in MIDI files that pad to a power of two.
  while (bars.length > 0 && bars[bars.length - 1].elements.every((e) => e.kind === 'rest')) {
    bars.pop();
  }

  // Build an instrument mapping from the MIDI notes we actually observed.
  // Reuse the per-song letter allocation so the mapping and the inline
  // pitches agree letter-for-letter even when fallback letters were used.
  const instrumentMapping = buildInstrumentMap(pitchByMidi);

  const globalMetadata: Metadata = {
    bpm,
    time: barSpans[0].time,
    instrumentMapping,
  };

  // [A9] No anacrusis is inferred; bars run from tick 0 onward.
  return {
    title: '',
    globalMetadata,
    voices: [{ bars }],
  };
}

// ---------- Helpers ----------

function toByteArray(buffer: Uint8Array | ArrayBuffer | ArrayLike<number>): Uint8Array {
  if (buffer instanceof Uint8Array) return buffer;
  if (buffer instanceof ArrayBuffer) return new Uint8Array(buffer);
  return new Uint8Array(Array.from(buffer as ArrayLike<number>));
}

type BarSpan = { startTick: number; endTick: number; time: TimeSignature };

function computeBarSpans(
  changes: Array<{ tick: number; time: TimeSignature }>,
  drumNotes: Array<{ tick: number }>,
  ticksPerBeat: number
): BarSpan[] {
  const lastNoteTick = drumNotes.length > 0 ? Math.max(...drumNotes.map((d) => d.tick)) : 0;
  // Always produce at least one bar so empty MIDIs are still well-formed jots.
  const spans: BarSpan[] = [];
  let cursor = 0;
  let changeIdx = 0;

  // Hard safety cap; a malformed MIDI shouldn't be able to spin us forever.
  const MAX_BARS = 16384;
  while (cursor <= lastNoteTick && spans.length < MAX_BARS) {
    while (
      changeIdx + 1 < changes.length &&
      changes[changeIdx + 1].tick <= cursor
    ) {
      changeIdx++;
    }
    const time = changes[changeIdx].time;
    const barBeats = (time.count * 4) / time.unit;
    const barTicks = Math.max(1, Math.round(ticksPerBeat * barBeats));
    spans.push({ startTick: cursor, endTick: cursor + barTicks, time });
    cursor += barTicks;
  }
  if (spans.length === 0) {
    const time = changes[0].time;
    const barBeats = (time.count * 4) / time.unit;
    spans.push({
      startTick: 0,
      endTick: Math.max(1, Math.round(ticksPerBeat * barBeats)),
      time,
    });
  }
  return spans;
}

function locateBar(spans: BarSpan[], tick: number): number {
  // Binary search keeps quantization cheap for long files.
  let lo = 0;
  let hi = spans.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const span = spans[mid];
    if (tick < span.startTick) hi = mid - 1;
    else if (tick >= span.endTick) lo = mid + 1;
    else return mid;
  }
  // Tick past the last bar; clamp to the final bar.
  return spans.length - 1;
}

function buildNote(
  pitch: string,
  baseModifiers: Modifier[],
  midi: number,
  velocity: number,
  opts: Required<FromMidiOptions>
): Note {
  const modifiers: Modifier[] = [...baseModifiers];
  // [A7] velocity -> accent/ghost decoration.
  if (velocity >= opts.accentThreshold && !modifiers.includes('a')) {
    modifiers.push('a');
  } else if (velocity < opts.ghostThreshold && !modifiers.includes('g')) {
    modifiers.push('g');
  }
  const note: Note = { kind: 'note', pitch };
  if (modifiers.length > 0) note.modifiers = modifiers;
  // [A6] Stash raw MIDI specifics for lossless round-trip.
  note.metadata = { midi: { note: midi, velocity } } as Metadata;
  return note;
}

function buildInstrumentMap(
  pitchByMidi: ReadonlyMap<number, string>
): Record<string, Instrument> {
  const out: Record<string, Instrument> = {};
  // Iterate in sorted MIDI order so the resulting mapping is stable.
  const midis = Array.from(pitchByMidi.keys()).sort((a, b) => a - b);
  for (const midi of midis) {
    const pitch = pitchByMidi.get(midi);
    if (!pitch || out[pitch]) continue;
    const entry = GM_PERCUSSION[midi];
    out[pitch] = {
      // GM entries carry an explicit kind; unknown MIDI notes get the
      // pitch-letter default (which falls back to `custom`).
      kind: entry?.kind ?? defaultKindForPitch(pitch),
      name: entry?.name ?? `MIDI ${midi}`,
      ...(entry?.limb ? { limb: entry.limb } : {}),
      midi: { note: midi },
    };
  }
  return out;
}
