/**
 * MIDI -> Drumjot conversion.
 *
 * Design decisions / assumptions (each tagged with [A#] inline below):
 *
 *  [A1] Only the drum channel (default GM channel 10, i.e. internal index 9)
 *       is read. Notes on other channels are ignored because the DSL is
 *       drums-only.
 *
 *  [A2] Notes are quantized to a fixed grid. The default `gridDivision`
 *       is 48 — i.e. 1/48-of-a-whole-note slots, twelve per quarter beat.
 *       12 = LCM(4, 3) is the coarsest grid that represents both straight
 *       16ths and triplet 8ths exactly, so a transcriber whose onset
 *       times have ~tens-of-ms jitter snaps to slots much closer to the
 *       true beat than a 16th grid (±20 ticks of tolerance at TPB=480 vs
 *       ±60). Multiple notes landing on the same slot collapse into a
 *       `simul` element. Slots at triplet positions are *representable*
 *       but currently render with the off-grid styling — wrapping them in
 *       tuplet groups is left to a downstream pass (deterministic
 *       clustering or an LLM refinement).
 *
 *  [A3] Note durations are discarded. Drums are modelled as one-shot strikes;
 *       the DSL has no concept of sustain (besides the `:l` modifier which we
 *       do not attempt to infer from MIDI).
 *
 *  [A4] Every `setTempo` is honoured at its precise (snapped) tick. The
 *       first tempo (latest event at or before tick 0) becomes
 *       `globalMetadata.bpm`. Subsequent setTempo events become
 *       `jot.tempoEvents` entries anchored at `(barIndex, beat-within-bar)`
 *       so mid-bar tempo changes survive round-trip with sub-bar precision.
 *       Ticks are quantized to the same 1/48 grid as note onsets, which
 *       gives ±~10 ms precision at common drum tempos.
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
  TempoEvent,
  TimeSignature,
} from 'src/schema/dsl/dsl';
import { ACCENT_THRESHOLD, GHOST_THRESHOLD } from 'src/dynamics/dynamics';
import { defaultKindForPitch } from 'src/instruments/instruments';
import {
  GENERIC_INSTRUMENT_NAME_BY_PITCH,
  GM_PERCUSSION,
  allocatePitchesForMidi,
} from './gm';

export type FromMidiOptions = {
  /** 1-based MIDI channel for drums. Defaults to GM convention (10). */
  drumChannel?: number;
  /**
   * Subdivisions per whole note used for quantization. The standard
   * musical naming applies: 16 = sixteenth note (4 slots per quarter
   * beat), 48 = 1/48 note (12 slots per quarter beat, supports both
   * 16ths and 8th triplets). See [A2] for why the default is 48.
   */
  gridDivision?: number;
  /** Velocity at and above which the note also gains a `:a` accent modifier. */
  accentThreshold?: number;
  /** Velocity below which the note also gains a `:g` ghost modifier. */
  ghostThreshold?: number;
  /**
   * Minimum |sub-slot residual| (in ms) for a note to keep a `note.offset`
   * rather than snapping silently to its grid slot. Below this, the
   * residual is treated as integer-tick round-trip noise and discarded so
   * clean hand-authored MIDI doesn't acquire spurious offsets. See
   * Pillar B of `docs/superpowers/specs/2026-05-29-geometric-quantise-design.md`.
   */
  offsetToleranceMs?: number;
};

const DEFAULTS: Required<FromMidiOptions> = {
  drumChannel: 10,
  gridDivision: 48,
  // Paired with ACCENT_BOOST / GHOST_REDUCTION in `src/dynamics.ts` so the
  // export boost and this import threshold can't drift apart.
  accentThreshold: ACCENT_THRESHOLD,
  ghostThreshold: GHOST_THRESHOLD,
  offsetToleranceMs: 5,
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
  const tempoChanges: Array<{ tick: number; bpm: number }> = [];
  const timeSigChanges: Array<{ tick: number; time: TimeSignature }> = [];
  const drumNotes: Array<{ tick: number; note: number; velocity: number }> = [];
  const drumChannelIdx = opts.drumChannel - 1;

  for (const { tick, ev } of events) {
    if ((ev as { meta?: true }).meta) {
      if (ev.type === 'setTempo') {
        // First setTempo wins for `globalMetadata.bpm` (preserves the
        // long-standing round-trip behaviour against `toMidi`, which writes
        // a single tick-0 tempo). The full list is kept for per-bar
        // attribution below; precision is preserved as a float because
        // micros-per-beat from the writer doesn't always land on an integer
        // BPM and rounding here would let multi-minute scores drift.
        const evBpm = Math.max(1, 60_000_000 / ev.microsecondsPerBeat);
        if (tempoChanges.length === 0) {
          bpm = Math.max(1, Math.round(evBpm));
        }
        tempoChanges.push({ tick, bpm: evBpm });
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
  // gridTicks = (ticksPerBeat * 4) / gridDivision; for the default 1/48
  // grid this is `ticksPerBeat / 12`.
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
      /**
       * Original absolute MIDI tick this note came from (pre-quantization).
       * Preserved on `note.metadata.midi.tick` so per-note debug provenance
       * sidecars (e.g. `note_provenance.json` from filter-mode transcribe)
       * can key by the unique `(tick, pitch)` identifier.
       */
      tick: number;
      /** Sub-slot timing residual in ms, or undefined when within tolerance. */
      offsetMs: number | undefined;
    }>;
  };
  const slotsByBar: Map<number, Map<number, Slot>> = new Map();

  // Tempo (float bpm) in force at an absolute tick, for the ms<->tick
  // conversion of sub-slot residuals. `tempoChanges` is tick-ascending
  // (built from the sorted event stream above).
  const bpmAtTick = (tick: number): number => {
    let result = tempoChanges.length > 0 ? tempoChanges[0].bpm : bpm;
    for (const tc of tempoChanges) {
      if (tc.tick <= tick) result = tc.bpm;
      else break;
    }
    return result > 0 ? result : 120;
  };

  for (const dn of drumNotes) {
    const snapped = Math.round(dn.tick / gridTicks) * gridTicks;
    const barIdx = locateBar(barSpans, snapped);
    if (barIdx < 0) continue;
    const bar = barSpans[barIdx];
    const slotIdx = Math.round((snapped - bar.startTick) / gridTicks);

    // Sub-slot residual: how far the raw onset sits from the slot it
    // snapped to, in ms at the local tempo. Kept as `note.offset` only
    // when it clears the tolerance (else discarded as round-trip noise).
    const residualTicks = dn.tick - snapped;
    const msPerTick = 60_000 / bpmAtTick(dn.tick) / ticksPerBeat;
    const residualMs = residualTicks * msPerTick;
    const offsetMs =
      Math.abs(residualMs) >= opts.offsetToleranceMs ? residualMs : undefined;

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
      tick: dn.tick,
      offsetMs,
    });
  }

  // [A4] Tempo events: snap each setTempo to the same 1/48 grid as note
  // onsets, anchor to (barIndex, beat-within-bar), dedup no-ops.
  // Initial tempo (latest event at or before tick 0) lives on
  // `globalMetadata.bpm`; later changes become `jot.tempoEvents` and are
  // honoured at sub-bar precision by the runtime tempo timeline.
  const initialBpm = tempoChanges.length > 0 ? tempoChanges[0].bpm : bpm;
  const tempoEvents: TempoEvent[] = [];
  {
    let currentBpm = initialBpm;
    for (let i = 0; i < tempoChanges.length; i++) {
      const tc = tempoChanges[i];
      if (i === 0) {
        // First event already accounted for in `initialBpm`.
        currentBpm = tc.bpm;
        continue;
      }
      if (tc.bpm === currentBpm) continue;
      const snapped = Math.round(tc.tick / gridTicks) * gridTicks;
      const barIndex = locateBar(barSpans, snapped);
      if (barIndex < 0) continue;
      const span = barSpans[barIndex];
      const beat = (snapped - span.startTick) / ticksPerBeat;
      tempoEvents.push({ barIndex, beat: Math.max(0, beat), bpm: tc.bpm });
      currentBpm = tc.bpm;
    }
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
        buildNote(s.pitch, s.modifiers, s.midi, s.velocity, s.tick, s.offsetMs, opts)
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
    const meta: Metadata = {};
    if (
      !lastEmittedTime ||
      lastEmittedTime.count !== span.time.count ||
      lastEmittedTime.unit !== span.time.unit
    ) {
      // Only emit on bars after the first; the global time is already on the jot.
      if (bi > 0) meta.time = span.time;
      lastEmittedTime = span.time;
    }
    if (meta.time !== undefined) {
      bar.metadata = meta;
    }
    bars.push(bar);
  }

  // Strip trailing all-rest bars - common in MIDI files that pad to a power of two.
  while (bars.length > 0 && bars[bars.length - 1].elements.every((e) => e.kind === 'rest')) {
    bars.pop();
  }

  // Count the leading run of all-rest bars: these are the pre-drum lead-in
  // the transcriber stamped to absorb the audio's drumless intro
  // (`transcriber/app/pipeline/onsets_midi.py:_inject_start_offset` aligned
  // bar 1 to a whole number of bar-0-length blocks past tick 0). They stay
  // in the bars list — the UI renders them with negative `bar.index` so
  // pre-drum audio still has a position on the timeline — but their count
  // is surfaced on `globalMetadata.leadBars` so consumers don't have to
  // recount them, and their cumulative audio duration becomes
  // `globalMetadata.drumsT0Sec` so playback / waveform alignment match the
  // transcriber's RLRR/DSL paths.
  let leadBars = 0;
  while (
    leadBars < bars.length &&
    bars[leadBars].elements.every((e) => e.kind === 'rest')
  ) {
    leadBars++;
  }
  // If the whole jot is rests (no drums in the file at all) we leave
  // leadBars = 0 / drumsT0Sec undefined — there's no "bar 1" to anchor
  // against and emitting a giant pre-drum offset would just hide the
  // entire content.
  if (leadBars >= bars.length) leadBars = 0;

  let drumsT0Sec = 0;
  if (leadBars > 0) {
    // Walk the pre-drum bars in lockstep with the tempo timeline (same
    // attribution rule as the main bar loop: "most recent tempo at or
    // before this bar's startTick"). Each bar contributes `barBeats * 60 /
    // bpm` seconds; bpm follows tempoChanges across the lead-in so a file
    // with a back-solved `lead_tempo` at tick 0 distinct from bar 1's
    // tempo still computes the correct audio duration.
    let leadTempoIdx = 0;
    for (let bi = 0; bi < leadBars; bi++) {
      const span = barSpans[bi];
      while (
        leadTempoIdx + 1 < tempoChanges.length &&
        tempoChanges[leadTempoIdx + 1].tick <= span.startTick
      ) {
        leadTempoIdx++;
      }
      const barBpm = tempoChanges[leadTempoIdx]?.bpm ?? initialBpm;
      const barBeats = (span.time.count * 4) / span.time.unit;
      if (barBpm > 0) {
        drumsT0Sec += (barBeats * 60) / barBpm;
      }
    }
  }

  // Build an instrument mapping from the MIDI notes we actually observed.
  // Reuse the per-song letter allocation so the mapping and the inline
  // pitches agree letter-for-letter even when fallback letters were used.
  const instrumentMapping = buildInstrumentMap(pitchByMidi);

  const globalMetadata: Metadata = {
    bpm,
    time: barSpans[0].time,
    instrumentMapping,
    // Record the grid density this load used so grid-aware consumers
    // (note-position readouts, the drum-offset slider, sub-slot offset
    // math) read the real resolution instead of assuming 48.
    gridDivision: opts.gridDivision,
    ...(leadBars > 0 ? { leadBars } : {}),
    ...(drumsT0Sec > 0 ? { drumsT0Sec } : {}),
  };

  // [A9] No anacrusis is inferred; bars run from tick 0 onward. Pre-drum
  // bars are preserved in `bars[0..leadBars-1]` and surfaced via
  // `globalMetadata.leadBars` / `globalMetadata.drumsT0Sec` so the
  // renderer can label them with negative indices.
  const jot: Jot = {
    title: '',
    globalMetadata,
    voices: [{ bars }],
  };
  if (tempoEvents.length > 0) jot.tempoEvents = tempoEvents;
  return jot;
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
  tick: number,
  offsetMs: number | undefined,
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
  if (offsetMs !== undefined) note.offset = offsetMs;
  // [A6] Stash raw MIDI specifics for lossless round-trip. `tick` is the
  // original absolute tick the noteOn arrived at (before grid snapping); it
  // is *not* written back by `to_midi.ts` (which recomputes tick from the
  // jot layout) and is purely diagnostic — it lets per-note debug
  // provenance sidecars key by the unique `(tick, pitch)` identifier.
  note.metadata = { midi: { note: midi, velocity, tick } } as Metadata;
  return note;
}

function buildInstrumentMap(
  pitchByMidi: ReadonlyMap<number, string>
): Record<string, Instrument> {
  const out: Record<string, Instrument> = {};
  // Count how many distinct MIDI notes mapped to each pitch; when a
  // pitch has more than one (e.g. 42 closed + 46 open hi-hat both →
  // pitch `h`, with the variant carried per-note as `:c` / `:o`), the
  // instrument-row label has to cover both, so we fall back to a
  // generic display name from `GENERIC_INSTRUMENT_NAME_BY_PITCH`
  // instead of whichever single GM entry happened to win the
  // first-iteration race.
  const midiCountByPitch = new Map<string, number>();
  for (const pitch of pitchByMidi.values()) {
    midiCountByPitch.set(pitch, (midiCountByPitch.get(pitch) ?? 0) + 1);
  }
  // Iterate in sorted MIDI order so the resulting mapping is stable.
  const midis = Array.from(pitchByMidi.keys()).sort((a, b) => a - b);
  for (const midi of midis) {
    const pitch = pitchByMidi.get(midi);
    if (!pitch || out[pitch]) continue;
    const entry = GM_PERCUSSION[midi];
    const hasMultipleVariants = (midiCountByPitch.get(pitch) ?? 0) > 1;
    const name = hasMultipleVariants
      ? GENERIC_INSTRUMENT_NAME_BY_PITCH[pitch] ?? entry?.name ?? `MIDI ${midi}`
      : entry?.name ?? `MIDI ${midi}`;
    out[pitch] = {
      // GM entries carry an explicit kind; unknown MIDI notes get the
      // pitch-letter default (which falls back to `custom`).
      kind: entry?.kind ?? defaultKindForPitch(pitch),
      name,
      ...(entry?.limb ? { limb: entry.limb } : {}),
      midi: { note: midi },
    };
  }
  return out;
}
