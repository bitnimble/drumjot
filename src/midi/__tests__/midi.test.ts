import { describe, expect, it } from 'bun:test';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { MidiData, MidiEvent, parseMidi, writeMidi } from 'midi-file';
import { fromMidi, toMidi } from 'src/midi';

const FIXTURES_DIR = join(dirname(fileURLToPath(import.meta.url)), 'fixtures');
const DRUM_CHANNEL_IDX = 9;

// ---------- helpers ----------

type DrumNote = {
  tick: number;
  note: number;
  velocity: number;
};

function collectDrumNotes(midi: MidiData): DrumNote[] {
  const out: DrumNote[] = [];
  for (const track of midi.tracks) {
    let t = 0;
    for (const ev of track) {
      t += ev.deltaTime;
      if (
        ev.type === 'noteOn' &&
        ev.channel === DRUM_CHANNEL_IDX &&
        ev.velocity > 0
      ) {
        out.push({ tick: t, note: ev.noteNumber, velocity: ev.velocity });
      }
    }
  }
  out.sort((a, b) => a.tick - b.tick || a.note - b.note);
  return out;
}

function firstTempoBpm(midi: MidiData): number | null {
  for (const track of midi.tracks) {
    for (const ev of track) {
      if (ev.type === 'setTempo') {
        return Math.round(60_000_000 / ev.microsecondsPerBeat);
      }
    }
  }
  return null;
}

function firstTimeSig(midi: MidiData): { count: number; unit: number } | null {
  for (const track of midi.tracks) {
    for (const ev of track) {
      if (ev.type === 'timeSignature') {
        return { count: ev.numerator, unit: ev.denominator };
      }
    }
  }
  return null;
}

/**
 * Build a minimal Format-1 MIDI buffer from a list of {tick, note, velocity}
 * drum hits at the supplied tempo and time signature.
 */
function buildMidi(opts: {
  ticksPerBeat?: number;
  bpm?: number;
  time?: { count: number; unit: number };
  notes: DrumNote[];
}): Uint8Array {
  const ticksPerBeat = opts.ticksPerBeat ?? 480;
  const microsPerBeat = Math.round(60_000_000 / (opts.bpm ?? 120));
  const time = opts.time ?? { count: 4, unit: 4 };

  type Pending = { tick: number; build: (dt: number) => MidiEvent };
  const events: Pending[] = [
    {
      tick: 0,
      build: (dt) => ({
        deltaTime: dt,
        meta: true,
        type: 'setTempo',
        microsecondsPerBeat: microsPerBeat,
      }),
    },
    {
      tick: 0,
      build: (dt) => ({
        deltaTime: dt,
        meta: true,
        type: 'timeSignature',
        numerator: time.count,
        denominator: time.unit,
        metronome: 24,
        thirtyseconds: 8,
      }),
    },
  ];
  for (const n of opts.notes) {
    events.push({
      tick: n.tick,
      build: (dt) => ({
        deltaTime: dt,
        type: 'noteOn',
        noteNumber: n.note,
        velocity: n.velocity,
        channel: DRUM_CHANNEL_IDX,
      }),
    });
    events.push({
      tick: n.tick + 1,
      build: (dt) => ({
        deltaTime: dt,
        type: 'noteOff',
        noteNumber: n.note,
        velocity: 0,
        channel: DRUM_CHANNEL_IDX,
      }),
    });
  }
  events.sort((a, b) => a.tick - b.tick);

  const track: MidiEvent[] = [];
  let last = 0;
  for (const { tick, build } of events) {
    track.push(build(tick - last));
    last = tick;
  }
  track.push({ deltaTime: 0, meta: true, type: 'endOfTrack' });

  return new Uint8Array(
    writeMidi({
      header: { format: 1, numTracks: 1, ticksPerBeat },
      tracks: [track],
    })
  );
}

// ---------- synthetic baseline tests ----------

describe('MIDI <-> Jot synthetic baseline', () => {
  it('reads a simple 4/4 backbeat into a single voice with one bar', () => {
    const tpq = 480;
    const bytes = buildMidi({
      ticksPerBeat: tpq,
      bpm: 120,
      notes: [
        { tick: 0, note: 36, velocity: 100 }, // k
        { tick: tpq, note: 38, velocity: 100 }, // s
        { tick: tpq * 2, note: 36, velocity: 100 }, // k
        { tick: tpq * 3, note: 38, velocity: 100 }, // s
      ],
    });

    const jot = fromMidi(bytes);
    expect(jot.globalMetadata.bpm).toBe(120);
    expect(jot.globalMetadata.time).toEqual({ count: 4, unit: 4 });
    expect(jot.voices).toHaveLength(1);
    expect(jot.voices[0].bars).toHaveLength(1);

    const els = jot.voices[0].bars[0].elements;
    // 16-slot grid: kick on 1,9; snare on 5,13.
    expect(els).toHaveLength(16);
    expect(els[0].kind).toBe('note');
    expect(els[4].kind).toBe('note');
    expect(els[8].kind).toBe('note');
    expect(els[12].kind).toBe('note');
    expect(els[1].kind).toBe('rest');
  });

  it('collapses simultaneous hits into a single simul element', () => {
    const bytes = buildMidi({
      notes: [
        { tick: 0, note: 36, velocity: 100 },
        { tick: 0, note: 42, velocity: 100 },
      ],
    });
    const jot = fromMidi(bytes);
    const first = jot.voices[0].bars[0].elements[0];
    expect(first.kind).toBe('simul');
  });

  it('round-trips note count and pitches losslessly', () => {
    const tpq = 480;
    const inputNotes: DrumNote[] = [
      { tick: 0, note: 36, velocity: 100 },
      { tick: tpq / 2, note: 42, velocity: 90 },
      { tick: tpq, note: 38, velocity: 110 },
      { tick: tpq * 3 / 2, note: 42, velocity: 50 },
      { tick: tpq * 2, note: 36, velocity: 100 },
      { tick: tpq * 5 / 2, note: 42, velocity: 90 },
      { tick: tpq * 3, note: 38, velocity: 110 },
      { tick: tpq * 7 / 2, note: 42, velocity: 50 },
    ];
    const bytes = buildMidi({ ticksPerBeat: tpq, notes: inputNotes });
    const jot = fromMidi(bytes);
    const out = parseMidi(toMidi(jot));

    const outNotes = collectDrumNotes(out);
    expect(outNotes).toHaveLength(inputNotes.length);

    const inSet = new Set(inputNotes.map((n) => n.note));
    const outSet = new Set(outNotes.map((n) => n.note));
    expect([...outSet].sort()).toEqual([...inSet].sort());

    // Velocity is preserved exactly via per-note metadata.
    const inVels = inputNotes.map((n) => n.velocity).sort();
    const outVels = outNotes.map((n) => n.velocity).sort();
    expect(outVels).toEqual(inVels);
  });

  it('preserves tempo and time signature across round trip', () => {
    const bytes = buildMidi({
      bpm: 92,
      time: { count: 7, unit: 8 },
      notes: [
        { tick: 0, note: 36, velocity: 100 },
        { tick: 240, note: 38, velocity: 100 },
        { tick: 480, note: 36, velocity: 100 },
      ],
    });
    const jot = fromMidi(bytes);
    expect(jot.globalMetadata.bpm).toBe(92);
    expect(jot.globalMetadata.time).toEqual({ count: 7, unit: 8 });

    const re = parseMidi(toMidi(jot));
    expect(firstTempoBpm(re)).toBe(92);
    expect(firstTimeSig(re)).toEqual({ count: 7, unit: 8 });
  });

  it('honours time-signature changes on subsequent bars', () => {
    const tpq = 480;
    const bytes = buildMidi({
      ticksPerBeat: tpq,
      time: { count: 4, unit: 4 },
      notes: [
        { tick: 0, note: 36, velocity: 100 },
        { tick: tpq * 4, note: 38, velocity: 100 }, // start of bar 2
      ],
    });
    // Inject a 3/4 change at the start of bar 2 by editing the events directly.
    // We do this by reparsing, splicing in a new timeSignature event, and rewriting.
    const parsed = parseMidi(bytes);
    const meta = parsed.tracks[0];
    // Find the noteOn at tick tpq*4 and inject a timeSignature before it.
    let cursor = 0;
    let injected = false;
    for (let i = 0; i < meta.length; i++) {
      cursor += meta[i].deltaTime;
      if (!injected && cursor >= tpq * 4 && meta[i].type === 'noteOn') {
        const dt = meta[i].deltaTime;
        const tsEvent: MidiEvent = {
          deltaTime: dt,
          meta: true,
          type: 'timeSignature',
          numerator: 3,
          denominator: 4,
          metronome: 24,
          thirtyseconds: 8,
        };
        meta.splice(i, 0, tsEvent);
        meta[i + 1].deltaTime = 0;
        injected = true;
        break;
      }
    }
    const reBytes = new Uint8Array(writeMidi(parsed));
    const jot = fromMidi(reBytes);
    // Bar 1 default 4/4 (16 slots), bar 2 should carry a time-sig override.
    expect(jot.voices[0].bars[0].elements).toHaveLength(16);
    expect(jot.voices[0].bars[1].metadata?.time).toEqual({ count: 3, unit: 4 });
    expect(jot.voices[0].bars[1].elements).toHaveLength(12);
  });

  it('round-trips a Jot built from the DSL parser', async () => {
    const { parse } = await import('src/parser');
    const src = `
      {{ bpm: 120, time: "4/4" }}
      | h:c h:c h:c h:c h:c h:c h:c h:c |
      | k . s . k . s . |
    `;
    const jot = parse(src);
    const bytes = toMidi(jot);
    const reparsed = parseMidi(bytes);
    const notes = collectDrumNotes(reparsed);
    // 8 hi-hat + 4 kick/snare = 12 hits.
    expect(notes.length).toBeGreaterThanOrEqual(12);
    expect(new Set(notes.map((n) => n.note))).toEqual(new Set([36, 38, 42]));
  });

  it('skips non-drum channels on read', () => {
    const tpq = 480;
    const track: MidiEvent[] = [
      { deltaTime: 0, meta: true, type: 'setTempo', microsecondsPerBeat: 500_000 },
      {
        deltaTime: 0,
        meta: true,
        type: 'timeSignature',
        numerator: 4,
        denominator: 4,
        metronome: 24,
        thirtyseconds: 8,
      },
      { deltaTime: 0, type: 'noteOn', noteNumber: 60, velocity: 100, channel: 0 },
      { deltaTime: 1, type: 'noteOff', noteNumber: 60, velocity: 0, channel: 0 },
      { deltaTime: tpq - 1, type: 'noteOn', noteNumber: 36, velocity: 100, channel: 9 },
      { deltaTime: 1, type: 'noteOff', noteNumber: 36, velocity: 0, channel: 9 },
      { deltaTime: 0, meta: true, type: 'endOfTrack' },
    ];
    const bytes = new Uint8Array(
      writeMidi({ header: { format: 1, numTracks: 1, ticksPerBeat: tpq }, tracks: [track] })
    );
    const jot = fromMidi(bytes);
    // The piano hit (channel 1) is dropped; only the kick survives.
    const all = jot.voices[0].bars.flatMap((b) => b.elements);
    const notesOnly = all.filter((e) => e.kind === 'note');
    expect(notesOnly).toHaveLength(1);
    expect((notesOnly[0] as { pitch: string }).pitch).toBe('k');
  });
});

// ---------- fixture-driven round trips ----------

describe('MIDI fixture round trips', () => {
  const files = existsSync(FIXTURES_DIR)
    ? readdirSync(FIXTURES_DIR).filter((f) => f.endsWith('.mid') || f.endsWith('.midi'))
    : [];

  if (files.length === 0) {
    it.skip('no fixtures present (add .mid files to src/midi/__tests__/fixtures/)', () => {});
    return;
  }

  for (const file of files) {
    describe(file, () => {
      const path = join(FIXTURES_DIR, file);
      const bytes = new Uint8Array(readFileSync(path));
      const inputMidi = parseMidi(bytes);
      const inputNotes = collectDrumNotes(inputMidi);

      it('parses without throwing', () => {
        const jot = fromMidi(bytes);
        expect(jot.voices.length).toBeGreaterThan(0);
      });

      it('preserves first tempo and time signature', () => {
        const jot = fromMidi(bytes);
        const inputBpm = firstTempoBpm(inputMidi) ?? 120;
        expect(jot.globalMetadata.bpm).toBe(inputBpm);

        const inputTime = firstTimeSig(inputMidi) ?? { count: 4, unit: 4 };
        expect(jot.globalMetadata.time).toEqual(inputTime);
      });

      it('preserves the set of MIDI note numbers used', () => {
        const jot = fromMidi(bytes);
        const out = parseMidi(toMidi(jot));
        const outNotes = collectDrumNotes(out);

        const inSet = new Set(inputNotes.map((n) => n.note));
        const outSet = new Set(outNotes.map((n) => n.note));
        expect([...outSet].sort()).toEqual([...inSet].sort());
      });

      it('preserves the drum hit count within quantization tolerance', () => {
        const jot = fromMidi(bytes);
        const out = parseMidi(toMidi(jot));
        const outNotes = collectDrumNotes(out);

        // We allow up to 10% drift to account for sixteenth-note quantization
        // collapsing hits that landed in the same slot. Tighten if your input
        // fixtures are pre-quantized.
        const tolerance = Math.max(2, Math.ceil(inputNotes.length * 0.1));
        expect(Math.abs(outNotes.length - inputNotes.length)).toBeLessThanOrEqual(tolerance);
      });
    });
  }
});
