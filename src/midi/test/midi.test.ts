import { describe, expect, it } from 'bun:test';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { MidiData, MidiEvent, parseMidi, writeMidi } from 'midi-file';
import { Jot } from 'src/schema/dsl/dsl';
import { buildStructural } from 'src/editing/jot_editor_store';
import { fromMidi } from 'src/midi/from_midi';
import { allocateLanesForMidi } from 'src/midi/gm';
import { toMidi } from 'src/midi/to_midi';

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
  it('reads a simple 4/4 backbeat into a single layer with one bar', () => {
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
    expect(jot.layers).toHaveLength(1);
    expect(jot.layers[0].bars).toHaveLength(1);

    const els = jot.layers[0].bars[0].elements;
    // Default 1/48 grid (12 slots per quarter): kicks on slot 0/24,
    // snares on slot 12/36.
    expect(els).toHaveLength(48);
    expect(els[0].kind).toBe('note');
    expect(els[12].kind).toBe('note');
    expect(els[24].kind).toBe('note');
    expect(els[36].kind).toBe('note');
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
    const first = jot.layers[0].bars[0].elements[0];
    expect(first.kind).toBe('simul');
  });

  it('round-trips note count and lanes losslessly', () => {
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
    // Default 1/48 grid: 4/4 -> 48 slots, 3/4 -> 36 slots.
    expect(jot.layers[0].bars[0].elements).toHaveLength(48);
    expect(jot.layers[0].bars[1].metadata?.time).toEqual({ count: 3, unit: 4 });
    expect(jot.layers[0].bars[1].elements).toHaveLength(36);
  });

  it('lifts a mid-bar setTempo into jot.tempoEvents at the snapped (bar, beat)', () => {
    // Build a 4/4 bar of 8 eighth notes at 120 bpm, then splice a
    // setTempo event in at tick = ticksPerBeat * 2 (= beat 2.0, the 5th
    // slot's onset). After fromMidi: globalMetadata.bpm = 120; tempoEvents
    // carries one entry at (barIndex: 0, beat: 2.0, bpm: 60).
    const tpq = 480;
    const baseBytes = buildMidi({
      ticksPerBeat: tpq,
      bpm: 120,
      notes: [
        { tick: 0, note: 36, velocity: 100 },
        { tick: tpq / 2, note: 38, velocity: 100 },
        { tick: tpq, note: 36, velocity: 100 },
        { tick: (tpq * 3) / 2, note: 38, velocity: 100 },
        { tick: tpq * 2, note: 36, velocity: 100 },
        { tick: tpq * 2 + tpq / 2, note: 38, velocity: 100 },
        { tick: tpq * 3, note: 36, velocity: 100 },
        { tick: tpq * 3 + tpq / 2, note: 38, velocity: 100 },
      ],
    });
    // Splice a setTempo (60 bpm = 1_000_000 µs/qn) at tick tpq*2.
    const parsed = parseMidi(baseBytes);
    const track = parsed.tracks[0];
    let cursor = 0;
    for (let i = 0; i < track.length; i++) {
      cursor += track[i].deltaTime;
      if (cursor >= tpq * 2 && track[i].type === 'noteOn') {
        const dt = track[i].deltaTime;
        const tempoEvent: MidiEvent = {
          deltaTime: dt,
          meta: true,
          type: 'setTempo',
          microsecondsPerBeat: 1_000_000,
        };
        track.splice(i, 0, tempoEvent);
        track[i + 1].deltaTime = 0;
        break;
      }
    }
    const spliced = new Uint8Array(writeMidi(parsed));
    const jot = fromMidi(spliced);
    expect(jot.globalMetadata.bpm).toBe(120);
    expect(jot.tempoEvents).toEqual([
      { barIndex: 0, beat: 2, bpm: 60 },
    ]);
  });

  it('round-trips mid-bar tempoEvents as setTempo at the precise tick', () => {
    // 4/4 bar with a tempo change at beat 2.0. After toMidi the setTempo
    // sits at exactly that tick (= 2 * TICKS_PER_BEAT = 960).
    const tpq = 480;
    const baseBytes = buildMidi({
      ticksPerBeat: tpq,
      bpm: 120,
      notes: [
        { tick: 0, note: 36, velocity: 100 },
        { tick: tpq * 2, note: 36, velocity: 100 },
      ],
    });
    const parsed = parseMidi(baseBytes);
    const track = parsed.tracks[0];
    let cursor = 0;
    for (let i = 0; i < track.length; i++) {
      cursor += track[i].deltaTime;
      if (cursor >= tpq * 2 && track[i].type === 'noteOn') {
        const dt = track[i].deltaTime;
        const tempoEvent: MidiEvent = {
          deltaTime: dt,
          meta: true,
          type: 'setTempo',
          microsecondsPerBeat: 1_000_000,
        };
        track.splice(i, 0, tempoEvent);
        track[i + 1].deltaTime = 0;
        break;
      }
    }
    const spliced = new Uint8Array(writeMidi(parsed));
    const jot = fromMidi(spliced);
    const out = parseMidi(toMidi(jot));
    // Walk the output track, summing deltaTimes, and assert a setTempo
    // (microsecondsPerBeat=1_000_000) fires at tick 2 * TICKS_PER_BEAT = 960.
    let t = 0;
    let foundTick = -1;
    for (const ev of out.tracks[0]) {
      t += ev.deltaTime;
      if (
        ev.type === 'setTempo' &&
        ev.microsecondsPerBeat === 1_000_000
      ) {
        foundTick = t;
        break;
      }
    }
    expect(foundTick).toBe(2 * 480);
  });

  it('stamps the default grid division (48) onto globalMetadata', () => {
    const bytes = buildMidi({ notes: [{ tick: 0, note: 36, velocity: 100 }] });
    const jot = fromMidi(bytes);
    expect(jot.globalMetadata.gridDivision).toBe(48);
  });

  it('honours a denser grid division and stamps it onto globalMetadata', () => {
    const tpq = 480;
    const bytes = buildMidi({
      ticksPerBeat: tpq,
      bpm: 120,
      notes: [
        { tick: 0, note: 36, velocity: 100 }, // slot 0
        { tick: tpq, note: 38, velocity: 100 }, // beat 2 -> slot 24 of 96
      ],
    });
    const jot = fromMidi(bytes, { gridDivision: 96 });
    expect(jot.globalMetadata.gridDivision).toBe(96);
    // 4/4 at 1/96 -> 96 slots per bar; quarter beat = 24 slots.
    const els = jot.layers[0].bars[0].elements;
    expect(els).toHaveLength(96);
    expect(els[0].kind).toBe('note');
    expect(els[24].kind).toBe('note');
    expect(els[12].kind).toBe('rest');
  });

  it('round-trips note ticks at a denser (1/96) grid', () => {
    const tpq = 480;
    const inputNotes: DrumNote[] = [
      { tick: 0, note: 36, velocity: 100 },
      { tick: tpq / 4, note: 42, velocity: 90 }, // 1/16 -> slot 6 of 96
      { tick: tpq, note: 38, velocity: 100 },
      { tick: (tpq * 3) / 2, note: 36, velocity: 100 },
    ];
    const bytes = buildMidi({ ticksPerBeat: tpq, notes: inputNotes });
    const jot = fromMidi(bytes, { gridDivision: 96 });
    const out = collectDrumNotes(parseMidi(toMidi(jot)));
    expect(out.map((n) => n.tick)).toEqual(inputNotes.map((n) => n.tick));
  });

  it('reads a sub-slot onset as a note.offset in ms', () => {
    // 120 BPM, 1/48 grid: one slot = 480/12 = 40 ticks = 41.667 ms.
    // Put a kick 12 ticks (= 12.5 ms) past beat 1, well over the 5 ms
    // tolerance, but inside half a slot so it snaps to slot 0 and the
    // residual surfaces as the offset.
    const tpq = 480;
    const bytes = buildMidi({
      ticksPerBeat: tpq,
      bpm: 120,
      notes: [{ tick: 12, note: 36, velocity: 100 }],
    });
    const jot = fromMidi(bytes);
    const el = jot.layers[0].bars[0].elements[0];
    expect(el.kind).toBe('note');
    const offset = (el as { offset?: number }).offset;
    // 12 ticks at 120 BPM = 12 / 480 * 500 ms = 12.5 ms.
    expect(offset).toBeCloseTo(12.5, 1);
  });

  it('leaves notes within tolerance free of an offset', () => {
    // 2 ticks past the beat = ~2.08 ms, under the 5 ms tolerance.
    const bytes = buildMidi({
      ticksPerBeat: 480,
      bpm: 120,
      notes: [{ tick: 2, note: 36, velocity: 100 }],
    });
    const jot = fromMidi(bytes);
    const el = jot.layers[0].bars[0].elements[0];
    expect(el.kind).toBe('note');
    expect((el as { offset?: number }).offset).toBeUndefined();
  });

  it('round-trips a note.offset through toMidi -> fromMidi within tolerance', () => {
    // Hand-author a jot with a kick on beat 1 carrying +20 ms, emit to
    // MIDI, re-read: the offset should survive (the emitted tick is
    // beat-1 + 20 ms, which re-reads as the same residual).
    const jot: Jot = {
      title: '',
      globalMetadata: { bpm: 120, time: { count: 4, unit: 4 }, gridDivision: 48 },
      layers: [
        {
          bars: [
            {
              elements: [
                { kind: 'note', lane: 'k', offset: 20, metadata: { midi: { note: 36, velocity: 100 } } },
                ...Array.from({ length: 47 }, () => ({ kind: 'rest' as const })),
              ],
            },
          ],
        },
      ],
    };
    const reread = fromMidi(toMidi(jot));
    const el = reread.layers[0].bars[0].elements[0];
    expect(el.kind).toBe('note');
    expect((el as { offset?: number }).offset).toBeCloseTo(20, 0);
  });

  it('counts the leading rest run as leadBars and stamps songLeadIn', () => {
    // Two empty 4/4 bars at 120 bpm (1.0s/beat / 2.0s/bar) followed by a
    // kick on bar-3 downbeat: leadBars=2, songLeadIn=-4.0s exactly.
    const tpq = 480;
    const bytes = buildMidi({
      ticksPerBeat: tpq,
      bpm: 120,
      notes: [
        { tick: tpq * 8, note: 36, velocity: 100 }, // bar 3 downbeat
      ],
    });

    const jot = fromMidi(bytes);
    expect(jot.globalMetadata.leadBars).toBe(2);
    expect(jot.globalMetadata.songLeadIn).toBeCloseTo(-4.0, 6);
    // First non-rest bar is bars[2].
    expect(jot.layers[0].bars).toHaveLength(3);
    expect(jot.layers[0].bars[0].elements.every((e) => e.kind === 'rest')).toBe(true);
    expect(jot.layers[0].bars[1].elements.every((e) => e.kind === 'rest')).toBe(true);
    expect(jot.layers[0].bars[2].elements[0].kind).toBe('note');
  });

  it('omits leadBars / songLeadIn when drums start at tick 0', () => {
    const bytes = buildMidi({
      notes: [{ tick: 0, note: 36, velocity: 100 }],
    });
    const jot = fromMidi(bytes);
    expect(jot.globalMetadata.leadBars).toBeUndefined();
    expect(jot.globalMetadata.songLeadIn).toBeUndefined();
  });

  it('numbers pre-drum bars negatively and bar 1 starts the drums', () => {
    // Same shape as the leadBars test above (2 empty bars then a kick at
    // bar 3 downbeat); verify the rendered jot exposes the new index
    // convention: bars[0]=-2, bars[1]=-1, bars[2]=1 (skip 0; no anacrusis).
    const tpq = 480;
    const bytes = buildMidi({
      ticksPerBeat: tpq,
      bpm: 120,
      notes: [{ tick: tpq * 8, note: 36, velocity: 100 }],
    });
    const bars = buildStructural(fromMidi(bytes)).layers[0].bars;
    expect(bars.map((b) => b.index)).toEqual([-2, -1, 1]);
  });

  it('round-trips a Jot built from the DSL parser', async () => {
    const { parse } = await import('src/schema/dsl/parser/parser');
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

  it('assigns unique fallback letters for unknown MIDI notes', () => {
    // 60 (% 26 = 8, hint = 'r') and 86 (% 26 = 8, hint = 'r') share a hint.
    // Both are outside GM_PERCUSSION, so the allocator must give them
    // different letters.
    const map = allocateLanesForMidi([60, 86, 36]);
    const a = map.get(60);
    const b = map.get(86);
    expect(a).toBeDefined();
    expect(b).toBeDefined();
    expect(a).not.toBe(b);
    // The GM-mapped note (36 -> kick -> 'k') wins for that entry.
    expect(map.get(36)).toBe('k');
    // Neither fallback may collide with a canonical lane from
    // GM_PERCUSSION (e.g. 'k' for kick) when that lane is in use.
    expect(a).not.toBe('k');
    expect(b).not.toBe('k');
  });

  it('preserves all unknown MIDI notes in the instrument mapping', () => {
    const tpq = 480;
    const bytes = buildMidi({
      ticksPerBeat: tpq,
      bpm: 120,
      notes: [
        { tick: 0, note: 60, velocity: 100 }, // unknown
        { tick: tpq, note: 86, velocity: 100 }, // unknown, shares hint with 60
        { tick: tpq * 2, note: 36, velocity: 100 }, // kick
      ],
    });
    const jot = fromMidi(bytes);
    const mapping = jot.globalMetadata.instrumentMapping ?? {};
    // Three distinct letters: kick + two unique unknowns.
    expect(Object.keys(mapping).sort().length).toBe(3);
    // Each unknown carries its source MIDI number so a round-trip survives.
    const midiNotes = Object.values(mapping).map((i) => i.midi?.note).sort();
    expect(midiNotes).toEqual([36, 60, 86]);
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
    const all = jot.layers[0].bars.flatMap((b) => b.elements);
    const notesOnly = all.filter((e) => e.kind === 'note');
    expect(notesOnly).toHaveLength(1);
    expect((notesOnly[0] as { lane: string }).lane).toBe('k');
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
        expect(jot.layers.length).toBeGreaterThan(0);
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
