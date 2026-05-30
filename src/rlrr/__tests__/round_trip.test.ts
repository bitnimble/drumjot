import { describe, expect, it } from 'bun:test';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { MidiEvent, parseMidi, writeMidi } from 'midi-file';
import { Jot } from 'src/dsl';
import {
  DEFAULT_INSTRUMENTS,
  RlrrFile,
  allocateFallbackLetters,
  eventTimeSeconds,
  jotToRlrr,
  midiToRlrr,
  rlrrToJot,
  rlrrToMidi,
} from 'src/rlrr';

const FIXTURES_DIR = join(dirname(fileURLToPath(import.meta.url)), 'fixtures');
const DRUM_CHANNEL_IDX = 9;

// ---------- helpers ----------

type DrumHit = { tick: number; note: number; velocity: number };

function collectDrumNotes(bytes: Uint8Array): DrumHit[] {
  const midi = parseMidi(bytes);
  const out: DrumHit[] = [];
  for (const track of midi.tracks) {
    let t = 0;
    for (const ev of track) {
      t += ev.deltaTime;
      if (ev.type === 'noteOn' && ev.velocity > 0 && ev.channel === DRUM_CHANNEL_IDX) {
        out.push({ tick: t, note: ev.noteNumber, velocity: ev.velocity });
      }
    }
  }
  out.sort((a, b) => a.tick - b.tick || a.note - b.note);
  return out;
}

function buildMidi(opts: {
  ticksPerBeat?: number;
  bpm?: number;
  notes: Array<{ tick: number; note: number; velocity: number }>;
}): Uint8Array {
  const ticksPerBeat = opts.ticksPerBeat ?? 480;
  const microsPerBeat = Math.round(60_000_000 / (opts.bpm ?? 120));

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
        numerator: 4,
        denominator: 4,
        metronome: 24,
        thirtyseconds: 8,
      }),
    },
    {
      tick: 0,
      build: (dt) => ({ deltaTime: dt, meta: true, type: 'trackName', text: 'Drums' }),
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
  for (const e of events) {
    track.push(e.build(e.tick - last));
    last = e.tick;
  }
  track.push({ deltaTime: 0, meta: true, type: 'endOfTrack' });
  return new Uint8Array(
    writeMidi({ header: { format: 1, numTracks: 1, ticksPerBeat }, tracks: [track] })
  );
}

function makeRlrr(events: Array<{ name: string; vel: number; time: string }>, bpm = 120): RlrrFile {
  return {
    version: 0.7,
    authoringTool: 'test-harness',
    recordingMetadata: { title: 'Synthetic', complexity: 1 },
    audioFileData: { songTracks: [], drumTracks: [], songPreview: '', calibrationOffset: 0 },
    instruments: [...DEFAULT_INSTRUMENTS],
    events: events.map((e) => ({ ...e, loc: 0 })),
    bpmEvents: [{ bpm, time: 0 }],
  };
}

// ---------- MIDI <-> RLRR ----------

describe('midiToRlrr / rlrrToMidi', () => {
  it('converts a basic 4/4 backbeat to RLRR events', () => {
    const tpq = 480;
    const bytes = buildMidi({
      ticksPerBeat: tpq,
      bpm: 120,
      notes: [
        { tick: 0, note: 36, velocity: 100 },
        { tick: tpq, note: 38, velocity: 100 },
        { tick: tpq * 2, note: 36, velocity: 100 },
        { tick: tpq * 3, note: 38, velocity: 100 },
      ],
    });

    const rlrr = midiToRlrr(bytes);
    expect(rlrr.bpmEvents[0]).toEqual({ bpm: 120, time: 0 });
    expect(rlrr.events).toHaveLength(4);
    // Times at 120 bpm: 0, 0.5, 1.0, 1.5 seconds.
    expect(rlrr.events.map((e) => eventTimeSeconds(e))).toEqual([0, 0.5, 1.0, 1.5]);
    // Kick = BP_Kick_C_1, Snare = BP_Snare_C_1.
    expect(rlrr.events.map((e) => e.name)).toEqual([
      'BP_Kick_C_1',
      'BP_Snare_C_1',
      'BP_Kick_C_1',
      'BP_Snare_C_1',
    ]);
  });

  it('round-trips through MIDI -> RLRR -> MIDI preserving every note', () => {
    const tpq = 480;
    const hits = [
      { tick: 0, note: 36, velocity: 100 },
      { tick: tpq / 2, note: 42, velocity: 90 },
      { tick: tpq, note: 38, velocity: 110 },
      { tick: (tpq * 3) / 2, note: 42, velocity: 60 },
      { tick: tpq * 2, note: 36, velocity: 100 },
      { tick: (tpq * 5) / 2, note: 42, velocity: 90 },
      { tick: tpq * 3, note: 38, velocity: 110 },
      { tick: (tpq * 7) / 2, note: 46, velocity: 90 }, // open hi-hat
    ];
    const input = buildMidi({ ticksPerBeat: tpq, notes: hits });
    const rlrr = midiToRlrr(input);
    expect(rlrr.events).toHaveLength(hits.length);

    const back = rlrrToMidi(rlrr);
    const outHits = collectDrumNotes(back);
    expect(outHits).toHaveLength(hits.length);

    const inputNotes = hits.map((h) => h.note).sort();
    const outNotes = outHits.map((h) => h.note).sort();
    expect(outNotes).toEqual(inputNotes); // includes 46 vs 42 distinction via [M8]

    const inputVels = hits.map((h) => h.velocity).sort();
    const outVels = outHits.map((h) => h.velocity).sort();
    expect(outVels).toEqual(inputVels);
  });

  it('preserves tempo across a MIDI <-> RLRR round trip', () => {
    const bytes = buildMidi({
      bpm: 92,
      notes: [
        { tick: 0, note: 36, velocity: 100 },
        { tick: 240, note: 38, velocity: 100 },
      ],
    });
    const rlrr = midiToRlrr(bytes);
    expect(rlrr.bpmEvents[0].bpm).toBe(92);
    const back = rlrrToMidi(rlrr);
    const parsed = parseMidi(back);
    const tempo = parsed.tracks
      .flat()
      .find((ev): ev is Extract<MidiEvent, { type: 'setTempo' }> => ev.type === 'setTempo');
    expect(tempo).toBeDefined();
    expect(Math.round(60_000_000 / tempo!.microsecondsPerBeat)).toBe(92);
  });

  it('round-trips a mid-bar setTempo through MIDI -> RLRR -> MIDI', () => {
    // 4/4 bar at 120 bpm; tempo change to 60 bpm at beat 2 (= tick 960
    // = 1.0s in). Round-tripping through RLRR (which carries bpm events
    // as absolute seconds) must preserve both the initial and the
    // mid-bar value, with the second event landing back at exactly
    // tick 960 in the output MIDI.
    const tpq = 480;
    const base = buildMidi({
      ticksPerBeat: tpq,
      bpm: 120,
      notes: [
        { tick: 0, note: 36, velocity: 100 },
        { tick: tpq * 2, note: 36, velocity: 100 },
      ],
    });
    // Splice a 60 bpm setTempo (1_000_000 µs/qn) at tick tpq*2.
    const parsed = parseMidi(base);
    const track = parsed.tracks[0];
    let cursor = 0;
    for (let i = 0; i < track.length; i++) {
      cursor += track[i].deltaTime;
      if (cursor >= tpq * 2 && track[i].type === 'noteOn') {
        const dt = track[i].deltaTime;
        track.splice(i, 0, {
          deltaTime: dt,
          meta: true,
          type: 'setTempo',
          microsecondsPerBeat: 1_000_000,
        });
        track[i + 1].deltaTime = 0;
        break;
      }
    }
    const spliced = new Uint8Array(writeMidi(parsed));

    const rlrr = midiToRlrr(spliced);
    // Two bpm events: 120 at t=0, 60 at t=1.0s (= beat 2 of bar 0 at 120 bpm).
    expect(rlrr.bpmEvents.length).toBeGreaterThanOrEqual(2);
    expect(rlrr.bpmEvents[0]).toEqual({ bpm: 120, time: 0 });
    const second = rlrr.bpmEvents[1];
    expect(second.bpm).toBe(60);
    expect(second.time).toBeCloseTo(1.0, 5);

    const back = rlrrToMidi(rlrr);
    const reparsed = parseMidi(back);
    let t = 0;
    let mid: number | undefined;
    for (const ev of reparsed.tracks[0]) {
      t += ev.deltaTime;
      if (ev.type === 'setTempo' && ev.microsecondsPerBeat === 1_000_000) {
        mid = t;
        break;
      }
    }
    expect(mid).toBeDefined();
    // RLRR -> MIDI defaults to its own ticksPerBeat (see rlrr_to_midi.ts);
    // assert the mid-bar tempo lands at exactly 1.0s by walking the
    // tempo timeline rather than assuming the output's tpq.
    const outTpq = reparsed.header.ticksPerBeat ?? 480;
    expect(mid).toBe(2 * outTpq);
  });

  it('round-trips a mid-bar RLRR bpm event through Jot', () => {
    // Author an RLRR with a tempo change at t=1.0s (beat 2 of bar 0 at
    // 120 bpm). After rlrr -> jot the tempo change must surface as a
    // jot.tempoEvents entry at (barIndex: 0, beat: 2); jot -> rlrr must
    // re-emit it at the same absolute time, within sixteenth-note
    // quantization tolerance.
    const rlrr: RlrrFile = {
      version: 0.7,
      authoringTool: 'test-harness',
      recordingMetadata: { title: 'mid-bar', complexity: 1 },
      audioFileData: { songTracks: [], drumTracks: [], songPreview: '', calibrationOffset: 0 },
      instruments: [...DEFAULT_INSTRUMENTS],
      events: [
        { name: 'BP_Kick_C_1', vel: 100, time: 0, loc: 0 },
        // Notes spanning the bar at 120 bpm so the chart actually
        // contains the mid-bar tick the bpm event sits on.
        { name: 'BP_Snare_C_1', vel: 100, time: 0.5, loc: 0 },
        { name: 'BP_Kick_C_1', vel: 100, time: 1.0, loc: 0 },
        { name: 'BP_Snare_C_1', vel: 100, time: 1.5, loc: 0 },
      ],
      bpmEvents: [
        { bpm: 120, time: 0 },
        { bpm: 60, time: 1.0 },
      ],
    };
    const jot = rlrrToJot(rlrr);
    expect(jot.globalMetadata.bpm).toBe(120);
    expect(jot.tempoEvents).toBeDefined();
    expect(jot.tempoEvents!.length).toBeGreaterThanOrEqual(1);
    const ev = jot.tempoEvents![0];
    expect(ev.bpm).toBe(60);
    expect(ev.barIndex).toBe(0);
    expect(ev.beat).toBeCloseTo(2, 3);

    const back = jotToRlrr(jot);
    expect(back.bpmEvents[0]).toEqual({ bpm: 120, time: 0 });
    const second = back.bpmEvents.find((e) => Math.abs(e.bpm - 60) < 0.5);
    expect(second).toBeDefined();
    // Sixteenth at 120 bpm = 0.125s; the round-trip's quantization
    // budget is one sixteenth either side of the source time.
    expect(Math.abs(second!.time - 1.0)).toBeLessThanOrEqual(0.125);
  });
});

// ---------- RLRR <-> Jot ----------

describe('rlrrToJot / jotToRlrr', () => {
  it('quantizes RLRR events onto a 4/4 sixteenth-note grid', () => {
    const rlrr = makeRlrr(
      [
        { name: 'BP_Kick_C_1', vel: 100, time: '0.0000' },
        { name: 'BP_Snare_C_1', vel: 100, time: '0.5000' }, // beat 2 at 120 bpm
        { name: 'BP_Kick_C_1', vel: 100, time: '1.0000' },
        { name: 'BP_Snare_C_1', vel: 100, time: '1.5000' },
      ],
      120
    );
    const jot = rlrrToJot(rlrr);
    expect(jot.voices).toHaveLength(1);
    expect(jot.voices[0].bars).toHaveLength(1);
    const els = jot.voices[0].bars[0].elements;
    expect(els).toHaveLength(16);
    expect(els[0].kind).toBe('note');
    expect(els[4].kind).toBe('note');
    expect(els[8].kind).toBe('note');
    expect(els[12].kind).toBe('note');
  });

  it('collapses same-time hits into a simul', () => {
    const rlrr = makeRlrr([
      { name: 'BP_Kick_C_1', vel: 100, time: '0.0000' },
      { name: 'BP_HiHat_C_1', vel: 100, time: '0.0000' },
    ]);
    const jot = rlrrToJot(rlrr);
    expect(jot.voices[0].bars[0].elements[0].kind).toBe('simul');
  });

  it('round-trips RLRR events losslessly through Jot', () => {
    const events = [
      { name: 'BP_Kick_C_1', vel: 100, time: '0.0000' },
      { name: 'BP_HiHat_C_1', vel: 90, time: '0.2500' },
      { name: 'BP_Snare_C_1', vel: 110, time: '0.5000' },
      { name: 'BP_HiHat_C_1', vel: 60, time: '0.7500' },
      { name: 'BP_Kick_C_1', vel: 100, time: '1.0000' },
      { name: 'BP_HiHat_C_1', vel: 90, time: '1.2500' },
      { name: 'BP_Snare_C_1', vel: 110, time: '1.5000' },
      { name: 'BP_HiHat_C_1', vel: 60, time: '1.7500' },
    ];
    const rlrr = makeRlrr(events);
    const jot = rlrrToJot(rlrr);
    const back = jotToRlrr(jot);

    expect(back.events).toHaveLength(events.length);
    expect(back.events.map((e) => e.name).sort()).toEqual(events.map((e) => e.name).sort());
    // Velocities preserved exactly via metadata.rlrr.vel.
    expect(back.events.map((e) => e.vel).sort()).toEqual(events.map((e) => e.vel).sort());
    // Times preserved within sixteenth quantization (0.125s at 120 bpm).
    for (const e of back.events) {
      const t = eventTimeSeconds(e);
      const matching = events.find((src) => Math.abs(t - parseFloat(src.time)) < 0.13);
      expect(matching).toBeDefined();
    }
    // BPM survives.
    expect(back.bpmEvents[0].bpm).toBe(120);
    // Kit list is preserved.
    expect(back.instruments.length).toBe(DEFAULT_INSTRUMENTS.length);
  });

  it('assigns unique fallback letters to unknown instrument instances', () => {
    // Two synthetic instance names that share a hash hint must end up on
    // different letters, and neither may collide with canonical pitches
    // (kick = 'k') in use.
    const map = allocateFallbackLetters([
      'BP_Kick_C_1',
      'BP_MysteryDrumA_C_1',
      'BP_MysteryDrumB_C_1',
    ]);
    expect(map.get('BP_Kick_C_1')).toBe('k');
    const a = map.get('BP_MysteryDrumA_C_1');
    const b = map.get('BP_MysteryDrumB_C_1');
    expect(a).toBeDefined();
    expect(b).toBeDefined();
    expect(a).not.toBe(b);
    expect(a).not.toBe('k');
    expect(b).not.toBe('k');
  });

  it('preserves recordingMetadata and audioFileData across a Jot round trip', () => {
    const rlrr: RlrrFile = {
      ...makeRlrr([{ name: 'BP_Kick_C_1', vel: 100, time: '0.0000' }]),
      recordingMetadata: {
        title: 'My Song',
        artist: 'Anon',
        creator: 'Tester',
        description: 'desc',
        coverImagePath: 'cover.png',
        length: 1.0,
        complexity: 2,
      },
      audioFileData: {
        songTracks: ['song.ogg'],
        drumTracks: ['drums.ogg'],
        songPreview: 'preview.ogg',
        calibrationOffset: 0,
      },
    };
    const jot = rlrrToJot(rlrr);
    expect(jot.title).toBe('My Song');
    const back = jotToRlrr(jot);
    expect(back.recordingMetadata.title).toBe('My Song');
    expect(back.recordingMetadata.artist).toBe('Anon');
    expect(back.recordingMetadata.complexity).toBe(2);
    expect(back.audioFileData?.songTracks).toEqual(['song.ogg']);
    expect(back.audioFileData?.drumTracks).toEqual(['drums.ogg']);
  });

  it('integrates: MIDI -> RLRR -> Jot -> RLRR -> MIDI', () => {
    const tpq = 480;
    const hits = [
      { tick: 0, note: 36, velocity: 100 },
      { tick: tpq, note: 38, velocity: 100 },
      { tick: tpq * 2, note: 36, velocity: 100 },
      { tick: tpq * 3, note: 38, velocity: 100 },
    ];
    const input = buildMidi({ ticksPerBeat: tpq, notes: hits });
    const rlrr1 = midiToRlrr(input);
    const jot = rlrrToJot(rlrr1);
    const rlrr2 = jotToRlrr(jot);
    const output = rlrrToMidi(rlrr2);
    const outHits = collectDrumNotes(output);
    // Quantization keeps every kick/snare; counts must match.
    expect(outHits).toHaveLength(hits.length);
    expect(new Set(outHits.map((h) => h.note))).toEqual(new Set([36, 38]));
  });

  it('applies a note.offset to the exported RLRR event time', () => {
    // RLRR event times are real seconds, so a +30 ms sub-slot offset on a
    // beat-1 kick at 120 BPM charts at t = 0.030 (not snapped to 0).
    const jot: Jot = {
      title: '',
      globalMetadata: {
        bpm: 120,
        time: { count: 4, unit: 4 },
        instrumentMapping: { k: { kind: 'kick', name: 'Kick', midi: { note: 36 } } },
      },
      voices: [
        {
          bars: [
            {
              elements: [
                { kind: 'note', pitch: 'k', offset: 30 },
                { kind: 'rest' },
                { kind: 'rest' },
                { kind: 'rest' },
              ],
            },
          ],
        },
      ],
    };
    const rlrr = jotToRlrr(jot);
    expect(rlrr.events).toHaveLength(1);
    expect(parseFloat(rlrr.events[0].time as string)).toBeCloseTo(0.03, 4);
  });

  it('integrates: DSL parser -> Jot -> RLRR -> Jot', async () => {
    const { parse } = await import('src/parser');
    const src = `
      {{ bpm: 120, time: "4/4" }}
      | h:c h:c h:c h:c h:c h:c h:c h:c |
      | k . s . k . s . |
    `;
    const jot = parse(src);
    const rlrr = jotToRlrr(jot);
    // 8 hi-hats + 4 kick/snare hits per the snippet (single voice).
    expect(rlrr.events.length).toBeGreaterThanOrEqual(4);
    const classes = new Set(rlrr.events.map((e) => e.name.replace(/_\d+$/, '')));
    expect(classes.has('BP_Kick_C')).toBe(true);
    expect(classes.has('BP_Snare_C')).toBe(true);
    expect(classes.has('BP_HiHat_C')).toBe(true);
  });
});

// ---------- fixture-driven tests ----------

describe('RLRR fixture round trips', () => {
  const files = existsSync(FIXTURES_DIR)
    ? readdirSync(FIXTURES_DIR).filter((f) => f.endsWith('.rlrr'))
    : [];

  if (files.length === 0) {
    it.skip('no fixtures present (add .rlrr files to src/rlrr/__tests__/fixtures/)', () => {});
    return;
  }

  for (const file of files) {
    describe(file, () => {
      const raw = readFileSync(join(FIXTURES_DIR, file), 'utf-8');
      const rlrr = JSON.parse(raw) as RlrrFile;

      it('parses without throwing', () => {
        const jot = rlrrToJot(rlrr);
        expect(jot.voices.length).toBeGreaterThan(0);
      });

      it('Jot round trip preserves note count exactly', () => {
        const jot = rlrrToJot(rlrr);
        const back = jotToRlrr(jot);
        expect(back.events.length).toBe(rlrr.events.length);
      });

      it('MIDI round trip preserves event count', () => {
        const midi = rlrrToMidi(rlrr);
        const outHits = collectDrumNotes(midi);
        // RLRR -> MIDI is exact, modulo events whose drum class isn't in our
        // kit (which we drop). Tolerate up to one missing per 20 events.
        const tolerance = Math.max(1, Math.ceil(rlrr.events.length / 20));
        expect(Math.abs(outHits.length - rlrr.events.length)).toBeLessThanOrEqual(tolerance);
      });
    });
  }
});
