import { describe, expect, it } from 'bun:test';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Jot } from 'src/schema/dsl/dsl';
import { allocateFallbackLetters } from 'src/schema/rlrr/fallback';
import { writeRlrr } from 'src/schema/rlrr/writer';
import { parseRlrr } from 'src/schema/rlrr/parser';
import { DEFAULT_INSTRUMENTS, RlrrFile, eventTimeSeconds } from 'src/schema/rlrr/schema';
import { parse } from 'src/schema/dsl/parser/parser';

const FIXTURES_DIR = join(dirname(fileURLToPath(import.meta.url)), 'fixtures');

// ---------- helpers ----------

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

// ---------- mid-bar tempo round trip (RLRR <-> Jot) ----------

describe('mid-bar tempo round trip through Jot', () => {
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
    const jot = parseRlrr(rlrr);
    expect(jot.globalMetadata.bpm).toBe(120);
    expect(jot.tempoEvents).toBeDefined();
    expect(jot.tempoEvents!.length).toBeGreaterThanOrEqual(1);
    const ev = jot.tempoEvents![0];
    expect(ev.bpm).toBe(60);
    expect(ev.barIndex).toBe(0);
    expect(ev.beat).toBeCloseTo(2, 3);

    const back = writeRlrr(jot);
    expect(back.bpmEvents[0]).toEqual({ bpm: 120, time: 0 });
    const second = back.bpmEvents.find((e) => Math.abs(e.bpm - 60) < 0.5);
    expect(second).toBeDefined();
    // Sixteenth at 120 bpm = 0.125s; the round-trip's quantization
    // budget is one sixteenth either side of the source time.
    expect(Math.abs(second!.time - 1.0)).toBeLessThanOrEqual(0.125);
  });
});

describe('writeRlrr gradual tempo ramp', () => {
  it('subdivides a BpmTransition into stepwise bpmEvents along the curve', () => {
    const jot = parse(
      '{{ bpm: 60, time: "4/4", instrumentMapping: { k:{name:"K"} } }}\n' +
        '{{ bpm: { start: 60, end: 120, duration: 4 } }}\n' +
        '| k k k k |\n| k k k k |'
    );
    const { bpmEvents } = writeRlrr(jot);
    const bpms = bpmEvents.map((e) => e.bpm);
    // Initial 60 at t=0, then many steps (1/32-beat over 4 beats ≈ 128), not
    // one flat step at `start`.
    expect(bpmEvents[0]).toEqual({ bpm: 60, time: 0 });
    expect(bpmEvents.length).toBeGreaterThan(40);
    expect(Math.max(...bpms)).toBeCloseTo(120, 0);
    // Monotonically non-decreasing (an accelerando).
    for (let i = 1; i < bpms.length; i++) expect(bpms[i]).toBeGreaterThanOrEqual(bpms[i - 1] - 1e-9);
    // The final step lands on `end` at the ramp's end time (4 beats at the
    // average 90 BPM = 2.6667s).
    const last = bpmEvents[bpmEvents.length - 1];
    expect(last.bpm).toBeCloseTo(120, 3);
    expect(last.time).toBeCloseTo((120 * 4) / (60 + 120), 2);
  });
});

// ---------- RLRR <-> Jot ----------

describe('parseRlrr / writeRlrr', () => {
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
    const jot = parseRlrr(rlrr);
    expect(jot.layers).toHaveLength(1);
    expect(jot.layers[0].bars).toHaveLength(1);
    const els = jot.layers[0].bars[0].elements;
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
    const jot = parseRlrr(rlrr);
    expect(jot.layers[0].bars[0].elements[0].kind).toBe('simul');
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
    const jot = parseRlrr(rlrr);
    const back = writeRlrr(jot);

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
    // different letters, and neither may collide with canonical lanes
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
    const jot = parseRlrr(rlrr);
    expect(jot.title).toBe('My Song');
    const back = writeRlrr(jot);
    expect(back.recordingMetadata.title).toBe('My Song');
    expect(back.recordingMetadata.artist).toBe('Anon');
    expect(back.recordingMetadata.complexity).toBe(2);
    expect(back.audioFileData?.songTracks).toEqual(['song.ogg']);
    expect(back.audioFileData?.drumTracks).toEqual(['drums.ogg']);
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
      layers: [
        {
          bars: [
            {
              elements: [
                { kind: 'note', lane: 'k', offset: 30 },
                { kind: 'rest' },
                { kind: 'rest' },
                { kind: 'rest' },
              ],
            },
          ],
        },
      ],
    };
    const rlrr = writeRlrr(jot);
    expect(rlrr.events).toHaveLength(1);
    expect(parseFloat(rlrr.events[0].time as string)).toBeCloseTo(0.03, 4);
  });

  it('integrates: DSL parser -> Jot -> RLRR -> Jot', async () => {
    const { parse } = await import('src/schema/dsl/parser/parser');
    const src = `
      {{ bpm: 120, time: "4/4" }}
      | h:c h:c h:c h:c h:c h:c h:c h:c |
      | k . s . k . s . |
    `;
    const jot = parse(src);
    const rlrr = writeRlrr(jot);
    // 8 hi-hats + 4 kick/snare hits per the snippet (single layer).
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
        const jot = parseRlrr(rlrr);
        expect(jot.layers.length).toBeGreaterThan(0);
      });

      it('Jot round trip preserves note count exactly', () => {
        const jot = parseRlrr(rlrr);
        const back = writeRlrr(jot);
        expect(back.events.length).toBe(rlrr.events.length);
      });
    });
  }
});
