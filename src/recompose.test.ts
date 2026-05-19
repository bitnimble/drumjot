import { describe, expect, it } from 'bun:test';
import { RenderedJot } from 'src/jot';
import { parse } from 'src/parser';
import {
  RecomposeStructure,
  recompose,
} from 'src/recompose';

// ---------- helpers ----------

function structure(
  nBars: number,
  opts: {
    sig?: [number, number];
    tempo?: number;
    sigs?: [number, number][];
    tempos?: number[];
  } = {}
): RecomposeStructure {
  const sig = opts.sig ?? [4, 4];
  const tempo = opts.tempo ?? 120;
  const bars = Array.from({ length: nBars }, (_, i) => ({
    index: i,
    timeSig: opts.sigs ? opts.sigs[i] : sig,
    tempoBpm: opts.tempos ? opts.tempos[i] : tempo,
  }));
  return {
    initialTempo: tempo,
    initialTimeSig: sig,
    hasTempoChanges: opts.tempos !== undefined,
    hasTimeSigChanges: opts.sigs !== undefined,
    bars,
  };
}

const NAMES = {
  k: 'Kick',
  s: 'Snare',
  h: 'HiHat',
  d: 'Ride',
  c: 'Crash',
  t: 'Tom',
};

/** {pitch -> sorted list of absolute onset times} for a Jot DSL. */
function onsetSet(dsl: string): Set<string> {
  const resolved = new RenderedJot(parse(dsl)).resolved;
  const out = new Set<string>();
  for (const voice of resolved.voices) {
    let barOffset = 0;
    for (const bar of voice.bars) {
      const secsPerBeat = 60 / 120; // global default; structure-anchored
      for (const pitch of voice.pitches) {
        const track = bar.tracks[pitch];
        if (!track) continue;
        for (const note of track.notes) {
          const t = barOffset + note.beat * secsPerBeat;
          out.add(`${pitch}@${t.toFixed(4)}`);
        }
      }
      barOffset += bar.beats * secsPerBeat;
    }
  }
  return out;
}

function run(lines: Record<string, string>, st: RecomposeStructure) {
  return recompose({
    lines,
    structure: st,
    feetPitches: ['k'],
    instrumentNames: NAMES,
  });
}

// ---------- merging ----------

describe('recompose merging', () => {
  it('merges concurrent hands with + and stays one voice', () => {
    const lines = {
      h: '| h h h h h h h h |',
      s: '| . . . . s . . . |',
    };
    const { dsl } = run(lines, structure(1));
    expect(dsl).not.toContain('||');
    expect(dsl).toMatch(/h\+s|s\+h/);
    expect(onsetSet(dsl)).toEqual(
      new Set([...onsetSet(lines.h), ...onsetSet(lines.s)])
    );
  });

  it('routes kick to a second || voice', () => {
    const lines = {
      h: '| h h h h h h h h |',
      k: '| k . . . k . . . |',
    };
    const { dsl } = run(lines, structure(1));
    expect(dsl).toContain('||');
    const [hands, feet] = dsl.split('||');
    expect(hands).toContain('h');
    expect(feet).toContain('k');
    expect(onsetSet(dsl)).toEqual(
      new Set([...onsetSet(lines.h), ...onsetSet(lines.k)])
    );
  });

  it('feet-only stays a single voice (no ||)', () => {
    const lines = { k: '| k . . . k . . . |' };
    const { dsl } = run(lines, structure(1));
    expect(dsl).not.toContain('||');
    expect(onsetSet(dsl)).toEqual(onsetSet(lines.k));
  });

  it('emits genuine polyrhythm as +-joined groups', () => {
    const lines = {
      d: '| d d d d d d d d |',
      s: '| (s s s) (s s s) (s s s) (s s s) |',
    };
    const { dsl } = run(lines, structure(1));
    expect(dsl).toContain(' + ');
    expect(onsetSet(dsl)).toEqual(
      new Set([...onsetSet(lines.d), ...onsetSet(lines.s)])
    );
  });

  it('end-to-end: straight 8ths + triplets -> exact merged Jot', () => {
    const straight = '| h h h h h h h h |';
    const triplet = '| (s s s) (s s s) (s s s) (s s s) |';
    const { dsl } = run({ h: straight, s: triplet }, structure(1));
    expect(dsl).toBe(
      '{{ bpm: 120.00, time: "4/4", instrumentMapping: ' +
        '{ h: { name: "HiHat" }, s: { name: "Snare" } } }}\n' +
        '| (h h h h h h h h) + (s s s s s s s s s s s s) |'
    );
  });

  it('preserves modifiers', () => {
    const { dsl } = run({ s: '| . . . . s:a . . . |' }, structure(1));
    const resolved = new RenderedJot(parse(dsl)).resolved;
    const notes = resolved.voices[0].bars[0].tracks['s'].notes;
    expect(notes).toHaveLength(1);
    expect(notes[0].modifiers.has('a')).toBe(true);
  });
});

// ---------- robustness ----------

describe('recompose robustness', () => {
  it('drops extra bars without offsetting earlier ones', () => {
    const lines = { k: '| k . . . | . . k . | k k k k |' };
    const { dsl } = run(lines, structure(2));
    const resolved = new RenderedJot(parse(dsl)).resolved;
    expect(resolved.voices[0].bars).toHaveLength(2);
    const b0 = resolved.voices[0].bars[0].tracks['k'].notes.map(
      (n) => n.beat
    );
    const b1 = resolved.voices[0].bars[1].tracks['k'].notes.map(
      (n) => n.beat
    );
    expect(b0).toEqual([0]);
    expect(b1).toEqual([2]);
  });

  it('drops an unparseable fragment and keeps the rest', () => {
    const lines = {
      h: '| h h h h h h h h |',
      s: '| this is not (((valid |',
    };
    const { dsl, dropped } = run(lines, structure(1));
    expect(dropped).toEqual(['s']);
    expect(dsl).toContain('HiHat');
    expect(dsl).not.toContain('Snare');
    expect(onsetSet(dsl)).toEqual(onsetSet(lines.h));
  });

  it('maps exactly the present pitches', () => {
    const { dsl } = run(
      { h: '| h h h h h h h h |', k: '| k . . . k . . . |' },
      structure(1)
    );
    expect(dsl).toContain('HiHat');
    expect(dsl).toContain('Kick');
    for (const absent of ['Snare', 'Ride', 'Crash', 'Tom']) {
      expect(dsl).not.toContain(absent);
    }
  });

  it('keeps voice-local bar indices aligned with the structure', () => {
    const lines = {
      h: '| h h h h | h h h h |',
      k: '| k . . . | . . k . |',
    };
    const resolved = new RenderedJot(
      parse(run(lines, structure(2)).dsl)
    ).resolved;
    expect(resolved.voices[0].bars).toHaveLength(2);
    expect(resolved.voices[1].bars).toHaveLength(2);
  });

  it('renders an empty bar as a rest and preserves alignment', () => {
    const lines = { s: '| . . . . . . . . | . . . . s . . . |' };
    const resolved = new RenderedJot(
      parse(run(lines, structure(2)).dsl)
    ).resolved;
    const b0 = resolved.voices[0].bars[0].tracks['s'];
    const b1 = resolved.voices[0].bars[1].tracks['s'];
    expect(b0 ? b0.notes.length : 0).toBe(0);
    expect(b1.notes).toHaveLength(1);
  });

  it('emits inline {{ time }} on a time-signature change', () => {
    const lines = { s: '| s . s . | s . s |' };
    const st = structure(2, { sigs: [[4, 4], [3, 4]] });
    const { dsl } = run(lines, st);
    expect(dsl).toContain('{{ time: "3/4" }}');
  });

  it('returns metadata only when the structure has no bars', () => {
    const { dsl } = run({ k: '| k . . . |' }, structure(0));
    expect(dsl.trim().startsWith('{{')).toBe(true);
  });
});
