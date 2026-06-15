import { describe, expect, it } from 'bun:test';
import { Element, Group, Note, PatternRef, Simultaneity } from 'src/schema/dsl/dsl';
import { buildJotModel } from 'src/editing/jot_editor_store';
import { ParseError } from 'src/schema/dsl/parser/errors';
import { parse } from 'src/schema/dsl/parser/parser';

// ---------- helpers ----------

function singleLayer(src: string) {
  const jot = parse(src);
  expect(jot.layers).toHaveLength(1);
  return jot.layers[0];
}

function asNote(el: Element): Note {
  if (el.kind !== 'note') throw new Error(`Expected note, got ${el.kind}`);
  return el;
}
function asGroup(el: Element): Group {
  if (el.kind !== 'group') throw new Error(`Expected group, got ${el.kind}`);
  return el;
}
function asSimul(el: Element): Simultaneity {
  if (el.kind !== 'simul') throw new Error(`Expected simul, got ${el.kind}`);
  return el;
}
function asPatternRef(el: Element): PatternRef {
  if (el.kind !== 'patternRef') throw new Error(`Expected patternRef, got ${el.kind}`);
  return el;
}

// ---------- primitives ----------

describe('primitives', () => {
  it('parses a single note', () => {
    const v = singleLayer('k');
    expect(v.bars).toHaveLength(1);
    expect(v.bars[0].elements).toEqual([
      { kind: 'note', lane: 'k', range: { start: 0, end: 1 } },
    ]);
  });

  it('parses a rest', () => {
    const v = singleLayer('.');
    expect(v.bars[0].elements).toEqual([{ kind: 'rest' }]);
  });

  it('parses a sequence of notes and rests', () => {
    const v = singleLayer('k.s.');
    expect(v.bars[0].elements.map((e) => e.kind)).toEqual(['note', 'rest', 'note', 'rest']);
  });

  it('is tolerant of whitespace between elements', () => {
    const v = singleLayer('  k   .\n s  . ');
    expect(v.bars[0].elements).toHaveLength(4);
  });
});

// ---------- groups & weights & repeats ----------

describe('groups, weights, repeats', () => {
  it('parses a parenthesised group', () => {
    const v = singleLayer('(k.s.)');
    expect(v.bars[0].elements).toHaveLength(1);
    const g = asGroup(v.bars[0].elements[0]);
    expect(g.elements.map((e) => e.kind)).toEqual(['note', 'rest', 'note', 'rest']);
  });

  it('parses _N weight on notes, rests, and groups', () => {
    const v = singleLayer('k_2 ._3 (a b)_4');
    expect(asNote(v.bars[0].elements[0]).weight).toBe(2);
    expect((v.bars[0].elements[1] as { weight?: number }).weight).toBe(3);
    expect(asGroup(v.bars[0].elements[2]).weight).toBe(4);
  });

  it('parses *N repeat on notes and groups', () => {
    const v = singleLayer('k*4 (k.s.)*2');
    expect(asNote(v.bars[0].elements[0]).repeat).toBe(4);
    const g = asGroup(v.bars[0].elements[1]);
    expect(g.repeat).toBe(2);
  });

  it('supports nested groups', () => {
    const v = singleLayer('((k k) (s s))');
    const outer = asGroup(v.bars[0].elements[0]);
    expect(outer.elements).toHaveLength(2);
    asGroup(outer.elements[0]);
    asGroup(outer.elements[1]);
  });
});

// ---------- modifiers, sticking, rolls ----------

describe('modifiers, sticking, rolls', () => {
  it('parses a chained modifier', () => {
    const v = singleLayer('s:a:r');
    const n = asNote(v.bars[0].elements[0]);
    expect(n.modifiers).toEqual(['a', 'r']);
  });

  it('parses multi-char modifiers (fl, dr, rf)', () => {
    const v = singleLayer('s:fl k:dr c:rf');
    expect(asNote(v.bars[0].elements[0]).modifiers).toEqual(['fl']);
    expect(asNote(v.bars[0].elements[1]).modifiers).toEqual(['dr']);
    expect(asNote(v.bars[0].elements[2]).modifiers).toEqual(['rf']);
  });

  it('parses sticking and combines with modifiers', () => {
    const v = singleLayer('s:fl@l k@r s@r:a');
    const [a, b, c] = v.bars[0].elements.map(asNote);
    expect(a.modifiers).toEqual(['fl']);
    expect(a.sticking).toBe('l');
    expect(b.sticking).toBe('r');
    expect(c.modifiers).toEqual(['a']);
    expect(c.sticking).toBe('r');
  });

  it('parses rolls with optional weight', () => {
    const v = singleLayer('a~ a~_4 (k k)~');
    expect(asNote(v.bars[0].elements[0]).roll).toBe(true);
    expect(asNote(v.bars[0].elements[1]).roll).toBe(true);
    expect(asNote(v.bars[0].elements[1]).weight).toBe(4);
    expect(asGroup(v.bars[0].elements[2]).roll).toBe(true);
  });

  it('rejects sticking on non-notes', () => {
    expect(() => parse('(k k)@r')).toThrow(ParseError);
  });

  it('rejects rolls on rests', () => {
    expect(() => parse('.~')).toThrow(ParseError);
  });
});

// ---------- simultaneity ----------

describe('simultaneity (+)', () => {
  it('parses a + b as two-element simul', () => {
    const v = singleLayer('a+b');
    const s = asSimul(v.bars[0].elements[0]);
    expect(s.elements.map(asNote).map((n) => n.lane)).toEqual(['a', 'b']);
  });

  it('flattens chained simultaneities', () => {
    const v = singleLayer('a+b+c+d');
    const s = asSimul(v.bars[0].elements[0]);
    expect(s.elements.map(asNote).map((n) => n.lane)).toEqual(['a', 'b', 'c', 'd']);
  });

  it('binds suffixes tightly to each operand', () => {
    const v = singleLayer('a:a + b:g');
    const s = asSimul(v.bars[0].elements[0]);
    const [left, right] = s.elements.map(asNote);
    expect(left.modifiers).toEqual(['a']);
    expect(right.modifiers).toEqual(['g']);
  });

  it('supports polyrhythm: (a a a)_4 + (b b b b)_4', () => {
    const v = singleLayer('(a a a)_4 + (b b b b)_4');
    const s = asSimul(v.bars[0].elements[0]);
    const [left, right] = s.elements.map(asGroup);
    expect(left.elements).toHaveLength(3);
    expect(left.weight).toBe(4);
    expect(right.elements).toHaveLength(4);
    expect(right.weight).toBe(4);
  });
});

// ---------- bars & layers ----------

describe('bars and layers', () => {
  it('splits content between | into bars', () => {
    const v = singleLayer('| k . s . | k . k . s . |');
    expect(v.bars).toHaveLength(2);
    expect(v.bars[0].elements).toHaveLength(4);
    expect(v.bars[1].elements).toHaveLength(6);
    expect(v.anacrusis).toBeUndefined();
  });

  it('treats content before the first | as anacrusis', () => {
    const v = singleLayer('k k k | s . k . s . k . | s . k . s . k . |');
    expect(v.anacrusis).toHaveLength(3);
    expect(v.bars).toHaveLength(2);
  });

  it('splits layers on ||', () => {
    const jot = parse('| h h h h | || | k . s . |');
    expect(jot.layers).toHaveLength(2);
    expect(jot.layers[0].bars[0].elements).toHaveLength(4);
    expect(jot.layers[1].bars[0].elements).toHaveLength(4);
  });
});

// ---------- metadata ----------

describe('metadata', () => {
  it('parses global metadata with time signature and bpm', () => {
    const jot = parse('{{ bpm: 120, time: "4/4" }} | k . s . k . s . |');
    expect(jot.globalMetadata.bpm).toBe(120);
    expect(jot.globalMetadata.time).toEqual({ count: 4, unit: 4 });
  });

  it('parses nested instrumentMapping objects', () => {
    const jot = parse(
      '{{ instrumentMapping: { k:{name:"Kick"}, s:{name:"Snare", limb:"lh"} } }} | k . s . |'
    );
    const map = jot.globalMetadata.instrumentMapping as Record<
      string,
      { name: string; limb?: string }
    >;
    expect(map.k.name).toBe('Kick');
    expect(map.s.limb).toBe('lh');
  });

  it('parses drumsT0Sec from a separately-prepended global block', () => {
    // The transcriber stamps the audio lead-in as its own `{{ }}` block at
    // the top of the DSL so the LLM-emitted metadata block doesn't have to
    // be rewritten. Both blocks must merge into a single `globalMetadata`.
    const jot = parse(
      '{{ drumsT0Sec: 5.321 }}\n{{ bpm: 120, time: "4/4" }}\n| k . s . k . s . |'
    );
    expect(jot.globalMetadata.drumsT0Sec).toBe(5.321);
    expect(jot.globalMetadata.bpm).toBe(120);
    expect(jot.globalMetadata.time).toEqual({ count: 4, unit: 4 });
  });

  it('parses crescendo as a transition object', () => {
    const jot = parse(
      '{{ vol: { start: "mp", end: "ff", duration: 2 } }} | h h h h | h h h h |'
    );
    expect(jot.globalMetadata.vol).toEqual({ start: 'mp', end: 'ff', duration: 2 });
  });

  it('parses per-note metadata', () => {
    const v = singleLayer('| s{vol: f} . k . |');
    const n = asNote(v.bars[0].elements[0]);
    expect(n.metadata).toEqual({ vol: 'f' });
  });

  it('extracts title from global metadata', () => {
    const jot = parse('{{ title: "Hello world" }} | k . s . |');
    expect(jot.title).toBe('Hello world');
    expect(jot.globalMetadata.title).toBeUndefined();
  });
});

// ---------- patterns ----------

describe('patterns', () => {
  it('captures pattern definitions and references', () => {
    const jot = parse('[Groove=(k.s.kks.)] | [Groove]*3 |');
    expect(jot.patterns?.Groove).toBeDefined();
    expect(jot.patterns?.Groove.elements).toHaveLength(1);
    const ref = asPatternRef(jot.layers[0].bars[0].elements[0]);
    expect(ref.name).toBe('Groove');
    expect(ref.repeat).toBe(3);
  });

  it('does not play a pattern definition at its position', () => {
    // A bare `[Groove=(...)]` defines the pattern silently — it must not
    // contribute an in-line played reference. The layer only contains
    // the explicit `[Groove]` after the bar separator.
    const jot = parse('[Groove=(k.s.)]| [Groove] |');
    expect(jot.patterns?.Groove.elements).toHaveLength(1);
    expect(jot.layers[0].bars).toHaveLength(1);
    expect(jot.layers[0].bars[0].elements).toHaveLength(1);
    expect(asPatternRef(jot.layers[0].bars[0].elements[0]).name).toBe('Groove');
  });

  it('supports defining and using a pattern in the same expression', () => {
    // To play a pattern at the definition site, follow the definition
    // with an explicit reference: `[Name=(...)][Name]`.
    const jot = parse('| [Groove=(k.s.)][Groove] |');
    expect(jot.patterns?.Groove.elements).toHaveLength(1);
    expect(jot.layers[0].bars).toHaveLength(1);
    expect(jot.layers[0].bars[0].elements).toHaveLength(1);
    expect(asPatternRef(jot.layers[0].bars[0].elements[0]).name).toBe('Groove');
  });

  it('parses position substitutions [Name#N=...]', () => {
    const jot = parse('[Groove=(k.s.kks.)] [Groove#3=(k+s)]');
    const refs = jot.layers[0].bars[0].elements
      .filter((e): e is PatternRef => e.kind === 'patternRef');
    const subRef = refs.find((r) => r.substitutions && r.substitutions.length > 0);
    expect(subRef).toBeDefined();
    expect(subRef!.substitutions![0].path).toEqual([3]);
  });

  it('parses range substitutions [Name#N-M=...]', () => {
    const jot = parse('[Groove=(k.s.kks.)] [Groove#5-8=(k+s k+s k+s)_4]');
    const refs = jot.layers[0].bars[0].elements
      .filter((e): e is PatternRef => e.kind === 'patternRef');
    const subRef = refs.find((r) => r.substitutions && r.substitutions.length > 0)!;
    expect(subRef.substitutions![0].path).toEqual([[5, 8]]);
  });

  it('parses nested substitutions [Name#3#2=...]', () => {
    const jot = parse('[Groove=((k.s.))] [Groove#1#2=(x)]');
    const refs = jot.layers[0].bars[0].elements
      .filter((e): e is PatternRef => e.kind === 'patternRef');
    const subRef = refs.find((r) => r.substitutions && r.substitutions.length > 0)!;
    expect(subRef.substitutions![0].path).toEqual([1, 2]);
  });
});

// ---------- macros ----------

describe('macros', () => {
  it('expands macros before parsing', () => {
    const jot = parse('[$grv=k.s.kks.] | ([$grv])*4 |');
    const els = jot.layers[0].bars[0].elements;
    const grp = asGroup(els[0]);
    expect(grp.repeat).toBe(4);
    expect(grp.elements).toHaveLength(8);
  });
});

// ---------- SPEC examples end-to-end ----------

describe('SPEC.md examples', () => {
  it('example 1: basic groove with two layers and an instrument mapping', () => {
    const src = `
      {{ bpm: 120, time: "4/4",
         instrumentMapping: { k:{name:"Kick"}, s:{name:"Snare"}, h:{name:"HiHat"} } }}
      | h:c h:c h:c h:c h:c h:c h:c h:c |
      ||
      | k . s . k . s . |
    `;
    const jot = parse(src);
    expect(jot.layers).toHaveLength(2);
    expect(jot.layers[0].bars[0].elements).toHaveLength(8);
    expect(jot.layers[1].bars[0].elements).toHaveLength(8);
    expect(jot.globalMetadata.time).toEqual({ count: 4, unit: 4 });
  });

  it('example 2: half-bar triplet', () => {
    const src = '| k . s . (k+s k+s k+s)_4 |';
    const jot = parse(src);
    const els = jot.layers[0].bars[0].elements;
    expect(els).toHaveLength(5);
    const triplet = asGroup(els[4]);
    expect(triplet.weight).toBe(4);
    expect(triplet.elements).toHaveLength(3);
    asSimul(triplet.elements[0]);
  });

  it('example 4: 3:4 polyrhythm', () => {
    const jot = parse('| (a a a)_4 + (b b b b)_4 |');
    const simul = asSimul(jot.layers[0].bars[0].elements[0]);
    expect(simul.elements.map((e) => asGroup(e).elements.length)).toEqual([3, 4]);
  });

  it('example 5: flam, accent, sticking', () => {
    const jot = parse('| s:fl@l k@r s@r . k@r s@l k@r k@r |');
    const els = jot.layers[0].bars[0].elements;
    expect(els).toHaveLength(8);
    expect(asNote(els[0]).modifiers).toEqual(['fl']);
    expect(asNote(els[0]).sticking).toBe('l');
  });

  it('example 7: anacrusis preserved', () => {
    const jot = parse('{{ time: "4/4" }} k k k | s . k . s . k . | s . k . s . k . |');
    expect(jot.layers[0].anacrusis).toHaveLength(3);
    expect(jot.layers[0].bars).toHaveLength(2);
  });

  it('example 8: mixed meter sections', () => {
    const jot = parse(
      '{{ time: "7/8" }} | k . s . k k s | {{ time: "4/4" }} | k . s . k . s . |'
    );
    expect(jot.globalMetadata.time).toEqual({ count: 4, unit: 4 });
    expect(jot.layers[0].bars).toHaveLength(2);
    expect(jot.layers[0].bars[0].elements).toHaveLength(7);
    // Per-bar metadata snapshots: bar 0 retains its 7/8 effective time, bar 1
    // picks up the inline 4/4 change. This is what makes the AST round-trippable
    // for tempo/time-sig changes.
    expect(jot.layers[0].bars[0].metadata?.time).toEqual({ count: 7, unit: 8 });
    expect(jot.layers[0].bars[1].metadata?.time).toEqual({ count: 4, unit: 4 });
  });

  it('hoists inline bpm changes between bars into tempoEvents', () => {
    const jot = parse(
      '{{ bpm: 120, time: "4/4" }} | k . s . | {{ bpm: 140 }} | k . s . |'
    );
    // Initial tempo lives on globalMetadata.
    expect(jot.globalMetadata.bpm).toBe(120);
    // Mid-track change becomes a tempoEvent anchored at the bar 1 downbeat
    // (the {{bpm:140}} sits between bars, so it lands at bar 1 beat 0).
    expect(jot.tempoEvents).toEqual([{ barIndex: 1, beat: 0, bpm: 140 }]);
    // Bar metadata no longer carries bpm post-hoist.
    expect(jot.layers[0].bars[0].metadata?.bpm).toBeUndefined();
    expect(jot.layers[0].bars[1].metadata?.bpm).toBeUndefined();
    // Time signature carries forward unchanged.
    expect(jot.layers[0].bars[0].metadata?.time).toEqual({ count: 4, unit: 4 });
    expect(jot.layers[0].bars[1].metadata?.time).toEqual({ count: 4, unit: 4 });
  });

  it('hoists a mid-bar {{bpm}} marker into tempoEvents anchored at the next element', () => {
    // In an 8-slot 4/4 bar, each slot is 0.5 beats. `{{bpm:140}}` after
    // the snare (slot 2, beat 1.0) anchors at the next element, the
    // rest at slot 3, beat 1.5.
    const jot = parse(
      '{{ bpm: 120, time: "4/4" }} | k . s {{ bpm: 140 }} . k . s . |'
    );
    expect(jot.globalMetadata.bpm).toBe(120);
    expect(jot.tempoEvents).toEqual([{ barIndex: 0, beat: 1.5, bpm: 140 }]);
  });

  it('hoists {bpm} on a note into a tempoEvent at the note onset', () => {
    const jot = parse(
      '{{ bpm: 120, time: "4/4" }} | k . s{bpm: 140} . k . s . |'
    );
    expect(jot.globalMetadata.bpm).toBe(120);
    // The snare sits at slot 2 = beat 1.0 in an 8-slot 4/4 bar.
    expect(jot.tempoEvents).toEqual([{ barIndex: 0, beat: 1, bpm: 140 }]);
  });

  it('example 9: macro + pattern combination', () => {
    const src = `
      [$std=k.s.kks.]
      [Verse=([$std])*4]
      [Chorus=([$std])*2 (k+s k+s k+s k+s)_8]
      | [Verse] | [Chorus] | [Verse] |
    `;
    const jot = parse(src);
    expect(jot.patterns?.Verse).toBeDefined();
    expect(jot.patterns?.Chorus).toBeDefined();
    expect(jot.layers[0].bars).toHaveLength(3);
    expect(asPatternRef(jot.layers[0].bars[0].elements[0]).name).toBe('Verse');
  });

  it('example 10: roll then choke', () => {
    const jot = parse('| c~_8:o | c:k . . . k . s . |');
    const bars = jot.layers[0].bars;
    expect(bars).toHaveLength(2);
    const roll = asNote(bars[0].elements[0]);
    expect(roll.roll).toBe(true);
    expect(roll.weight).toBe(8);
    expect(roll.modifiers).toEqual(['o']);
    expect(asNote(bars[1].elements[0]).modifiers).toEqual(['k']);
  });
});

// ---------- end-to-end with the structural derivation ----------

describe('integration with the structural derivation', () => {
  it('produces a layout for a parsed jot', () => {
    const jot = parse(
      '{{ bpm: 120, time: "4/4" }} | h h h h h h h h | || | k . s . k . s . |'
    );
    const layers = buildJotModel(jot).structural.layers;
    expect(layers).toHaveLength(2);
    expect(layers[0].bars.length).toBeGreaterThan(0);
  });

  it('expands patterns at render time', () => {
    const jot = parse('[Groove=(k.s.kks.)] | [Groove] |');
    const layers = buildJotModel(jot).structural.layers;
    const bar = layers[0].bars[0];
    const kickNotes = bar.tracks.k?.notes ?? [];
    const snareNotes = bar.tracks.s?.notes ?? [];
    expect(kickNotes.length).toBe(3); // k . s . k k s . -> 3 kicks
    expect(snareNotes.length).toBe(2);
  });
});
