/**
 * End-to-end linter tests. Each `it` parses a DSL string, runs the linter,
 * and asserts which rule(s) fired (or didn't). Positive examples confirm a
 * rule triggers; negative examples confirm the rule stays quiet for
 * legitimate inputs.
 */
import { describe, expect, it } from 'bun:test';
import { parse } from 'src/parser';
import { lint, LintDiagnostic, LintSeverity } from 'src/linter';

function lintSource(src: string): LintDiagnostic[] {
  const jot = parse(src);
  return lint(jot, src).diagnostics;
}

function diagsByRule(diags: LintDiagnostic[], ruleId: string): LintDiagnostic[] {
  return diags.filter((d) => d.ruleId === ruleId);
}

function countBy(
  diags: LintDiagnostic[],
  severity: LintSeverity
): number {
  return diags.filter((d) => d.severity === severity).length;
}

// ---------- instrument/invalid-modifier ----------

describe('instrument/invalid-modifier', () => {
  it('flags :o (open) on a kick', () => {
    const diags = lintSource(
      '{{ instrumentMapping: { k:{name:"Kick"} } }} | k:o k:o k:o k:o |'
    );
    const hits = diagsByRule(diags, 'instrument/invalid-modifier');
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0].severity).toBe('error');
    expect(hits[0].kind).toBe('instrument');
  });

  it('flags :r (rim shot) on a ride', () => {
    const diags = lintSource(
      '{{ instrumentMapping: { d:{name:"Ride"} } }} | d:r d:r |'
    );
    const hits = diagsByRule(diags, 'instrument/invalid-modifier');
    expect(hits.length).toBeGreaterThan(0);
  });

  it('does not flag :a (accent) on a snare', () => {
    const diags = lintSource(
      '{{ instrumentMapping: { s:{name:"Snare"} } }} | s:a s:a s:a s:a |'
    );
    expect(diagsByRule(diags, 'instrument/invalid-modifier')).toHaveLength(0);
  });

  it('skips checks for custom instruments', () => {
    const diags = lintSource(
      '{{ instrumentMapping: { x:{kind:"custom", name:"Anvil"} } }} | x:o x:r x:c |'
    );
    expect(diagsByRule(diags, 'instrument/invalid-modifier')).toHaveLength(0);
  });
});

// ---------- instrument/discouraged-modifier ----------

describe('instrument/discouraged-modifier', () => {
  it('warns on :o (open) on a crash', () => {
    const diags = lintSource(
      '{{ instrumentMapping: { c:{name:"Crash"} } }} | c:o c:o |'
    );
    const hits = diagsByRule(diags, 'instrument/discouraged-modifier');
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0].severity).toBe('warning');
  });

  it('does not warn on a vanilla crash hit', () => {
    const diags = lintSource(
      '{{ instrumentMapping: { c:{name:"Crash"} } }} | c c c c |'
    );
    expect(diagsByRule(diags, 'instrument/discouraged-modifier')).toHaveLength(0);
  });
});

// ---------- performance/roll-on-kick ----------

describe('performance/roll-on-kick', () => {
  it('warns on a kick roll', () => {
    const diags = lintSource(
      '{{ instrumentMapping: { k:{name:"Kick"} } }} | k~_4 . . . |'
    );
    const hits = diagsByRule(diags, 'performance/roll-on-kick');
    expect(hits.length).toBe(1);
    expect(hits[0].severity).toBe('warning');
  });

  it('does not warn on a snare roll', () => {
    const diags = lintSource(
      '{{ instrumentMapping: { s:{name:"Snare"} } }} | s~_4 . . . |'
    );
    expect(diagsByRule(diags, 'performance/roll-on-kick')).toHaveLength(0);
  });
});

// ---------- performance/roll-on-multi-instrument ----------

describe('performance/roll-on-multi-instrument', () => {
  it('errors when a group roll spans snare and ride', () => {
    const diags = lintSource(
      '{{ instrumentMapping: { s:{name:"Snare"}, d:{name:"Ride"} } }} ' +
        '| (s d s d)~ . . . |'
    );
    const hits = diagsByRule(diags, 'performance/roll-on-multi-instrument');
    expect(hits.length).toBe(1);
    expect(hits[0].severity).toBe('error');
  });

  it('does not fire when a group roll is single-instrument', () => {
    const diags = lintSource(
      '{{ instrumentMapping: { s:{name:"Snare"} } }} | (s s s s)~ . . . |'
    );
    expect(diagsByRule(diags, 'performance/roll-on-multi-instrument')).toHaveLength(0);
  });
});

// ---------- performance/too-many-hands ----------

describe('performance/too-many-hands', () => {
  it('errors on a 3-hand simultaneity', () => {
    const diags = lintSource(
      '{{ instrumentMapping: { s:{name:"Snare"}, d:{name:"Ride"}, c:{name:"Crash"} } }} ' +
        '| s+d+c . . . . . . . |'
    );
    const hits = diagsByRule(diags, 'performance/too-many-hands');
    expect(hits.length).toBe(1);
    expect(hits[0].severity).toBe('error');
  });

  it('does NOT fire when one of the three is a foot (kick)', () => {
    const diags = lintSource(
      '{{ instrumentMapping: { s:{name:"Snare"}, d:{name:"Ride"}, k:{name:"Kick"} } }} ' +
        '| s+d+k . . . . . . . |'
    );
    expect(diagsByRule(diags, 'performance/too-many-hands')).toHaveLength(0);
  });

  it('does NOT fire on the standard hi-hat + snare + kick triple', () => {
    const diags = lintSource(
      '{{ instrumentMapping: { h:{name:"HiHat"}, s:{name:"Snare"}, k:{name:"Kick"} } }} ' +
        '| h+s+k . . . . . . . |'
    );
    expect(diagsByRule(diags, 'performance/too-many-hands')).toHaveLength(0);
  });
});

// ---------- performance/same-hand-conflict ----------

describe('performance/same-hand-conflict', () => {
  it('errors when two notes share an explicit left-hand sticking at one onset', () => {
    const diags = lintSource(
      '{{ instrumentMapping: { s:{name:"Snare"}, h:{name:"HiHat"} } }} ' +
        '| s@l+h@l . . . . . . . |'
    );
    const hits = diagsByRule(diags, 'performance/same-hand-conflict');
    expect(hits.length).toBeGreaterThanOrEqual(1);
    expect(hits[0].severity).toBe('error');
  });

  it('does not fire when hi-hat + snare are at the same onset without conflicts', () => {
    const diags = lintSource(
      '{{ instrumentMapping: { h:{name:"HiHat"}, s:{name:"Snare"} } }} ' +
        '| h+s . . . . . . . |'
    );
    expect(diagsByRule(diags, 'performance/same-hand-conflict')).toHaveLength(0);
  });
});

// ---------- clean Jot ----------

describe('clean jots emit nothing', () => {
  it('rock loop has no diagnostics', () => {
    const diags = lintSource(
      '{{ bpm: 120, time: "4/4", instrumentMapping: { ' +
        'h:{name:"HiHat"}, s:{name:"Snare"}, k:{name:"Kick"} } }} ' +
        '| h:c h:c h:c h:c h:c h:c h:c h:c | ' +
        '|| ' +
        '| k . s . k . s . |'
    );
    expect(countBy(diags, 'error')).toBe(0);
  });
});

// ---------- position info ----------

describe('positions', () => {
  it('attaches a range to each note diagnostic', () => {
    const diags = lintSource(
      '{{ instrumentMapping: { k:{name:"Kick"} } }} | k:o . . . . . . . |'
    );
    const hit = diagsByRule(diags, 'instrument/invalid-modifier')[0];
    expect(hit).toBeDefined();
    expect(hit.range).toBeDefined();
    expect(hit.range!.start).toBeGreaterThan(0);
    expect(hit.range!.end).toBeGreaterThan(hit.range!.start);
  });

  it('attaches a range to the leftmost operand of a + simultaneity', () => {
    // Regression: rules that anchor on `hands[0]` (e.g. too-many-hands)
    // previously reported "(no position)" because the parser only
    // attached `range` to the rightmost operand of a `+` chain — left
    // operands were swept into the Simultaneity wrapper before they
    // could be tagged.
    const src =
      '{{ instrumentMapping: { s:{name:"Snare"}, d:{name:"Ride"}, c:{name:"Crash"} } }} ' +
      '| s+d+c . . . . . . . |';
    const diags = lintSource(src);
    const hit = diagsByRule(diags, 'performance/too-many-hands')[0];
    expect(hit).toBeDefined();
    expect(hit.range).toBeDefined();
    // Anchor is `hands[0]`, the leftmost in source order (`s`).
    expect(src.slice(hit.range!.start, hit.range!.end)).toBe('s');
  });
});
