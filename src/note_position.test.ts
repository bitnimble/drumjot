import { describe, expect, it } from 'bun:test';
import { NotePosition } from 'src/note_position';

describe('NotePosition.slotsPerBar', () => {
  it('returns 48 for a 4/4 bar at the default 1/48 grid (12 per quarter)', () => {
    const p = new NotePosition({
      barIndex: 1,
      beatInBar: 1,
      timeSig: { count: 4, unit: 4 },
      slotsPerQuarter: 12,
    });
    expect(p.slotsPerBar).toBe(48);
  });

  it('returns 36 for a 3/4 bar at the default grid', () => {
    const p = new NotePosition({
      barIndex: 1,
      beatInBar: 1,
      timeSig: { count: 3, unit: 4 },
      slotsPerQuarter: 12,
    });
    expect(p.slotsPerBar).toBe(36);
  });

  it('scales with a denser grid (24 per quarter -> 96 in 4/4)', () => {
    const p = new NotePosition({
      barIndex: 1,
      beatInBar: 1,
      timeSig: { count: 4, unit: 4 },
      slotsPerQuarter: 24,
    });
    expect(p.slotsPerBar).toBe(96);
  });

  it('returns null when no time signature is supplied', () => {
    const p = new NotePosition({ barIndex: 1, beatInBar: 1, slotsPerQuarter: 12 });
    expect(p.slotsPerBar).toBeNull();
  });
});

describe('NotePosition.formatOffset', () => {
  it('returns null when no offset was supplied', () => {
    const p = new NotePosition({ barIndex: 1, beatInBar: 1, slotsPerQuarter: 12 });
    expect(p.formatOffset()).toBeNull();
  });

  it('formats a positive offset with a sign and ms unit', () => {
    const p = new NotePosition({ barIndex: 1, beatInBar: 1, slotsPerQuarter: 12, offsetMs: 12.34 });
    expect(p.formatOffset()).toBe('+12.3 ms');
  });

  it('formats a negative offset', () => {
    const p = new NotePosition({ barIndex: 1, beatInBar: 1, slotsPerQuarter: 12, offsetMs: -7 });
    expect(p.formatOffset()).toBe('-7.0 ms');
  });

  it('includes the offset in the dense toString readout', () => {
    const p = new NotePosition({
      barIndex: 1,
      beatInBar: 1,
      slotsPerQuarter: 12,
      timeSig: { count: 4, unit: 4 },
      offsetMs: 12.34,
    });
    expect(p.toString()).toContain('+12.3 ms');
  });
});

describe('NotePosition.formatBarBeat48ths', () => {
  it('denominator reflects the dynamic slot count', () => {
    const p = new NotePosition({
      barIndex: 1,
      beatInBar: 1,
      timeSig: { count: 4, unit: 4 },
      slotsPerQuarter: 24,
    });
    // beat 1 -> slot 1 of 96
    expect(p.formatBarBeat48ths()).toBe('1/96');
  });
});
