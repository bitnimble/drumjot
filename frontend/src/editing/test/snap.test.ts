import { describe, expect, it } from 'bun:test';
import { enabledDivisors, snapBeat } from 'src/editing/snap';

const near = (a: number, b: number) => expect(Math.abs(a - b)).toBeLessThan(1e-9);

describe('enabledDivisors', () => {
  it('maps enabled grid families to per-beat divisors', () => {
    expect(
      enabledDivisors({
        mainBeat: true,
        subBeat16: true,
        subBeatQuarterTriplet: false,
        subBeatTriplet: false,
        subBeat48: false,
      })
    ).toEqual([1, 4]);
  });

  it('returns nothing when no families are enabled', () => {
    expect(
      enabledDivisors({
        mainBeat: false,
        subBeat16: false,
        subBeatQuarterTriplet: false,
        subBeatTriplet: false,
        subBeat48: false,
      })
    ).toEqual([]);
  });
});

describe('snapBeat', () => {
  it('snaps to the nearest 16th', () => {
    near(snapBeat(0.3, [4]), 0.25);
    near(snapBeat(0.4, [4]), 0.5);
  });

  it('snaps to the nearest main beat', () => {
    near(snapBeat(1.4, [1]), 1);
    near(snapBeat(1.6, [1]), 2);
  });

  it('snaps to the nearest point in the UNION of families (16ths + 8th-triplets)', () => {
    // 0.3 is closer to a triplet (1/3 ≈ 0.333, dist 0.033) than a 16th (0.25, dist 0.05)
    near(snapBeat(0.3, [4, 3]), 1 / 3);
    // 0.2 is closer to a 16th (0.25) than a triplet (1/3)
    near(snapBeat(0.2, [4, 3]), 0.25);
  });

  it('leaves the beat untouched when no divisors are given', () => {
    near(snapBeat(0.37, []), 0.37);
  });

  it('clamps the snapped result to [0, maxBeat]', () => {
    near(snapBeat(3.9, [1], 4), 4);
    near(snapBeat(-0.1, [1], 4), 0);
    // would round up to 4 but the bar is only 3.5 beats long
    near(snapBeat(3.6, [1], 3.5), 3.5);
  });
});
