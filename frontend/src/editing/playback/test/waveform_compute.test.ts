import { describe, expect, it } from 'bun:test';
import {
  BarSlice,
  computeWaveformPeaksFromChannels,
} from 'src/editing/playback/waveform_compute';

/** Mono audio of `length` samples with a single unit impulse at `at`. */
function impulse(sampleRate: number, length: number, at: number) {
  const ch = new Float32Array(length);
  ch[at] = 1;
  return { channels: [ch], sampleRate, length };
}

/** Pixel column holding the largest peak (the rendered impulse). */
function peakPixel(peaks: Float32Array): number {
  let best = -1;
  let bestVal = 0;
  for (let p = 0; p * 2 + 1 < peaks.length; p++) {
    const v = Math.abs(peaks[p * 2 + 1]);
    if (v > bestVal) {
      bestVal = v;
      best = p;
    }
  }
  return best;
}

describe('computeWaveformPeaksFromChannels, render-time drift stretch', () => {
  const SR = 1000;
  const W = 100; // bar pixel width
  const D = 2.0; // bar jot duration (s)
  // 0.2 s of recorded drift: the bar's real downbeat is 200 ms later than
  // the uniform grid says.
  const DRIFT = 0.2;
  const audio = impulse(SR, 3000, Math.round(DRIFT * SR)); // impulse at the real downbeat

  it('renders a transient at the bar line when the bar carries its drift', () => {
    const slice: BarSlice = {
      x: 0,
      width: W,
      startSec: 0,
      durationSec: D,
      driftSec: DRIFT,
      nextDriftSec: DRIFT,
    };
    // The impulse sits at the bar's REAL downbeat, so with drift applied it
    // renders right at the bar line (pixel ~0).
    expect(peakPixel(computeWaveformPeaksFromChannels(audio, [slice], W, 0))).toBeLessThanOrEqual(2);
  });

  it('without drift the same transient is misaligned (right of the bar line)', () => {
    const slice: BarSlice = { x: 0, width: W, startSec: 0, durationSec: D };
    // 0.2 s / 2.0 s * 100 px = 10 px to the right.
    expect(peakPixel(computeWaveformPeaksFromChannels(audio, [slice], W, 0))).toBeGreaterThan(6);
  });

  it('is identical to the plain mapping when drift is zero', () => {
    const at = impulse(SR, 3000, 1000); // audio 1.0 s
    const plain = computeWaveformPeaksFromChannels(
      at,
      [{ x: 0, width: W, startSec: 0, durationSec: D }],
      W,
      0,
    );
    const zeroDrift = computeWaveformPeaksFromChannels(
      at,
      [{ x: 0, width: W, startSec: 0, durationSec: D, driftSec: 0, nextDriftSec: 0 }],
      W,
      0,
    );
    expect(peakPixel(zeroDrift)).toBe(peakPixel(plain));
    expect(peakPixel(plain)).toBeCloseTo(50, 0); // 1.0 s / 2.0 s * 100 px
  });
});
