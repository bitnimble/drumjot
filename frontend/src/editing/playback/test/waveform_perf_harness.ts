/**
 * Isolated main-thread perf harness for the waveform render path. Loaded by the
 * blank `waveform_perf.html` page (NO app, NO worker), it synthesises a track,
 * builds the pyramid, and times {@link computeWaveformPeaks} + the shared
 * {@link paintWaveform} across a few zooms -- all on the page's main thread, so
 * `waveform_render.perf.e2e.ts` can measure them directly (the production path
 * runs the same code off-thread in a worker, where it's invisible to a
 * frame-budget probe).
 */
import {
  buildTrackPeaks,
  computeWaveformPeaks,
  type BarSlice,
} from 'src/editing/playback/waveform_compute';
import { paintWaveform } from 'src/editing/playback/waveform_paint';

const SAMPLE_RATE = 44100;

/** Deterministic pseudo-audio (220Hz tone under a 1Hz amplitude envelope + a
 *  cheap LCG noise floor) so the waveform isn't degenerate. Content only
 *  affects how it looks; the compute scans every sample regardless. */
function synthesize(seconds: number): { channels: Float32Array[]; sampleRate: number; length: number } {
  const length = Math.floor(SAMPLE_RATE * seconds);
  const c0 = new Float32Array(length);
  const c1 = new Float32Array(length);
  let seed = 1;
  for (let i = 0; i < length; i++) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    const noise = (seed / 0x7fffffff - 0.5) * 0.1;
    const t = i / SAMPLE_RATE;
    c0[i] = Math.sin(t * 220 * Math.PI * 2) * Math.abs(Math.sin(t * Math.PI * 2)) + noise;
    c1[i] = c0[i] * 0.92;
  }
  return { channels: [c0, c1], sampleRate: SAMPLE_RATE, length };
}

export type WfRenderSample = {
  tileSeconds: number;
  samplesPerPx: number;
  computeMs: number;
  paintMs: number;
};

export type WfPerfResult = {
  trackSeconds: number;
  samples: number;
  buildMs: number;
  renders: WfRenderSample[];
};

export type WfPerfOpts = {
  trackSeconds?: number;
  widthPx?: number;
  height?: number;
  dpr?: number;
  iterations?: number;
  warmup?: number;
  tileDurations?: number[];
};

function run(opts: WfPerfOpts = {}): WfPerfResult {
  const trackSeconds = opts.trackSeconds ?? 240;
  const W = opts.widthPx ?? 400;
  const H = opts.height ?? 64;
  const dpr = opts.dpr ?? 2;
  const iterations = opts.iterations ?? 300;
  const warmup = opts.warmup ?? 10;
  const tiles = opts.tileDurations ?? [2, 8, 30];

  const data = synthesize(trackSeconds);
  const tb = performance.now();
  const peaks = buildTrackPeaks(data);
  const buildMs = performance.now() - tb;

  const canvas = document.getElementById('wf') as HTMLCanvasElement;
  const backingW = W * dpr;
  const backingH = H * dpr;
  canvas.width = backingW;
  canvas.height = backingH;
  canvas.style.width = `${W}px`;
  canvas.style.height = `${H}px`;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('no 2d context');

  const renders: WfRenderSample[] = [];
  for (const tileSeconds of tiles) {
    const bars: BarSlice[] = [{ x: 0, width: W, startSec: 0, durationSec: tileSeconds }];
    let pk = computeWaveformPeaks(peaks, bars, W, 0);
    for (let k = 0; k < warmup; k++) {
      pk = computeWaveformPeaks(peaks, bars, W, 0);
      paintWaveform(ctx, pk, W, H, backingW, backingH, '#5ba8e8', 1);
    }
    const tc = performance.now();
    for (let k = 0; k < iterations; k++) pk = computeWaveformPeaks(peaks, bars, W, 0);
    const computeMs = (performance.now() - tc) / iterations;
    const tp = performance.now();
    for (let k = 0; k < iterations; k++) paintWaveform(ctx, pk, W, H, backingW, backingH, '#5ba8e8', 1);
    const paintMs = (performance.now() - tp) / iterations;
    renders.push({
      tileSeconds,
      samplesPerPx: Math.round((tileSeconds * SAMPLE_RATE) / W),
      computeMs,
      paintMs,
    });
  }
  return { trackSeconds, samples: data.length, buildMs, renders };
}

(window as unknown as { __waveformPerf: { run: typeof run } }).__waveformPerf = { run };
