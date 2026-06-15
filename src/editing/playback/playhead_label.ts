import { JotTimeline } from 'src/editing/playback/timeline';

/**
 * Format a playhead time (seconds) as `M:SS.cc`. Negative times (the
 * audio lead-in before bar 0) render with a leading `-`. Pure; unit-tested
 * in `playhead_label.test.ts`.
 */
export function formatPlayheadTime(seconds: number): string {
  const negative = seconds < 0;
  const abs = Math.abs(seconds);
  const totalSec = Math.floor(abs);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  const cs = Math.floor((abs - totalSec) * 100);
  return `${negative ? '-' : ''}${min}:${String(sec).padStart(2, '0')}.${String(cs).padStart(2, '0')}`;
}

/**
 * Convert the playhead's jot-time position to `Bar N, X.XXb` for the
 * second line of the label. Walks the timeline's per-bar timings to
 * find the bar containing `jotTime`, then computes beat-in-bar in the
 * bar's time-signature beats (1-indexed at the downbeat). Returns
 * `null` when no bar can be resolved (empty timeline / no rendered
 * layer). Pure; unit-tested in `playhead_label.test.ts`.
 */
export function playheadBarBeat(timeline: JotTimeline, jotTime: number): string | null {
  const renderedBars = timeline.rendered?.layers[0]?.bars ?? [];
  if (renderedBars.length === 0 || timeline.bars.length === 0) return null;
  for (let i = 0; i < timeline.bars.length; i++) {
    const t = timeline.bars[i]!;
    if (jotTime < t.startSec + t.durationSec) {
      const rb = renderedBars[i];
      if (!rb || t.durationSec <= 0) return null;
      const beatInBar = 1 + ((jotTime - t.startSec) / t.durationSec) * rb.tsCount;
      // Truncate (not round) so the tail of a bar never rounds up to the
      // next bar's downbeat, e.g. 4/4 must go 4.99 → 1.00, never 5.00.
      const beatDisplay = (Math.floor(beatInBar * 100) / 100).toFixed(2);
      return `Bar ${rb.index}, ${beatDisplay}b`;
    }
  }
  // Past the end of the last bar; pin to its final beat so the label
  // doesn't blank out when scrubbing slightly past the score's tail.
  const last = renderedBars[renderedBars.length - 1];
  if (!last) return null;
  return `Bar ${last.index}, ${(last.tsCount + 0.99).toFixed(2)}b`;
}
