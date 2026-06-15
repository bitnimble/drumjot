/**
 * Time-to-pixel mapping for the playback playhead.
 *
 * Tempo follows `jot.tempoEvents` (sticky tempo changes anchored at
 * `(barIndex, beat)`, mid-bar precision) on top of the initial
 * `globalMetadata.bpm`. Browser playback does NOT go through the MIDI
 * bytes; the scheduler in `events.ts` walks the same per-bar tempo
 * segments so the playhead, the audio-track waveform and the scheduled
 * drums all share one clock.
 *
 * Jot-time anchor: bar 1 (= the first drum bar, `layer.bars[leadBars]`)
 * sits at jot time 0 by construction. That coincides with the audio time
 * stored in `globalMetadata.songLeadIn` so the player's
 * `media = jot - songLeadIn` identity lines the synth drums up with the
 * recorded audio drums. Pre-drum lead-in bars (if any) sit at negative
 * startSec; their notes (rare, but possible if an upstream generator
 * stamps drums into a pre-drum bar) fire at negative jot time, which
 * maps back to positive media time during playback. Anacrusis, when
 * present, is itself drum content; under the drums-t0 convention it
 * sits at jot 0 alongside bar 1's downbeat.
 *
 * Pixel offsets are looked up against the LIVE {@link LaidOutJot} (the
 * `StructuralPresenter`) on every call to {@link timeToX}, not cached at
 * build time, so the playhead stays in sync with the score even when the
 * user zooms during playback (the `ViewConfig.barWidth` change reflows the
 * layout reactively and the playhead re-reads the new `pxPerBeat`).
 */
import { Jot, TimeSignature } from 'src/schema/dsl/dsl';
import { toTempoBars, type StructLayer } from 'src/editing/structure/structure_store';
import { Pixels, px, type ViewConfig } from 'src/editing/viewport/view_config';
import { buildBarTempos, resolveBpm } from 'src/schema/dsl/tempo';

export { resolveBpm };

/**
 * The minimal laid-out-jot surface the timeline maths needs: the
 * zoom-invariant `layers` + `source` for the tempo walk, and the pixel
 * scale + `config` for the playhead mapping. {@link StructuralPresenter}
 * satisfies this structurally, so the timeline (and everything that holds a
 * {@link JotTimeline}) depends on this interface rather than the concrete
 * presenter. */
export interface LaidOutJot {
  layers: readonly StructLayer[];
  source: Jot;
  pxPerBeat: number;
  config: ViewConfig;
}

/**
 * Pick the BPM and time signature the song spends the largest proportion
 * of its audio duration in, excluding pre-drum lead-in bars (negative
 * `index`) whose bpm is artificially scaled to fit the audio pre-roll.
 *
 * Used by the subtitle formatter (cosmetic), and by `store.setDrumOffset`
 * to convert a beat-shift delta to the audio-compensation seconds.
 * `globalMetadata.bpm` is the initial tempo (before any tempoEvent
 * fires), which for transcribed bundles is the back-solved lead-in
 * tempo and can differ markedly from the song's playing tempo.
 */
export function pickDominantBpmAndTime(jot: LaidOutJot): {
  dominantBpm: number | undefined;
  dominantTime: TimeSignature | undefined;
} {
  const layer = jot.layers[0];
  if (!layer || layer.bars.length === 0) {
    return { dominantBpm: undefined, dominantTime: undefined };
  }
  const tempos = buildBarTempos(jot.source, toTempoBars(layer.bars));
  const bpmDur = new Map<number, number>();
  const timeDur = new Map<string, { time: TimeSignature; duration: number }>();
  for (let i = 0; i < layer.bars.length; i++) {
    const bar = layer.bars[i];
    if (bar.index < 0) continue;
    const barTempos = tempos[i];
    for (const seg of barTempos.segments) {
      const segDuration = (seg.endBeat - seg.startBeat) * (60 / seg.bpm);
      const bpmKey = Math.round(seg.bpm);
      bpmDur.set(bpmKey, (bpmDur.get(bpmKey) ?? 0) + segDuration);
    }
    const timeKey = `${bar.tsCount}/${bar.tsUnit}`;
    const prev = timeDur.get(timeKey);
    if (prev) prev.duration += barTempos.durationSec;
    else
      timeDur.set(timeKey, {
        time: { count: bar.tsCount, unit: bar.tsUnit },
        duration: barTempos.durationSec,
      });
  }
  let dominantBpm: number | undefined;
  let bestBpmDur = -Infinity;
  for (const [bpm, dur] of bpmDur) {
    if (dur > bestBpmDur) {
      bestBpmDur = dur;
      dominantBpm = bpm;
    }
  }
  let dominantTime: TimeSignature | undefined;
  let bestTimeDur = -Infinity;
  for (const { time, duration } of timeDur.values()) {
    if (duration > bestTimeDur) {
      bestTimeDur = duration;
      dominantTime = time;
    }
  }
  return { dominantBpm, dominantTime };
}

export type BarTiming = {
  startSec: number;
  durationSec: number;
};

export type JotTimeline = {
  totalDurationSec: number;
  bars: BarTiming[];
  /**
   * Reference to the laid-out jot whose bars drive playback. Held by
   * reference so `timeToX` always reads current `bar.x` / `bar.width` —
   * critical for zoom changes during playback.
   *
   * `undefined` only on {@link EMPTY_TIMELINE}.
   */
  rendered: LaidOutJot | undefined;
};

export const EMPTY_TIMELINE: JotTimeline = {
  totalDurationSec: 0,
  bars: [],
  rendered: undefined,
};

export function buildTimeline(rendered: LaidOutJot): JotTimeline {
  // The audio-time fields we need (`bar.beats`, `bar.index`,
  // `jot.tempoEvents`, `globalMetadata.bpm`) all live on the structural
  // cache or on the source jot, which are zoom-invariant. Reading from
  // `rendered.structure` rather than `rendered.resolved` means observers
  // calling `buildTimeline` don't pick up a spurious dependency on
  // `viewConfig.barWidth`, so the per-bar timings stay stable across
  // wheel ticks.
  // All layers in a Jot share the same bar grid (they're laid out from the
  // same global metadata), so the first layer's timing is canonical.
  const layer = rendered.layers[0];
  if (!layer || layer.bars.length === 0) return EMPTY_TIMELINE;

  // Per-bar tempo segments come from `jot.tempoEvents`. Mid-bar tempo
  // changes are honoured natively: each bar's `durationSec` is the sum
  // of its constant-tempo intra-bar segments.
  const tempos = buildBarTempos(rendered.source, toTempoBars(layer.bars));
  const durations: number[] = new Array(layer.bars.length);
  for (let i = 0; i < layer.bars.length; i++) durations[i] = tempos[i].durationSec;

  // Anchor bar 1 (= the first non-lead-in bar) at jot time 0, so the
  // audio scheduler's "media = jot - songLeadIn" identity lines up the
  // synth drums with the recorded audio drums. Pre-drum bars (the
  // lead-in, identified by `bar.index < 0`) get negative `startSec`;
  // playback's rAF loop already accepts negative jot times for the
  // pre-roll scrub, and `timeToX` resolves them via the per-bar loop.
  // Lead-in count is read directly from the structure (counting the
  // leading run of negative-indexed bars) rather than from
  // `globalMetadata.leadBars`: `structureForLayer` materialises both
  // the explicit-leadBars and the chrome-only (`songLeadIn` without
  // `leadBars`) source shapes into the same negative-indexed-bar form,
  // so reading the count from the bars themselves keeps the timeline
  // path single-source-of-truth.
  let leadBars = 0;
  for (const b of layer.bars) {
    if (b.index >= 0) break;
    leadBars++;
  }
  let leadOffsetSec = 0;
  for (let i = 0; i < leadBars; i++) leadOffsetSec += durations[i];

  const bars: BarTiming[] = new Array(layer.bars.length);
  let cursor = -leadOffsetSec;
  for (let i = 0; i < layer.bars.length; i++) {
    bars[i] = { startSec: cursor, durationSec: durations[i] };
    cursor += durations[i];
  }
  // `cursor` now sits at the end of the last bar in jot time. The total
  // playable duration (used as the playback stop sentinel) covers
  // [-leadOffsetSec, cursor].
  return { totalDurationSec: cursor, bars, rendered };
}

/**
 * Map an absolute playback time (seconds from start) to the playhead's
 * pixel x within a layer's `barsRow`. Derives bar widths on the fly from
 * `structure.bars[i].beats * pxPerBeat`, deliberately avoiding
 * `rendered.resolved`: `timeToX` is called from non-reactive useLayoutEffect
 * contexts (PlayheadPosVar), where the MobX `computed` cache for `resolved`
 * is not kept warm, so each call would otherwise re-run the full pixel pass
 * (layoutJot → pixelLayer for every layer, every frame). Clamps to the bar
 * grid at both ends so the playhead never escapes the score.
 */
export function timeToX(timeline: JotTimeline, seconds: number): Pixels {
  const rendered = timeline.rendered;
  if (!rendered) return px(0);
  const layer = rendered.layers[0];
  const structBars = layer?.bars ?? [];
  const bars = timeline.bars;
  if (bars.length === 0 || structBars.length === 0) return px(0);
  const pxPerBeat = rendered.pxPerBeat;
  // Notes sit `pad` inside their bar's left edge (engraving inset). The
  // playhead marks when an onset *sounds*, so it has to ride the note
  // grid, not the time-anchored bar box; every branch below adds `pad`
  // so it lands on the note instead of a constant `pad` px to its left.
  const pad = rendered.config.barNotePaddingBeats * pxPerBeat;
  // Lead-in (when present) lives in `layer.bars` as negative-indexed
  // bars with negative `startSec`, so the per-bar loop below resolves
  // negative seek targets that fall inside them without a separate
  // chrome branch.
  const firstStartSec = bars[0]?.startSec ?? 0;
  // Clamp `seconds` earlier than the first bar's start (the pre-pre-roll
  // window) to that bar's left edge; keeps the playhead anchored
  // somewhere visible if a seek lands before any bar's range.
  if (seconds < firstStartSec) return px(pad);

  // Linear scan; jots are typically under ~64 bars so binary search adds
  // complexity without measurable benefit at the rAF rate. Walking the
  // bars also lets us accumulate `cursor` (the live `bar.x` analogue)
  // without a separate prefix-sum pass.
  let cursor = 0;
  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];
    const widthPx = (structBars[i]?.beats ?? 0) * pxPerBeat;
    if (seconds < bar.startSec + bar.durationSec) {
      const within = bar.durationSec > 0 ? (seconds - bar.startSec) / bar.durationSec : 0;
      return px(cursor + pad + within * widthPx);
    }
    cursor += widthPx;
  }
  return px(cursor + pad);
}

/**
 * Inverse of {@link timeToX}: map a pixel x within the layer's
 * `barsRow` (same coordinate space `bar.x` is in; origin at the left
 * edge of the bars region, *after* the gutter) back to an absolute
 * playback time in seconds. Used for click-to-seek on the score and
 * the audio-track waveforms. Clamps to the bar grid at both ends so a click
 * in the margins lands on 0 / the final bar boundary. Derives bar widths
 * from the structural cache + `pxPerBeat`; same rationale as
 * {@link timeToX}.
 */
export function xToTime(timeline: JotTimeline, x: number): number {
  const rendered = timeline.rendered;
  if (!rendered) return 0;
  const layer = rendered.layers[0];
  const structBars = layer?.bars ?? [];
  const bars = timeline.bars;
  if (bars.length === 0 || structBars.length === 0) return 0;
  const pxPerBeat = rendered.pxPerBeat;
  // Inverse of timeToX's note-grid shift: the clickable axis is offset
  // `pad` px right of the bar boxes, so subtract it before mapping x
  // back to time.
  const pad = rendered.config.barNotePaddingBeats * pxPerBeat;
  const lastTiming = bars[bars.length - 1];
  const endSec = lastTiming.startSec + lastTiming.durationSec;

  let cursor = 0;
  for (let i = 0; i < structBars.length; i++) {
    const widthPx = (structBars[i]?.beats ?? 0) * pxPerBeat;
    const x0 = cursor + pad;
    if (x < x0 + widthPx) {
      const timing = bars[i];
      if (!timing) return endSec;
      const within = widthPx > 0 ? (x - x0) / widthPx : 0;
      return timing.startSec + within * timing.durationSec;
    }
    cursor += widthPx;
  }
  return endSec;
}
