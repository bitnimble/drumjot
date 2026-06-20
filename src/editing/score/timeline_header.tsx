import { observer } from 'mobx-react-lite';
import React from 'react';
import { jotPlayer } from 'src/editing/playback/player';
import type { TempoRamp } from 'src/editing/playback/tempo_presenter';
import { GutterResizeHandle } from 'src/ui/gutter_resize_handle/gutter_resize_handle';
import { StructuralContext, TempoContext } from '../jot_editor_contexts';
import { ViewportStoreContext } from '../viewport/viewport_contexts';
import { Playhead } from '../playback/playhead';
import styles from './score.module.css';
import { barsRowWidthSeed, intersectsBeatRange } from '../utils/windowing';
import { seekFromClick } from './seek';

/**
 * Sticky-gutter header above the audio tracks / score that labels each
 * bar boundary with its 1-based bar number and the playback time at that
 * boundary (mm:ss). Tick marks sit on the same `bar.x` line as the
 * score's barlines below so the header reads as a ruler over the
 * timeline. Click-to-seek mirrors the score and audio-track rows.
 *
 * Per-bar timings come from the live playback timeline whenever it
 * matches the current jot (so tempo overrides and the lead-in offset
 * stay in sync with the playhead); otherwise we build a one-shot
 * timeline so the header still labels everything correctly before the
 * user hits Play.
 */
export const TimelineHeader = observer(
  ({
    onSeek,
    onResizeGutterStart,
  }: {
    onSeek: (x: number) => void;
    /** Pointer-down handler for the gutter resize affordance rendered
     * on the right edge of this header's gutter. */
    onResizeGutterStart: (e: React.PointerEvent<HTMLDivElement>) => void;
  }) => {
    const structural = React.useContext(StructuralContext);
    const tempo = React.useContext(TempoContext);
    // Reading the structural layers (not pixels) keeps this header stable
    // across zoom; the per-tick `--bar-start-beat` is set inline, and CSS
    // calc() multiplies by the score-root's `--px-per-beat` to get the
    // final pixel position. Without this the header re-rendered every wheel
    // tick, re-creating 100+ tick marks just to reposition each by one
    // calc-arithmetic step.
    // `hasContent` is the stable geometry-spine check (doesn't churn on note
    // edits); the tick layout below reads the tempo timeline, not the layers.
    if (!structural || !tempo || !structural.hasContent) return null;

    const liveTimeline = jotPlayer.timeline;
    const timeline =
      liveTimeline.bars.length > 0 && liveTimeline.rendered === structural
        ? liveTimeline
        : tempo.timeline;

    // Lead-in is materialised as negative-indexed bars by
    // `structureForLayer`, so a single sum over `bar.beats` covers
    // both pre-drum and drum content with no separate chrome offset.
    // Cached on the jot (`layerBeats`) so all observers share one walk.
    const layerBeats = structural.layerBeats;

    // Effective tempo at each bar's downbeat, derived from the shared
    // tempo timeline. Mid-bar tempo changes inside a bar aren't shown
    // separately by the header pill; the displayed value tracks the
    // tempo in force at the bar's downbeat. Reading the cached
    // `barTempos` computed avoids rebuilding the layout on every
    // header render (the tempo timeline is structure-only input, so a
    // zoom tick doesn't invalidate it).
    const tempos = tempo.barTempos;

    // Gradual tempo changes render as solid ramp bars (handled in
    // WindowedTicks). The flat bpm pills the segment walk would otherwise
    // paint across a ramp's span (now that `buildBarTempos` emits varying
    // ramp segments) are suppressed; the ramp's own captions carry the
    // start/end values.
    const ramps = tempo.tempoRamps;

    // Full (non-windowed) walk to build the per-bar tick descriptors.
    // The time-sig / bpm "changed since the previous bar" flags depend on
    // running state across every bar, so the walk can't be windowed; but
    // it only produces plain data (no DOM / React), so it stays cheap on
    // a long song. The DOM is windowed separately in {@link WindowedTicks}.
    let cumBeats = 0;
    let prevTime: { count: number; unit: number } | undefined;
    // Tempo "carried out" of the previous bar (= its last segment's bpm).
    // Rounded so float jitter (119.97 vs 120.03) doesn't paint a change.
    let prevBpm: number | undefined;
    const ticks: TickDescriptor[] = [];
    // Stable geometry spine (incl. the virtual lead-in bar); the per-bar
    // timing/tempo come from the tempo timeline alongside.
    const geometry = structural.viewGeometry;
    for (let i = 0; i < geometry.length; i++) {
      const bar = geometry[i];
      const timing = timeline.bars[i];
      const timeSec = timing?.startSec ?? 0;
      const startBeat = cumBeats;
      cumBeats += bar.beats;
      const showTimeSig =
        !prevTime || bar.tsCount !== prevTime.count || bar.tsUnit !== prevTime.unit;
      prevTime = { count: bar.tsCount, unit: bar.tsUnit };
      // Walk the bar's tempo segments and emit a label whenever the
      // bpm changes (relative to the running bpm). The label at
      // segment.startBeat=0 sits in the bar tick's top row alongside
      // the bar number; later labels float at their beat-anchored
      // position so a mid-bar tempo change renders where it actually
      // takes effect, not at the next downbeat.
      const segments = tempos[i]?.segments ?? [];
      let downbeatBpm: number | undefined;
      const midBpmChanges: Array<{ beat: number; bpm: number }> = [];
      for (let s = 0; s < segments.length; s++) {
        const seg = segments[s];
        // A segment inside a ramp is drawn as the ramp bar, not a flat pill;
        // it still advances the running tempo to its local end so the first
        // flat segment after the ramp doesn't paint a redundant pill.
        if (inRampSpan(ramps, startBeat + seg.startBeat)) {
          prevBpm = Math.round(seg.endBpm ?? seg.bpm);
          continue;
        }
        const bpm = Math.round(seg.bpm);
        if (prevBpm === undefined || bpm !== prevBpm) {
          if (seg.startBeat === 0) downbeatBpm = bpm;
          else midBpmChanges.push({ beat: seg.startBeat, bpm });
          prevBpm = bpm;
        }
      }
      ticks.push({
        barIndex: bar.index,
        startBeat,
        beats: bar.beats,
        timeSec,
        showTimeSig,
        timeCount: bar.tsCount,
        timeUnit: bar.tsUnit,
        downbeatBpm,
        midBpmChanges,
      });
    }
    return (
      <div className={styles.timelineHeader}>
        <div className={styles.timelineHeaderGutter}>
          <span className={styles.timelineHeaderLabel}>Bar / Time</span>
          <GutterResizeHandle onResizeStart={onResizeGutterStart} />
        </div>
        <div
          className={styles.timelineHeaderBarsRow}
          data-bars-row
          style={
            {
              ['--layer-beats' as string]: layerBeats,
              ['--bars-row-width' as string]: barsRowWidthSeed(structural, layerBeats),
            } as React.CSSProperties
          }
          onClick={(e) => seekFromClick(e, onSeek)}
        >
          <WindowedTicks ticks={ticks} ramps={ramps} />
          <Playhead showLabel onSeek={onSeek} />
        </div>
      </div>
    );
  }
);

/** Whether a global beat falls inside any tempo ramp's span. The end is
 *  exclusive (with a small epsilon) so the first segment after a ramp reads
 *  as a normal flat region. */
function inRampSpan(ramps: readonly TempoRamp[], gBeat: number): boolean {
  const eps = 1e-6;
  for (const r of ramps) {
    if (gBeat >= r.startBeat - eps && gBeat < r.endBeat - eps) return true;
  }
  return false;
}

/** Clamp `v` to `[lo, hi]`. */
function clampRange(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** One timeline-header tick's render data, precomputed by the full bar
 *  walk in {@link TimelineHeader} so {@link WindowedTicks} can window the
 *  DOM without re-deriving the running tempo / time-sig change flags. */
type TickDescriptor = {
  /** Clone-stable bar index; the React key (survives the window sliding). */
  barIndex: number;
  startBeat: number;
  beats: number;
  timeSec: number;
  showTimeSig: boolean;
  timeCount: number;
  timeUnit: number;
  downbeatBpm: number | undefined;
  midBpmChanges: Array<{ beat: number; bpm: number }>;
};

/**
 * Windowed DOM for the timeline-header ticks. Split out of {@link
 * TimelineHeader} so only this map (not the header gutter or its label)
 * re-renders on a scroll / zoom tick. Renders only ticks whose bar span
 * intersects {@link JotEditorStore.visibleBeatRange}; the descriptor list
 * is precomputed and stable, so the parent doesn't re-render on scroll.
 */
const WindowedTicks = observer(function WindowedTicks({
  ticks,
  ramps,
}: {
  ticks: TickDescriptor[];
  ramps: readonly TempoRamp[];
}) {
  const viewport = React.useContext(ViewportStoreContext);
  const structural = React.useContext(StructuralContext);
  const range = viewport?.visibleBeatRange ?? null;
  // The score has no real scroll container (`.jotContainer` is
  // `overflow: hidden`; scroll is a CSS `transform` on `.scrollViewport`),
  // so `position: sticky` is inert. To keep both bpm captions visible while
  // a ramp wider than the viewport runs off either edge, we clamp each label
  // to the visible window in JS off store observables (scrollX / viewport
  // width / pxPerBeat, no DOM layout reads). This component already
  // re-renders on scroll + zoom (its `visibleBeatRange` read depends on
  // both), and only the handful of windowed ramps pay the cost.
  const pxPerBeat = structural?.pxPerBeat ?? 0;
  const scrollX = viewport?.scrollX ?? 0;
  // Bars-row-local x of the visible window: its left edge sits at `scrollX`
  // (the sticky gutter overlays content-x 0..gutterWidth), and its width is
  // the viewport minus that gutter.
  const usableWidth = (viewport?._viewportWidth ?? 0) - (viewport?.gutterWidth ?? 0);
  const canPin = pxPerBeat > 0 && usableWidth > 0;
  return (
    <>
      {ramps.map((r) => {
        if (!intersectsBeatRange(range, r.startBeat, r.endBeat - r.startBeat)) return null;
        const widthPx = (r.endBeat - r.startBeat) * pxPerBeat;
        const leftPx = r.startBeat * pxPerBeat;
        const rightPx = r.endBeat * pxPerBeat;
        // Start caption rides the left visible edge; end caption the right;
        // each clamped so it can't leave its own ramp box.
        const startShift = canPin ? clampRange(scrollX - leftPx, 0, widthPx) : 0;
        const endShift = canPin
          ? -clampRange(rightPx - (scrollX + usableWidth), 0, widthPx)
          : 0;
        return (
          <div
            key={`ramp-${r.startBeat}`}
            className={styles.timelineHeaderBpmRamp}
            data-testid="bpm-ramp"
            style={
              {
                ['--bar-start-beat' as string]: r.startBeat,
                ['--bar-span-beats' as string]: r.endBeat - r.startBeat,
              } as React.CSSProperties
            }
          >
            <div className={styles.timelineHeaderBpmRampLine} />
            <span
              className={styles.timelineHeaderBpmRampLabel}
              data-testid="bpm-ramp-start"
              style={{ transform: `translateX(${startShift}px)` }}
            >
              {Math.round(r.startBpm)} bpm
            </span>
            <span
              className={`${styles.timelineHeaderBpmRampLabel} ${styles.timelineHeaderBpmRampEnd}`}
              data-testid="bpm-ramp-end"
              style={{ transform: `translateX(${endShift}px)` }}
            >
              {Math.round(r.endBpm)} bpm
            </span>
          </div>
        );
      })}
      {ticks.map((t) => {
        if (!intersectsBeatRange(range, t.startBeat, t.beats)) return null;
        return (
          <React.Fragment key={t.barIndex}>
            <div
              className={styles.timelineHeaderTick}
              style={{ ['--bar-start-beat' as string]: t.startBeat } as React.CSSProperties}
            >
              <div className={styles.timelineHeaderTopRow}>
                <span className={styles.timelineHeaderBar}>{t.barIndex}</span>
                {t.showTimeSig && (
                  <span className={styles.timelineHeaderTimeSig}>
                    {t.timeCount}/{t.timeUnit}
                  </span>
                )}
                {t.downbeatBpm !== undefined && (
                  <span className={styles.timelineHeaderBpm}>{t.downbeatBpm} bpm</span>
                )}
              </div>
              <span className={styles.timelineHeaderTime}>{formatTime(t.timeSec)}</span>
            </div>
            {t.midBpmChanges.map((c, j) => (
              <div
                key={`bpm-${t.barIndex}-${j}`}
                className={styles.timelineHeaderBpmAnchor}
                style={
                  { ['--bar-start-beat' as string]: t.startBeat + c.beat } as React.CSSProperties
                }
              >
                <div className={styles.timelineHeaderTopRow}>
                  <span className={styles.timelineHeaderBpm}>{c.bpm} bpm</span>
                </div>
              </div>
            ))}
          </React.Fragment>
        );
      })}
    </>
  );
});

function formatTime(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const min = Math.floor(total / 60);
  const sec = total % 60;
  return `${min}:${String(sec).padStart(2, '0')}`;
}
