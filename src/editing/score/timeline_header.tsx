import { observer } from 'mobx-react-lite';
import React from 'react';
import { jotPlayer } from 'src/editing/playback/player';
import type { TempoRamp } from 'src/editing/playback/tempo_presenter';
import type { BpmMarker, TempoEditPresenter } from 'src/editing/playback/tempo_edit_presenter';
import { ActionMenuItem } from 'src/ui/dropdown/dropdown';
import { ContextMenu } from 'src/ui/context_menu/context_menu';
import { GutterResizeHandle } from 'src/ui/gutter_resize_handle/gutter_resize_handle';
import { StructuralContext, TempoContext, TempoEditContext } from '../jot_editor_contexts';
import { ViewportStoreContext } from '../viewport/viewport_contexts';
import { Playhead } from '../playback/playhead';
import styles from './score.module.css';
import { barsRowWidthSeed, intersectsBeatRange } from '../utils/windowing';
import { seekFromClick } from './seek';
import { BpmPill } from './bpm_pill';

const EPS = 1e-6;

/**
 * Sticky-gutter header above the audio tracks / score that labels each
 * bar boundary with its 1-based bar number and the playback time at that
 * boundary (mm:ss). Tick marks sit on the same `bar.x` line as the
 * score's barlines below so the header reads as a ruler over the
 * timeline. Click-to-seek mirrors the score and audio-track rows.
 *
 * The BPM pills are editable: each flat tempo change (and the song's initial
 * tempo) is a {@link BpmPill} that edits in place; right-clicking empty header
 * space opens a "Change BPM here" menu that drops a new change at the clicked
 * beat (see {@link TempoEditPresenter}). Gradual `BpmTransition` ramps render
 * read-only as captioned ramp bars.
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
    const tempoEdit = React.useContext(TempoEditContext);
    // The cursor-anchored "Change BPM here" menu, and the id of a pill freshly
    // created through it (so that pill mounts straight into edit mode). Both
    // are transient UI state, so they stay React-local.
    const [menu, setMenu] = React.useState<{
      clientX: number;
      clientY: number;
      barsRowX: number;
    } | null>(null);
    const [autoFocusId, setAutoFocusId] = React.useState<string | null>(null);
    const clearAutoFocus = React.useCallback(() => setAutoFocusId(null), []);

    // Reading the structural layers (not pixels) keeps this header stable
    // across zoom; the per-tick `--bar-start-beat` is set inline, and CSS
    // calc() multiplies by the score-root's `--px-per-beat` to get the
    // final pixel position.
    if (!structural || !tempo || !structural.hasContent) return null;

    const liveTimeline = jotPlayer.timeline;
    const timeline =
      liveTimeline.bars.length > 0 && liveTimeline.rendered === structural
        ? liveTimeline
        : tempo.timeline;

    const layerBeats = structural.layerBeats;
    // Gradual tempo changes render as solid ramp bars (handled in
    // WindowedTicks). Flat changes + the initial tempo are editable pills
    // sourced from the tempo-edit presenter (each carries its event id /
    // initial-marker tag), bucketed onto the bar they fall in below.
    const ramps = tempo.tempoRamps;
    const markers: BpmMarker[] = tempoEdit?.bpmMarkers ?? [];

    // Full (non-windowed) walk to build the per-bar tick descriptors. The
    // time-sig "changed since the previous bar" flags depend on running state
    // across every bar, so it can't be windowed; it produces only plain data
    // (no DOM), so it stays cheap. The DOM is windowed in {@link WindowedTicks}.
    let cumBeats = 0;
    let prevTime: { count: number; unit: number } | undefined;
    const ticks: TickDescriptor[] = [];
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
      // Pick the markers anchored in this bar: the one on its downbeat sits in
      // the tick's top row (alongside the bar number); later ones float at
      // their beat-anchored position.
      let downbeatMarker: BpmMarker | undefined;
      const midMarkers: Array<{ beatInBar: number; marker: BpmMarker }> = [];
      for (const m of markers) {
        if (Math.abs(m.globalBeat - startBeat) < EPS) downbeatMarker = m;
        else if (m.globalBeat > startBeat + EPS && m.globalBeat < startBeat + bar.beats - EPS) {
          midMarkers.push({ beatInBar: m.globalBeat - startBeat, marker: m });
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
        downbeatMarker,
        midMarkers,
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
          data-testid="timeline-bars-row"
          style={
            {
              ['--layer-beats' as string]: layerBeats,
              ['--bars-row-width' as string]: barsRowWidthSeed(structural, layerBeats),
            } as React.CSSProperties
          }
          onClick={(e) => seekFromClick(e, onSeek)}
          onContextMenu={(e) => {
            // Pills stop propagation, so this only fires on empty header space.
            if (!tempoEdit) return;
            e.preventDefault();
            const rect = e.currentTarget.getBoundingClientRect();
            setMenu({ clientX: e.clientX, clientY: e.clientY, barsRowX: e.clientX - rect.left });
          }}
        >
          <WindowedTicks
            ticks={ticks}
            ramps={ramps}
            tempoEdit={tempoEdit}
            autoFocusId={autoFocusId}
            onAutoFocusConsumed={clearAutoFocus}
          />
          <Playhead showLabel onSeek={onSeek} />
        </div>
        {menu && tempoEdit && (
          <ContextMenu x={menu.clientX} y={menu.clientY} onClose={() => setMenu(null)}>
            <ActionMenuItem
              label="Change BPM here"
              testId="bpm-menu-change"
              onClick={() => {
                const id = tempoEdit.createTempoChangeAtX(menu.barsRowX);
                setMenu(null);
                if (id) setAutoFocusId(id);
              }}
            />
          </ContextMenu>
        )}
      </div>
    );
  }
);

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
  /** The editable bpm pill on this bar's downbeat, if any. */
  downbeatMarker: BpmMarker | undefined;
  /** Editable bpm pills inside the bar (mid-bar tempo changes). */
  midMarkers: Array<{ beatInBar: number; marker: BpmMarker }>;
};

/**
 * Windowed DOM for the timeline-header ticks. Split out of {@link
 * TimelineHeader} so only this map (not the header gutter or its label)
 * re-renders on a scroll / zoom tick. Renders only ticks whose bar span
 * intersects the visible beat range; the descriptor list is precomputed and
 * stable, so the parent doesn't re-render on scroll.
 */
const WindowedTicks = observer(function WindowedTicks({
  ticks,
  ramps,
  tempoEdit,
  autoFocusId,
  onAutoFocusConsumed,
}: {
  ticks: TickDescriptor[];
  ramps: readonly TempoRamp[];
  tempoEdit: TempoEditPresenter | null;
  autoFocusId: string | null;
  onAutoFocusConsumed: () => void;
}) {
  const viewport = React.useContext(ViewportStoreContext);
  const structural = React.useContext(StructuralContext);
  const range = viewport?.visibleBeatRange ?? null;
  // The score has no real scroll container (scroll is a CSS `transform` on
  // `.scrollViewport`), so a ramp wider than the viewport clamps each caption
  // to the visible window in JS off store observables (no DOM layout reads).
  const pxPerBeat = structural?.pxPerBeat ?? 0;
  const scrollX = viewport?.scrollX ?? 0;
  const usableWidth = (viewport?._viewportWidth ?? 0) - (viewport?.gutterWidth ?? 0);
  const canPin = pxPerBeat > 0 && usableWidth > 0;

  const markerAutoFocus = (m: BpmMarker): boolean =>
    autoFocusId !== null && m.source.kind === 'event' && m.source.id === autoFocusId;

  return (
    <>
      {ramps.map((r) => {
        if (!intersectsBeatRange(range, r.startBeat, r.endBeat - r.startBeat)) return null;
        const widthPx = (r.endBeat - r.startBeat) * pxPerBeat;
        const leftPx = r.startBeat * pxPerBeat;
        const rightPx = r.endBeat * pxPerBeat;
        const startShift = canPin ? clampRange(scrollX - leftPx, 0, widthPx) : 0;
        const endShift = canPin ? -clampRange(rightPx - (scrollX + usableWidth), 0, widthPx) : 0;
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
                {t.downbeatMarker && tempoEdit && (
                  <BpmPill
                    marker={t.downbeatMarker}
                    presenter={tempoEdit}
                    autoFocus={markerAutoFocus(t.downbeatMarker)}
                    onAutoFocusConsumed={onAutoFocusConsumed}
                  />
                )}
              </div>
              <span className={styles.timelineHeaderTime}>{formatTime(t.timeSec)}</span>
            </div>
            {t.midMarkers.map((c) => (
              <div
                key={`bpm-${t.barIndex}-${
                  c.marker.source.kind === 'event' ? c.marker.source.id : 'initial'
                }`}
                className={styles.timelineHeaderBpmAnchor}
                style={
                  { ['--bar-start-beat' as string]: t.startBeat + c.beatInBar } as React.CSSProperties
                }
              >
                <div className={styles.timelineHeaderTopRow}>
                  {tempoEdit && (
                    <BpmPill
                      marker={c.marker}
                      presenter={tempoEdit}
                      autoFocus={markerAutoFocus(c.marker)}
                      onAutoFocusConsumed={onAutoFocusConsumed}
                    />
                  )}
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
