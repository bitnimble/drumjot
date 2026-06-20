import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { Instrument } from 'src/schema/dsl/dsl';
import type { StructBar } from 'src/editing/structure/structure_store';
import { ViewConfig } from 'src/editing/viewport/view_config';
import { InstrumentTrack, trackKey } from 'src/editing/tracks/tracks';
import { GutterResizeHandle } from 'src/ui/gutter_resize_handle/gutter_resize_handle';
import { MuteButton, SoloButton } from 'src/ui/icon_button/icon_button';
import { StructuralContext } from '../jot_editor_contexts';
import { MixerStoreContext } from './mixer_contexts';
import { NoteProvenanceContext } from '../provenance/provenance_contexts';
import { ViewportStoreContext } from '../viewport/viewport_contexts';
import styles from './mixer.module.css';
import { Playhead } from '../playback/playhead';
import { seekFromClick } from '../score/seek';
import { FilteredOnsetView } from '../score/filtered_onset_view';
import { DragPreviewView, PlaceholderNoteView } from '../score/placeholder_note';
import { BarView } from '../score/bar_view';
import { EditingStoreContext, EditingPresenterContext } from '../editing_contexts';
import type { PlaceholderNote } from '../editing_store';
import { ViewportStore } from '../viewport/viewport_store';
import { barsRowWidthSeed, intersectsBeatRange } from '../utils/windowing';
import { InstrumentTrackOverflowMenu } from './overflow_menus';
import { RowVolumeSlider } from './gutter_controls';
import { MixerRowDragProps, useMixerRowDropTarget, MixerDragHandle } from './mixer_drag';
import type { LayerControls } from './mixer_controls';

/**
 * The windowed bar list for one instrument row. Split out of
 * {@link InstrumentTrackView} so the only thing that re-renders on a scroll /
 * zoom tick is this bar map, the row gutter (label, fader, M/S, overflow
 * menu) reads no scroll observable and stays put. Mirrors the
 * waveform-chunk visibility pattern ({@link AudioTrackWaveformCanvas}):
 * read the visible beat window from the store and render only the bars
 * whose span intersects it (plus the buffer baked into
 * `visibleBeatRange`).
 *
 * Bars key on the clone-stable `bar.index` (not the array position) so
 * the window sliding by one bar reuses every surviving bar's DOM instead
 * of re-keying the whole list. The per-bar props handed to {@link
 * BarView} are referentially stable across scroll (the caller memoises
 * `lanes` / `colorForLane`), so `BarView`'s `observer` memo holds and
 * an unchanged visible bar pays nothing on a scroll tick that doesn't
 * move the window, only newly-revealed bars mount.
 */
const WindowedBarList = observer(function WindowedBarList({
  viewport,
  laneBars,
  startBeats,
  lane,
  config,
  showBrackets,
  laneOrder,
  highlightedPattern,
  onPatternClick,
  isLaneAudible,
  lanes,
  colorForLane,
  instrumentForLane,
}: {
  viewport: ViewportStore | null;
  laneBars: readonly StructBar[];
  startBeats: readonly number[];
  lane: string;
  config: ViewConfig;
  showBrackets: boolean;
  laneOrder: readonly string[];
  highlightedPattern: string | undefined;
  onPatternClick: (name: string) => void;
  isLaneAudible: (lane: string) => boolean;
  lanes: string[];
  colorForLane: (lane: string) => string | undefined;
  instrumentForLane: (lane: string) => Instrument;
}) {
  const range = viewport?.visibleBeatRange ?? null;
  return (
    <>
      {laneBars.map((bar, i) => {
        const startBeat = startBeats[i];
        if (!intersectsBeatRange(range, startBeat, bar.beats)) return null;
        return (
          <BarView
            key={bar.index}
            bar={bar}
            barStartBeat={startBeat}
            lanes={lanes}
            config={config}
            isAnacrusis={bar.index === 0}
            highlightedPattern={highlightedPattern}
            onPatternClick={onPatternClick}
            isLaneAudible={isLaneAudible}
            showBrackets={showBrackets}
            rowLane={lane}
            laneOrder={laneOrder}
            colorForLane={colorForLane}
            instrumentForLane={instrumentForLane}
          />
        );
      })}
    </>
  );
});

export const InstrumentTrackView = observer(
  ({
    lane,
    layerId,
    mergeLaneLayerIds,
    config,
    showBrackets,
    laneOrder,
    highlightedPattern,
    onPatternClick,
    onSeek,
    layerControls,
    idx,
    dragFromIdx,
    dropTargetIdx,
    onDragStartIdx,
    onDropTargetIdx,
    onMoveTrack,
    onResetDrag,
    groupStart,
    groupEnd,
    inGroup,
    onResizeGutterStart,
  }: {
    lane: string;
    /** When set, the row shows ONLY this layer's notes on the lane (the
     *  per-track render path, via `barsForTrack`); absent = merged across
     *  layers (`barsForLane`, the legacy lane-row path). */
    layerId?: string;
    /** Merge view only: the layers this collapsed row aggregates. Mute/solo/
     *  volume then act on every `${layerId}/${lane}` track at once. */
    mergeLaneLayerIds?: readonly string[];
    config: ViewConfig;
    showBrackets: boolean;
    laneOrder: readonly string[];
    highlightedPattern: string | undefined;
    onPatternClick: (name: string) => void;
    onSeek: (x: number) => void;
    layerControls: LayerControls;
  } & MixerRowDragProps) => {
    const structural = React.useContext(StructuralContext);
    // `hasContent` is a stable boolean (it reads the geometry spine, not the
    // note-content `layers`), so this row never re-renders just because some
    // OTHER bar/lane changed; the per-lane `barsForLane(lane)` read below is
    // the sole structural dependency, and it's granular per (bar, lane).
    if (!structural || !structural.hasContent) return null;
    // Per-lane derived data (bars, layer-wide totals, cumulative
    // bar-start offsets, label color/instrument name); all memoised on
    // the jot via `barsForLane(lane)`, so each row reads its slice
    // from the MobX cache instead of recomputing on every render.
    // `barBeatStart` and `startBeats` are the same array; the keyed
    // names just disambiguate the two historical use sites.
    const {
      bars: laneBars,
      layerBeats,
      leadInBarsBeats,
      barBeatStart,
      startBeats,
      instrumentName,
    } = layerId !== undefined ? structural.barsForTrack(layerId, lane) : structural.barsForLane(lane);
    // Resolve the row's note colour through the store-owned
    // `InstrumentTrack`. The structural `barsForLane().laneColor` is
    // now palette-only (overrides moved off the jot in the colour-
    // picker refactor), so layering happens here: the InstrumentTrack
    // returns the override if set, otherwise the jot's palette default,
    // otherwise the neutral fallback grey. Reading it inside this
    // observer is the dependency that drives a row re-render when the
    // user picks a new colour.
    const mixer = React.useContext(MixerStoreContext);
    const viewport = React.useContext(ViewportStoreContext);
    const instrumentTrack = mixer?.getInstrumentTrack(lane);
    const laneColor = instrumentTrack?.color ?? 'var(--color-text-faint-strong)';

    // Insert-mode plumbing. Read off context as plain objects (not tracked
    // MobX reads), so the row never re-renders on a mode toggle or a cursor
    // move; `editing.mode` is read only inside the event handlers (untracked),
    // and the placeholder preview re-renders in its own isolated observer.
    const editing = React.useContext(EditingStoreContext);
    const editingPresenter = React.useContext(EditingPresenterContext);
    // Bars-row left edge, cached lazily on the first pointer move of a hover
    // session and cleared on leave, so the per-move x→beat math never reads
    // layout (no `getBoundingClientRect` per pointer move; see AGENTS.md §5.9).
    const barsRowLeftRef = React.useRef<number | null>(null);
    const placeholderAt = (clientX: number, left: number): PlaceholderNote | undefined => {
      const px = structural.pxPerBeat;
      if (px <= 0) return undefined;
      // Invert the note `left` calc: x = (notePadBeats + absBeat) * px.
      const cont = (clientX - left) / px - (config.barNotePaddingBeats as number);
      for (let i = 0; i < laneBars.length; i++) {
        const start = startBeats[i];
        if (cont < start + laneBars[i].beats || i === laneBars.length - 1) {
          const beat = Math.min(Math.max(cont - start, 0), laneBars[i].beats);
          return { lane, layerId, barId: laneBars[i].id, beat, absBeat: start + beat, barBeats: laneBars[i].beats };
        }
      }
      return undefined;
    };
    const onBarsRowPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
      if (!editingPresenter) return;
      // A drag-move in flight: report the lane the cursor is over (this row) +
      // the cursor x; the presenter recomputes the preview top-down. The lane
      // comes from event routing, not hit-testing, so this needs no DOM read.
      if (editing?.dragActive) {
        editingPresenter.updateDragMove(lane, e.clientX, laneOrder);
        return;
      }
      // A paste placement in flight: feed the cursor's absolute beat (same
      // clientX→beat mapping as the insert placeholder) + the row's lane; the
      // copied cluster follows via the shared placement core.
      if (editing?.pasteActive) {
        let pLeft = barsRowLeftRef.current;
        if (pLeft === null) {
          pLeft = e.currentTarget.getBoundingClientRect().left;
          barsRowLeftRef.current = pLeft;
        }
        const ph = placeholderAt(e.clientX, pLeft);
        if (ph) editingPresenter.updatePaste(ph.absBeat, lane, laneOrder);
        return;
      }
      if (editing?.mode !== 'insert') return;
      let left = barsRowLeftRef.current;
      if (left === null) {
        left = e.currentTarget.getBoundingClientRect().left;
        barsRowLeftRef.current = left;
      }
      const placeholder = placeholderAt(e.clientX, left);
      if (placeholder) editingPresenter.movePlaceholder(placeholder);
      else editingPresenter.clearPlaceholder();
    };
    const onBarsRowPointerLeave = () => {
      barsRowLeftRef.current = null;
      if (editing?.mode === 'insert') editingPresenter?.clearPlaceholder();
    };
    const onBarsRowClick = (e: React.MouseEvent<HTMLDivElement>) => {
      // A click during a paste placement commits the cluster at its previewed
      // position (and is swallowed here so it doesn't also seek).
      if (editing?.pasteActive && editingPresenter) {
        editingPresenter.commitPaste();
        return;
      }
      if (editing?.mode === 'insert' && editingPresenter) {
        editingPresenter.insertNote();
        return;
      }
      seekFromClick(e, onSeek);
    };

    // Filtered-onset ghost overlays (debug bundle + checkbox gated).
    // Resolve once per row so the per-entry render below is just a map.
    const provenance = React.useContext(NoteProvenanceContext);
    const showFiltered = provenance?.showFiltered ?? false;
    const rejectedForLane = showFiltered ? (provenance!.rejectedByLane.get(lane) ?? []) : [];

    // Mute/solo/volume key by track (layer+lane) so the same lane in two
    // layers is controlled independently. A normal row controls one track; a
    // merged row aggregates every layer's track of the lane (mute = mute each).
    const rowKeys =
      mergeLaneLayerIds && mergeLaneLayerIds.length > 0
        ? mergeLaneLayerIds.map((l) => trackKey(l, lane))
        : [layerId !== undefined ? trackKey(layerId, lane) : lane];
    // The row reads as audible if ANY underlying track would sound; muted/soloed
    // only when ALL its tracks are (so the toggle has a clear next state).
    const audible = rowKeys.some((k) => layerControls.isTrackAudible(k));
    const muted = rowKeys.every((k) => layerControls.mutedTracks.has(k));
    const soloed = rowKeys.every((k) => layerControls.soloedTracks.has(k));
    // Set every aggregated track to `target` via the per-key toggle.
    const setAll = (
      get: (k: string) => boolean,
      toggle: (k: string) => void,
      target: boolean
    ) => {
      for (const k of rowKeys) if (get(k) !== target) toggle(k);
    };
    const drop = useMixerRowDropTarget({
      idx,
      dragFromIdx,
      dropTargetIdx,
      onDropTargetIdx,
      onMoveTrack,
      onResetDrag,
    });
    const isDragging = dragFromIdx === idx;
    const labelText = instrumentName ?? `Lane ${lane}`;
    // Stable per-bar props so the windowed bar list's scroll re-renders
    // don't bust `BarView`'s observer memo for bars that didn't move.
    const lanesMemo = React.useMemo(() => [lane], [lane]);
    const colorForLane = React.useCallback(
      (p: string) => mixer?.getInstrumentTrack(p).color,
      [mixer]
    );
    const instrumentForLane = React.useCallback(
      (p: string): Instrument =>
        structural.source.globalMetadata.instrumentMapping?.[p] ?? { kind: 'custom' },
      [structural]
    );
    return (
      <div
        className={classNames(
          styles.instrumentTrack,
          groupStart && styles.mixerRowGroupStart,
          groupEnd && styles.mixerRowGroupEnd,
          inGroup && styles.mixerRowInGroup,
          isDragging && styles.mixerRowDragging,
          drop.isDropIndicatorAbove && styles.mixerDropIndicatorAbove,
          drop.isDropIndicatorBelow && styles.mixerDropIndicatorBelow
        )}
        data-testid={`instrument-track-${lane}`}
        data-lane={lane}
        data-layer-id={layerId}
        onDragOver={drop.onDragOver}
        onDragLeave={drop.onDragLeave}
        onDrop={drop.onDrop}
      >
        <div className={styles.instrumentTrackGutter}>
          <MixerDragHandle
            idx={idx}
            onDragStartIdx={onDragStartIdx}
            onResetDrag={onResetDrag}
            ariaLabel={labelText}
          />
          <GutterResizeHandle onResizeStart={onResizeGutterStart} />
          {/* Two-row stack mirroring the audio-track row: header (label +
              overflow trigger) on top, slider + M/S on a second line
              below. */}
          <div className={styles.instrumentTrackContent}>
            <div className={styles.instrumentTrackHeader}>
              <div
                className={classNames(styles.instrumentTrackLabel, !audible && styles.musicTrackLabelDim)}
                title={instrumentName ? `${instrumentName} (lane ${lane})` : `Lane ${lane}`}
              >
                <span className={styles.gutterLane}>{lane}</span>
                {instrumentName && <span className={styles.instrumentTrackName}>{instrumentName}</span>}
              </div>
              {instrumentTrack && (
                <InstrumentTrackOverflowMenu
                  instrumentTrack={instrumentTrack}
                  trackLabel={labelText}
                />
              )}
            </div>
            <div className={styles.instrumentTrackControls}>
              <RowVolumeSlider
                value={layerControls.volumeFor(rowKeys[0])}
                onChange={(v) => {
                  for (const k of rowKeys) layerControls.onSetVolume(k, v);
                }}
                label={labelText}
              />
              <MuteButton
                active={muted}
                onToggle={() =>
                  setAll((k) => layerControls.mutedTracks.has(k), layerControls.onToggleMute, !muted)
                }
                offTitle={`Mute ${lane}`}
                onTitle={`Unmute ${lane}`}
              />
              <SoloButton
                active={soloed}
                onToggle={() =>
                  setAll((k) => layerControls.soloedTracks.has(k), layerControls.onToggleSolo, !soloed)
                }
                offTitle={`Solo ${lane}`}
                onTitle={`Unsolo ${lane}`}
              />
            </div>
          </div>
        </div>
        <div
          className={styles.barsRow}
          data-bars-row
          style={
            {
              ['--layer-beats' as string]: layerBeats,
              ['--bars-row-width' as string]: barsRowWidthSeed(structural, layerBeats),
            } as React.CSSProperties
          }
          onClick={onBarsRowClick}
          onPointerMove={onBarsRowPointerMove}
          onPointerLeave={onBarsRowPointerLeave}
        >
          {/* Lead-in label overlay floating across the negative-indexed
              bars. The bars themselves carry the hatched background
              (`.barLeadIn` / `.barLeadInLast`); this overlay just adds
              the centered "lead-in" caption. Topmost-row only
              (`showBrackets`) so the label doesn't repeat on every
              instrument row. */}
          {leadInBarsBeats > 0 && showBrackets && (
            <div
              className={styles.leadInOverlay}
              style={
                {
                  ['--lead-in-bars-beats' as string]: leadInBarsBeats,
                } as React.CSSProperties
              }
            >
              <span className={styles.leadInLabel}>lead-in</span>
            </div>
          )}
          {/* Cumulative quarter-note position of each bar's left edge
              within the layer (drives the bar's absolute left via
              `--bar-start-beat`; see `.bar` in score.module.css) is
              precomputed by `jot.barsForLane(lane)` as `startBeats`,
              so this map is just a render. */}
          <WindowedBarList
            viewport={viewport ?? null}
            laneBars={laneBars}
            startBeats={startBeats}
            lane={lane}
            config={config}
            showBrackets={showBrackets}
            laneOrder={laneOrder}
            highlightedPattern={highlightedPattern}
            onPatternClick={onPatternClick}
            isLaneAudible={() => audible}
            lanes={lanesMemo}
            colorForLane={colorForLane}
            instrumentForLane={instrumentForLane}
          />
          {rejectedForLane.map((entry, i) => {
            // The MIDI lays `leadBars` empty bar-0-sized blocks before
            // struct bar 0, so the struct bar index maps to the
            // rendered jot's bars array as `leadBars + entry.bar`.
            // Out-of-range entries are already filtered out upstream.
            const barIdx = provenance!.leadBars + entry.bar;
            if (barIdx < 0 || barIdx >= laneBars.length) return null;
            // beat_in_bar is 1-indexed in the provenance (per the
            // transcriber's OnsetCandidate convention); the CSS calc
            // expects a 0-indexed beat offset within the bar.
            const beatInBar = Math.max(0, entry.beat_in_bar - 1);
            const beatOffset = barBeatStart[barIdx] + beatInBar;
            return (
              <FilteredOnsetView
                key={`f-${entry.bar}-${i}-${entry.detected_time_sec}`}
                entry={entry}
                beatOffset={beatOffset}
                color={laneColor}
              />
            );
          })}
          <PlaceholderNoteView
            rowLane={lane}
            color={laneColor}
            noteDiameter={config.noteDiameter as number}
          />
          <DragPreviewView
            rowLane={lane}
            color={laneColor}
            noteDiameter={config.noteDiameter as number}
          />
          <Playhead onSeek={onSeek} />
        </div>
      </div>
    );
  }
);

/**
 * Tiled waveform row for one audio track, aligned to the score's bar
 * timeline. The row is split into `BEATS_PER_CHUNK`-beat windows;
 * each window renders as its own absolutely-positioned canvas tile
 * (see {@link AudioTrackWaveformChunk}). Tiling buys:
 *
 *  1. **Unbounded effective resolution.** Each tile picks its own
 *     backing-store size, so the cross-browser 16 384 px per-axis
 *     canvas cap no longer rolls a long or zoomed track off into
 *     lower-resolution rendering. `BEATS_PER_CHUNK = 4` (see
 *     `waveform_chunks.ts`) keeps the worst-case backing well under
 *     the cap (max zoom × max densityFactor × DPR 3 ≈ 8 600 px).
 *  2. **Stable chunk identity across zoom.** Chunks key on a beat-
 *     aligned bucket index, so a zoom change only resizes existing
 *     chunks via JS-recomputed `left` / `width`; React never unmounts
 *     / remounts them. No bucket-transition churn, no stretched-
 *     bitmap holdover gap.
 *  3. **Parent-driven visibility (no `IntersectionObserver`).** This
 *     component is an `observer()` that reads `scrollX` +
 *     `_viewportWidth` from `JotEditorStore` (no DOM layout reads, see
 *     AGENTS.md §5.9) and only mounts the chunks whose CSS box
 *     currently intersects the viewport (plus a prefetch margin).
 *     Off-screen chunks unmount cleanly; the worker keeps the PCM,
 *     so re-entering the viewport draws fresh from stored peaks in
 *     ~5 ms.
 *
 * Chunk layout is memoised on `jot`, so scroll / zoom re-renders of
 * this observer only walk the filtered visibility check, not the
 * structure.
 */

/**
 * Score-px margin around the visible viewport that still counts as
 * "in viewport" for chunk-mount purposes. Generous enough that a
 * moderate horizontal scroll never reveals a blank tile before its
 * draw completes (one chunk at typical zoom ≈ 4 beats × 112 px/beat
 * ≈ 450 score-px, so a 1200 px margin covers ~2-3 chunks of lookahead
 * on either side of the visible range).
 */
