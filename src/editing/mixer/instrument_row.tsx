import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { Instrument } from 'src/schema/dsl/dsl';
import type { StructBar } from 'src/editing/structure/structure_store';
import { ViewConfig } from 'src/editing/viewport/view_config';
import { InstrumentTrack } from 'src/editing/tracks/tracks';
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
import { BarView } from '../score/bar_view';
import { ViewportStore } from '../viewport/viewport_store';
import { barsRowWidthSeed, intersectsBeatRange } from '../utils/windowing';
import { InstrumentRowOverflowMenu } from './overflow_menus';
import { RowVolumeSlider } from './gutter_controls';
import { MixerRowDragProps, useMixerRowDropTarget, MixerDragHandle } from './mixer_drag';
import type { VoiceControls } from './mixer_controls';

/**
 * The windowed bar list for one instrument row. Split out of
 * {@link InstrumentRow} so the only thing that re-renders on a scroll /
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
 * `pitches` / `colorForPitch`), so `BarView`'s `observer` memo holds and
 * an unchanged visible bar pays nothing on a scroll tick that doesn't
 * move the window, only newly-revealed bars mount.
 */
const WindowedBarList = observer(function WindowedBarList({
  viewport,
  pitchBars,
  startBeats,
  pitch,
  config,
  showBrackets,
  pitchOrder,
  highlightedPattern,
  onPatternClick,
  isPitchAudible,
  pitches,
  colorForPitch,
  instrumentForPitch,
}: {
  viewport: ViewportStore | null;
  pitchBars: readonly StructBar[];
  startBeats: readonly number[];
  pitch: string;
  config: ViewConfig;
  showBrackets: boolean;
  pitchOrder: readonly string[];
  highlightedPattern: string | undefined;
  onPatternClick: (name: string) => void;
  isPitchAudible: (pitch: string) => boolean;
  pitches: string[];
  colorForPitch: (pitch: string) => string | undefined;
  instrumentForPitch: (pitch: string) => Instrument;
}) {
  const range = viewport?.visibleBeatRange ?? null;
  return (
    <>
      {pitchBars.map((bar, i) => {
        const startBeat = startBeats[i];
        if (!intersectsBeatRange(range, startBeat, bar.beats)) return null;
        return (
          <BarView
            key={bar.index}
            bar={bar}
            barStartBeat={startBeat}
            pitches={pitches}
            config={config}
            isAnacrusis={bar.index === 0}
            highlightedPattern={highlightedPattern}
            onPatternClick={onPatternClick}
            isPitchAudible={isPitchAudible}
            showBrackets={showBrackets}
            rowPitch={pitch}
            pitchOrder={pitchOrder}
            colorForPitch={colorForPitch}
            instrumentForPitch={instrumentForPitch}
          />
        );
      })}
    </>
  );
});

export const InstrumentRow = observer(
  ({
    pitch,
    config,
    showBrackets,
    pitchOrder,
    highlightedPattern,
    onPatternClick,
    onSeek,
    voiceControls,
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
    pitch: string;
    config: ViewConfig;
    showBrackets: boolean;
    pitchOrder: readonly string[];
    highlightedPattern: string | undefined;
    onPatternClick: (name: string) => void;
    onSeek: (x: number) => void;
    voiceControls: VoiceControls;
  } & MixerRowDragProps) => {
    const structural = React.useContext(StructuralContext);
    const voice0 = structural?.primaryVoice;
    if (!structural || !voice0) return null;
    const trackHeight = config.trackHeight as number;
    // Per-pitch derived data (bars, voice-wide totals, cumulative
    // bar-start offsets, label color/instrument name); all memoised on
    // the jot via `barsForPitch(pitch)`, so each row reads its slice
    // from the MobX cache instead of recomputing on every render.
    // `barBeatStart` and `startBeats` are the same array; the keyed
    // names just disambiguate the two historical use sites.
    const {
      bars: pitchBars,
      voiceBeats,
      leadInBarsBeats,
      barBeatStart,
      startBeats,
      instrumentName,
    } = structural.barsForPitch(pitch);
    // Resolve the row's note colour through the store-owned
    // `InstrumentTrack`. The structural `barsForPitch().pitchColor` is
    // now palette-only (overrides moved off the jot in the colour-
    // picker refactor), so layering happens here: the InstrumentTrack
    // returns the override if set, otherwise the jot's palette default,
    // otherwise the neutral fallback grey. Reading it inside this
    // observer is the dependency that drives a row re-render when the
    // user picks a new colour.
    const mixer = React.useContext(MixerStoreContext);
    const viewport = React.useContext(ViewportStoreContext);
    const instrumentTrack = mixer?.getInstrumentTrack(pitch);
    const pitchColor = instrumentTrack?.color ?? 'var(--color-text-faint-strong)';

    // Filtered-onset ghost overlays (debug bundle + checkbox gated).
    // Resolve once per row so the per-entry render below is just a map.
    const provenance = React.useContext(NoteProvenanceContext);
    const showFiltered = provenance?.showFiltered ?? false;
    const rejectedForPitch = showFiltered ? (provenance!.rejectedByPitch.get(pitch) ?? []) : [];

    const audible = voiceControls.isPitchAudible(pitch);
    const muted = voiceControls.mutedPitches.has(pitch);
    const soloed = voiceControls.soloedPitches.has(pitch);
    const drop = useMixerRowDropTarget({
      idx,
      dragFromIdx,
      dropTargetIdx,
      onDropTargetIdx,
      onMoveTrack,
      onResetDrag,
    });
    const isDragging = dragFromIdx === idx;
    const labelText = instrumentName ?? `Pitch ${pitch}`;
    // Stable per-bar props so the windowed bar list's scroll re-renders
    // don't bust `BarView`'s observer memo for bars that didn't move.
    const pitchesMemo = React.useMemo(() => [pitch], [pitch]);
    const colorForPitch = React.useCallback(
      (p: string) => mixer?.getInstrumentTrack(p).color,
      [mixer]
    );
    const instrumentForPitch = React.useCallback(
      (p: string): Instrument =>
        structural.source.globalMetadata.instrumentMapping?.[p] ?? { kind: 'custom' },
      [structural]
    );
    return (
      <div
        className={classNames(
          styles.instrumentRow,
          groupStart && styles.mixerRowGroupStart,
          groupEnd && styles.mixerRowGroupEnd,
          inGroup && styles.mixerRowInGroup,
          isDragging && styles.mixerRowDragging,
          drop.isDropIndicatorAbove && styles.mixerDropIndicatorAbove,
          drop.isDropIndicatorBelow && styles.mixerDropIndicatorBelow
        )}
        data-testid={`instrument-row-${pitch}`}
        onDragOver={drop.onDragOver}
        onDragLeave={drop.onDragLeave}
        onDrop={drop.onDrop}
      >
        <div className={styles.instrumentRowGutter}>
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
          <div className={styles.instrumentRowContent}>
            <div className={styles.instrumentRowHeader}>
              <div
                className={classNames(styles.instrumentRowLabel, !audible && styles.musicTrackLabelDim)}
                title={instrumentName ? `${instrumentName} (pitch ${pitch})` : `Pitch ${pitch}`}
              >
                <span className={styles.gutterPitch}>{pitch}</span>
                {instrumentName && <span className={styles.instrumentRowName}>{instrumentName}</span>}
              </div>
              {instrumentTrack && (
                <InstrumentRowOverflowMenu
                  instrumentTrack={instrumentTrack}
                  trackLabel={labelText}
                />
              )}
            </div>
            <div className={styles.instrumentRowControls}>
              <RowVolumeSlider
                value={voiceControls.volumeFor(pitch)}
                onChange={(v) => voiceControls.onSetVolume(pitch, v)}
                label={labelText}
              />
              <MuteButton
                active={muted}
                onToggle={() => voiceControls.onToggleMute(pitch)}
                offTitle={`Mute ${pitch}`}
                onTitle={`Unmute ${pitch}`}
              />
              <SoloButton
                active={soloed}
                onToggle={() => voiceControls.onToggleSolo(pitch)}
                offTitle={`Solo ${pitch}`}
                onTitle={`Unsolo ${pitch}`}
              />
            </div>
          </div>
        </div>
        <div
          className={styles.barsRow}
          data-bars-row
          style={
            {
              ['--voice-beats' as string]: voiceBeats,
              ['--bars-row-width' as string]: barsRowWidthSeed(structural, voiceBeats),
            } as React.CSSProperties
          }
          onClick={(e) => seekFromClick(e, onSeek)}
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
              within the voice (drives the bar's absolute left via
              `--bar-start-beat`; see `.bar` in score.module.css) is
              precomputed by `jot.barsForPitch(pitch)` as `startBeats`,
              so this map is just a render. */}
          <WindowedBarList
            viewport={viewport ?? null}
            pitchBars={pitchBars}
            startBeats={startBeats}
            pitch={pitch}
            config={config}
            showBrackets={showBrackets}
            pitchOrder={pitchOrder}
            highlightedPattern={highlightedPattern}
            onPatternClick={onPatternClick}
            isPitchAudible={voiceControls.isPitchAudible}
            pitches={pitchesMemo}
            colorForPitch={colorForPitch}
            instrumentForPitch={instrumentForPitch}
          />
          {rejectedForPitch.map((entry, i) => {
            // The MIDI lays `leadBars` empty bar-0-sized blocks before
            // struct bar 0, so the struct bar index maps to the
            // rendered jot's bars array as `leadBars + entry.bar`.
            // Out-of-range entries are already filtered out upstream.
            const barIdx = provenance!.leadBars + entry.bar;
            if (barIdx < 0 || barIdx >= pitchBars.length) return null;
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
                color={pitchColor}
                trackHeight={trackHeight as number}
              />
            );
          })}
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
