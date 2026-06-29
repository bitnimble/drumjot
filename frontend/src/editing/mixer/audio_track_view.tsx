import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import React from 'react';
import type { StructuralPresenter } from 'src/editing/structure/structural_presenter';
import { AudioTrack, AudioTrackId } from 'src/editing/playback/audio_tracks';
import { jotPlayer } from 'src/editing/playback/player';
import { waveformWorker, BarSlice } from 'src/editing/playback/waveform_worker_client';
import { BarBeat, WaveformChunk, buildChunkLayout } from './waveform_chunks';
import { MuteButton, SoloButton } from 'src/ui/icon_button/icon_button';
import { TrackGutter, type TrackBusy } from 'src/editing/track_gutter/track_gutter';
import { TranscribeStoreContext, TranscribePresenterContext } from 'src/editing/transcribe/transcribe_contexts';
import { STAGE_ORDER } from 'src/editing/transcribe/transcriber';
import type { TranscribeTrackStatus } from 'src/editing/transcribe/transcribe_store';
import { JotContext, StructuralContext } from '../jot_editor_contexts';
import { UniformWaveformsContext, WaveformGridLinesContext } from './mixer_contexts';
import { MixerStoreContext } from './mixer_contexts';
import { ViewportStoreContext } from '../viewport/viewport_contexts';
import styles from './mixer.module.css';
import { Playhead } from '../playback/playhead';
import { seekFromClick } from '../score/seek';
import { BarView } from '../score/bar_view';
import { barsRowWidthSeed, intersectsBeatRange } from '../utils/windowing';
import { AudioTrackOverflowMenu } from './overflow_menus';
import { RowVolumeSlider } from './gutter_controls';
import { MixerRowDragProps, useMixerRowDropTarget, MixerDragHandle } from './mixer_drag';
import { useLiveJotPxPerBeat } from './use_live_px_per_beat';

import type { AudioTrackControls } from './mixer_controls';
// Re-exported so existing `from '.../mixer/mixer'` importers (jot_editor) keep
// working; the definitions live in the leaf `mixer_controls.ts`.

/** Fixed row height shared by the gutter (label + filename + button
 *  cluster) and the bars-row waveform on the right. Sized to fit the
 *  worst-case gutter content: a 2-line clamped name (~32px) + the
 *  filename row (~14px) + the M/S/X button cluster (~22px) + the
 *  gutter's 8px vertical padding. Bumping this also bumps the
 *  waveform height, which is a desirable side effect, taller peaks
 *  read more clearly. */
const AUDIO_TRACK_HEIGHT = 76;

/** Audio-track display name: filename with its extension stripped. */
function audioTrackLabel(filename: string): string {
  return filename.replace(/\.[^./\\]+$/, '') || filename;
}

/** Tooltip for the gutter spinner while a transcription runs: "Transcribing
 *  (stage X of Y)…", with the live substage appended when present. */
function transcribeTrackTooltip(status: TranscribeTrackStatus): string {
  if (!status.stage) return 'Transcribing…';
  const x = STAGE_ORDER.indexOf(status.stage) + 1;
  const base = `Transcribing (stage ${x} of ${STAGE_ORDER.length})…`;
  return status.substage ? `${base} ${status.substage}` : base;
}

export const AudioTrackView = observer(
  ({
    id,
    track,
    controls,
    onSeek,
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
    id: AudioTrackId;
    track: AudioTrack;
    controls: AudioTrackControls;
    onSeek: (x: number) => void;
  } & MixerRowDragProps) => {
    const structural = React.useContext(StructuralContext);
    // Layer-level total beats for the bars-row width (in beats, the
    // row's pixel width is `layerBeats × --px-per-beat` via CSS calc).
    // `layerBeats` reads off the structural layers (not pixels) so the
    // value is stable across zoom changes; pixel width updates via CSS
    // variable on the score root. The waveform canvas reads the
    // zoom-dependent pixel width itself so only IT re-renders on zoom.
    const layerBeats = structural?.layerBeats ?? 0;
    const audible = controls.isAudioTrackAudible(id);
    const muted = controls.mutedAudioTracks.has(id);
    const soloed = controls.soloedAudioTracks.has(id);
    const label = audioTrackLabel(track.filename);
    const lc = `"${track.filename}"`;
    const drop = useMixerRowDropTarget({
      idx,
      dragFromIdx,
      dropTargetIdx,
      onDropTargetIdx,
      onMoveTrack,
      onResetDrag,
    });
    const isDragging = dragFromIdx === idx;
    const mixer = React.useContext(MixerStoreContext);
    const transcribe = React.useContext(TranscribeStoreContext);
    const transcribePresenter = React.useContext(TranscribePresenterContext);
    const splitStatus = mixer?.audioTrackSplitStatuses.get(id);
    const splittingTitle =
      splitStatus?.kind === 'mix'
        ? 'Splitting into drums + backing…'
        : splitStatus?.kind === 'pieces'
          ? 'Splitting into per-instrument pieces…'
          : undefined;
    const transcribeStatus = transcribe?.trackStatuses.get(id);
    // A transcribe-in-flight wins the spinner over a split (the two won't
    // realistically overlap, but transcribe is the more recent intent).
    const busy: TrackBusy | undefined = transcribeStatus
      ? { tooltip: transcribeTrackTooltip(transcribeStatus), testId: `audio-track-transcribe-spinner-${id}` }
      : splitStatus
        ? { tooltip: splittingTitle ?? 'Splitting…', testId: `audio-track-split-spinner-${id}` }
        : undefined;
    return (
      <div
        className={classNames(
          styles.musicTrack,
          groupStart && styles.mixerRowGroupStart,
          groupEnd && styles.mixerRowGroupEnd,
          inGroup && styles.mixerRowInGroup,
          isDragging && styles.mixerRowDragging,
          drop.isDropIndicatorAbove && styles.mixerDropIndicatorAbove,
          drop.isDropIndicatorBelow && styles.mixerDropIndicatorBelow
        )}
        data-testid={`audio-track-row-${id}`}
        onDragOver={drop.onDragOver}
        onDragLeave={drop.onDragLeave}
        onDrop={drop.onDrop}
      >
        <TrackGutter
          variant="cream"
          height={AUDIO_TRACK_HEIGHT}
          dim={!audible}
          onResizeGutterStart={onResizeGutterStart}
          dragHandle={
            <MixerDragHandle
              idx={idx}
              onDragStartIdx={onDragStartIdx}
              onResetDrag={onResetDrag}
              ariaLabel={`${label} audio track`}
            />
          }
          primary={
            <span className={styles.musicTrackName} title={label}>
              {label}
            </span>
          }
          secondary={
            <span className={styles.musicTrackFile} title={track.filename}>
              {track.filename}
            </span>
          }
          busy={busy}
          overflow={
            <AudioTrackOverflowMenu
              track={track}
              trackLabel={label}
              transcribing={transcribeStatus !== undefined}
              onTranscribe={
                transcribePresenter ? () => void transcribePresenter.openAppendDialog(id) : undefined
              }
              onCancelTranscribe={
                transcribePresenter ? () => transcribePresenter.cancelTrackTranscribe(id) : undefined
              }
              onSplitFromMix={controls.onSplitFromMix}
              onSplitDrumPieces={controls.onSplitDrumPieces}
              onClear={controls.onClear}
            />
          }
          body={
            <div className={styles.musicTrackButtons}>
              <RowVolumeSlider
                value={controls.volumeFor(id)}
                onChange={(v) => controls.onSetVolume(id, v)}
                label={`${label} audio track`}
              />
              <MuteButton
                active={muted}
                onToggle={() => controls.onToggleMute(id)}
                offTitle={`Mute ${lc} audio track`}
                onTitle={`Unmute ${lc} audio track`}
                testId={`audio-track-mute-${id}`}
              />
              <SoloButton
                active={soloed}
                onToggle={() => controls.onToggleSolo(id)}
                offTitle={`Solo ${lc} audio track`}
                onTitle={`Unsolo ${lc} audio track`}
                testId={`audio-track-solo-${id}`}
              />
            </div>
          }
        />
        <div
          className={styles.musicTrackBarsRow}
          data-bars-row
          style={
            {
              ['--layer-beats' as string]: layerBeats,
              ['--bars-row-width' as string]: structural
                ? barsRowWidthSeed(structural, layerBeats)
                : '0px',
              height: AUDIO_TRACK_HEIGHT,
            } as React.CSSProperties
          }
          onClick={(e) => seekFromClick(e, onSeek)}
        >
          <AudioTrackWaveformCanvas
            structural={structural}
            track={track}
            height={AUDIO_TRACK_HEIGHT}
            dim={!audible}
            testId={`audio-track-waveform-${id}`}
          />
          <WaveformGridOverlay structural={structural} testId={`audio-track-grid-${id}`} />
          <Playhead onSeek={onSeek} />
        </div>
      </div>
    );
  }
);

/**
 * One drum-instrument row in the unified mixer; exactly one DSL lane
 * (kick, snare, hi-hat, …). Mirrors `AudioTrackView`: same gutter
 * geometry, M/S/volume controls, drag handle, bars-row + barlines +
 * beat dividers; the lane content is this lane's notes (drawn through
 * `BarView` with `lanes=[lane]`). The topmost instrument row in the
 * mixer (`showBrackets={true}`) also paints the pattern + tuplet
 * brackets so the score chrome stays visible regardless of where the
 * user has dragged the rows.
 *
 * Multi-layer jots: lanes can belong to any layer (e.g. kick lives in
 * the "Feet" layer). The bar geometry is taken from layer[0] (every layer
 * shares the same bar grid), and per-bar tracks are looked up across all
 * layers for this lane, so the row works whether the lane lives in
 * layer 0 or 1.
 */
/**
 * Bar + beat reference grid for one audio-track waveform row. Mounts only
 * when the View → Waveforms → "Bar & beat lines" toggle is on (read off
 * {@link WaveformGridLinesContext}); off ⇒ renders nothing.
 *
 * Renders one `.waveformGridBar` per visible bar in the layer's
 * beat-space, identical to the score's `.bar` positioning, so the bar
 * line (the div's right border) and the composed `.gridLayer*` overlays
 * land directly beneath the matching score bar/beat lines above. Which
 * sub-beat families show is inherited from the score root's
 * `--grid-display-*` vars (the same ones the score grid reads), so the
 * waveform mirrors whatever grid the score is showing.
 *
 * Windowed the same way as {@link AudioTrackWaveformCanvas} and the
 * instrument rows: only bars whose beat span intersects the visible
 * range mount, so a long song doesn't pay for thousands of off-screen
 * grid bars. Lead-in bars (negative index) are skipped, like the score,
 * which draws no grid over its hatched lead-in chrome.
 */
const WaveformGridOverlay = observer(
  ({ structural, testId }: { structural: StructuralPresenter | null; testId?: string }) => {
    const show = React.useContext(WaveformGridLinesContext);
    const viewport = React.useContext(ViewportStoreContext);
    const layer = structural?.layers[0];
    // Cumulative beat position per bar, recomputed only when the bar
    // structure changes (zoom-invariant, so scroll re-renders reuse it).
    const bars = React.useMemo(() => {
      if (!layer) return [];
      const out: { index: number; startBeat: number; beats: number }[] = [];
      let cursor = 0;
      for (const b of layer.bars) {
        out.push({ index: b.index, startBeat: cursor, beats: b.beats });
        cursor += b.beats;
      }
      return out;
    }, [layer]);
    if (!show || bars.length === 0) return null;
    const range = viewport?.visibleBeatRange ?? null;
    return (
      <div className={styles.waveformGrid} aria-hidden="true" data-testid={testId}>
        {bars.map((bar) => {
          if (bar.index < 0) return null;
          if (!intersectsBeatRange(range, bar.startBeat, bar.beats)) return null;
          return (
            <div
              key={bar.index}
              className={styles.waveformGridBar}
              style={
                {
                  ['--bar-start-beat' as string]: bar.startBeat,
                  ['--bar-beats' as string]: bar.beats,
                } as React.CSSProperties
              }
            >
              <div className={styles.waveformGridMain} />
              <div className={styles.waveformGridSubBeat16} />
              <div className={styles.waveformGridSubBeatQuarterTriplet} />
              <div className={styles.waveformGridSubBeatTriplet} />
              <div className={styles.waveformGridSubBeat48} />
            </div>
          );
        })}
      </div>
    );
  }
);

const CHUNK_VIEWPORT_MARGIN_PX = 1200;

const AudioTrackWaveformCanvas = observer(
  ({
    structural,
    track,
    height,
    dim,
    testId,
  }: {
    structural: StructuralPresenter | null;
    track: AudioTrack;
    height: number;
    dim: boolean;
    testId?: string;
  }) => {
    const viewport = React.useContext(ViewportStoreContext);
    const uniformWaveforms = React.useContext(UniformWaveformsContext);
    const jot = React.useContext(JotContext);
    const padBeats = structural?.config.barNotePaddingBeats ?? 0.125;
    // Waveform tint reads straight off the AudioTrack instance; the
    // class's `color` getter resolves the user override -> grouped
    // instrument inheritance -> neutral chain itself (see
    // `resolveAudioInheritedColor`), and is MobX-observable so picker
    // commits repaint chunks reactively. Always returns a `#rrggbb`
    // string the chunk worker can consume directly.
    const laneColor = track.color;
    // Beat-stable chunk layout (zoom-invariant), built off the reactive model's
    // derived fields (`renderedLayers` / `tempoSource` / `barDrift`). Memoed on
    // `jot` so scroll / zoom re-renders of this observer don't rebuild it.
    const layout = React.useMemo(
      () => (jot ? buildChunkLayout(jot) : { bars: [], totalBeats: 0, chunks: [] }),
      [jot]
    );
    const livePxPerBeat = useLiveJotPxPerBeat();

    if (!viewport || layout.chunks.length === 0) return null;

    // Visibility: derive the score-px x-range currently on screen
    // from `JotEditorStore` observables. The score uses a virtualised
    // scroll model (`.scrollViewport` translated by `(-scrollX, 0)`),
    // so `[scrollX, scrollX + viewportWidth]` is exactly the score-px
    // window the user sees. Each chunk's score-px left mirrors the
    // formula the chunk component below uses for its inline `left`
    // (`chunk.startBeat * livePxPerBeat + padBeats * livePxPerBeat`);
    // any chunk whose box intersects the viewport plus prefetch
    // margin is mounted, anything else is unmounted.
    const scrollX = viewport.scrollX;
    const viewportWidth = viewport._viewportWidth;
    if (viewportWidth <= 0 || livePxPerBeat <= 0) return null;
    const visibleLeft = scrollX - CHUNK_VIEWPORT_MARGIN_PX;
    const visibleRight = scrollX + viewportWidth + CHUNK_VIEWPORT_MARGIN_PX;
    const padPx = padBeats * livePxPerBeat;

    const visibleChunks: WaveformChunk[] = [];
    for (const c of layout.chunks) {
      const left = c.startBeat * livePxPerBeat + padPx;
      const right = left + c.totalBeats * livePxPerBeat;
      if (right > visibleLeft && left < visibleRight) visibleChunks.push(c);
    }
    if (visibleChunks.length === 0) return null;

    // Per-track amplitude scale for uniform mode (resolved once at
    // track registration, identical for every chunk of this track,
    // so neighbouring chunks render at the same vertical scale and no
    // amplitude seam shows at the chunk boundary).
    const ampScale = uniformWaveforms ? waveformWorker.getAmpScale(track.id) : 1;

    return (
      <>
        {visibleChunks.map((chunk, i) => (
          <AudioTrackWaveformChunk
            key={chunk.key}
            track={track}
            chunk={chunk}
            bars={layout.bars}
            height={height}
            dim={dim}
            laneColor={laneColor}
            ampScale={ampScale}
            testId={i === 0 ? testId : undefined}
          />
        ))}
      </>
    );
  }
);

/**
 * One tile in the tiled waveform row. Owns the `<canvas>` and its
 * rasterised bitmap (sized to `chunk.totalBeats × livePxPerBeat`,
 * snapped to integer CSS px).
 *
 * Visibility is decided by the parent
 * (`AudioTrackWaveformCanvas`), which only mounts chunks intersecting
 * the viewport; so mount = visible, and there's no
 * `IntersectionObserver` round-trip to wait through. The first render
 * after mount draws immediately so a newly-visible chunk paints on
 * the same frame as the parent's visibility decision; subsequent
 * renders triggered by zoom / `songLeadInSec` / etc. rAF-coalesce so a
 * sustained wheel-zoom gesture triggers at most one worker call per
 * displayed frame.
 */
const AudioTrackWaveformChunk = observer(
  ({
    track,
    chunk,
    bars,
    height,
    dim,
    laneColor,
    ampScale,
    testId,
  }: {
    track: AudioTrack;
    chunk: WaveformChunk;
    bars: BarBeat[];
    height: number;
    dim: boolean;
    laneColor: string | undefined;
    ampScale: number;
    testId?: string;
  }) => {
    const canvasRef = React.useRef<HTMLCanvasElement>(null);
    // Live drum↔audio offset; chunks re-render (and so re-rasterise
    // on the next rAF) when the user nudges the Offset control.
    const songLeadInSec = jotPlayer.songLeadInSec;
    const livePxPerBeat = useLiveJotPxPerBeat();
    const padBeats = React.useContext(StructuralContext)?.config.barNotePaddingBeats ?? 0.125;
    // Globally-unique worker-side slot identifier for this tile.
    // `chunk.key` alone collides across audio tracks (it's
    // `startBeat / BEATS_PER_CHUNK`, defined per-layer); prefixing
    // with `track.id` (a string per audio track) disambiguates.
    const chunkKey = `${track.id}:${chunk.key}`;

    // Snap the chunk's CSS left / width to integer CSS pixels in JS so
    // the canvas's backing-store width (= cssWidth × dpr) and the peak
    // buffer length (= 2 × cssWidth) are both whole integers, and so
    // adjacent chunks share an *exactly* aligned boundary (chunk N+1's
    // left = chunk N's right by construction, no asymmetric rounding
    // gap or overlap). Without this, each chunk's CSS width came from
    // `round(right_edge) - round(left_edge)` in CSS, and the two edges
    // would round in different directions for adjacent chunks (one to
    // -0.5, one to +0.5), leaving each chunk's canvas bitmap stretched
    // by a slightly different ratio, which renders as a visible
    // brightness / density step at the chunk boundary. Same snapped
    // width feeds the canvas backing-store, the inline CSS width, and
    // the peak buffer length so all three agree to the pixel.
    const chunkLayout = React.useMemo(() => {
      const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
      const padPx = Math.round(padBeats * livePxPerBeat * dpr) / dpr;
      const leftRaw = chunk.startBeat * livePxPerBeat + padPx;
      const rightRaw = leftRaw + chunk.totalBeats * livePxPerBeat;
      const left = Math.round(leftRaw);
      const right = Math.round(rightRaw);
      return { left, width: Math.max(0, right - left) };
    }, [chunk.startBeat, chunk.totalBeats, livePxPerBeat, padBeats]);

    // Transfer control of the `<canvas>` to the worker once on mount;
    // release on unmount. After this point the main thread can no
    // longer draw into the canvas (any attempt throws); the worker
    // owns the bitmap, sized via `canvas.width` / `canvas.height` set
    // on its `OffscreenCanvas` handle inside `renderChunk`. CSS box
    // dimensions are still controlled here via inline `style.left` /
    // `style.width` (CSS properties of the `<canvas>` element are
    // separate from the backing bitmap).
    React.useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      if (typeof canvas.transferControlToOffscreen !== 'function') {
        console.warn(
          '[mixer] OffscreenCanvas not supported; waveform chunk will not render',
        );
        return;
      }
      const offscreen = canvas.transferControlToOffscreen();
      waveformWorker.attachChunk(chunkKey, offscreen, track.id);
      return () => {
        waveformWorker.releaseChunk(chunkKey);
      };
    }, [chunkKey, track.id]);

    // First render after mount paints immediately so a newly-visible
    // chunk shows up on the same frame as the parent's visibility
    // decision; subsequent paints (zoom tick, songLeadInSec change, etc.)
    // rAF-coalesce so a sustained wheel-zoom gesture triggers at most
    // one worker call per displayed frame. The paint itself is
    // fire-and-forget: the worker computes peaks and paints into the
    // chunk's `OffscreenCanvas` directly, no bytes cross back to the
    // main thread.
    const isFirstDrawRef = React.useRef(true);

    React.useEffect(() => {
      if (chunk.totalBeats <= 0 || livePxPerBeat <= 0) return;
      const widthPx = chunkLayout.width;
      if (widthPx <= 0) return;
      const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
      // `BEATS_PER_CHUNK` is sized so the worst-case backing
      // dimensions stay well under the 16 384 px cross-browser
      // canvas cap; this clamp is defensive only and shouldn't fire
      // in normal use.
      const MAX_CANVAS_DIM = 16384;
      // `widthPx` is integer (from `chunkLayout.width`) so
      // `widthPx * dpr` is also integer, no `Math.floor` needed, and
      // the backing-store width is exactly `cssWidth × dpr`, so
      // `image-rendering: pixelated` displays the bitmap 1:1 with no
      // stretch.
      const backingW = Math.min(Math.max(1, widthPx * dpr), MAX_CANVAS_DIM);
      const backingH = Math.min(Math.max(1, Math.floor(height * dpr)), MAX_CANVAS_DIM);
      // Effective per-beat scale for this bitmap. Differs from
      // `livePxPerBeat` by at most ~0.5 CSS px / chunk.totalBeats
      // (because `widthPx` is rounded to integer above); using the
      // bitmap's actual per-beat ratio for the bar slice mapping
      // keeps each column's audio time aligned to the chunk's CSS
      // box, so a transient at beat B in the source audio lands
      // exactly under beat B in the snapped chunk geometry.
      const renderedScale = widthPx / chunk.totalBeats;
      // Bar slices in chunk-local pixel coordinates: bars to the left
      // of the chunk get a negative `x`, bars to the right get `x >=
      // widthPx`; the worker's clamp drops both groups without an
      // explicit filter on our side.
      const barSlices: BarSlice[] = bars.map((b, idx) => ({
        x: (b.startBeat - chunk.startBeat) * renderedScale,
        width: b.beats * renderedScale,
        startSec: b.startSec,
        durationSec: b.durationSec,
        driftSec: b.driftSec,
        // The next bar's drift, so the worker can map this bar's pixels onto
        // its REAL audio span (its width covers `durationSec + (nextDrift -
        // drift)` of recording). Last bar: no next → same drift (no stretch).
        nextDriftSec: bars[idx + 1]?.driftSec ?? b.driftSec,
      }));
      const fire = () => {
        waveformWorker.renderChunk(
          chunkKey,
          barSlices,
          widthPx,
          height,
          backingW,
          backingH,
          songLeadInSec,
          laneColor ?? '#5BA8E8',
          ampScale,
        );
      };
      if (isFirstDrawRef.current) {
        isFirstDrawRef.current = false;
        fire();
        return;
      }
      const id = requestAnimationFrame(fire);
      return () => cancelAnimationFrame(id);
    }, [
      chunkKey,
      chunk,
      bars,
      height,
      songLeadInSec,
      livePxPerBeat,
      laneColor,
      ampScale,
      chunkLayout.width,
    ]);

    // Canvas `left` / `width` come from `chunkLayout` (JS-snapped to
    // integer CSS px). Inline styles override CSS, and we need the
    // canvas's CSS box to match the bitmap's pixel count
    // (`widthPx * dpr` backing) exactly so adjacent chunks share a
    // pixel-perfect boundary and `image-rendering: pixelated`
    // displays the bitmap 1:1. During the one-rAF gap between a zoom
    // event and the next rasterisation, the chunk's CSS width grows
    // with `livePxPerBeat` while the bitmap's intrinsic size is
    // still at the previous scale; the canvas element scales the
    // bitmap nearest-neighbour (via image-rendering: pixelated)
    // until the rAF redraw catches up.
    return (
      <canvas
        ref={canvasRef}
        className={classNames(styles.musicTrackWaveformChunk, dim && styles.musicTrackWaveformDim)}
        style={
          {
            height,
            left: `${chunkLayout.left}px`,
            width: `${chunkLayout.width}px`,
          } as React.CSSProperties
        }
        data-testid={testId}
      />
    );
  }
);
