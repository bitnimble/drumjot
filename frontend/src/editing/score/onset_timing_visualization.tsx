import { observer } from 'mobx-react-lite';
import React from 'react';
import { NoteProvenanceEntry } from 'src/editing/provenance/debug_zip';
import { AudioTrack } from 'src/editing/playback/audio_tracks';
import { jotPlayer } from 'src/editing/playback/player';
import { waveformWorker } from 'src/editing/playback/waveform_worker_client';
import { WAVEFORM_PAINT_COLOR } from '../utils/waveform_color';
import type { RenderedNoteContext } from './provenance_timing_math';
import { formatSignedMs, formatSignedSlots } from './provenance_format';
import styles from './score.module.css';

/** Total pixel width of the timing-visualization panel, wide enough to
 * resolve a ±150 ms window without crowding the bars in the diff
 * rows, narrow enough that the parent label popover doesn't run off
 * the score for selections near the right edge. */
const TIMING_VIZ_WIDTH = 320;
const TIMING_VIZ_WAVE_HEIGHT = 40;
const TIMING_VIZ_ROW_HEIGHT = 18;
/** Minimum half-window so a near-zero snap/alignment still shows
 * waveform context around the detected onset; otherwise the window
 * collapses to a single pixel. Wider than the snap deltas typically
 * encountered so the snippet reads as "where in the song are we"
 * rather than a tight zoom on the transient alone. */
const TIMING_VIZ_MIN_HALF_WINDOW_SEC = 0.3;

/**
 * One shift in the detected → final chain. Built by
 * {@link NoteProvenanceDetails} from the per-onset provenance entry +
 * the file-level alignment fields + the live drum-offset slider, then
 * handed to {@link OnsetTimingVisualization} which renders the
 * chain in pipeline order with each stage occupying its own coloured
 * bar. Allowing the parent to assemble the list (instead of
 * hard-coding every named stage in the visualization) is what lets
 * the popup faithfully decompose multi-pass shifts (the four quantise
 * passes, the coarse/fine alignment split) into one bar each.
 */
export type StageShift = {
  /** Stable React key. */
  key: string;
  /** Human-readable label shown inside the bar + on the textual row. */
  label: string;
  /** CSS class name from `score.module.css` driving the bar's colour. */
  className: string;
  /** Audio-time displacement this stage contributed, in seconds. Zero
   * shifts are suppressed by the visualization. */
  deltaSec: number;
  /** Same delta expressed in ts-beats of the original bar. `undefined`
   * when the bar's tempo can't be resolved. */
  deltaBeats?: number;
  /** Same delta as an exact integer slot count, when the stage's native
   * unit is slots (per-pass quantise + MIDI 1/48 snap). `undefined` for
   * stages whose native unit is seconds (alignment, anchor drift, drum
   * offset). */
  deltaSlots?: number;
  /** When set, render this stage as an independent bar anchored at the
   * given audio-time position (extending by `deltaSec`) instead of
   * cumulating into the chain. Used by frame-shift stages (envelope
   * refine, beat-grid alignment): they don't move the onset's
   * audio-time position, they shift the reference frame the chain is
   * measured against, so they sit visually beside the chain rather
   * than threading through it. See the `barGridFrameShiftSec` block
   * in {@link NoteProvenanceDetails} for the full rationale on why
   * these are kept out of `accountedSec`. */
  anchorSec?: number;
};

/**
 * Per-onset timing diagram, rendered above the textual {@link
 * NoteProvenanceDetails} grid. Layers, top to bottom:
 *
 *   1. Header, bar number + the audio-time window the snippet spans.
 *   2. Waveform canvas (drawn from the first loaded audio track) with
 *      overlay vertical lines for each bar boundary inside the window,
 *      the original detected onset, and the final-position landing
 *      (post-quantization, post-alignment).
 *   3. One or more diff rows: each row's bar starts at the end of the
 *      previous row's bar (the detected onset for row 1) and extends
 *      by that stage's delta in seconds. Bar colour encodes the stage;
 *      label inside reads stage + delta in beats and ms.
 *
 * Skips the waveform row when no audio is loaded so the panel still
 * conveys the deltas; if no rendering context (filtered onsets) is
 * provided the parent suppresses this component entirely.
 */
export const OnsetTimingVisualization = observer(
  ({
    entry,
    rendered,
    stages,
    finalSec,
    displayedBarIndex,
    gridDivision,
  }: {
    entry: NoteProvenanceEntry;
    rendered: RenderedNoteContext;
    /** Grid density (1/N-of-whole-note) of the jot, for slot readouts. */
    gridDivision: number;
    /** Ordered chain of shifts from the detected onset to where the
     * score plays it now. Each stage's bar starts at the previous
     * stage's end; the residual between cumulative-stages and
     * `finalSec` becomes the trailing "Unknown drift" bar. The parent
     * is responsible for ordering by pipeline stage (raw model →
     * envelope refine → beat-grid alignment → quantise passes → MIDI
     * snap → bar anchor → drum offset). */
    stages: StageShift[];
    /** Final-position audio time (post-quantization, post-alignment).
     * `undefined` when the playback timeline can't resolve the bar's
     * start. */
    finalSec: number | undefined;
    displayedBarIndex: number | undefined;
  }) => {
    const canvasRef = React.useRef<HTMLCanvasElement>(null);

    const detectedSec = entry.detected_time_sec;

    // Window: span the detected onset + every cumulative chain position
    // + the final landing, plus a small symmetric pad. Clamped to a
    // minimum half-window so a zero-shift onset still gets visible
    // waveform context.
    const stagePositions: number[] = [detectedSec];
    let chainAcc = detectedSec;
    for (const s of stages) {
      if (s.anchorSec !== undefined) {
        // Frame-shift stage: bar lives at a fixed audio-time position,
        // doesn't advance the chain. Include both edges so the window
        // expands to keep them visible.
        stagePositions.push(s.anchorSec, s.anchorSec + s.deltaSec);
      } else {
        chainAcc += s.deltaSec;
        stagePositions.push(chainAcc);
      }
    }
    if (finalSec !== undefined) stagePositions.push(finalSec);
    const minPos = Math.min(...stagePositions);
    const maxPos = Math.max(...stagePositions);
    const span = maxPos - minPos;
    const halfWindow = Math.max(TIMING_VIZ_MIN_HALF_WINDOW_SEC, span * 1.2);
    const center = (minPos + maxPos) / 2;
    const windowStart = center - halfWindow;
    const windowEnd = center + halfWindow;
    const windowDur = windowEnd - windowStart;

    // Returns the audio-time `t`'s position inside the snippet as a
    // percentage of the timing-viz row width. Percentages (not pixels)
    // for every overlay (bar boundaries, grid ticks, detected/final
    // lines, diff-row bars, inverted-text clip) so the waveform and
    // every aligned overlay stretch with the popover instead of being
    // pinned to a 320 px box on the left.
    const timeToPct = (t: number): number => ((t - windowStart) / windowDur) * 100;

    // Pick the audio track most likely to expose this onset clearly.
    // The debug bundle's manifest carries an authoritative lane →
    // audio-filename map (set up server-side in
    // `transcriber/app/debug_bundle.py`); the isolated stem for the
    // note's lane is the right source; it isolates the drum we're
    // inspecting from the rest of the kit. Fallback chain when the
    // mapping doesn't resolve (legacy bundle, manual file load, etc.):
    //   1. Any other mapped stem, still a per-lane isolated source.
    //   2. Any loaded track other than `no_drums.mp3`; `no_drums` by
    //      definition has the drum content removed and shows nothing.
    //   3. Whatever's loaded.
    // `undefined` when no audio is loaded; the row collapses to a
    // "(no audio loaded)" placeholder in that case.
    //
    // O(1) per-onset cost: every `audioTracks` walk has been hoisted
    // off the per-onset path. `jotPlayer.audioTracksByFilename` is a
    // MobX computed shared across every visible onset; the only thing
    // we recompute here is the small mapping-derived state, memoized
    // on the per-onset inputs.
    const audioTracksByFilename = jotPlayer.audioTracksByFilename;
    const mapping = rendered.provenance.audioFilenameByLane;
    const audioTrack = React.useMemo<AudioTrack | undefined>(() => {
      if (audioTracksByFilename.size === 0) return undefined;
      const wantedFilename = mapping.get(entry.lane);
      if (wantedFilename) {
        const exact = audioTracksByFilename.get(wantedFilename.toLowerCase());
        if (exact) return exact;
      }
      // Any other mapped per-lane stem; skip `no_drums`, which is the
      // backing track and won't show drum hits.
      for (const [key, filename] of mapping.entries()) {
        if (key === 'no_drums') continue;
        const t = audioTracksByFilename.get(filename.toLowerCase());
        if (t) return t;
      }
      const noDrumsName = mapping.get('no_drums')?.toLowerCase();
      for (const t of audioTracksByFilename.values()) {
        if (t.filename.toLowerCase() !== noDrumsName) return t;
      }
      // Last fallback: whatever's loaded (first by insertion order).
      return audioTracksByFilename.values().next().value;
    }, [audioTracksByFilename, mapping, entry.lane]);

    // Always uniform-normalise the snippet: the debug-details popover is
    // a fixed-size inspector window the user can't enlarge to compensate
    // for a quiet recording, so the global mixer's `uniformWaveforms`
    // toggle isn't sufficient. Read here so MobX subscribes; thread into
    // the effect via deps so a late-arriving scale repaints.
    const ampScale = audioTrack ? waveformWorker.getAmpScale(audioTrack.id) : 1;

    React.useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas || !audioTrack) return;
      let cancelled = false;
      (async () => {
        // Off-main-thread peak compute (the worker holds a copy of the
        // track's PCM; this call only ships the window + width). Canvas
        // keeps its previous paint during the worker round-trip.
        let peaks: Float32Array;
        try {
          peaks = await waveformWorker.computeWindow(
            audioTrack.id,
            windowStart,
            windowDur,
            TIMING_VIZ_WIDTH
          );
        } catch (err) {
          if (!cancelled) console.warn('[score] timing-viz peaks failed:', err);
          return;
        }
        if (cancelled) return;
        const c = canvasRef.current;
        if (!c) return;
        const dpr = window.devicePixelRatio || 1;
        c.width = Math.max(1, Math.floor(TIMING_VIZ_WIDTH * dpr));
        c.height = Math.max(1, Math.floor(TIMING_VIZ_WAVE_HEIGHT * dpr));
        const ctx = c.getContext('2d');
        if (!ctx) return;
        ctx.imageSmoothingEnabled = false;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, TIMING_VIZ_WIDTH, TIMING_VIZ_WAVE_HEIGHT);
        const mid = TIMING_VIZ_WAVE_HEIGHT / 2;
        const scale = mid * 0.9;
        ctx.fillStyle = WAVEFORM_PAINT_COLOR;
        // Always paint at least a 1 px centerline per column (no skip-zero
        // shortcut) so silent ranges render as a continuous ground line
        // instead of empty gaps; in the long mixer waveform the skip is
        // fine because silent regions fade into the chrome around them,
        // but in the snippet it reads as broken rendering.
        for (let p = 0; p < TIMING_VIZ_WIDTH; p++) {
          // Clamp post-scale so an aggressive normalisation on a track
          // with a couple of full-scale transients doesn't shoot peaks
          // off the row's top/bottom edge.
          const mn = Math.max(-1, peaks[p * 2] * ampScale);
          const mx = Math.min(1, peaks[p * 2 + 1] * ampScale);
          const y0 = mid - mx * scale;
          const y1 = mid - mn * scale;
          ctx.fillRect(p, y0, 1, Math.max(1, y1 - y0));
        }
      })();
      return () => {
        cancelled = true;
      };
    }, [audioTrack, windowStart, windowDur, ampScale]);

    // Bar boundaries + per-grid-slot subdivisions visible inside the
    // window. Walk the timeline once; boundaries sit at each bar's
    // audio-time start (labeled with the rendered bar index so the
    // operator can orient on the snippet), and within each visible bar
    // we drop a subtle tick at every grid slot so the operator can see
    // where the detected/final onset positions land against the jot's
    // 1/N grid. Per-slot timing assumes constant tempo within the bar
    // (matches `tempo.ts`'s uniform-spread when no mid-bar bpm block is
    // present, see AGENTS.md §8.7); this is a visual aid, not a
    // precise timing readout.
    const timeline = jotPlayer.timeline;
    const songLeadInSec = jotPlayer.songLeadInSec;
    // Per-overlay positions are stored as percentage values (`pct`,
    // 0..100) of the timing-viz row width, not pixels. The waveform +
    // every aligned overlay stretch with the popover; using pixels
    // would pin them to a fixed 320 px box while the labels around
    // them grew.
    const barBoundaries: { pct: number; label: number | null }[] = [];
    const gridLines: number[] = [];
    if (timeline.rendered) {
      const structBars = timeline.rendered.layers[0]?.bars ?? [];
      for (let i = 0; i < timeline.bars.length; i++) {
        const bar = timeline.bars[i]!;
        const structBar = structBars[i];
        const barStart = bar.startSec - songLeadInSec;
        const barEnd = barStart + bar.durationSec;
        if (barStart >= windowStart && barStart <= windowEnd) {
          barBoundaries.push({ pct: timeToPct(barStart), label: structBar?.index ?? null });
        }
        if (
          !structBar ||
          bar.durationSec <= 0 ||
          structBar.tsUnit <= 0 ||
          barEnd < windowStart ||
          barStart > windowEnd
        ) {
          continue;
        }
        const slots = Math.round((structBar.tsCount * gridDivision) / structBar.tsUnit);
        if (slots <= 0) continue;
        const slotDur = bar.durationSec / slots;
        // j starts at 1 so the bar-boundary line (j=0) isn't doubled up.
        for (let j = 1; j < slots; j++) {
          const t = barStart + j * slotDur;
          if (t >= windowStart && t <= windowEnd) gridLines.push(timeToPct(t));
        }
      }
    }

    const detectedPct = timeToPct(detectedSec);
    const finalPct = finalSec !== undefined ? timeToPct(finalSec) : undefined;

    // Each diff row: a coloured bar from `anchorPct` to `endPct` (always
    // drawn left-to-right via min/abs), with an inline label. The chain
    // anchors at the detected line and advances by each stage's
    // `deltaSec`; the trailing residual (`finalSec` minus the cumulative
    // sum) surfaces as the "Unknown drift" bar so every gap between
    // named stages and the actual final landing remains visible rather
    // than silently absorbed. Positions are in row-width percentages so
    // they stay aligned with the waveform's overlays at any popover
    // width.
    type DiffRow = {
      key: string;
      label: string;
      deltaBeats: number | undefined;
      deltaSlots: number | undefined;
      deltaSec: number;
      anchorPct: number;
      endPct: number;
      className: string;
    };
    const diffRows: DiffRow[] = [];
    let cursorSec = detectedSec;
    let cursorPct = detectedPct;
    for (const s of stages) {
      if (Math.abs(s.deltaSec) < 1e-9) continue;
      let anchorSec: number;
      let anchorPct: number;
      let endSec: number;
      if (s.anchorSec !== undefined) {
        // Frame-shift bar: positioned at the stage's explicit anchor
        // (env refine: raw model peak → detected; alignment: by
        // convention from the detected line). Doesn't advance the
        // chain cursor, the chain math (and the trailing
        // unknown-drift residual) only sees genuine audio-time
        // displacements.
        anchorSec = s.anchorSec;
        anchorPct = timeToPct(anchorSec);
        endSec = anchorSec + s.deltaSec;
      } else {
        anchorSec = cursorSec;
        anchorPct = cursorPct;
        endSec = cursorSec + s.deltaSec;
      }
      const endPct = timeToPct(endSec);
      diffRows.push({
        key: s.key,
        label: s.label,
        deltaBeats: s.deltaBeats,
        deltaSlots: s.deltaSlots,
        deltaSec: s.deltaSec,
        anchorPct,
        endPct,
        className: s.className,
      });
      if (s.anchorSec === undefined) {
        cursorSec = endSec;
        cursorPct = endPct;
      }
    }
    if (finalSec !== undefined && finalPct !== undefined) {
      const unknownSec = finalSec - cursorSec;
      if (Math.abs(unknownSec) > 1e-6) {
        diffRows.push({
          key: 'unknown',
          label: 'Unknown drift source',
          deltaBeats: undefined,
          deltaSlots: undefined,
          deltaSec: unknownSec,
          anchorPct: cursorPct,
          endPct: finalPct,
          className: styles.timingVizDiffBarUnknown,
        });
      }
    }

    const renderSignedBeats = (b: number) => `${b >= 0 ? '+' : ''}${b.toFixed(3)} beats`;
    const renderSignedSlots = (slots: number) => formatSignedSlots(slots, gridDivision);
    const renderSignedMs = (ms: number) => formatSignedMs(ms);

    return (
      <div className={styles.timingViz}>
        <div className={styles.timingVizHeader}>
          {displayedBarIndex !== undefined && (
            <span className={styles.timingVizHeaderBar}>bar {displayedBarIndex}</span>
          )}
          <span className={styles.timingVizHeaderRange}>
            {windowStart.toFixed(3)}s ─ {windowEnd.toFixed(3)}s
          </span>
        </div>
        <div
          className={styles.timingVizWaveformRow}
          style={{ height: TIMING_VIZ_WAVE_HEIGHT } as React.CSSProperties}
        >
          {audioTrack ? (
            <canvas
              ref={canvasRef}
              className={styles.timingVizCanvas}
              // Canvas backing buffer stays at `TIMING_VIZ_WIDTH × dpr`
              // (set in the effect); CSS stretches it to the row's
              // actual width. The horizontal scale is unimportant for
              // a context snippet, and the time-mapped overlays use
              // percentages so they stay aligned with the peaks under
              // them regardless of stretch.
              style={{ width: '100%', height: TIMING_VIZ_WAVE_HEIGHT }}
            />
          ) : (
            <div className={styles.timingVizNoAudio}>(no audio loaded)</div>
          )}
          {gridLines.map((pct, i) => (
            <div
              key={`grid-${i}`}
              className={styles.timingVizGridLine}
              style={{ left: `${pct}%` }}
            />
          ))}
          {barBoundaries.map((b, i) => (
            <React.Fragment key={`bar-${i}`}>
              <div
                className={styles.timingVizBarLine}
                style={{ left: `${b.pct}%` }}
              />
              {b.label !== null && (
                <div
                  className={styles.timingVizBarLabel}
                  style={{ left: `${b.pct}%` }}
                >
                  {b.label}
                </div>
              )}
            </React.Fragment>
          ))}
          <div
            className={styles.timingVizDetectedLine}
            style={{ left: `${detectedPct}%` }}
            title={`Detected · ${detectedSec.toFixed(3)}s`}
          />
          {finalPct !== undefined && (
            <div
              className={styles.timingVizFinalLine}
              style={{ left: `${finalPct}%` }}
              title={`Final · ${finalSec!.toFixed(3)}s`}
            />
          )}
        </div>
        {diffRows.map((row) => {
          const leftPct = Math.min(row.anchorPct, row.endPct);
          const widthPct = Math.abs(row.endPct - row.anchorPct);
          const beatsPart =
            row.deltaBeats !== undefined ? `· ${renderSignedBeats(row.deltaBeats)} ` : '';
          const slotsPart =
            row.deltaSlots !== undefined ? `· ${renderSignedSlots(row.deltaSlots)} ` : '';
          const fullText = `${row.label} ${beatsPart}${slotsPart}· ${renderSignedMs(row.deltaSec * 1000)}`;
          return (
            <div
              key={row.key}
              className={styles.timingVizDiffRow}
              style={{
                height: TIMING_VIZ_ROW_HEIGHT,
                '--bar-left': `${leftPct}%`,
                '--bar-width': `${widthPct}%`,
              } as React.CSSProperties}
              title={fullText}
            >
              <div
                className={`${styles.timingVizDiffBar} ${row.className}`}
                style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
              />
              <div className={styles.timingVizDiffRowText}>{fullText}</div>
              <div
                className={styles.timingVizDiffRowTextInverted}
                aria-hidden="true"
              >
                {fullText}
              </div>
            </div>
          );
        })}
      </div>
    );
  }
);
