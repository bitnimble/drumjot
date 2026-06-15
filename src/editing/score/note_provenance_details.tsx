import { ChevronDown, ChevronRight } from 'lucide-react';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { NotePosition } from 'src/editing/score/note_position';
import { NoteProvenanceEntry } from 'src/editing/provenance/debug_zip';
import { DEFAULT_GRID_DIVISION, gridDivisionFor } from 'src/grid/grid';
import type { StructBar, StructNote } from 'src/editing/structure/structure_store';
import { TICKS_PER_BEAT } from 'src/midi/to_midi';
import { AudioTrack } from 'src/editing/playback/audio_tracks';
import { jotPlayer } from 'src/editing/playback/player';
import { waveformWorker } from 'src/editing/playback/waveform_worker_client';
import { BarTimingsContext, StructuralContext } from '../jot_editor_contexts';
import { NoteProvenanceContextValue } from '../provenance/provenance_contexts';
import styles from './score.module.css';
import { WAVEFORM_PAINT_COLOR } from '../utils/waveform_color';

/**
 * Optional rendering context for {@link NoteProvenanceDetails}.
 *
 * Present only when the panel is hosted by a kept note (i.e. by
 * NoteView). Lets the panel compare the detector's view of the onset
 * against where the score actually drew it after all post-detection
 * processing — beat-tracker alignment and the MIDI→Jot quantization
 * pass in {@link from_midi.ts} that snaps every onset to the 16th
 * grid. FilteredOnsetView omits this prop (a rejected onset has no
 * rendered note) and the panel skips the rendered/snap rows.
 */
type RenderedNoteContext = {
  note: StructNote;
  bar: StructBar;
  provenance: NoteProvenanceContextValue;
};

/**
 * Computes the rendered (post-quantization) beat position in the same
 * 1-indexed `beat_in_bar` convention the provenance entry uses, so the
 * "Detected beat" → "Quantized to" comparison reads natively.
 *
 * `note.beat` is in quarter-notes from bar start (0-indexed). The
 * provenance's `beat_in_bar` counts beats of the bar's time
 * signature (1-indexed, where downbeat = 1.0). The conversion is
 * `1 + note.beat × (time.count / bar.beats)` — equals
 * `1 + note.beat` in 4/4, scales correctly for 6/8, 7/8, etc.
 */
function renderedBeatInBar(note: StructNote, bar: StructBar): number {
  if (bar.beats <= 0) return 1;
  return 1 + (note.beat / bar.beats) * bar.tsCount;
}

/** Grid step in MIDI ticks for a given grid division, matching
 * `from_midi.ts`'s `gridTicks = ticksPerBeat * 4 / gridDivision`. Used by
 * the per-note Snap-delta computation in {@link NoteProvenanceDetails} so
 * the value depends only on the immutable detected tick, not on the
 * rendered note's current bar (which moves under the Beat-offset slider). */
function midiGridTicks(gridDivision: number): number {
  return (TICKS_PER_BEAT * 4) / gridDivision;
}

/**
 * Render a delta in whole-note-subdivision slots with a sign and a
 * `/${gridDivision}` denominator (matching {@link
 * NotePosition.formatBarBeat48ths}'s absolute-position format).
 * Integer-rounded when the slot count is effectively whole (within 0.05)
 * so jitter-class deltas read as `+1/48` rather than `+0.97/48`;
 * fractional otherwise so a sub-slot snap delta (e.g. `+0.3/48`) still
 * surfaces its magnitude.
 */
function formatSignedSlots(slots: number, gridDivision: number): string {
  const sign = slots >= 0 ? '+' : '';
  const rounded = Math.round(slots);
  if (Math.abs(slots - rounded) < 0.05) return `${sign}${rounded}/${gridDivision}`;
  return `${sign}${slots.toFixed(1)}/${gridDivision}`;
}

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
 * Per-onset timing diagram, rendered above the textual {@link
 * NoteProvenanceDetails} grid. Layers, top to bottom:
 *
 *   1. Header — bar number + the audio-time window the snippet spans.
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
type StageShift = {
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

const OnsetTimingVisualization = observer(
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
    const renderSignedMs = (ms: number) => `${ms >= 0 ? '+' : ''}${ms.toFixed(1)} ms`;

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

/**
 * Collapsible "Debug details" block that surfaces a single onset's full
 * provenance (detected time, strength, beat-tracker placement, MIDI
 * quantization, filter decision, MIDI tick, …) inside the selection
 * label. Shared between {@link NoteView} (where it appears under the
 * human-readable description with the post-quantization comparison
 * filled in) and {@link FilteredOnsetView} (where it IS the label —
 * filtered onsets have no rendered counterpart so the rendered/snap
 * rows are hidden).
 *
 * Toggle state is local to this component, so it resets every time the
 * label remounts. Closing the popover (de-selecting / un-hovering the
 * note) collapses the details again next time — acceptable for v1
 * since the block is short.
 */
export const NoteProvenanceDetails = observer(
  ({
    entry,
    rendered,
    startOpen = false,
  }: {
    entry: NoteProvenanceEntry;
    /** Present when hosted by a kept note; absent for filtered ghosts. */
    rendered?: RenderedNoteContext;
    /** Open by default (used by FilteredOnsetView so the user immediately
     * sees why the onset was rejected — for kept notes the toggle is
     * collapsed by default so it doesn't crowd the basic description). */
    startOpen?: boolean;
  }) => {
    const [open, setOpen] = React.useState(startOpen);
    // Subsection toggles. Both default open so opening the Debug
    // details panel surfaces everything at once; collapse to focus
    // on the other half.
    const [timingOpen, setTimingOpen] = React.useState(true);
    const [acousticOpen, setAcousticOpen] = React.useState(true);
    // Stop the container's mousedown handler so clicks on the toggle don't
    // begin a marquee selection (which would clear the surrounding note's
    // selection and immediately unmount this component).
    const stop = (e: React.MouseEvent) => e.stopPropagation();

    // Eager per-bar timings table provided once at JotEditor. Lets the
    // "Final position" row resolve a bar's absolute jot-time start without
    // waiting for the player's timeline to be built — pre-Play the
    // player's timeline is `EMPTY_TIMELINE`, but the math doesn't actually
    // need any playback state.
    const barTimings = React.useContext(BarTimingsContext);
    // The current song's structure, used to read `effectiveDrumOffsetBeats`,
    // the user-applied Beat-offset slider value, as a labelled stage in
    // the detected → final timing-drift chain.
    const structural = React.useContext(StructuralContext);
    // Grid density the jot was produced at; drives every slot readout in
    // this panel. Falls back to the default when no structure is in
    // context (filtered-ghost rendering).
    const gridDivision = structural ? gridDivisionFor(structural.source) : DEFAULT_GRID_DIVISION;
    const slotsPerQuarterNote = gridDivision / 4;

    // Two coordinate frames are tracked here:
    //
    //   1. ORIGINAL — the bar the detector placed the onset in
    //      (`entry.bar + 1` in jot's 1-indexed convention) and the
    //      post-MIDI-snap beat position inside it. Immutable for a given
    //      note: derived purely from `entry.tick` (the unsnapped MIDI
    //      tick preserved through `from_midi.ts`) and the 1/48 grid
    //      snap, neither of which depend on the Beat-offset slider.
    //   2. CURRENT — `rendered.bar` and the note's current quantized
    //      position inside it, after `applyDrumOffsetStructure` has
    //      re-bucketed the note under the user's slider value. This is
    //      what the score is drawing right now and what the playback
    //      scheduler will fire.
    //
    // The Snap-delta row reads ORIGINAL (so dragging the Beat slider
    // doesn't change it — it's a property of the MIDI quantization, not
    // the user's view); the Final-position row reads CURRENT (so the
    // operator sees where the note actually plays after the slider);
    // the Drum-offset row is the transition between them.
    let displayedBarIndex: number | undefined;
    let currentQuantizedBeat: number | undefined;
    let currentSecPerQuarterNote: number | undefined;
    let originalBar: StructBar | undefined;
    let originalBarIndex: number | undefined;
    let originalSecPerQuarterNote: number | undefined;
    let originalQuantizedBeat: number | undefined;
    let originalQuantizedSec: number | undefined;
    let snapBeats: number | undefined;
    let snapSec: number | undefined;
    let snapMs: number | undefined;
    let finalSec: number | undefined;
    let drumOffsetBeats: number | undefined;
    let drumOffsetSec: number | undefined;
    let anchorDriftSec: number | undefined;
    let anchorDriftBeats: number | undefined;

    if (rendered) {
      currentQuantizedBeat = renderedBeatInBar(rendered.note, rendered.bar);
      displayedBarIndex = rendered.bar.index;

      const currentBarTiming = barTimings?.get(rendered.bar.index);
      if (currentBarTiming && rendered.bar.beats > 0) {
        currentSecPerQuarterNote = currentBarTiming.durationSec / rendered.bar.beats;
      }

      // The ORIGINAL bar lives in the rendered jot's structure at array
      // position `provenance.leadBars + entry.bar` (see the docstring on
      // `NoteProvenanceContextValue.leadBars` and `note_provenance.py`:
      // the MIDI lays `lead_bars` empty bar-0-sized blocks before
      // transcriber bar 0). We look it up by *array position*, NOT by
      // `bar.index === entry.bar + 1`, because the naïve index mapping
      // breaks whenever `from_midi.ts` counts more leading all-rest bars
      // than the transcriber's `lead_bars` did; e.g. when the filter
      // LLM kept no onsets in transcriber bars 0..N-1, those bars come
      // through the MIDI as all-rest and `from_midi`'s leading-rest walk
      // folds them into the lead-in, inflating `jot.globalMetadata.leadBars`
      // past `provenance.leadBars` and shifting every drum bar's
      // `bar.index` left by the difference. Array position survives that:
      // bar 0 of the transcriber's structure is always at array index
      // `provenance.leadBars`, with or without the inflation. `applyDrumOffsetStructure`
      // shallow-clones bars and preserves both array order and `index`,
      // so this lookup is also drum-offset-invariant.
      const structBars = structural?.layers[0]?.bars;
      const originalBarArrayPos = rendered.provenance.leadBars + entry.bar;
      originalBar =
        structBars && originalBarArrayPos >= 0 && originalBarArrayPos < structBars.length
          ? structBars[originalBarArrayPos]
          : undefined;
      originalBarIndex = originalBar?.index;
      const originalBarTiming =
        originalBarIndex !== undefined ? barTimings?.get(originalBarIndex) : undefined;
      if (originalBar && originalBarTiming && originalBar.beats > 0) {
        originalSecPerQuarterNote = originalBarTiming.durationSec / originalBar.beats;
      }

      // Snap delta is the audio-time displacement from the detected onset
      // to where the 1/48 MIDI grid landed; both expressed in the
      // ORIGINAL bar's tempo frame. Derived from `entry.tick`
      // (post-`onsets_midi.py` rounding to integer ticks, pre-`from_midi`
      // grid snap), so the value is a fixed property of the MIDI and
      // doesn't move when the Beat-offset slider re-buckets the rendered
      // note into a different bar.
      if (
        originalBar &&
        originalBarTiming &&
        originalSecPerQuarterNote !== undefined &&
        entry.tick !== null
      ) {
        const gridTicks = midiGridTicks(gridDivision);
        const postSnapTick = Math.round(entry.tick / gridTicks) * gridTicks;
        const snapDeltaQn = (postSnapTick - entry.tick) / TICKS_PER_BEAT;
        // qn → ts-beats: 1 ts-beat = 4/unit qn, so 1 qn = unit/4 ts-beats.
        snapBeats = snapDeltaQn * (originalBar.tsUnit / 4);
        snapSec = snapDeltaQn * originalSecPerQuarterNote;
        snapMs = snapSec * 1000;
        // Post-Python-quantise position, sourced from the provenance
        // sidecar's `quantised_time_sec` (audio time, set by every pass
        // that placed this onset). The earlier `entry.beat_in_bar +
        // snapBeats` formula combined the *detector* beat with only the
        // JS-side 1/48 snap, it ignored the Python geometric / envelope
        // / grid / LLM shifts, so the "Quantized to" readout could print
        // a value past the bar's last slot (e.g. "49/48") while the MIDI
        // bytes carried the true clamped position. Falls back to the
        // detected time when quantise didn't shift this onset; that
        // matches what the MIDI tick was emitted from (see
        // `onsets_to_midi_bytes`).
        const quantSourceSec = entry.quantised_time_sec ?? entry.detected_time_sec;
        const originalBarAudioStart = originalBarTiming.startSec - jotPlayer.songLeadInSec;
        const intraBarQn =
          (quantSourceSec - originalBarAudioStart) / originalSecPerQuarterNote;
        originalQuantizedBeat = 1 + intraBarQn * (originalBar.tsUnit / 4);
        originalQuantizedSec = quantSourceSec;
      }

      // Final position uses the CURRENT bar's timing; where the note
      // plays now after the slider has applied its shift.
      if (currentBarTiming && currentSecPerQuarterNote !== undefined) {
        const intra =
          (currentQuantizedBeat - 1) * (4 / rendered.bar.tsUnit) * currentSecPerQuarterNote;
        finalSec = currentBarTiming.startSec + intra - jotPlayer.songLeadInSec;
      }

      // Bar-anchor drift: the difference between where the JOT places
      // the note's original bar in audio time and where the detector's
      // post-alignment view of beat_in_bar implies it should be. In a
      // perfect round-trip these match exactly; in practice the most
      // common source of mismatch is `transcriber/app/pipeline/onsets_midi.py`'s
      // `compute_bar_tick_grid` rounding `lead_bars = round(lead_in_secs *
      // initial_tempo_ticks_per_sec / bar0_ticks)` to zero when the
      // pre-roll is shorter than ~half a bar. The MIDI then carries no
      // lead-in, `from_midi.ts` reconstructs `songLeadIn = 0`, and every
      // bar is anchored `lead_in_secs` early in audio time. Secondary
      // sources (per-bar bpm attribution drift, integer-tick rounding)
      // can also contribute small amounts but are dominated by lead-in
      // rounding when it triggers.
      if (
        originalBar &&
        originalBarTiming &&
        originalSecPerQuarterNote !== undefined &&
        originalSecPerQuarterNote > 0
      ) {
        // Where the detector says `originalBar` starts in audio time:
        // back out the intra-bar offset from the detected onset.
        const intraSecFromDetected =
          (entry.beat_in_bar - 1) * (4 / originalBar.tsUnit) * originalSecPerQuarterNote;
        const transcriberBarAudioTime = entry.detected_time_sec - intraSecFromDetected;
        // Where the JOT places that same bar in audio time.
        const jotBarAudioTime = originalBarTiming.startSec - jotPlayer.songLeadInSec;
        const drift = jotBarAudioTime - transcriberBarAudioTime;
        if (Math.abs(drift) > 1e-6) {
          anchorDriftSec = drift;
          anchorDriftBeats = (drift / originalSecPerQuarterNote) * (originalBar.tsUnit / 4);
        }
      }

      // Manual drum-offset slider (`effectiveDrumOffsetBeats` is in
      // quarter notes by convention, `applyDrumOffset` shifts each
      // `note.beat` by this amount, and `note.beat` itself is in quarter
      // notes from bar start). The audio-time displacement uses the
      // CURRENT (destination) bar's tempo, since that's the tempo the
      // note's new intra-bar position is interpreted under. Cross-bar
      // shifts under a per-bar bpm change are approximate; the residual
      // surfaces in the Unknown-drift row.
      const offsetQn = structural?.effectiveDrumOffsetBeats;
      if (typeof offsetQn === 'number' && Math.abs(offsetQn) > 1e-9) {
        drumOffsetBeats = offsetQn;
        if (currentSecPerQuarterNote !== undefined) {
          drumOffsetSec = offsetQn * currentSecPerQuarterNote;
        }
      }
    }

    // ─── New per-stage shifts (provenance format v3) ───
    //
    // Envelope refine inside the `onsets` stage: shift the ADTOF model
    // peak time picked up when `_refine_peak_times_audio` snapped to the
    // audio envelope's local max. `null`/`undefined` on legacy bundles
    // or non-ADTOF detection paths (none in production today).
    const rawModelSec = entry.raw_model_time_sec;
    const envelopeRefineSec =
      rawModelSec !== null && rawModelSec !== undefined
        ? entry.detected_time_sec - rawModelSec
        : undefined;

    // Beat-grid alignment split (v3). When the bundle predates the
    // split, both fields are null; we fall back to surfacing the
    // combined `beatAlignmentOffsetSec` as a single "Beat alignment"
    // stage. Both halves can be negative; only suppress when literally
    // 0.0 (or the bundle didn't surface them).
    const coarseAlignSec = rendered?.provenance.beatAlignCoarseOffsetSec ?? null;
    const fineAlignSec = rendered?.provenance.beatAlignFineOffsetSec ?? null;
    const hasAlignSplit = coarseAlignSec !== null || fineAlignSec !== null;
    const combinedAlignSec = rendered?.provenance.beatAlignmentOffsetSec ?? 0;

    // Per-pass quantise contributions (v3). `null`/`undefined` means
    // the pass didn't run for this onset (off-grid for any pass after
    // geometric; envelope pass skipped because no envelope was
    // available; grid/LLM toggled off; LLM cancelled/errored). `0`
    // means the pass ran but didn't shift (or its shift was rejected
    // by the monotonic-injective guard).
    type QuantPass = {
      key: string;
      label: string;
      className: string;
      slots: number | null | undefined;
    };
    const quantisePasses: QuantPass[] = [
      {
        key: 'q-geo',
        label: 'Quantise · geometric snap',
        className: styles.timingVizDiffBarQuantGeo,
        slots: entry.geometric_shift_slots,
      },
      {
        key: 'q-env',
        label: 'Quantise · envelope re-snap',
        className: styles.timingVizDiffBarQuantEnv,
        slots: entry.envelope_shift_slots,
      },
      {
        key: 'q-grid',
        label: 'Quantise · musical grid',
        className: styles.timingVizDiffBarQuantGrid,
        slots: entry.grid_shift_slots,
      },
      {
        key: 'q-llm',
        label: 'Quantise · LLM residual',
        className: styles.timingVizDiffBarQuantLlm,
        slots: entry.llm_shift_slots,
      },
    ];
    const hasPerPassQuantise = quantisePasses.some(
      (p) => p.slots !== null && p.slots !== undefined
    );

    // Legacy v1/v2 fallback: a single combined quantise row using the
    // summed `quantised_shift_slots`. Only emitted when the per-pass
    // split is absent so we never double-count.
    const fallbackQuantSec =
      !hasPerPassQuantise &&
      entry.quantised_time_sec !== null &&
      entry.quantised_time_sec !== undefined
        ? entry.quantised_time_sec - entry.detected_time_sec
        : undefined;
    const fallbackQuantSlots =
      !hasPerPassQuantise &&
      entry.quantised_shift_slots !== null &&
      entry.quantised_shift_slots !== undefined
        ? entry.quantised_shift_slots
        : undefined;

    // ─── Unit conversion helpers (shared by chain build + dl render) ───
    //
    // ts-beats of the original bar from audio seconds. Returns
    // `undefined` when the bar's tempo can't be resolved.
    const secToOrigBeats = (sec: number): number | undefined =>
      originalBar !== undefined &&
      originalSecPerQuarterNote !== undefined &&
      originalSecPerQuarterNote > 0
        ? (sec / originalSecPerQuarterNote) * (originalBar.tsUnit / 4)
        : undefined;
    // Integer slot count → audio seconds in the original bar's frame.
    // 1 slot = 1/(gridDivision/4) qn, so `slots / slotsPerQn × secPerQn`.
    const slotsToOrigSec = (slots: number): number | undefined =>
      originalSecPerQuarterNote !== undefined && originalSecPerQuarterNote > 0
        ? (slots / slotsPerQuarterNote) * originalSecPerQuarterNote
        : undefined;
    // ts-beats → slot count, using `gridDivision / unit` slots-per-ts-beat.
    const origBeatsToSlots = (beats: number | undefined): number | undefined =>
      beats !== undefined && originalBar !== undefined
        ? (beats * gridDivision) / originalBar.tsUnit
        : undefined;

    // ─── Bar-grid frame shift attribution ───
    //
    // Two values that LOOK like they should advance the onset along the
    // detected → final chain actually don't, they shift the *frame*
    // we're measuring against, not the onset itself:
    //
    //   - **Envelope refine** is pre-canonical. `entry.detected_time_sec`
    //     is set in `note_provenance.py` to `float(c.time)`, and `c.time`
    //     was already snapped to the audio-envelope local-max by
    //     `_refine_peak_times_audio` *before* being stored on the
    //     candidate. The chain starts AT detected_time_sec, so adding
    //     `envelopeRefineSec = detected − raw_model` on top would
    //     double-count the move from raw model peak to canonical time.
    //   - **Beat-grid alignment** shifts every beat's `time` (and thus
    //     each bar's `start_time`) by `align_offset_sec`, but the
    //     onset's audio time `c.time` is unchanged. The post-alignment
    //     bar positions ARE reflected in the JOT's bar audio anchors
    //     (via the transcriber's per-bar `set_tempo` map +
    //     `from_midi.ts`'s `songLeadIn` reconstruction), so
    //     `finalSec = currentBarTiming.startSec + intra + songLeadIn`
    //     is already in the post-alignment frame.
    //
    // Their combined value `(envelopeRefineSec ?? 0) + combinedAlignSec`
    // is what previously surfaced unlabeled as "Unknown drift" on every
    // onset of a track with a non-zero net frame shift, the chain
    // included those shifts in `accountedSec` while `finalSec` (rightly)
    // did not, leaving the gap as a confusing residual of magnitude
    // `|envelopeRefineSec + combinedAlignSec|` (often 50-80 ms once
    // both contribute). We now keep them OUT of `accountedSec` (so the
    // residual collapses to just the genuinely-unknown rounding noise)
    // and surface their sum as its own labeled "Bar-grid frame shift"
    // row in the debug details below.
    const barGridFrameShiftSec: number | undefined =
      envelopeRefineSec !== undefined || Math.abs(combinedAlignSec) > 1e-9
        ? (envelopeRefineSec ?? 0) + combinedAlignSec
        : undefined;

    // ─── Unknown drift residual ───
    //
    // `finalSec` minus everything the named-stages chain accounts for.
    // After the bar-grid frame-shift fix above this should sit at ~0
    // for well-formed bundles; any remaining gap is a genuine drift
    // source we haven't enumerated (per-bar BPM attribution mismatches
    // between the transcriber and from_midi, cross-bar drum-offset
    // re-bucketing under varying tempo, etc.).
    let unknownDriftSec: number | undefined;
    let unknownDriftBeats: number | undefined;
    let unknownDriftMs: number | undefined;
    if (finalSec !== undefined && rendered) {
      let accountedSec = entry.detected_time_sec;
      // `envelopeRefineSec` NOT added, see the `barGridFrameShiftSec`
      // comment above. The chain starts at `entry.detected_time_sec`,
      // which is already the post-envelope-refine audio time.
      if (hasPerPassQuantise) {
        for (const p of quantisePasses) {
          if (p.slots === null || p.slots === undefined || p.slots === 0) continue;
          const sec = slotsToOrigSec(p.slots);
          if (sec !== undefined) accountedSec += sec;
        }
      } else if (fallbackQuantSec !== undefined) {
        accountedSec += fallbackQuantSec;
      }
      if (snapSec !== undefined) accountedSec += snapSec;
      // `combinedAlignSec` NOT added, see the `barGridFrameShiftSec`
      // comment above. The alignment shifts the beat grid (and
      // thereby `bar.start_time`), not the onset's audio time; the
      // post-alignment bar positions are already reflected in
      // `originalBarTiming.startSec + songLeadIn`.
      if (anchorDriftSec !== undefined) accountedSec += anchorDriftSec;
      if (drumOffsetSec !== undefined) accountedSec += drumOffsetSec;
      const residual = finalSec - accountedSec;
      if (Math.abs(residual) > 1e-6) {
        unknownDriftSec = residual;
        unknownDriftMs = residual * 1000;
        unknownDriftBeats = secToOrigBeats(residual);
      }
    }

    // ─── Build the ordered chain in pipeline order ───
    //
    // Each entry is one named shift that actually advances the onset
    // along the detected → final chain in audio time; the trailing
    // unknown-drift residual is appended inside
    // `OnsetTimingVisualization` so it always lands at the chain's end.
    // Pipeline order: quantise passes → bar anchor drift → MIDI snap →
    // drum offset.
    //
    // **Not in the chain:** envelope refine and beat-grid alignment.
    // Both are frame shifts (pre-canonical for env refine, bar-grid for
    // alignment), so they don't accumulate into `finalSec` and adding
    // them as forward chain stages would create the same -64..-68 ms
    // residual mismatch this code's `barGridFrameShiftSec` attribution
    // is meant to surface explicitly. The two are still visible as
    // their own rows in the textual debug details (under "Onset
    // detection" → envelope refine and "Beat-grid alignment" →
    // coarse/fine), and their summed contribution is the "Bar-grid
    // frame shift" row.
    const stages: StageShift[] = [];
    // Frame-shift stages first: they render at the top of the diff rows
    // and are anchored to fixed audio-time positions (envelope refine
    // literally spans `rawModelSec → detectedSec`; alignment shifts the
    // grid without a literal "from" position, anchored at the detected
    // line by convention). Excluded from `accountedSec` above, so the
    // chain math and the trailing unknown-drift residual are unaffected.
    if (
      envelopeRefineSec !== undefined &&
      Math.abs(envelopeRefineSec) > 1e-9 &&
      rawModelSec !== null &&
      rawModelSec !== undefined
    ) {
      stages.push({
        key: 'env-refine',
        label: 'Envelope refine',
        className: styles.timingVizDiffBarEnvRefine,
        deltaSec: envelopeRefineSec,
        deltaBeats: secToOrigBeats(envelopeRefineSec),
        anchorSec: rawModelSec,
      });
    }
    if (hasAlignSplit) {
      if (coarseAlignSec !== null && Math.abs(coarseAlignSec) > 1e-9) {
        stages.push({
          key: 'align-coarse',
          label: 'Coarse · envelope phase',
          className: styles.timingVizDiffBarAlignCoarse,
          deltaSec: coarseAlignSec,
          deltaBeats: secToOrigBeats(coarseAlignSec),
          anchorSec: entry.detected_time_sec,
        });
      }
      if (fineAlignSec !== null && Math.abs(fineAlignSec) > 1e-9) {
        stages.push({
          key: 'align-fine',
          label: 'Fine · onset-snap',
          className: styles.timingVizDiffBarAlignFine,
          deltaSec: fineAlignSec,
          deltaBeats: secToOrigBeats(fineAlignSec),
          anchorSec: entry.detected_time_sec,
        });
      }
    } else if (Math.abs(combinedAlignSec) > 1e-9) {
      stages.push({
        key: 'align-combined',
        label: 'Beat alignment (combined)',
        className: styles.timingVizDiffBarAlignCoarse,
        deltaSec: combinedAlignSec,
        deltaBeats: secToOrigBeats(combinedAlignSec),
        anchorSec: entry.detected_time_sec,
      });
    }
    if (hasPerPassQuantise) {
      for (const p of quantisePasses) {
        if (p.slots === null || p.slots === undefined || p.slots === 0) continue;
        const deltaSec = slotsToOrigSec(p.slots);
        if (deltaSec === undefined) continue;
        stages.push({
          key: p.key,
          label: p.label,
          className: p.className,
          deltaSec,
          deltaBeats: secToOrigBeats(deltaSec),
          deltaSlots: p.slots,
        });
      }
    } else if (fallbackQuantSec !== undefined && Math.abs(fallbackQuantSec) > 1e-9) {
      stages.push({
        key: 'quantise-combined',
        label: 'Quantise (combined)',
        className: styles.timingVizDiffBarQuantGeo,
        deltaSec: fallbackQuantSec,
        deltaBeats: secToOrigBeats(fallbackQuantSec),
        deltaSlots: fallbackQuantSlots,
      });
    }
    if (anchorDriftSec !== undefined && Math.abs(anchorDriftSec) > 1e-9) {
      stages.push({
        key: 'anchor',
        label: 'Bar anchor drift',
        className: styles.timingVizDiffBarAnchor,
        deltaSec: anchorDriftSec,
        deltaBeats: anchorDriftBeats,
      });
    }
    if (snapSec !== undefined && Math.abs(snapSec) > 1e-9) {
      stages.push({
        key: 'midi-snap',
        label: 'MIDI 1/48 snap',
        className: styles.timingVizDiffBarMidiSnap,
        deltaSec: snapSec,
        deltaBeats: snapBeats,
        deltaSlots: origBeatsToSlots(snapBeats),
      });
    }
    if (drumOffsetSec !== undefined && Math.abs(drumOffsetSec) > 1e-9) {
      stages.push({
        key: 'drum-offset',
        label: 'Drum offset',
        className: styles.timingVizDiffBarDrumOffset,
        deltaSec: drumOffsetSec,
        deltaBeats: drumOffsetBeats,
        // `drumOffsetBeats` is in quarter notes; 1 qn = gridDivision/4
        // slots independent of the bar's time signature.
        deltaSlots:
          drumOffsetBeats !== undefined ? drumOffsetBeats * slotsPerQuarterNote : undefined,
      });
    }

    const renderSignedMs = (ms: number) => `${ms >= 0 ? '+' : ''}${ms.toFixed(1)} ms`;
    const renderSignedBeats = (b: number) => `${b >= 0 ? '+' : ''}${b.toFixed(3)}`;
    const renderSignedSec = (s: number) => `${s >= 0 ? '+' : ''}${s.toFixed(3)}s`;
    // Format an integer slot count as a per-slot label like "+3/48".
    const formatSlots = (slots: number): string => formatSignedSlots(slots, gridDivision);

    const hasAcoustic = entryHasAcousticFields(entry);
    const renderSubsectionToggle = (
      label: string,
      isOpen: boolean,
      onToggle: () => void,
    ) => (
      <button
        type="button"
        className={styles.debugDetailsToggle}
        onClick={(e) => {
          stop(e);
          onToggle();
        }}
        onMouseDown={stop}
        aria-expanded={isOpen}
      >
        {isOpen ? (
          <ChevronDown size={12} aria-hidden="true" />
        ) : (
          <ChevronRight size={12} aria-hidden="true" />
        )}
        {label}
      </button>
    );
    return (
      <div className={styles.debugDetails} onMouseDown={stop}>
        <button
          type="button"
          className={styles.debugDetailsToggle}
          onClick={(e) => {
            stop(e);
            setOpen((o) => !o);
          }}
          onMouseDown={stop}
          aria-expanded={open}
        >
          {open ? (
            <ChevronDown size={12} aria-hidden="true" />
          ) : (
            <ChevronRight size={12} aria-hidden="true" />
          )}
          Debug details
        </button>
        {open && (
          <>
            {renderSubsectionToggle(
              'Timing',
              timingOpen,
              () => setTimingOpen((o) => !o),
            )}
            {timingOpen && rendered && (
              <OnsetTimingVisualization
                entry={entry}
                rendered={rendered}
                gridDivision={gridDivision}
                stages={stages}
                finalSec={finalSec}
                displayedBarIndex={displayedBarIndex}
              />
            )}
            {timingOpen &&
              renderDebugStageSections({
                entry,
                rendered,
                originalBar,
                originalBarIndex,
                originalQuantizedBeat,
                originalQuantizedSec,
                currentQuantizedBeat,
                displayedBarIndex,
                finalSec,
                snapBeats,
                snapMs,
                envelopeRefineSec,
                rawModelSec,
                coarseAlignSec,
                fineAlignSec,
                combinedAlignSec,
                hasAlignSplit,
                quantisePasses,
                hasPerPassQuantise,
                fallbackQuantSec,
                fallbackQuantSlots,
                anchorDriftSec,
                anchorDriftBeats,
                drumOffsetSec,
                drumOffsetBeats,
                barGridFrameShiftSec,
                unknownDriftSec,
                unknownDriftBeats,
                unknownDriftMs,
                gridDivision,
                slotsPerQuarterNote,
                secToOrigBeats,
                origBeatsToSlots,
                renderSignedMs,
                renderSignedBeats,
                renderSignedSec,
                formatSlots,
              })}
            {hasAcoustic && renderSubsectionToggle(
              'Acoustic properties',
              acousticOpen,
              () => setAcousticOpen((o) => !o),
            )}
            {hasAcoustic && acousticOpen && renderAcousticSection(entry)}
          </>
        )}
      </div>
    );
  }
);

/** Per-lane acoustic measurements the cymbal_split + hihat_split
 *  passes capture. `null`/`undefined` everywhere else (other lanes,
 *  bundles predating v4). At least one populated field means the
 *  "Acoustic properties" subsection is worth surfacing. */
function entryHasAcousticFields(entry: NoteProvenanceEntry): boolean {
  return (
    entry.decay_s != null ||
    entry.flatness != null ||
    entry.centroid_hz != null ||
    entry.gap_s != null ||
    entry.attack_s != null ||
    entry.late_rms != null ||
    entry.pre_rms != null ||
    entry.tail_end_s != null
  );
}

/** Render the per-onset acoustic measurements the cymbal / hi-hat
 *  classifiers saw. Only rows whose value is populated are emitted; the
 *  caller is responsible for skipping the whole section when none are.
 *  Reuses the `debugStageSections` grid + `debugDetailsList`
 *  contents-flow pair so labels line up with the Timing subsection's
 *  rows above. */
function renderAcousticSection(entry: NoteProvenanceEntry): React.ReactNode {
  const rows: React.ReactNode[] = [];
  const row = (key: string, label: string, value: string) => {
    rows.push(
      <React.Fragment key={key}>
        <dt>{label}</dt>
        <dd>{value}</dd>
      </React.Fragment>,
    );
  };
  if (entry.attack_s != null) {
    // ms reads better than fractional seconds for typical attacks
    // (5-50ms range on hi-hats); ride / crash attacks fall in the
    // same range so the unit choice is uniform.
    row('attack', 'Attack rise', `${(entry.attack_s * 1000).toFixed(1)} ms`);
  }
  if (entry.decay_s != null) {
    row('decay', 'Decay (-20 dB)', `${entry.decay_s.toFixed(3)} s`);
  }
  if (entry.tail_end_s != null) {
    row('tail', 'Ring tail end', `${entry.tail_end_s.toFixed(3)} s`);
  }
  if (entry.late_rms != null) {
    row('late', 'Late RMS ratio', entry.late_rms.toFixed(3));
  }
  if (entry.pre_rms != null) {
    row('pre', 'Pre RMS ratio', entry.pre_rms.toFixed(3));
  }
  if (entry.flatness != null) {
    row('flat', 'Spectral flatness', entry.flatness.toFixed(4));
  }
  if (entry.centroid_hz != null) {
    row('cen', 'Spectral centroid', `${(entry.centroid_hz / 1000).toFixed(2)} kHz`);
  }
  if (entry.gap_s != null) {
    row('gap', 'Gap to neighbour', `${entry.gap_s.toFixed(3)} s`);
  }
  return (
    <div className={styles.debugStageSections}>
      <dl className={styles.debugDetailsList}>{rows}</dl>
    </div>
  );
}

/**
 * Render the Debug details body as a vertical stack of per-stage
 * sections (Onset detection → Beat-grid alignment → Quantise → MIDI
 * render → User adjustment → Final → Filter). Lifted out of the
 * component body so the JSX stays readable; every value is
 * pre-computed by the parent and passed in.
 *
 * Stages with no relevant data for this onset are omitted entirely;
 * the section only renders when at least one row inside has something
 * worth showing. That way the popup stays compact for a clean
 * round-trip and only grows when shifts actually accumulated.
 */
type DebugStageSectionsProps = {
  entry: NoteProvenanceEntry;
  rendered: RenderedNoteContext | undefined;
  originalBar: StructBar | undefined;
  originalBarIndex: number | undefined;
  originalQuantizedBeat: number | undefined;
  originalQuantizedSec: number | undefined;
  currentQuantizedBeat: number | undefined;
  displayedBarIndex: number | undefined;
  finalSec: number | undefined;
  snapBeats: number | undefined;
  snapMs: number | undefined;
  envelopeRefineSec: number | undefined;
  rawModelSec: number | null | undefined;
  coarseAlignSec: number | null;
  fineAlignSec: number | null;
  combinedAlignSec: number;
  hasAlignSplit: boolean;
  quantisePasses: ReadonlyArray<{
    key: string;
    label: string;
    className: string;
    slots: number | null | undefined;
  }>;
  hasPerPassQuantise: boolean;
  fallbackQuantSec: number | undefined;
  fallbackQuantSlots: number | undefined;
  anchorDriftSec: number | undefined;
  anchorDriftBeats: number | undefined;
  drumOffsetSec: number | undefined;
  drumOffsetBeats: number | undefined;
  /** Summed envelope-refine + combined-align frame shift, the
   * properly-attributed source of what previously surfaced as
   * "Unknown drift" on tracks with a non-zero net frame shift. See the
   * `barGridFrameShiftSec` comment in {@link NoteProvenanceDetails}. */
  barGridFrameShiftSec: number | undefined;
  unknownDriftSec: number | undefined;
  unknownDriftBeats: number | undefined;
  unknownDriftMs: number | undefined;
  gridDivision: number;
  slotsPerQuarterNote: number;
  secToOrigBeats: (sec: number) => number | undefined;
  origBeatsToSlots: (beats: number | undefined) => number | undefined;
  renderSignedMs: (ms: number) => string;
  renderSignedBeats: (b: number) => string;
  renderSignedSec: (s: number) => string;
  formatSlots: (slots: number) => string;
};

function renderDebugStageSections(p: DebugStageSectionsProps): React.ReactNode {
  const {
    entry, rendered, originalBar, originalBarIndex, originalQuantizedBeat,
    originalQuantizedSec, currentQuantizedBeat,
    displayedBarIndex, finalSec, snapBeats, snapMs, envelopeRefineSec,
    rawModelSec, coarseAlignSec, fineAlignSec, combinedAlignSec,
    hasAlignSplit, quantisePasses, hasPerPassQuantise, fallbackQuantSec,
    fallbackQuantSlots, anchorDriftSec, anchorDriftBeats, drumOffsetSec,
    drumOffsetBeats, barGridFrameShiftSec, unknownDriftSec,
    unknownDriftBeats, unknownDriftMs,
    gridDivision, slotsPerQuarterNote, secToOrigBeats, origBeatsToSlots,
    renderSignedMs, renderSignedBeats, renderSignedSec, formatSlots,
  } = p;
  const velocity = rendered?.note.velocity;

  // Helper: render a `<dt>/<dd>` pair displaying a per-stage shift in
  // {beats, slots, ms} triple form. The `slots` annotation is omitted
  // when neither an explicit count nor a derivable one resolves.
  const shiftRow = (
    label: string,
    deltaSec: number,
    deltaBeats: number | undefined,
    explicitSlots?: number,
  ) => {
    const derivedSlots = explicitSlots ?? origBeatsToSlots(deltaBeats);
    return (
      <>
        <dt>{label}</dt>
        <dd>
          {deltaBeats !== undefined && `${renderSignedBeats(deltaBeats)} beats `}
          {derivedSlots !== undefined && `· ${formatSlots(derivedSlots)} `}
          ({renderSignedMs(deltaSec * 1000)})
        </dd>
      </>
    );
  };

  return (
    <div className={styles.debugStageSections}>
      <section className={styles.stageGroup}>
        <h4 className={styles.stageHeading}>Onset detection</h4>
        <dl className={styles.debugDetailsList}>
          <dt>Detected beat</dt>
          <dd>
            {/* `entry.bar` is the transcriber's 0-indexed structural
              bar. We display the rendered jot's `bar.index` for the
              same bar (resolved via the array-position lookup above)
              so the value matches where the note actually lives in
              the score; `from_midi.ts` can fold leading all-rest
              drum bars into the lead-in, so the naïve `entry.bar + 1`
              mapping no longer always equals the displayed bar. Falls
              back to `entry.bar + 1` when the lookup couldn't resolve
              (no rendered jot context). */}
            {new NotePosition({
              barIndex: originalBarIndex ?? entry.bar + 1,
              beatInBar: entry.beat_in_bar,
              slotsPerQuarter: slotsPerQuarterNote,
              timeSig: originalBar
                ? { count: originalBar.tsCount, unit: originalBar.tsUnit }
                : undefined,
              audioSec: entry.detected_time_sec,
            }).toString()}
          </dd>
          {entry.amplitude !== null && entry.amplitude !== undefined && (
            <>
              <dt>Amplitude</dt>
              <dd>{entry.amplitude.toFixed(3)}</dd>
            </>
          )}
          <dt>Onset confidence</dt>
          <dd>{entry.strength.toFixed(3)}</dd>
          {entry.midi_note !== null && entry.midi_note !== undefined && (
            <>
              <dt>MIDI note</dt>
              <dd>{entry.midi_note}</dd>
            </>
          )}
          {/* Raw MIDI velocity is preserved on `note.metadata.midi.velocity`
              by from_midi.ts; the same file's [A7] step maps it into the
              `:a` (≥100) / `:g` (<40) modifiers visible in the description
              above. Surface the raw value so the operator can see exactly
              why a note picked up (or didn't) the accent/ghost decoration. */}
          {typeof velocity === 'number' && (
            <>
              <dt>Velocity</dt>
              <dd>{velocity}</dd>
            </>
          )}
          {envelopeRefineSec !== undefined && Math.abs(envelopeRefineSec) > 1e-9 && (
            <>
              <dt>Raw model peak</dt>
              <dd>{rawModelSec !== null && rawModelSec !== undefined ? `${rawModelSec.toFixed(3)}s` : ', '}</dd>
              {shiftRow(
                'Envelope refine',
                envelopeRefineSec,
                secToOrigBeats(envelopeRefineSec),
              )}
            </>
          )}
        </dl>
      </section>

      {(hasAlignSplit || Math.abs(combinedAlignSec) > 1e-9) && (
        <section className={styles.stageGroup}>
          <h4 className={styles.stageHeading}>Beat-grid alignment</h4>
          <dl className={styles.debugDetailsList}>
            {hasAlignSplit ? (
              <>
                {coarseAlignSec !== null &&
                  shiftRow(
                    'Coarse · envelope phase',
                    coarseAlignSec,
                    secToOrigBeats(coarseAlignSec),
                  )}
                {fineAlignSec !== null &&
                  shiftRow(
                    'Fine · onset-snap',
                    fineAlignSec,
                    secToOrigBeats(fineAlignSec),
                  )}
              </>
            ) : (
              shiftRow(
                'Beat alignment (combined)',
                combinedAlignSec,
                secToOrigBeats(combinedAlignSec),
              )
            )}
          </dl>
        </section>
      )}

      {/* Bar-grid frame shift: explicit attribution of the previously-
        * "Unknown drift" residual to its source, the sum of envelope
        * refine + combined alignment. Both shift the frame the chain
        * is measured against (envelope refine: raw model peak →
        * canonical onset time; alignment: pre- → post-alignment bar
        * grid) without moving the onset's final audio-time position,
        * so the chain math keeps them out of `accountedSec` while
        * this row makes their combined contribution visible. See the
        * `barGridFrameShiftSec` block in NoteProvenanceDetails for
        * the full rationale. */}
      {barGridFrameShiftSec !== undefined &&
        Math.abs(barGridFrameShiftSec) > 1e-9 && (
          <section className={styles.stageGroup}>
            <h4 className={styles.stageHeading}>Bar-grid frame shift</h4>
            <dl className={styles.debugDetailsList}>
              {shiftRow(
                'Frame shift (envelope refine + align)',
                barGridFrameShiftSec,
                secToOrigBeats(barGridFrameShiftSec),
              )}
            </dl>
          </section>
        )}

      {(hasPerPassQuantise ||
        (fallbackQuantSec !== undefined && Math.abs(fallbackQuantSec) > 1e-9) ||
        entry.off_grid === true ||
        (entry.quantised_residual_slots !== null &&
          entry.quantised_residual_slots !== undefined)) && (
        <section className={styles.stageGroup}>
          <h4 className={styles.stageHeading}>Quantise</h4>
          <dl className={styles.debugDetailsList}>
            {hasPerPassQuantise &&
              quantisePasses.map((pass) =>
                pass.slots === null || pass.slots === undefined ? null : (
                  <React.Fragment key={pass.key}>
                    <dt>{pass.label.replace(/^Quantise · /, '')}</dt>
                    <dd>{formatSlots(pass.slots)}</dd>
                  </React.Fragment>
                ),
              )}
            {!hasPerPassQuantise &&
              fallbackQuantSec !== undefined &&
              Math.abs(fallbackQuantSec) > 1e-9 &&
              shiftRow(
                'Quantise (combined)',
                fallbackQuantSec,
                secToOrigBeats(fallbackQuantSec),
                fallbackQuantSlots,
              )}
            {entry.quantised_residual_slots !== null &&
              entry.quantised_residual_slots !== undefined && (
                <>
                  <dt>Sub-slot residual</dt>
                  <dd>{renderSignedBeats(entry.quantised_residual_slots)} slot</dd>
                </>
              )}
            {entry.off_grid === true && (
              <>
                <dt>Off-grid</dt>
                <dd>yes (no free slot within match band)</dd>
              </>
            )}
            {originalQuantizedBeat !== undefined && originalBarIndex !== undefined && (
              <>
                <dt>Quantized to</dt>
                <dd>
                  {new NotePosition({
                    barIndex: originalBarIndex,
                    beatInBar: originalQuantizedBeat,
                    slotsPerQuarter: slotsPerQuarterNote,
                    timeSig: originalBar
                ? { count: originalBar.tsCount, unit: originalBar.tsUnit }
                : undefined,
                    audioSec: originalQuantizedSec,
                  }).toString()}
                </dd>
              </>
            )}
          </dl>
        </section>
      )}

      {(snapBeats !== undefined || anchorDriftSec !== undefined) && (
        <section className={styles.stageGroup}>
          <h4 className={styles.stageHeading}>MIDI render</h4>
          <dl className={styles.debugDetailsList}>
            {anchorDriftSec !== undefined &&
              shiftRow('Bar anchor drift', anchorDriftSec, anchorDriftBeats)}
            {snapBeats !== undefined && snapMs !== undefined && (
              <>
                <dt>1/48 snap</dt>
                <dd>
                  {`${renderSignedBeats(snapBeats)} beats `}
                  {origBeatsToSlots(snapBeats) !== undefined &&
                    `· ${formatSlots(origBeatsToSlots(snapBeats)!)} `}
                  ({renderSignedMs(snapMs)})
                </dd>
              </>
            )}
          </dl>
        </section>
      )}

      {(drumOffsetBeats !== undefined || unknownDriftSec !== undefined) && (
        <section className={styles.stageGroup}>
          <h4 className={styles.stageHeading}>User adjustment</h4>
          <dl className={styles.debugDetailsList}>
            {drumOffsetBeats !== undefined && (
              <>
                <dt>Drum offset</dt>
                <dd>
                  {renderSignedBeats(drumOffsetBeats)} beats{' '}
                  · {formatSlots(drumOffsetBeats * slotsPerQuarterNote)}
                  {drumOffsetSec !== undefined && ` (${renderSignedMs(drumOffsetSec * 1000)})`}
                </dd>
              </>
            )}
            {unknownDriftSec !== undefined && unknownDriftMs !== undefined && (
              <>
                <dt>Unknown drift</dt>
                <dd>
                  {unknownDriftBeats !== undefined &&
                    `${renderSignedBeats(unknownDriftBeats)} beats `}
                  {origBeatsToSlots(unknownDriftBeats) !== undefined &&
                    `· ${formatSlots(origBeatsToSlots(unknownDriftBeats)!)} `}
                  ({renderSignedMs(unknownDriftMs)})
                </dd>
              </>
            )}
          </dl>
        </section>
      )}

      {finalSec !== undefined &&
        currentQuantizedBeat !== undefined &&
        displayedBarIndex !== undefined && (
          <section className={styles.stageGroup}>
            <h4 className={styles.stageHeading}>Final</h4>
            <dl className={styles.debugDetailsList}>
              <dt>Position</dt>
              <dd>
                {new NotePosition({
                  barIndex: displayedBarIndex,
                  beatInBar: currentQuantizedBeat,
                  slotsPerQuarter: slotsPerQuarterNote,
                  timeSig: rendered
                    ? { count: rendered.bar.tsCount, unit: rendered.bar.tsUnit }
                    : undefined,
                  audioSec: finalSec,
                  offsetMs: rendered?.note.offsetMs,
                }).toString()}
              </dd>
            </dl>
          </section>
        )}

      <section className={styles.stageGroup}>
        <h4 className={styles.stageHeading}>Filter</h4>
        <dl className={styles.debugDetailsList}>
          <dt>Decision</dt>
          <dd>
            {entry.kept
              ? 'kept'
              : (() => {
                  const source =
                    entry.rejected_by ?? (entry.out_of_range ? 'out of range' : 'rejected');
                  const reasonLabel = entry.reason_code
                    ? (FILTER_REASON_LABELS[entry.reason_code] ?? entry.reason_code)
                    : null;
                  return reasonLabel ? (
                    <>
                      {source} · <span className={styles.reasonCode}>{reasonLabel}</span>
                    </>
                  ) : (
                    source
                  );
                })()}
          </dd>
          {!entry.kept && !entry.out_of_range && entry.reason_text && (
            <>
              <dt>Reason</dt>
              <dd>
                <span className={styles.reasonText}>{entry.reason_text}</span>
              </dd>
            </>
          )}
        </dl>
      </section>
    </div>
  );
}

/**
 * Display labels for the filter LLM's short reason codes. Keep in
 * lockstep with `REASON_CODES` in
 * `transcriber/app/pipeline/filter_llm.py`; unknown codes fall through
 * to the raw value so a new backend code still renders something
 * meaningful before the frontend catches up.
 */
const FILTER_REASON_LABELS: Record<string, string> = {
  bleed: 'Bleed from louder instrument',
  double_trigger: 'Detector double-trigger',
  noise: 'Isolated noise',
  custom: 'Custom',
};
