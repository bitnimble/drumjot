import classNames from 'classnames';
import { observer, useLocalObservable } from 'mobx-react-lite';
import React from 'react';
import { RenderedJot } from 'src/jot';
import {
  LYRICS_OFFSET_MAX_SEC,
  LYRICS_OFFSET_MIN_SEC,
  LYRICS_OFFSET_STEP_SEC,
  LyricLine,
  LyricWord,
  LyricsTrackId,
  activeLineIndexAt,
  activeWordIndexAt,
  audioSecToBeat,
  lyricsStore,
} from 'src/lyrics';
import { JotTimeline, jotPlayer } from 'src/playback';
import { JotViewStoreContext } from './contexts';
import { DropdownButton, dropdownStyles } from './components/dropdown';
import { GutterResizeHandle } from './components/gutter_resize_handle';
import { NumberStepper } from './components/number_stepper';
import {
  LyricLineMeasureInput,
  computeLyricShifts,
  lyricShiftKey,
  lyricsMeasurer,
} from './lyrics_measure';
import styles from './lyrics_row.module.css';
import mixerStyles from './mixer.module.css';
import { Playhead } from './playback';
import { seekFromClick } from './score';

/** Same fixed height as the audio-track row so adjacent rows align flush. */
const LYRICS_ROW_HEIGHT = 56;

/** Treat sub-millisecond gaps between rendered and raw model times as
 *  noise (floating-point round-trip through JSON, tiny rounding inside
 *  the aligner). Keeps the tooltip from screaming "Δ +0ms" on words
 *  where the model and our render agree. */
const TIMING_NOISE_FLOOR_SEC = 1e-4;

/** Build the per-word hover tooltip showing the model's raw output
 *  alongside the rendered cell timings. Surfaces:
 *
 *    - The line of text (in quotes; some words are punctuation-heavy
 *      and the quotes help disambiguate edge whitespace).
 *    - Rendered start/end and duration as a sanity baseline.
 *    - The model's raw start/end when present, plus the per-edge delta
 *      vs the rendered value (so the user can see whether drift came
 *      from the model itself or from our fallback chain).
 *    - The fallback marker (`endFallback`) when the rendered `endSec`
 *      came from substitution rather than from wav2vec2. Distinct from
 *      "model says X but we render Y" - this is "the model said
 *      nothing usable, and we filled in via rule Z".
 *
 *  Returns a `\n`-joined string. The browser's native `title` tooltip
 *  preserves newlines in modern engines; we accept the styling
 *  limitations of that surface in exchange for zero extra DOM. */
function buildWordDebugTitle(w: LyricWord): string {
  const fmtSec = (s: number) => `${s.toFixed(3)}s`;
  const fmtMs = (sec: number) => {
    const ms = Math.round(sec * 1000);
    const sign = ms > 0 ? '+' : '';
    return `${sign}${ms}ms`;
  };
  const lines: string[] = [];
  lines.push(`"${w.text}"`);
  if (w.romaji !== undefined) {
    lines.push(`aligned as: ${w.romaji}`);
  }
  lines.push(
    `rendered: ${fmtSec(w.startSec)} – ${fmtSec(w.endSec)}  (${fmtSec(w.endSec - w.startSec)})`,
  );
  if (w.rawStartSec !== undefined) {
    const d = w.startSec - w.rawStartSec;
    const note = Math.abs(d) > TIMING_NOISE_FLOOR_SEC ? `  Δ ${fmtMs(d)}` : '';
    lines.push(`model start: ${fmtSec(w.rawStartSec)}${note}`);
  } else {
    lines.push('model start: (substituted from segment)');
  }
  if (w.rawEndSec !== undefined) {
    const d = w.endSec - w.rawEndSec;
    const note = Math.abs(d) > TIMING_NOISE_FLOOR_SEC ? `  Δ ${fmtMs(d)}` : '';
    lines.push(`model end:   ${fmtSec(w.rawEndSec)}${note}`);
  } else {
    lines.push('model end:   (substituted)');
  }
  if (w.endFallback !== undefined) {
    lines.push(`end fallback: ${w.endFallback}`);
  }
  return lines.join('\n');
}

/** Per-word position metadata derived from the lyrics store + timeline.
 *  Stable under playhead movement; rebuilt only when `lines`, `offsetSec`,
 *  `timeline`, `drumsT0Sec`, `structuralBeats`, or `voiceBeats` change. */
type PositionedWord = {
  /** Index back into the source `line.words` array, so the JSX can
   *  compare against `activeWordIndexAt`'s return value even when
   *  out-of-range words at the line edges have been dropped. */
  sourceIdx: number;
  text: string;
  beatOffset: number;
  /** Width of this word's cell in beats: `endBeat - startBeat`.
   *  Drives the trailing-rule render in CSS via the `--lyric-word-
   *  beat-width` var; combined with `--lyric-word-shift` the cell's
   *  right edge stays anchored to the word's `endSec`. */
  beatWidth: number;
  /** Original word entry from the lyrics store, kept by reference
   *  so the JSX can build the debug tooltip (model raw times,
   *  fallback marker) without re-indexing into `line.words`. */
  source: LyricWord;
};
type PositionedLine = {
  i: number;
  text: string;
  startBeat: number;
  endBeat: number;
  /** When defined, the row renders one absolutely-positioned span
   *  per word inside the line container (beat offsets are relative
   *  to `startBeat`). When undefined, the line falls back to the
   *  inline text (LRCLIB-style). */
  wordPositions: PositionedWord[] | undefined;
};

/** Floor for a word's cell width in beats when the aligner emits an
 *  end-time we can't resolve against the timeline (out-of-range, or
 *  collapsed by upstream clamping). Matches the Python aligner's
 *  0.05 s last-ditch epsilon scaled to "noticeable but not silly":
 *  a quarter of a beat is small enough to read as a point on the
 *  bars row at any reasonable zoom. */
const MIN_BEAT_WIDTH = 0.05;

/** Pure beat-positioning pass for the lyrics row. Walks every line and
 *  every word once, resolving audio-sec → beat against the supplied
 *  timeline. Extracted out of the render so the result can be memoised
 *  on its real dependencies (lines / offset / timeline / structure)
 *  rather than rebuilt on every playhead tick. */
function positionLyricLines(
  lines: readonly LyricLine[],
  timeline: JotTimeline,
  drumsT0Sec: number,
  structuralBeats: readonly number[],
  offsetSec: number,
  voiceBeats: number,
): PositionedLine[] {
  const out: PositionedLine[] = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    // Blank lines (LRC instrumental gap stamps with no text or words)
    // produce no visible chip - rendering an empty span just leaves a
    // bare start-beat tick floating above the audio waveform.
    if (line.text.trim() === '' && (!line.words || line.words.length === 0)) {
      continue;
    }
    const lineSec = line.startSec + offsetSec;

    let startBeat: number | undefined;
    let endBeat: number | undefined;
    let wordPositions: PositionedWord[] | undefined;

    if (line.words && line.words.length > 0) {
      // Walk the words once, dropping any whose start beat falls
      // outside the timeline (rare; usually the whole line is in-
      // range or out). End-beats are resolved against the timeline
      // too; an out-of-range end falls back to `startBeat +
      // MIN_BEAT_WIDTH` so the cell has a defined, visible width.
      // The sourceIdx is preserved so word-level highlighting still
      // matches `activeWordIndexAt` (indexed against the unfiltered
      // source array) when edge words are dropped.
      const inRange: {
        sourceIdx: number;
        source: LyricWord;
        startBeat: number;
        endBeat: number;
      }[] = [];
      for (let wi = 0; wi < line.words.length; wi++) {
        const w = line.words[wi];
        const ws = audioSecToBeat(
          w.startSec + offsetSec,
          timeline,
          drumsT0Sec,
          structuralBeats,
        );
        if (ws === undefined) continue;
        const weRaw = audioSecToBeat(
          w.endSec + offsetSec,
          timeline,
          drumsT0Sec,
          structuralBeats,
        );
        const we =
          weRaw !== undefined && weRaw > ws ? weRaw : ws + MIN_BEAT_WIDTH;
        inRange.push({ sourceIdx: wi, source: w, startBeat: ws, endBeat: we });
      }
      if (inRange.length > 0) {
        startBeat = inRange[0].startBeat;
        endBeat = inRange[inRange.length - 1].endBeat;
        wordPositions = inRange.map((w) => ({
          sourceIdx: w.sourceIdx,
          text: w.source.text,
          beatOffset: w.startBeat - startBeat!,
          beatWidth: w.endBeat - w.startBeat,
          source: w.source,
        }));
      }
    } else {
      startBeat = audioSecToBeat(lineSec, timeline, drumsT0Sec, structuralBeats);
      if (startBeat !== undefined) {
        // End beat = next-line's start (clamped to voiceBeats) so the
        // text has a defined max-width region. The final line uses
        // voiceBeats as the bound.
        endBeat = voiceBeats;
        for (let j = i + 1; j < lines.length; j++) {
          const next = audioSecToBeat(
            lines[j].startSec + offsetSec,
            timeline,
            drumsT0Sec,
            structuralBeats,
          );
          if (next !== undefined) {
            endBeat = next;
            break;
          }
        }
      }
    }

    if (startBeat === undefined || endBeat === undefined) continue;
    // Tiny non-zero floor so consecutive same-timestamp lines (or a
    // single-word line) still establish a visible positioning context
    // rather than collapsing to width 0.
    if (endBeat - startBeat < 0.05) endBeat = startBeat + 0.05;
    out.push({
      i,
      text: line.text,
      startBeat,
      endBeat,
      wordPositions,
    });
  }
  return out;
}

/** Common drag/drop props passed to every mixer row. Subset of MixerRowDragProps
 *  from mixer.tsx; we re-declare here to avoid a circular import.
 *  See mixer.tsx::MixerRowDragProps for the canonical doc. */
type LyricsRowDragProps = {
  idx: number;
  dragFromIdx: number | undefined;
  dropTargetIdx: number | undefined;
  onDragStartIdx: (i: number) => void;
  onDropTargetIdx: (i: number | undefined) => void;
  onMoveTrack: (from: number, to: number) => void;
  onResetDrag: () => void;
  groupStart: boolean;
  groupEnd: boolean;
  inGroup: boolean;
  onResizeGutterStart: (e: React.PointerEvent<HTMLDivElement>) => void;
};

/** Per-row overflow menu on lyrics tracks. Hosts the time-offset stepper
 *  (replacing the inline gutter control) and the "Remove lyrics" action;
 *  same trigger position as the audio-track row's overflow so the chrome
 *  reads identically across the mixer. */
const LyricsOverflowMenu = ({
  id,
  offsetSec,
  onSetOffset,
  onRemove,
}: {
  id: LyricsTrackId;
  offsetSec: number;
  onSetOffset: (sec: number) => void;
  onRemove: () => void;
}) => (
  <DropdownButton
    label="⋯"
    className={mixerStyles.overflowTrigger}
    title="More actions for this lyrics track"
  >
    {(close) => (
      <>
        <label
          className={styles.offsetStepperRow}
          title="Lyrics offset (seconds). Positive values delay the lyric chips relative to the audio."
        >
          <span>Offset</span>
          <span className={styles.offsetStepperControl}>
            <NumberStepper
              value={offsetSec}
              onChange={onSetOffset}
              step={LYRICS_OFFSET_STEP_SEC}
              min={LYRICS_OFFSET_MIN_SEC}
              max={LYRICS_OFFSET_MAX_SEC}
              ariaLabel="Lyrics time offset (seconds)"
              title="Lyrics offset (seconds)"
              testId={`lyrics-offset-input-${id}`}
            />
            <span className={styles.offsetStepperUnit}>s</span>
          </span>
        </label>
        <span className={dropdownStyles.dropdownDivider} aria-hidden="true" />
        <button
          type="button"
          className={dropdownStyles.dropdownItem}
          role="menuitem"
          onClick={() => {
            onRemove();
            close();
          }}
          data-testid="lyrics-clear"
          title="Remove this lyrics track from the mixer"
        >
          Remove track
        </button>
      </>
    )}
  </DropdownButton>
);

/** One absolutely-positioned line chip on the bars row. Pure props +
 *  `observer` (so `React.memo` short-circuits when nothing changed): the
 *  parent re-renders on every line/word transition and re-keys this child,
 *  but identical props mean the body never runs unless `isActive` or
 *  `activeWordIdx` (this line's word-level highlight target) actually
 *  flipped. */
const LyricLineChip = observer(
  ({
    lineIdx,
    startBeat,
    endBeat,
    text,
    wordPositions,
    shifts,
    isActive,
    activeWordIdx,
  }: {
    lineIdx: number;
    startBeat: number;
    endBeat: number;
    text: string;
    wordPositions: PositionedWord[] | undefined;
    shifts: Map<string, number>;
    isActive: boolean;
    /** Defined only when this line is the active line; otherwise undefined
     *  so non-active lines stay memo-stable across word transitions. */
    activeWordIdx: number | undefined;
  }) => {
    const wordAligned = wordPositions !== undefined;
    return (
      <span
        className={classNames(
          styles.lyricLine,
          wordAligned && styles.lyricLineWordAligned,
          isActive && styles.lyricLineActive,
        )}
        style={
          {
            ['--lyric-start-beat' as string]: startBeat,
            ['--lyric-end-beat' as string]: endBeat,
          } as React.CSSProperties
        }
        title={text}
        data-testid={`lyrics-line-${lineIdx}`}
      >
        {wordAligned
          ? wordPositions!.map((w) => (
              <LyricWordChip
                key={w.sourceIdx}
                lineIdx={lineIdx}
                wordIdx={w.sourceIdx}
                word={w}
                shift={shifts.get(lyricShiftKey(lineIdx, w.sourceIdx)) ?? 0}
                isActive={activeWordIdx === w.sourceIdx}
              />
            ))
          : text}
      </span>
    );
  },
);

const LyricWordChip = observer(
  ({
    lineIdx,
    wordIdx,
    word,
    shift,
    isActive,
  }: {
    lineIdx: number;
    wordIdx: number;
    word: PositionedWord;
    shift: number;
    isActive: boolean;
  }) => {
    const wordStyle: Record<string, string | number> = {
      '--lyric-word-beat-offset': word.beatOffset,
      '--lyric-word-beat-width': word.beatWidth,
    };
    if (shift > 0) wordStyle['--lyric-word-shift'] = `${shift}px`;
    return (
      <span
        className={classNames(
          styles.lyricWord,
          isActive && styles.lyricWordActive,
        )}
        style={wordStyle as React.CSSProperties}
        title={buildWordDebugTitle(word.source)}
        data-testid={`lyrics-word-${lineIdx}-${wordIdx}`}
      >
        <span className={styles.lyricWordText}>{word.text}</span>
      </span>
    );
  },
);

/**
 * The time-aligned lyrics row in the unified mixer. Same gutter geometry
 * as `AudioTrackRow` / `InstrumentRow`: drag handle on the leftmost edge, a
 * stacked label + controls column on the right, sticky-left so it stays
 * pinned during horizontal scroll. The right-hand bars row carries one
 * `<span>` per lyric line, absolutely positioned at the beat offset
 * derived from the line's `startSec + offsetSec` against the rendered
 * jot's per-bar timeline.
 *
 * The row is session-only; no jot metadata, no persistence. The
 * `lyricsStore` clears on every jot replace via the existing
 * "wholesale song change" loaders, so a stale lyric set can't bleed
 * onto a new song.
 */
export const LyricsRow = observer(
  ({
    id,
    jot,
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
    id: LyricsTrackId;
    jot: RenderedJot;
    onSeek: (x: number) => void;
  } & LyricsRowDragProps) => {
    const store = React.useContext(JotViewStoreContext);
    const track = lyricsStore.get(id);
    // Guard: the reaction in JotViewStore drops dead lyrics ids on the
    // same MobX tick a `remove()` happens, so this gap is one-frame at
    // most. Render nothing rather than crash if the maps race.
    if (!track) return null;
    const lines = track.lines;
    const offsetSec = track.offsetSec;
    const sourceLabel = track.sourceLabel;
    const alignPhase = store?.lyricsAlignStatuses.get(id)?.phase;
    const isAligning = alignPhase === 'aligning' || alignPhase === 'queued';
    const alignLabel =
      alignPhase === 'queued'
        ? 'Queued, waiting for the GPU'
        : 'Aligning lyrics to audio';

    // Voice-level total beats for the bars-row width. Same pattern as
    // AudioTrackRow / InstrumentRow: read off the structural cache (zoom-
    // invariant) so this row doesn't re-render on every wheel tick;
    // CSS calc handles the per-zoom pixel scaling.
    const structureVoice = jot.structure.voices[0];
    let voiceBeats = 0;
    if (structureVoice) {
      for (const b of structureVoice.bars) voiceBeats += b.beats;
    }
    const structuralBeats = React.useMemo(
      () => (structureVoice?.bars ?? []).map((b) => b.beats),
      [structureVoice],
    );

    // The playback timeline is the canonical source for audio-sec → beat
    // mapping. `jot.timeline` is a MobX computed that mirrors what the
    // bars header / audio waveforms use; depending on `jot` (not on
    // `jotPlayer.timeline`) keeps this row off the per-frame playback
    // observable graph.
    const timeline = jot.timeline;
    const drumsT0Sec = jotPlayer.drumsT0Sec;
    const pxPerBeat = jot.pxPerBeat;

    // Pre-compute each line's beat positions. For lines with `words`,
    // each word resolves to its own [startBeat, endBeat] cell; the
    // line's bounding box stretches from the first word's start to the
    // last word's end (true line duration). For word-less lines (LRCLIB
    // / plain LRC) we fall back to the legacy single-stamp chip width:
    // bound by the next line's start so text doesn't run into the
    // following line at low zoom.
    //
    // Memoised on the pure inputs - none of which tick per frame - so
    // the active-line/word highlight (driven imperatively below) doesn't
    // pull this walk along with it.
    const positioned = React.useMemo(
      () =>
        positionLyricLines(
          lines,
          timeline,
          drumsT0Sec,
          structuralBeats,
          offsetSec,
          voiceBeats,
        ),
      [lines, timeline, drumsT0Sec, structuralBeats, offsetSec, voiceBeats],
    );

    // Word-collision avoidance. Word spans are absolutely positioned at
    // `beatOffset * --px-per-beat`, so when two words land on nearly
    // identical beats (e.g. a Japanese sokuon `ッ` immediately before
    // its host syllable), their glyphs render on top of each other.
    // `computeLyricShifts` runs the same left-to-right walk the legacy
    // DOM round-trip did, but measures each glyph's true text width via
    // an off-screen canvas whose font string mirrors the variable-font
    // axes (wdth / wght / letter-spacing) the CSS clamps against
    // `--px-per-beat`. Result is the source of truth; the JSX writes
    // each shift into `--lyric-word-shift` as a render sink, and the
    // CSS calc subtracts it from the cell width so the shifted cell's
    // right edge stays anchored to the word's `endSec`.
    //
    // Read `fontReady` so this row re-renders once Bricolage Grotesque
    // loads; canvas measurement before that uses the fallback stack and
    // is off by a glyph or two on long words.
    const fontReady = lyricsMeasurer.fontReady;
    // Deliberate decision: measure every word with `isActive=false`. The
    // active-word weight override (wght 600) only shifts canvas-measured
    // widths by sub-pixel amounts on the active line, well inside
    // `MIN_GAP_PX = 4` in `computeLyricShifts`. Dropping the active
    // dependency keeps shift recomputation off the playhead cadence
    // entirely - shifts now rebuild only when geometry (positioned,
    // pxPerBeat) or font readiness change.
    const shifts = React.useMemo(() => {
      const measureInputs: LyricLineMeasureInput[] = positioned
        .filter((p) => p.wordPositions !== undefined)
        .map((p) => ({
          lineIdx: p.i,
          activeWordSourceIdx: undefined,
          words: p.wordPositions!.map((w) => ({
            sourceIdx: w.sourceIdx,
            text: w.text,
            beatOffset: w.beatOffset,
          })),
        }));
      return computeLyricShifts(measureInputs, pxPerBeat);
      // `fontReady` is intentionally in the deps so a font-load
      // completion re-derives shifts against real glyph widths.
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [positioned, pxPerBeat, fontReady]);

    // Reactive active-line/word state. Reading `playhead.activeLineIdx`
    // here re-renders `LyricsRow` only when the active line index *flips*
    // (a few times per second), not on every `currentTime` tick - the
    // computed dedupes by output value. Child chips are `observer`s
    // wrapped in `React.memo`, so identical props bail; only the two
    // chips whose `isActive` actually flipped re-run. Getters read the
    // live track from `lyricsStore` (rather than closing over `lines` /
    // `offsetSec`) so updates to those propagate without re-initialising
    // the local observable.
    const playhead = useLocalObservable(() => ({
      get audioTimeNow(): number {
        return jotPlayer.currentTime + jotPlayer.drumsT0Sec;
      },
      get activeLineIdx(): number | undefined {
        const t = lyricsStore.get(id);
        if (!t) return undefined;
        return activeLineIndexAt(t.lines, this.audioTimeNow, t.offsetSec);
      },
      get activeWordIdx(): number | undefined {
        const lineIdx = this.activeLineIdx;
        if (lineIdx === undefined) return undefined;
        const t = lyricsStore.get(id);
        if (!t) return undefined;
        return activeWordIndexAt(t.lines, lineIdx, this.audioTimeNow, t.offsetSec);
      },
    }));

    const drop = useDropTarget({
      idx,
      dragFromIdx,
      dropTargetIdx,
      onDropTargetIdx,
      onMoveTrack,
      onResetDrag,
    });
    const isDragging = dragFromIdx === idx;

    return (
      <div
        className={classNames(
          styles.lyricsRow,
          groupStart && styles.mixerRowGroupStart,
          groupEnd && styles.mixerRowGroupEnd,
          inGroup && styles.mixerRowInGroup,
          isDragging && styles.mixerRowDragging,
          drop.isDropIndicatorAbove && styles.mixerDropIndicatorAbove,
          drop.isDropIndicatorBelow && styles.mixerDropIndicatorBelow,
        )}
        data-testid="lyrics-row"
        onDragOver={drop.onDragOver}
        onDragLeave={drop.onDragLeave}
        onDrop={drop.onDrop}
      >
        <div className={styles.lyricsGutter} style={{ height: LYRICS_ROW_HEIGHT }}>
          <div
            className={styles.dragHandle}
            draggable={true}
            onMouseDown={(e) => e.stopPropagation()}
            onDragStart={(e) => {
              e.dataTransfer.setData(
                'application/x-drumjot-mixer-row',
                String(idx),
              );
              e.dataTransfer.setData('text/plain', String(idx));
              e.dataTransfer.effectAllowed = 'move';
              onDragStartIdx(idx);
            }}
            onDragEnd={onResetDrag}
            title="Lyrics row (drag to reorder)"
            aria-label="Reorder lyrics row"
            role="button"
          >
            ⋮⋮
          </div>
          <GutterResizeHandle onResizeStart={onResizeGutterStart} />
          <div className={styles.lyricsContent}>
            <div className={styles.lyricsHeader}>
              <div className={styles.lyricsLabel}>
                <span className={styles.lyricsTitle}>Lyrics</span>
                <span className={styles.lyricsSourceRow}>
                  <span className={styles.lyricsSource} title={sourceLabel}>
                    {sourceLabel}
                  </span>
                  {isAligning && (
                    <span
                      className={styles.lyricsAlignSpinner}
                      title={`${alignLabel}…`}
                      aria-label={alignLabel}
                      role="status"
                      data-testid={`lyrics-align-spinner-${id}`}
                    />
                  )}
                </span>
              </div>
              <LyricsOverflowMenu
                id={id}
                offsetSec={offsetSec}
                onSetOffset={(v) => lyricsStore.setOffsetSec(id, v)}
                onRemove={() => store?.removeLyricsTrack(id)}
              />
            </div>
          </div>
        </div>
        <div
          className={styles.lyricsBarsRow}
          style={
            {
              ['--voice-beats' as string]: voiceBeats,
              height: LYRICS_ROW_HEIGHT,
            } as React.CSSProperties
          }
          onClick={(e) => seekFromClick(e, onSeek)}
        >
          {positioned.map((p) => {
            const isActive = playhead.activeLineIdx === p.i;
            return (
              <LyricLineChip
                key={p.i}
                lineIdx={p.i}
                startBeat={p.startBeat}
                endBeat={p.endBeat}
                text={p.text}
                wordPositions={p.wordPositions}
                shifts={shifts}
                isActive={isActive}
                activeWordIdx={isActive ? playhead.activeWordIdx : undefined}
              />
            );
          })}
          <Playhead onSeek={onSeek} />
        </div>
      </div>
    );
  },
);

/** Locally-mirrored drop-target hook. The mixer's `useMixerRowDropTarget`
 *  is private to mixer.tsx; we duplicate the small body here rather than
 *  exporting it (and trading a tighter mixer.tsx surface for a circular
 *  cross-file dependency). The MIME constant is shared with the mixer
 *  via string literal so foreign drops are still rejected. */
function useDropTarget(opts: {
  idx: number;
  dragFromIdx: number | undefined;
  dropTargetIdx: number | undefined;
  onDropTargetIdx: (i: number | undefined) => void;
  onMoveTrack: (from: number, to: number) => void;
  onResetDrag: () => void;
}) {
  const { idx, dragFromIdx, dropTargetIdx, onDropTargetIdx, onMoveTrack, onResetDrag } = opts;
  const MIME = 'application/x-drumjot-mixer-row';
  const onDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    if (dragFromIdx === undefined) return;
    if (!e.dataTransfer.types.includes(MIME)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    const rect = e.currentTarget.getBoundingClientRect();
    const isTopHalf = e.clientY < rect.top + rect.height / 2;
    const target = isTopHalf ? idx : idx + 1;
    if (target !== dropTargetIdx) onDropTargetIdx(target);
  };
  const onDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    const next = e.relatedTarget as Node | null;
    if (next && e.currentTarget.contains(next)) return;
    if (dropTargetIdx === idx || dropTargetIdx === idx + 1) onDropTargetIdx(undefined);
  };
  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    if (dragFromIdx === undefined) return;
    const data = e.dataTransfer.getData(MIME);
    if (!data) return;
    e.preventDefault();
    const from = parseInt(data, 10);
    if (Number.isFinite(from) && dropTargetIdx !== undefined) {
      onMoveTrack(from, dropTargetIdx);
    }
    onResetDrag();
  };
  const isDropIndicatorAbove = dropTargetIdx === idx && dragFromIdx !== undefined;
  const isDropIndicatorBelow = dropTargetIdx === idx + 1 && dragFromIdx !== undefined;
  return { onDragOver, onDragLeave, onDrop, isDropIndicatorAbove, isDropIndicatorBelow };
}
