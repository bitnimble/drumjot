import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { RenderedJot } from 'src/jot';
import {
  LYRICS_OFFSET_MAX_SEC,
  LYRICS_OFFSET_MIN_SEC,
  LYRICS_OFFSET_STEP_SEC,
  LyricWord,
  audioSecToBeat,
  lyricsStore,
} from 'src/lyrics';
import { buildTimeline, jotPlayer } from 'src/playback';
import { ClearButton } from './components/icon_button';
import { GutterResizeHandle } from './components/gutter_resize_handle';
import { NumberStepper } from './components/number_stepper';
import styles from './lyrics_row.module.css';
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

/** Numeric stepper for the lyrics offset, with clamping baked in via
 *  the shared {@link NumberStepper}'s min/max contract. */
const OffsetInput = ({
  value,
  onChange,
}: {
  value: number;
  onChange: (v: number) => void;
}) => (
  <NumberStepper
    value={value}
    onChange={onChange}
    step={LYRICS_OFFSET_STEP_SEC}
    min={LYRICS_OFFSET_MIN_SEC}
    max={LYRICS_OFFSET_MAX_SEC}
    ariaLabel="Lyrics time offset (seconds)"
    title="Lyrics offset (seconds)"
    testId="lyrics-offset-input"
  />
);

/**
 * The time-aligned lyrics row in the unified mixer. Same gutter geometry
 * as `AudioTrackRow` / `PitchRow`: drag handle on the leftmost edge, a
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
    jot: RenderedJot;
    onSeek: (x: number) => void;
  } & LyricsRowDragProps) => {
    const lines = lyricsStore.lines;
    const offsetSec = lyricsStore.offsetSec;
    const sourceLabel = lyricsStore.sourceLabel ?? 'Lyrics';

    // Voice-level total beats for the bars-row width. Same pattern as
    // AudioTrackRow / PitchRow: read off the structural cache (zoom-
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
    // mapping. Live timeline when one is in flight (so per-bar tempo
    // overrides + lead-in stay in sync with the playhead); otherwise a
    // one-shot `buildTimeline` matches what the bars header / audio
    // waveforms use before Play.
    const liveTimeline = jotPlayer.timeline;
    const timeline =
      liveTimeline.bars.length > 0 && liveTimeline.rendered === jot
        ? liveTimeline
        : buildTimeline(jot);
    const drumsT0Sec = jotPlayer.drumsT0Sec;

    // Compute the active line against the live audio clock + offset. The
    // `currentTime` observable read in `jotPlayer.currentTime` ticks
    // every rAF during playback so this row re-renders at the playhead's
    // cadence; that's the same per-frame cost the score's other
    // playhead-driven elements already pay.
    const audioTimeNow = jotPlayer.currentTime + drumsT0Sec;
    const activeIdx = lyricsStore.activeLineIndexAt(audioTimeNow);
    const activeWordIdx =
      activeIdx !== undefined
        ? lyricsStore.activeWordIndexAt(activeIdx, audioTimeNow)
        : undefined;

    // Pre-compute each line's beat positions. For lines with `words`,
    // each word resolves to its own [startBeat, endBeat] cell; the
    // line's bounding box stretches from the first word's start to the
    // last word's end (true line duration). For word-less lines (LRCLIB
    // / plain LRC) we fall back to the legacy single-stamp chip width:
    // bound by the next line's start so text doesn't run into the
    // following line at low zoom.
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
    type Positioned = {
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
    const positioned: Positioned[] = [];
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
      positioned.push({
        i,
        text: line.text,
        startBeat,
        endBeat,
        wordPositions,
      });
    }

    // Word-collision avoidance. Word spans are absolutely positioned at
    // `beatOffset * --px-per-beat`, so when two words land on nearly
    // identical beats (e.g. a Japanese sokuon `ッ` immediately before
    // its host syllable), their glyphs render on top of each other.
    // We post-process the DOM: walk each word-aligned line left-to-
    // right, measure each glyph's true text width via a Range on the
    // inner `.lyricWordText` span (so the trailing rule and active-
    // state padding don't perturb the measurement), and write a per-
    // word `--lyric-word-shift` px offset that pushes colliding words
    // just enough to keep a minimum gap. The CSS calc that consumes
    // `--lyric-word-shift` also subtracts it from the cell width, so
    // the shifted cell's right edge stays anchored to the word's
    // `endSec` (the cell shrinks rather than slides).
    const barsRowRef = React.useRef<HTMLDivElement>(null);
    const adjustWordSpacing = React.useCallback(() => {
      const row = barsRowRef.current;
      if (!row) return;
      const pxPerBeat = parseFloat(
        getComputedStyle(row).getPropertyValue('--px-per-beat'),
      );
      if (!Number.isFinite(pxPerBeat) || pxPerBeat <= 0) return;
      const MIN_GAP_PX = 4;
      const lines = row.querySelectorAll<HTMLElement>('[data-lyric-word-line="1"]');
      const range = document.createRange();
      for (const line of lines) {
        const words = line.querySelectorAll<HTMLElement>('[data-lyric-word="1"]');
        let prevRight = -Infinity;
        for (const w of words) {
          const beatOffset = parseFloat(w.dataset.beatOffset ?? '0');
          const natural = beatOffset * pxPerBeat;
          const textNode =
            w.querySelector<HTMLElement>('[data-lyric-word-text="1"]') ?? w;
          range.selectNodeContents(textNode);
          const textWidth = range.getBoundingClientRect().width;
          const required = Math.max(natural, prevRight + MIN_GAP_PX);
          const shift = required - natural;
          // Write only when non-zero to keep the inline style attribute
          // tidy and avoid no-op style mutations on the first paint of
          // most words.
          if (shift > 0) {
            w.style.setProperty('--lyric-word-shift', `${shift}px`);
          } else if (w.style.getPropertyValue('--lyric-word-shift')) {
            w.style.removeProperty('--lyric-word-shift');
          }
          prevRight = required + textWidth;
        }
      }
    }, []);

    // Re-run after every render so word add/remove and active-state
    // changes both leave spacing consistent. The work is a pure
    // measure-then-write pass over the row's DOM; cheap because lyric
    // counts per row are small.
    React.useLayoutEffect(() => {
      adjustWordSpacing();
    });

    // Zoom is propagated via the `--px-per-beat` CSS variable rather
    // than React state (see ScoreZoomVar in jot_view.tsx), so this row
    // doesn't re-render on zoom. A ResizeObserver on the bars-row
    // catches the resulting width change and re-runs the spacing
    // adjustment so shifts stay correct across the full zoom range.
    React.useEffect(() => {
      const row = barsRowRef.current;
      if (!row) return;
      const ro = new ResizeObserver(() => adjustWordSpacing());
      ro.observe(row);
      return () => ro.disconnect();
    }, [adjustWordSpacing]);

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
            <div className={styles.lyricsLabel}>
              <span className={styles.lyricsTitle}>Lyrics</span>
              <span className={styles.lyricsSource} title={sourceLabel}>
                {sourceLabel}
              </span>
            </div>
            <div className={styles.lyricsControls}>
              <OffsetInput
                value={offsetSec}
                onChange={(v) => lyricsStore.setOffsetSec(v)}
              />
              <span className={styles.offsetUnit}>s</span>
              <ClearButton
                onClear={() => lyricsStore.clear()}
                label="Remove lyrics"
                testId="lyrics-clear"
              />
            </div>
          </div>
        </div>
        <div
          ref={barsRowRef}
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
            const isActive = p.i === activeIdx;
            const wordAligned = p.wordPositions !== undefined;
            return (
              <span
                key={p.i}
                className={classNames(
                  styles.lyricLine,
                  wordAligned && styles.lyricLineWordAligned,
                  isActive && styles.lyricLineActive,
                )}
                style={
                  {
                    ['--lyric-start-beat' as string]: p.startBeat,
                    ['--lyric-end-beat' as string]: p.endBeat,
                  } as React.CSSProperties
                }
                title={p.text}
                data-testid={`lyrics-line-${p.i}`}
                data-lyric-word-line={wordAligned ? '1' : undefined}
              >
                {wordAligned
                  ? p.wordPositions!.map((w) => (
                      <span
                        key={w.sourceIdx}
                        className={classNames(
                          styles.lyricWord,
                          isActive && w.sourceIdx === activeWordIdx && styles.lyricWordActive,
                        )}
                        style={
                          {
                            ['--lyric-word-beat-offset' as string]: w.beatOffset,
                            ['--lyric-word-beat-width' as string]: w.beatWidth,
                          } as React.CSSProperties
                        }
                        title={buildWordDebugTitle(w.source)}
                        data-testid={`lyrics-word-${p.i}-${w.sourceIdx}`}
                        data-lyric-word="1"
                        data-beat-offset={w.beatOffset}
                      >
                        <span className={styles.lyricWordText} data-lyric-word-text="1">
                          {w.text}
                        </span>
                      </span>
                    ))
                  : p.text}
              </span>
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
