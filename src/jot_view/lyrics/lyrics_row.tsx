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
  RubySegment,
  activeLineIndexAt,
  activeWordIndexAt,
  furiganaAnnotator,
  lyricsStore,
  serializeEnhancedLrc,
} from 'src/lyrics';
import { downloadTextFile } from 'src/download';
import { jotPlayer } from 'src/playback';
import {
  LyricsPresenterContext,
  LyricsAlignStoreContext,
  ViewportStoreContext,
} from '../contexts';
import { DropdownButton, dropdownStyles } from '../components/dropdown';
import { GutterResizeHandle } from '../components/gutter_resize_handle';
import { NumberStepper } from '../components/number_stepper';
import {
  LyricLineMeasureInput,
  computeLyricShifts,
  lyricShiftKey,
  lyricsMeasurer,
} from './lyrics_measure';
import { PositionedLine, PositionedWord, positionLyricLines } from './lyric_layout';
import styles from './lyrics_row.module.css';
import mixerStyles from '../mixer/mixer.module.css';
import { Playhead } from '../playback/playhead';
import { seekFromClick } from '../score/score';
import { barsRowWidthSeed, intersectsBeatRange } from '../utils/windowing';

/** Taller than the audio-track row (which is 56) to fit the enlarged 22px
 *  karaoke text plus the furigana strip stacked above it. Rows stack
 *  independently, so a taller lyrics row doesn't disturb the others. */
const LYRICS_ROW_HEIGHT = 64;

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

/** Build a filesystem-friendly `.lrc` name from a track's source label.
 *  Drops a leading `Source · ` prefix, strips an existing `.lrc`
 *  extension, and replaces filename-hostile characters. */
function lyricsExportFilename(sourceLabel: string): string {
  let base = sourceLabel.replace(/^[^·]*·\s*/, '').trim() || sourceLabel;
  base = base.replace(/\.lrc$/i, '');
  base = base
    .replace(/[^\p{L}\p{N}\-_. ]+/gu, '_')
    .replace(/\s+/g, ' ')
    .trim();
  return `${base || 'lyrics'}.lrc`;
}

/** Serialize a lyrics track to an enhanced-LRC file (with word-level
 *  durations + the offset nudge) and trigger a download. No-op if the
 *  track raced a removal. */
function exportLyricsTrack(id: LyricsTrackId): void {
  const track = lyricsStore.get(id);
  if (!track) return;
  const text = serializeEnhancedLrc(track.lines, { offsetSec: track.offsetSec });
  downloadTextFile(lyricsExportFilename(track.sourceLabel), text);
}

/** Per-row overflow menu on lyrics tracks. Hosts the time-offset stepper
 *  (replacing the inline gutter control), the enhanced-LRC export, and
 *  the "Remove lyrics" action; same trigger position as the audio-track
 *  row's overflow so the chrome reads identically across the mixer. */
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
            exportLyricsTrack(id);
            close();
          }}
          data-testid={`lyrics-export-${id}`}
          title="Download this track as an enhanced LRC file (round-trips word timings)"
        >
          Export enhanced LRC
        </button>
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
    // Surfaces of this line's words, joined for context-aware furigana.
    // Memoised on `wordPositions` so each `LyricWordChip` receives a
    // stable array reference and its memo still bails across playhead
    // transitions (the word texts only change when the line re-positions).
    const lineWordTexts = React.useMemo(
      () => (wordPositions ? wordPositions.map((w) => w.text) : []),
      [wordPositions],
    );
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
          ? wordPositions!.map((w, i) => (
              <LyricWordChip
                key={w.sourceIdx}
                lineIdx={lineIdx}
                wordIdx={w.sourceIdx}
                word={w}
                shift={shifts.get(lyricShiftKey(lineIdx, w.sourceIdx)) ?? 0}
                isActive={activeWordIdx === w.sourceIdx}
                lineWordTexts={lineWordTexts}
                wordPosIndex={i}
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
    lineWordTexts,
    wordPosIndex,
  }: {
    lineIdx: number;
    wordIdx: number;
    word: PositionedWord;
    shift: number;
    isActive: boolean;
    /** Surfaces of every (in-range) word on this line, in render order;
     *  the furigana annotator tokenizes them together for context. Stable
     *  identity (memoised by the parent) so this `observer`+memo chip
     *  still bails on word/playhead transitions. */
    lineWordTexts: readonly string[];
    /** This word's position within {@link lineWordTexts}. */
    wordPosIndex: number;
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
        <span className={styles.lyricWordText}>
          <WordText words={lineWordTexts} index={wordPosIndex} />
        </span>
      </span>
    );
  },
);

/** Renders a word's glyphs, stacking hiragana furigana over kanji runs
 *  when the annotator has a reading for the text. Takes the whole line's
 *  word surfaces plus this word's index (not the bare text) so the reading
 *  is tokenized with sentence context: a chip the aligner split off a
 *  compound (実 out of 実は) reads correctly (じつ) instead of its lone-
 *  token reading (み). `segmentsForWords` is synchronous (bare text until
 *  the kuromoji dictionary resolves) and its `revision` read makes this
 *  `observer` re-render in place once readings arrive. Falls back to a
 *  plain text node when there's no ruby, so non-Japanese words render
 *  exactly as before. */
const WordText = observer(
  ({ words, index }: { words: readonly string[]; index: number }) => {
    const segments =
      furiganaAnnotator.segmentsForWords(words)[index] ??
      ([{ base: words[index] ?? '' }] as RubySegment[]);
    const hasRuby = segments.some((s) => s.reading !== undefined);
    if (!hasRuby) return <>{words[index] ?? ''}</>;
    return (
      <ruby className={styles.ruby}>
        {segments.map((seg: RubySegment, i) =>
          seg.reading !== undefined ? (
            <React.Fragment key={i}>
              {seg.base}
              <rt>{seg.reading}</rt>
            </React.Fragment>
          ) : (
            // Bare run (okurigana / kana / punctuation): base on the
            // baseline, no annotation column.
            <React.Fragment key={i}>{seg.base}</React.Fragment>
          ),
        )}
      </ruby>
    );
  },
);

/** The active-line/word state {@link WindowedLines} reads to mark its
 *  chips. Subset of the row's local playhead observable. */
type LyricsPlayhead = {
  readonly activeLineIdx: number | undefined;
  readonly activeWordIdx: number | undefined;
};

/**
 * Windowed DOM for the lyric-line chips. Split out of {@link LyricsRow}
 * so a scroll / zoom tick re-renders only this map, not the row gutter
 * (label, controls, overflow menu). Renders only lines whose beat span
 * intersects {@link JotViewStore.visibleBeatRange}. Reads the row's
 * playhead observable for the active-line/word highlight, so it also
 * re-renders on a line transition (a few times per second), the precise
 * thing each child {@link LyricLineChip}'s memo then short-circuits.
 */
const WindowedLines = observer(function WindowedLines({
  positioned,
  shifts,
  playhead,
}: {
  positioned: PositionedLine[];
  shifts: Map<string, number>;
  playhead: LyricsPlayhead;
}) {
  const viewport = React.useContext(ViewportStoreContext);
  const range = viewport?.visibleBeatRange ?? null;
  return (
    <>
      {positioned.map((p) => {
        if (!intersectsBeatRange(range, p.startBeat, p.endBeat - p.startBeat)) return null;
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
    </>
  );
});

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
    const presenter = React.useContext(LyricsPresenterContext);
    const lyricsAlign = React.useContext(LyricsAlignStoreContext);
    const track = lyricsStore.get(id);
    // Guard: the reaction in JotViewStore drops dead lyrics ids on the
    // same MobX tick a `remove()` happens, so this gap is one-frame at
    // most. Render nothing rather than crash if the maps race.
    if (!track) return null;
    const lines = track.lines;
    const offsetSec = track.offsetSec;
    const sourceLabel = track.sourceLabel;
    const alignPhase = lyricsAlign?.lyricsAlignStatuses.get(id)?.phase;
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
    // Re-derive shifts when furigana readings resolve: ruby widens a word
    // whose reading outruns its kanji, so the collision walk must re-run
    // once `segmentsFor` upgrades from bare text. Reading the counter here
    // (this row is an `observer`) re-renders on each resolution burst.
    const furiganaRevision = furiganaAnnotator.revision;
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
        .map((p) => {
          const wp = p.wordPositions!;
          // Context-aware: tokenize the line's words together so the
          // ruby widths the collision walk measures match what renders
          // (実 in 実は widens to じつ, not the lone-token み).
          const lineSegs = furiganaAnnotator.segmentsForWords(
            wp.map((w) => w.text),
          );
          return {
            lineIdx: p.i,
            activeWordSourceIdx: undefined,
            words: wp.map((w, j) => ({
              sourceIdx: w.sourceIdx,
              text: w.text,
              beatOffset: w.beatOffset,
              segments: lineSegs[j],
            })),
          };
        });
      return computeLyricShifts(measureInputs, pxPerBeat);
      // `fontReady` and `furiganaRevision` are intentionally in the deps:
      // a font-load completion or a furigana resolution re-derives shifts
      // against real glyph widths.
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [positioned, pxPerBeat, fontReady, furiganaRevision]);

    // True once any word in the track has a resolved furigana reading. Drives
    // the `.lyricsBarsRowFurigana` modifier, which reserves the ruby
    // annotation strip above the plain (no-ruby) words too, so every base
    // sits on one line instead of the kanji dropping below its neighbours.
    // Re-derives on `furiganaRevision` (readings resolve async).
    const trackHasFurigana = React.useMemo(() => {
      for (const p of positioned) {
        if (!p.wordPositions) continue;
        const lineSegs = furiganaAnnotator.segmentsForWords(
          p.wordPositions.map((w) => w.text),
        );
        if (
          lineSegs.some((wordSegs) =>
            wordSegs.some((s) => s.reading !== undefined),
          )
        ) {
          return true;
        }
      }
      return false;
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [positioned, furiganaRevision]);

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
                onRemove={() => presenter?.removeLyricsTrack(id)}
              />
            </div>
          </div>
        </div>
        <div
          className={classNames(
            styles.lyricsBarsRow,
            trackHasFurigana && styles.lyricsBarsRowFurigana,
          )}
          data-bars-row
          // Lyrics rows alone keep a scoped `--px-per-beat` (written by
          // `setBarsRowVars`) for their zoom-responsive font metrics +
          // word-cell geometry, which can't be percentages.
          data-lyrics-bars-row="1"
          style={
            {
              ['--voice-beats' as string]: voiceBeats,
              ['--bars-row-width' as string]: barsRowWidthSeed(jot, voiceBeats),
              height: LYRICS_ROW_HEIGHT,
            } as React.CSSProperties
          }
          onClick={(e) => seekFromClick(e, onSeek)}
        >
          <WindowedLines positioned={positioned} shifts={shifts} playhead={playhead} />
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
