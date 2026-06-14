import classNames from 'classnames';
import { observer, useLocalObservable } from 'mobx-react-lite';
import React from 'react';
import { RenderedJot } from 'src/jot';
import {
  LyricsTrackId,
  activeLineIndexAt,
  activeWordIndexAt,
  furiganaAnnotator,
  lyricsStore,
} from 'src/lyrics';
import { jotPlayer } from 'src/jot_view/playback';
import { LyricsPresenterContext, LyricsAlignStoreContext } from '../contexts';
import { GutterResizeHandle } from '../components/gutter_resize_handle';
import { MixerRowDragProps, useMixerRowDropTarget } from '../mixer/mixer_drag';
import {
  LyricLineMeasureInput,
  computeLyricShifts,
  lyricsMeasurer,
} from './lyrics_measure';
import { positionLyricLines } from './lyric_layout';
import { WindowedLines } from './lyric_chips';
import { LyricsOverflowMenu } from './lyrics_overflow_menu';
import styles from './lyrics_row.module.css';
import { Playhead } from '../playback/playhead';
import { seekFromClick } from '../score/seek';
import { barsRowWidthSeed } from '../utils/windowing';

/** Taller than the audio-track row (which is 56) to fit the enlarged 22px
 *  karaoke text plus the furigana strip stacked above it. Rows stack
 *  independently, so a taller lyrics row doesn't disturb the others. */
const LYRICS_ROW_HEIGHT = 64;

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
  } & MixerRowDragProps) => {
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

    const drop = useMixerRowDropTarget({
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
