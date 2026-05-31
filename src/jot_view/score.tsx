import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { NotePosition } from 'src/note_position';
import { NoteProvenanceEntry } from 'src/debug_zip';
import { Instrument, Modifier, Sticking } from 'src/dsl';
import { DEFAULT_GRID_DIVISION, gridDivisionFor } from 'src/grid';
import {
  RenderedJot,
  StructuralBar,
  StructuralNote,
  StructuralPatternSpan,
  StructuralTupletSpan,
  ViewConfig,
} from 'src/jot';
import { TICKS_PER_BEAT } from 'src/midi';
import { msOffsetToBeats } from 'src/tempo';
import { AudioTrack, jotPlayer } from 'src/playback';
import { waveformWorker } from 'src/playback/waveform_worker_client';
import sharedStyles from '../jot_view.module.css';
import { GutterResizeHandle } from './components/gutter_resize_handle';
import {
  BarTimingsContext,
  JotViewStoreContext,
  NoteProvenanceContext,
  NoteProvenanceContextValue,
  RenderedJotContext,
  SelectionContext,
} from './contexts';
import { Playhead } from './playback';
import styles from './score.module.css';

/**
 * Shared click-to-seek handler for the score bars row and the audio-track
 * waveforms. Bails on clicks that originated on a note, pattern label,
 * or anything else tagged `data-noseek` so those keep their own
 * behaviour. `e.currentTarget` is the bars-row element whose left edge
 * is x=0 in `bar.x` space, so `clientX - rect.left` is the bars-row-
 * local pixel regardless of horizontal scroll.
 */
export function seekFromClick(
  e: React.MouseEvent<HTMLDivElement>,
  onSeek: (x: number) => void
): void {
  if (e.button !== 0) return;
  if ((e.target as HTMLElement).closest('[data-noseek]')) return;
  const rect = e.currentTarget.getBoundingClientRect();
  onSeek(e.clientX - rect.left);
}

/**
 * Decides whether a label popover anchored to `anchorRef` should flip
 * above its anchor instead of below. Default placement is below; flips
 * when below-placement would extend past the score scroll viewport's
 * bottom edge; which sits flush with the playback bar's top, so
 * "extends past" means "hidden behind the playback bar / debug panel".
 *
 * Measured synchronously on open in `useLayoutEffect` so the flip class
 * is applied before paint (no one-frame flash of a wrongly-placed
 * label). Not re-measured on scroll/zoom: the popover is transient and
 * users dismiss + re-open if the score moves under them.
 *
 * Falls back to below-placement when neither side fits; better to
 * partially clip the bottom of the label than to cover the notehead.
 */
function usePopoverFlipAbove(
  anchorRef: React.RefObject<HTMLElement>,
  labelRef: React.RefObject<HTMLElement>,
  enabled: boolean
): boolean {
  const [flip, setFlip] = React.useState(false);
  React.useLayoutEffect(() => {
    if (!enabled) {
      setFlip(false);
      return;
    }
    const anchor = anchorRef.current;
    const label = labelRef.current;
    if (!anchor || !label) return;
    const aRect = anchor.getBoundingClientRect();
    const lRect = label.getBoundingClientRect();
    const SAFE = 8;
    const GAP = 16;
    const scroller = anchor.closest('[data-jot-scroller]') as HTMLElement | null;
    const scRect = scroller?.getBoundingClientRect();
    const bottomLimit = scRect?.bottom ?? window.innerHeight;
    const topLimit = scRect?.top ?? 0;
    const overflowsBelow = aRect.bottom + GAP + lRect.height > bottomLimit - SAFE;
    const fitsAbove = aRect.top - GAP - lRect.height > topLimit + SAFE;
    setFlip(overflowsBelow && fitsAbove);
  }, [enabled, anchorRef, labelRef]);
  return flip;
}

/**
 * True when `beat` falls inside any tuplet bracket on this bar. The
 * upper bound is inclusive because `endBeat` is now the last slot's
 * onset (see jot.ts); the final tuplet note sits exactly on it and is
 * still covered by the bracket.
 */
function coveredByTuplet(bar: StructuralBar, beat: number): boolean {
  const eps = 1e-6;
  return bar.tupletSpans.some((s) => beat >= s.startBeat - eps && beat <= s.endBeat + eps);
}

/**
 * Read the artist string from wherever a loader plausibly stashed it.
 * Today only the RLRR (Paradiddle map) loader surfaces an artist, on
 * `globalMetadata.rlrr.recordingMetadata.artist`; a top-level
 * `globalMetadata.artist` is accepted too so hand-authored DSL or a
 * future loader can populate it directly. Anything non-string or empty
 * returns `undefined`, which makes the call site fall back to the
 * title alone.
 */
export function extractArtist(jot: RenderedJot): string | undefined {
  const meta = jot.globalMetadata as Record<string, unknown>;
  const direct = meta.artist;
  if (typeof direct === 'string' && direct.trim() !== '') return direct.trim();
  const rlrr = meta.rlrr as { recordingMetadata?: { artist?: unknown } } | undefined;
  const rlrrArtist = rlrr?.recordingMetadata?.artist;
  if (typeof rlrrArtist === 'string' && rlrrArtist.trim() !== '') return rlrrArtist.trim();
  return undefined;
}

/**
 * Display string for the score's `<h2>`. Appends ` - <artist>` when the
 * artist is known (RLRR-loaded charts today), so the header reads
 * "Song Name - Artist Name". When no artist is known the title stands
 * alone. Empty when the jot has neither title nor artist; the caller
 * shows the "Untitled jot" placeholder in that case.
 */
export function formatDisplayTitle(jot: RenderedJot): string {
  const title = jot.title.trim();
  const artist = extractArtist(jot);
  if (title && artist) return `${title} - ${artist}`;
  if (title) return title;
  if (artist) return artist;
  return '';
}

export function formatSubtitle(jot: RenderedJot): string {
  const parts: string[] = [];
  const { bpm: globalBpm, time: globalTime, vol } = jot.globalMetadata;
  const { dominantBpm, dominantTime } = jot.dominantBpmAndTime;

  if (dominantBpm !== undefined) parts.push(`${dominantBpm} bpm`);
  else if (typeof globalBpm === 'number') parts.push(`${globalBpm} bpm`);
  else if (globalBpm) parts.push(`${globalBpm.start ?? '?'}-${globalBpm.end} bpm`);

  const time = dominantTime ?? globalTime;
  if (time) parts.push(`${time.count}/${time.unit}`);
  if (typeof vol === 'string') parts.push(vol);
  else if (vol) parts.push(`${vol.start ?? '?'} -> ${vol.end}`);
  return parts.join('  -  ');
}

export const Legend = observer(({ jot }: { jot: RenderedJot }) => {
  // Aggregate unique pitches across all voices, in first-seen order.
  // Cached on the jot itself (`legendPitches`) so the walk is shared
  // across observers and only recomputes when the structural cache
  // changes, not on every zoom tick.
  const entries = jot.legendPitches;
  if (entries.length === 0) return null;
  return (
    <div className={sharedStyles.legend}>
      {entries.map(([pitch, info]) => (
        <span key={pitch} className={sharedStyles.legendChip}>
          <span className={sharedStyles.legendSwatch} style={{ background: info.color }} />
          <strong>{pitch}</strong>
          {info.name ? <span>{info.name}</span> : null}
        </span>
      ))}
    </div>
  );
});

/**
 * Sticky-gutter header above the audio tracks / score that labels each
 * bar boundary with its 1-based bar number and the playback time at that
 * boundary (mm:ss). Tick marks sit on the same `bar.x` line as the
 * score's barlines below so the header reads as a ruler over the
 * timeline. Click-to-seek mirrors the score and audio-track rows.
 *
 * Per-bar timings come from the live playback timeline whenever it
 * matches the current jot (so tempo overrides and the lead-in offset
 * stay in sync with the playhead); otherwise we build a one-shot
 * timeline so the header still labels everything correctly before the
 * user hits Play.
 */
export const TimelineHeader = observer(
  ({
    jot,
    onSeek,
    onResizeGutterStart,
  }: {
    jot: RenderedJot;
    onSeek: (x: number) => void;
    /** Pointer-down handler for the gutter resize affordance rendered
     * on the right edge of this header's gutter. */
    onResizeGutterStart: (e: React.PointerEvent<HTMLDivElement>) => void;
  }) => {
    // Reading the structural cache (not `jot.resolved`) keeps this
    // header stable across zoom; the per-tick `--bar-start-beat` is
    // set inline, and CSS calc() multiplies by the score-root's
    // `--px-per-beat` to get the final pixel position. Without this
    // the header re-rendered every wheel tick, re-creating 100+ tick
    // marks just to reposition each by one calc-arithmetic step.
    const voice = jot.primaryStructuralVoice;
    if (!voice || voice.bars.length === 0) return null;

    const liveTimeline = jotPlayer.timeline;
    const timeline =
      liveTimeline.bars.length > 0 && liveTimeline.rendered === jot
        ? liveTimeline
        : jot.timeline;

    // Lead-in is materialised as negative-indexed bars by
    // `structureForVoice`, so a single sum over `bar.beats` covers
    // both pre-drum and drum content with no separate chrome offset.
    // Cached on the jot (`voiceBeats`) so all observers share one walk.
    const voiceBeats = jot.voiceBeats;

    // Effective tempo at each bar's downbeat, derived from the shared
    // tempo timeline. Mid-bar tempo changes inside a bar aren't shown
    // separately by the header pill; the displayed value tracks the
    // tempo in force at the bar's downbeat. Reading the cached
    // `barTempos` computed avoids rebuilding the layout on every
    // header render (the tempo timeline is structure-only input, so a
    // zoom tick doesn't invalidate it).
    const tempos = jot.barTempos;

    let cumBeats = 0;
    let prevTime: { count: number; unit: number } | undefined;
    // Tempo "carried out" of the previous bar (= its last segment's bpm).
    // Rounded so float jitter (119.97 vs 120.03) doesn't paint a change.
    let prevBpm: number | undefined;
    return (
      <div className={styles.timelineHeader}>
        <div className={styles.timelineHeaderGutter}>
          <span className={styles.timelineHeaderLabel}>Bar / Time</span>
          <GutterResizeHandle onResizeStart={onResizeGutterStart} />
        </div>
        <div
          className={styles.timelineHeaderBarsRow}
          style={{ ['--voice-beats' as string]: voiceBeats } as React.CSSProperties}
          onClick={(e) => seekFromClick(e, onSeek)}
        >
          {voice.bars.map((bar, i) => {
            const timing = timeline.bars[i];
            const timeSec = timing?.startSec ?? 0;
            const startBeat = cumBeats;
            cumBeats += bar.beats;
            const showTimeSig =
              !prevTime || bar.time.count !== prevTime.count || bar.time.unit !== prevTime.unit;
            prevTime = bar.time;
            // Walk the bar's tempo segments and emit a label whenever the
            // bpm changes (relative to the running bpm). The label at
            // segment.startBeat=0 sits in the bar tick's top row alongside
            // the bar number; later labels float at their beat-anchored
            // position so a mid-bar tempo change renders where it actually
            // takes effect, not at the next downbeat.
            const segments = tempos[i]?.segments ?? [];
            let downbeatBpm: number | undefined;
            const midBpmChanges: Array<{ beat: number; bpm: number }> = [];
            for (let s = 0; s < segments.length; s++) {
              const seg = segments[s];
              const bpm = Math.round(seg.bpm);
              if (prevBpm === undefined || bpm !== prevBpm) {
                if (seg.startBeat === 0) downbeatBpm = bpm;
                else midBpmChanges.push({ beat: seg.startBeat, bpm });
                prevBpm = bpm;
              }
            }
            return (
              <React.Fragment key={i}>
                <div
                  className={styles.timelineHeaderTick}
                  style={
                    {
                      ['--bar-start-beat' as string]: startBeat,
                    } as React.CSSProperties
                  }
                >
                  <div className={styles.timelineHeaderTopRow}>
                    <span className={styles.timelineHeaderBar}>{bar.index}</span>
                    {showTimeSig && (
                      <span className={styles.timelineHeaderTimeSig}>
                        {bar.time.count}/{bar.time.unit}
                      </span>
                    )}
                    {downbeatBpm !== undefined && (
                      <span className={styles.timelineHeaderBpm}>{downbeatBpm} bpm</span>
                    )}
                  </div>
                  <span className={styles.timelineHeaderTime}>{formatTime(timeSec)}</span>
                </div>
                {midBpmChanges.map((c, j) => (
                  <div
                    key={`bpm-${i}-${j}`}
                    className={styles.timelineHeaderBpmAnchor}
                    style={
                      {
                        ['--bar-start-beat' as string]: startBeat + c.beat,
                      } as React.CSSProperties
                    }
                  >
                    <div className={styles.timelineHeaderTopRow}>
                      <span className={styles.timelineHeaderBpm}>{c.bpm} bpm</span>
                    </div>
                  </div>
                ))}
              </React.Fragment>
            );
          })}
          <Playhead showLabel onSeek={onSeek} />
        </div>
      </div>
    );
  }
);

function formatTime(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const min = Math.floor(total / 60);
  const sec = total % 60;
  return `${min}:${String(sec).padStart(2, '0')}`;
}

export const BarView = observer(
  ({
    bar,
    barStartBeat,
    pitches,
    config,
    isAnacrusis,
    highlightedPattern,
    onPatternClick,
    isPitchAudible,
    showBrackets = true,
    rowPitch,
    pitchOrder,
    colorForPitch,
  }: {
    bar: StructuralBar;
    /** Cumulative quarter-note position of this bar's left edge within
     *  the voice (sum of `beats` for every bar before this one). Drives
     *  the bar's absolute CSS left; see `.bar` in score.module.css. */
    barStartBeat: number;
    pitches: string[];
    config: ViewConfig;
    isAnacrusis: boolean;
    highlightedPattern: string | undefined;
    onPatternClick: (name: string) => void;
    isPitchAudible: (pitch: string) => boolean;
    /**
     * Whether to draw bar chrome that belongs to the score as a whole
     * (tuplet brackets and lead-in label). Pattern brackets are drawn
     * per-row instead — see {@link rowPitch} — so this flag doesn't
     * gate them.
     */
    showBrackets?: boolean;
    /**
     * In the unified mixer, the DSL pitch this BarView's row represents.
     * Pattern brackets only render when this pitch is in the span's
     * `pitches` set — rows for pitches the pattern doesn't play get no
     * bracket on that span, so the outline visually "skips" them.
     * Undefined falls back to drawing every span (label always shown).
     */
    rowPitch?: string;
    /**
     * Drum pitches in mixer-row order. Used together with {@link rowPitch}
     * to decide which row is the topmost / bottommost contributor for a
     * given span; the topmost shows the pattern label and the top edge
     * of the bracket, the bottommost shows the bottom edge, middles show
     * only the left/right sides so the outline reads as one connected
     * box across all participating rows.
     */
    pitchOrder?: readonly string[];
    /**
     * Optional per-pitch colour override. The unified mixer uses this to
     * layer the user's per-instrument-track colour pick on top of the
     * jot's palette default; the jot's structural `track.color` is the
     * palette-only baseline, and this resolver returns the same value
     * augmented with the override (or undefined / empty when the row
     * should fall through to the palette default).
     */
    colorForPitch?: (pitch: string) => string | undefined;
  }) => {
    // Inline style carries only zoom-invariant data so React's prop
    // diff sees no change on a zoom tick: `--bar-start-beat` /
    // `--bar-beats` are the bar's cumulative left edge and length in
    // quarter notes. The bar is absolutely positioned within the bars
    // row (see `.bar` in score.module.css) rather than relying on flex
    // accumulation; that gives each barline a sub-pixel-precise left
    // edge driven directly by `--px-per-beat`, matching the timeline
    // ticks, waveform chunks, and lyric words. `minHeight` is
    // config-derived; the bar stretches top:0/bottom:0 to the bars
    // row's full height so the right-edge barline reaches the row
    // separator (the pitch gutter is taller than a single lane).
    const barStyle = {
      ['--bar-start-beat' as string]: barStartBeat,
      ['--bar-beats' as string]: bar.beats,
      minHeight: pitches.length * (config.trackHeight as number),
    } as React.CSSProperties;
    const isLeadIn = bar.index < 0;
    // `bar.index === -1` is the last lead-in bar (lead-in indices count
    // -leadBars..-1 inclusive). Tag it so its right border draws the
    // dashed lead-in→music boundary; the rest of the lead-in bars have
    // border-right suppressed so they merge into one visual block.
    const isLastLeadIn = bar.index === -1;
    // Each lead-in bar paints a 45° hatch as its own background, so the
    // pattern's phase resets at every row boundary; and the rows don't
    // stack at a fixed multiple of the pattern period (1px row borders,
    // gutter-driven row heights). Walk `offsetParent` up to the document
    // and expose the bar's accumulated top as `--leadin-y`; the CSS
    // shifts `background-position-y` by it so every lead-in's hatch
    // reads as a slice of one global gradient anchored at the document
    // origin, and diagonals run seamlessly through the row separators.
    const barRef = React.useRef<HTMLDivElement>(null);
    React.useLayoutEffect(() => {
      if (!isLeadIn) return;
      const el = barRef.current;
      if (!el) return;
      let y = 0;
      let node: HTMLElement | null = el;
      while (node) {
        y += node.offsetTop;
        node = node.offsetParent as HTMLElement | null;
      }
      el.style.setProperty('--leadin-y', `${y}px`);
    });
    return (
      <div
        ref={isLeadIn ? barRef : undefined}
        className={classNames(
          styles.bar,
          isAnacrusis && styles.barAnacrusis,
          isLeadIn && styles.barLeadIn,
          isLastLeadIn && styles.barLeadInLast
        )}
        style={barStyle}
      >
        {/* Four grid-line overlay layers (main beat / 16th / triplet /
            48th). Always present in the DOM so toggling a grid is a
            single `setProperty` on the score root with zero per-bar
            React work; visibility flows in through
            `display: var(--grid-display-<family>, none)`, which
            `GridLineVars` writes when the corresponding toggle is on
            and removes when it's off. Each layer paints a single
            `background-color` clipped to a dashed/dotted vertical-line
            pattern via two intersecting masks (see `.gridLayer*` in the
            CSS), so we keep the original dashed/dotted look without
            the per-divider DOM spam that made playback laggy at 48ths.
            Skipped on lead-in bars (the hatched intro chrome reads
            cleaner without a grid on top). */}
        {!isLeadIn && (
          <>
            <div className={styles.gridLayerMain} aria-hidden="true" />
            <div className={styles.gridLayerSubBeat16} aria-hidden="true" />
            <div className={styles.gridLayerSubBeatQuarterTriplet} aria-hidden="true" />
            <div className={styles.gridLayerSubBeatTriplet} aria-hidden="true" />
            <div className={styles.gridLayerSubBeat48} aria-hidden="true" />
          </>
        )}
        {pitches.map((pitch) => {
          const track = bar.tracks[pitch];
          const dim = !isPitchAudible(pitch);
          return (
            <div
              key={pitch}
              className={classNames(styles.lane, dim && styles.laneDim)}
              style={{ height: config.trackHeight }}
            >
              {track?.notes.map((note, i) => (
                <NoteView
                  key={i}
                  note={note}
                  bar={bar}
                  color={colorForPitch?.(pitch) ?? track.color}
                  config={config}
                  instrument={track.instrument}
                  // A non-straight note already inside a tuplet bracket
                  // is explained by that bracket, so only flag the
                  // strays (e.g. an off-grid note not authored as a
                  // group) individually.
                  offGrid={!note.straight && !coveredByTuplet(bar, note.beat)}
                />
              ))}
            </div>
          );
        })}
        {bar.patternSpans.map((span, i) => {
          const position = bracketPositionForRow(span, rowPitch, pitchOrder);
          if (position === 'hidden') return null;
          return (
            <PatternBracket
              key={i}
              span={span}
              highlighted={highlightedPattern === span.name}
              onClick={onPatternClick}
              position={position}
            />
          );
        })}
        {showBrackets && bar.tupletSpans.map((span, i) => <TupletBracket key={i} span={span} />)}
      </div>
    );
  }
);

/**
 * Where this row's slice of a pattern bracket sits within the
 * top-to-bottom span across all contributing rows.
 *
 *   - `single`: this is the only contributing row — render a full box.
 *   - `top`:    this is the topmost contributor — render top edge + sides + label.
 *   - `middle`: a contributor between top and bottom — render sides only;
 *               the bracket reads as continuous across stacked rows.
 *   - `bottom`: the bottommost contributor — render bottom edge + sides.
 *   - `hidden`: this row's pitch isn't in the pattern — render nothing.
 */
type BracketPosition = 'single' | 'top' | 'middle' | 'bottom' | 'hidden';

function bracketPositionForRow(
  span: StructuralPatternSpan,
  rowPitch: string | undefined,
  pitchOrder: readonly string[] | undefined
): BracketPosition {
  // No row context (non-mixer caller) → render as a self-contained box,
  // same as the pre-mixer behaviour.
  if (rowPitch === undefined || !pitchOrder) return 'single';
  if (!span.pitches.has(rowPitch)) return 'hidden';
  let firstIdx = -1;
  let lastIdx = -1;
  for (let i = 0; i < pitchOrder.length; i++) {
    if (span.pitches.has(pitchOrder[i])) {
      if (firstIdx === -1) firstIdx = i;
      lastIdx = i;
    }
  }
  // The pattern body could in principle include pitches the mixer doesn't
  // surface (e.g. a brand-new pitch type that hasn't been added to the
  // row order yet). Treat the row as a single contributor in that case
  // rather than silently producing an open-ended bracket.
  if (firstIdx === -1 || lastIdx === -1) return 'single';
  if (firstIdx === lastIdx) return 'single';
  const myIdx = pitchOrder.indexOf(rowPitch);
  if (myIdx === firstIdx) return 'top';
  if (myIdx === lastIdx) return 'bottom';
  return 'middle';
}

/**
 * Pattern bracket colors, cycled in `colorIndex` order. Each entry
 * resolves at runtime to the matching `--color-pattern-N` token in
 * src/design_tokens.css, so updating a shade only requires editing the
 * token — this array just names the slots. Length intentionally
 * matches the token list; if it ever shrinks, the modulo wrap below
 * silently recycles colors.
 */
const PATTERN_COLOR_VARS: readonly string[] = [
  'var(--color-pattern-1)',
  'var(--color-pattern-2)',
  'var(--color-pattern-3)',
  'var(--color-pattern-4)',
  'var(--color-pattern-5)',
  'var(--color-pattern-6)',
  'var(--color-pattern-7)',
  'var(--color-pattern-8)',
  'var(--color-pattern-9)',
  'var(--color-pattern-10)',
  'var(--color-pattern-11)',
  'var(--color-pattern-12)',
];

/**
 * Hex literal used by Canvas paint sites for the waveform's stroke colour.
 * Mirrors `--color-pattern-2` in `src/design_tokens.css` (sky blue);
 * Canvas 2D needs a literal RGB string and reading it via
 * `getComputedStyle(canvas).getPropertyValue('--color-pattern-2')` per
 * paint forces a style flush, so we duplicate the value here and accept
 * the carve-out from "no naked color literals" (AGENTS.md §5.8). If the
 * CSS token's hex changes, update this in lockstep.
 */
export const WAVEFORM_PAINT_COLOR = '#5ba8e8';

const PatternBracket = observer(
  ({
    span,
    highlighted,
    onClick,
    position,
  }: {
    span: StructuralPatternSpan;
    highlighted: boolean;
    onClick: (name: string) => void;
    position: Exclude<BracketPosition, 'hidden'>;
  }) => {
    const color = PATTERN_COLOR_VARS[span.colorIndex % PATTERN_COLOR_VARS.length];
    return (
      <div
        className={classNames(
          styles.patternBracket,
          (position === 'top' || position === 'single') && styles.patternBracketTop,
          (position === 'bottom' || position === 'single') && styles.patternBracketBottom,
          highlighted && styles.patternBracketHighlight
        )}
        style={
          {
            ['--span-start-beat' as string]: span.startBeat,
            ['--span-end-beat' as string]: span.endBeat,
            ['--pattern-color' as string]: color,
          } as React.CSSProperties
        }
      >
        {/* Label sits with the bracket's top edge — only render on the
            topmost contributing row so a multi-row bracket doesn't stack
            duplicate labels down the score. */}
        {(position === 'top' || position === 'single') && (
          <button
            type="button"
            data-noseek="true"
            className={classNames(styles.patternLabel, highlighted && styles.patternLabelHighlight)}
            onClick={(e) => {
              e.stopPropagation();
              onClick(span.name);
            }}
            onMouseDown={(e) => e.stopPropagation()}
            title={`Pattern usage: ${span.name} (click to highlight other usages)`}
          >
            {span.name}
          </button>
        )}
      </div>
    );
  }
);

/**
 * Classic engraved tuplet bracket: a thin line over the grouped notes
 * with the slot count (3 = triplet, 5 = quintuplet, ...) on it. Purely
 * decorative — no interaction, unlike the pattern bracket.
 */
const TupletBracket = observer(({ span }: { span: StructuralTupletSpan }) => (
  <div
    className={styles.tupletBracket}
    style={
      {
        ['--span-start-beat' as string]: span.startBeat,
        ['--span-end-beat' as string]: span.endBeat,
      } as React.CSSProperties
    }
    title={`${span.count}-tuplet (not a straight subdivision)`}
  >
    <span className={styles.tupletNumber}>{span.count}</span>
  </div>
));

const NoteView = observer(
  ({
    note,
    bar,
    color,
    config,
    instrument,
    offGrid,
  }: {
    note: StructuralNote;
    /**
     * The bar this note lives in. Carried through so the `Debug details`
     * panel can compute the rendered (post-quantization) beat position
     * — `note.beat` is in quarter-notes from bar start, and the bar's
     * time signature is needed to convert into the 1-indexed
     * `beat_in_bar` convention the provenance entry uses.
     */
    bar: StructuralBar;
    color: string;
    config: ViewConfig;
    instrument: Instrument;
    offGrid: boolean;
  }) => {
    const isAccent = note.modifiers.has('a');
    const isGhost = note.modifiers.has('g');
    const isFlam = note.modifiers.has('fl');
    const isDrag = note.modifiers.has('dr');
    const isCross = note.modifiers.has('x');
    const badge = pickBadge(note);
    const selection = React.useContext(SelectionContext);
    const selected = selection?.selectedNote === note;
    const [hovered, setHovered] = React.useState(false);
    const showLabel = selected || hovered;
    const description = offGrid
      ? `${describeNote(note, instrument)} — off the straight grid (triplet/tuplet)`
      : describeNote(note, instrument);
    // Per-note debug provenance, when a filter-mode debug bundle is
    // loaded. Keyed by the original MIDI tick preserved through
    // `from_midi.ts`. Falls back to `undefined` for notes that didn't
    // round-trip through MIDI (e.g. examples, hand-loaded jots) or
    // when no bundle is loaded — the `Debug details` section is hidden
    // in those cases.
    const provenance = React.useContext(NoteProvenanceContext);
    // Sub-slot timing offset: shift the glyph from its slot to where the
    // note actually plays. The score's x-axis is notational beats, so the
    // ms offset is converted via the bar's local sec-per-beat (from the
    // eager per-bar timings, keyed by the clone-stable `bar.index`).
    const barTimings = React.useContext(BarTimingsContext);
    const offsetMs = note.source.offset;
    let offsetBeats = 0;
    if (offsetMs !== undefined) {
      const timing = barTimings?.get(bar.index);
      if (timing && bar.beats > 0) {
        offsetBeats = msOffsetToBeats(offsetMs, timing.durationSec / bar.beats);
      }
    }
    const sourceMeta = note.source.metadata as { midi?: { tick?: number } } | undefined;
    const tick = sourceMeta?.midi?.tick;
    const provenanceEntry =
      provenance && typeof tick === 'number'
        ? provenance.byTick.get(`${note.pitch}:${tick}`)
        : undefined;

    const noteRef = React.useRef<HTMLDivElement>(null);
    const labelRef = React.useRef<HTMLDivElement>(null);
    const flipAbove = usePopoverFlipAbove(noteRef, labelRef, showLabel);

    return (
      <div
        ref={noteRef}
        // Notes opt out of click-to-seek so clicking a note keeps its
        // own meaning (selection / hover label) instead of moving the
        // playhead.
        data-noseek="true"
        className={classNames(
          styles.note,
          isAccent && styles.accent,
          isGhost && styles.ghost,
          isCross && styles.cross,
          note.roll && styles.roll,
          offGrid && styles.offGrid,
          selected && styles.noteSelected,
          showLabel && styles.noteShowingLabel,
          hovered && styles.noteHovered
        )}
        style={
          {
            // Beat is stable per note (set inline); the CSS rule on
            // `.note` derives `left` from `padLeft + beat × pxPerBeat`,
            // so a zoom tick changes one root var instead of mutating
            // this element's style. `offsetBeats` nudges an off-grid note
            // to its true sub-slot position (0 for on-grid notes).
            ['--note-beat' as string]: note.beat + offsetBeats,
            top: (config.trackHeight as number) / 2,
            width: config.noteDiameter,
            height: config.noteDiameter,
            background: isCross ? 'var(--color-bg-base)' : color,
            color,
            borderStyle: isCross ? 'solid' : undefined,
            border: isCross ? `2px solid ${color}` : undefined,
          } as React.CSSProperties
        }
        // Suppress the container's mousedown handler — it begins a
        // marquee selection that clears the existing state on every
        // press, which would wipe this note's selection before the
        // click ever fires.
        onMouseDown={(e) => e.stopPropagation()}
        onClick={(e) => {
          e.stopPropagation();
          selection?.selectNote(note);
        }}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        // Marks the row (via a `:has()` rule in mixer.module.css) as
        // carrying an open label popover so its whole row can be lifted
        // above the sibling rows the popover extends down into. A data
        // attribute rather than a class because the row selector lives
        // in a different CSS module's local scope.
        data-note-label-open={showLabel || undefined}
      >
        {isFlam && <FlamGrace color={color} config={config} />}
        {isDrag && <DragGrace color={color} config={config} />}
        {badge && <span className={styles.modifierBadge}>{badge}</span>}
        {note.sticking && (
          <span className={styles.stickingBadge}>{note.sticking.toUpperCase()}</span>
        )}
        {showLabel && (
          <div
            ref={labelRef}
            className={classNames(styles.noteLabel, flipAbove && styles.noteLabelAbove)}
          >
            <div className={styles.noteLabelText}>{description}</div>
            {provenanceEntry && (
              <NoteProvenanceDetails
                entry={provenanceEntry}
                rendered={{ note, bar, provenance: provenance! }}
                startOpen
              />
            )}
          </div>
        )}
      </div>
    );
  }
);

function FlamGrace({ color, config }: { color: string; config: ViewConfig }) {
  const size = (config.noteDiameter as number) * 0.55;
  return (
    <span
      style={{
        position: 'absolute',
        left: -size - 2,
        top: '50%',
        transform: 'translateY(-50%)',
        width: size,
        height: size,
        background: color,
        borderRadius: '50%',
        opacity: 0.7,
      }}
    />
  );
}

function DragGrace({ color, config }: { color: string; config: ViewConfig }) {
  const size = (config.noteDiameter as number) * 0.45;
  return (
    <>
      {[0, 1].map((i) => (
        <span
          key={i}
          style={{
            position: 'absolute',
            left: -((size + 2) * (i + 1)),
            top: '50%',
            transform: 'translateY(-50%)',
            width: size,
            height: size,
            background: color,
            borderRadius: '50%',
            opacity: 0.6,
          }}
        />
      ))}
    </>
  );
}

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
  note: StructuralNote;
  bar: StructuralBar;
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
function renderedBeatInBar(note: StructuralNote, bar: StructuralBar): number {
  if (bar.beats <= 0) return 1;
  return 1 + (note.beat / bar.beats) * bar.time.count;
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
/** Width of the per-stage label column to the left of the bar tracks.
 * The label sits outside the coloured bar because most stage bars are
 * a handful of px wide (the deltas being visualised are millisecond-
 * scale), so an inline label clips to ellipsis almost every time. */
const TIMING_VIZ_LABEL_WIDTH = 260;
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
      chainAcc += s.deltaSec;
      stagePositions.push(chainAcc);
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

    const timeToX = (t: number): number => ((t - windowStart) / windowDur) * TIMING_VIZ_WIDTH;

    // Pick the audio track most likely to expose this onset clearly.
    // The debug bundle's manifest carries an authoritative pitch →
    // audio-filename map (set up server-side in
    // `transcriber/app/debug_bundle.py`); the isolated stem for the
    // note's pitch is the right source; it isolates the drum we're
    // inspecting from the rest of the kit. Fallback chain when the
    // mapping doesn't resolve (legacy bundle, manual file load, etc.):
    //   1. Any other mapped stem, still a per-pitch isolated source.
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
    const mapping = rendered.provenance.audioFilenameByPitch;
    const audioTrack = React.useMemo<AudioTrack | undefined>(() => {
      if (audioTracksByFilename.size === 0) return undefined;
      const wantedFilename = mapping.get(entry.pitch);
      if (wantedFilename) {
        const exact = audioTracksByFilename.get(wantedFilename.toLowerCase());
        if (exact) return exact;
      }
      // Any other mapped per-pitch stem; skip `no_drums`, which is the
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
    }, [audioTracksByFilename, mapping, entry.pitch]);

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
    const drumsT0Sec = jotPlayer.drumsT0Sec;
    const barBoundaries: { x: number; label: number | null }[] = [];
    const gridLines: number[] = [];
    if (timeline.rendered) {
      const structBars = timeline.rendered.structure.voices[0]?.bars ?? [];
      for (let i = 0; i < timeline.bars.length; i++) {
        const bar = timeline.bars[i]!;
        const structBar = structBars[i];
        const barStart = bar.startSec + drumsT0Sec;
        const barEnd = barStart + bar.durationSec;
        if (barStart >= windowStart && barStart <= windowEnd) {
          barBoundaries.push({ x: timeToX(barStart), label: structBar?.index ?? null });
        }
        if (
          !structBar ||
          bar.durationSec <= 0 ||
          structBar.time.unit <= 0 ||
          barEnd < windowStart ||
          barStart > windowEnd
        ) {
          continue;
        }
        const slots = Math.round((structBar.time.count * gridDivision) / structBar.time.unit);
        if (slots <= 0) continue;
        const slotDur = bar.durationSec / slots;
        // j starts at 1 so the bar-boundary line (j=0) isn't doubled up.
        for (let j = 1; j < slots; j++) {
          const t = barStart + j * slotDur;
          if (t >= windowStart && t <= windowEnd) gridLines.push(timeToX(t));
        }
      }
    }

    const detectedX = timeToX(detectedSec);
    const finalX = finalSec !== undefined ? timeToX(finalSec) : undefined;

    // Each diff row: a coloured bar from `anchorX` to `endX` (always
    // drawn left-to-right via min/abs), with an inline label. The chain
    // anchors at the detected line and advances by each stage's
    // `deltaSec`; the trailing residual (`finalSec` minus the cumulative
    // sum) surfaces as the "Unknown drift" bar so every gap between
    // named stages and the actual final landing remains visible rather
    // than silently absorbed.
    type DiffRow = {
      key: string;
      label: string;
      deltaBeats: number | undefined;
      deltaSlots: number | undefined;
      deltaSec: number;
      anchorX: number;
      endX: number;
      className: string;
    };
    const diffRows: DiffRow[] = [];
    let cursorSec = detectedSec;
    let cursorX = detectedX;
    for (const s of stages) {
      if (Math.abs(s.deltaSec) < 1e-9) continue;
      const nextSec = cursorSec + s.deltaSec;
      const nextX = timeToX(nextSec);
      diffRows.push({
        key: s.key,
        label: s.label,
        deltaBeats: s.deltaBeats,
        deltaSlots: s.deltaSlots,
        deltaSec: s.deltaSec,
        anchorX: cursorX,
        endX: nextX,
        className: s.className,
      });
      cursorSec = nextSec;
      cursorX = nextX;
    }
    if (finalSec !== undefined && finalX !== undefined) {
      const unknownSec = finalSec - cursorSec;
      if (Math.abs(unknownSec) > 1e-6) {
        diffRows.push({
          key: 'unknown',
          label: 'Unknown drift source',
          deltaBeats: undefined,
          deltaSlots: undefined,
          deltaSec: unknownSec,
          anchorX: cursorX,
          endX: finalX,
          className: styles.timingVizDiffBarUnknown,
        });
      }
    }

    const renderSignedBeats = (b: number) => `${b >= 0 ? '+' : ''}${b.toFixed(3)} beats`;
    const renderSignedSlots = (slots: number) => formatSignedSlots(slots, gridDivision);
    const renderSignedMs = (ms: number) => `${ms >= 0 ? '+' : ''}${ms.toFixed(1)} ms`;

    return (
      <div
        className={styles.timingViz}
        style={{
          gridTemplateColumns: `${TIMING_VIZ_LABEL_WIDTH}px ${TIMING_VIZ_WIDTH}px`,
        } as React.CSSProperties}
      >
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
              style={{
                width: TIMING_VIZ_WIDTH,
                height: TIMING_VIZ_WAVE_HEIGHT,
              }}
            />
          ) : (
            <div className={styles.timingVizNoAudio}>(no audio loaded)</div>
          )}
          {gridLines.map((x, i) => (
            <div key={`grid-${i}`} className={styles.timingVizGridLine} style={{ left: x }} />
          ))}
          {barBoundaries.map((b, i) => (
            <React.Fragment key={`bar-${i}`}>
              <div className={styles.timingVizBarLine} style={{ left: b.x }} />
              {b.label !== null && (
                <div className={styles.timingVizBarLabel} style={{ left: b.x }}>
                  {b.label}
                </div>
              )}
            </React.Fragment>
          ))}
          <div
            className={styles.timingVizDetectedLine}
            style={{ left: detectedX }}
            title={`Detected · ${detectedSec.toFixed(3)}s`}
          />
          {finalX !== undefined && (
            <div
              className={styles.timingVizFinalLine}
              style={{ left: finalX }}
              title={`Final · ${finalSec!.toFixed(3)}s`}
            />
          )}
        </div>
        {diffRows.map((row) => {
          const left = Math.min(row.anchorX, row.endX);
          const width = Math.abs(row.endX - row.anchorX);
          const beatsPart =
            row.deltaBeats !== undefined ? `· ${renderSignedBeats(row.deltaBeats)} ` : '';
          const slotsPart =
            row.deltaSlots !== undefined ? `· ${renderSignedSlots(row.deltaSlots)} ` : '';
          const fullText = `${row.label} ${beatsPart}${slotsPart}· ${renderSignedMs(row.deltaSec * 1000)}`;
          return (
            <React.Fragment key={row.key}>
              <div
                className={styles.timingVizDiffRowLabel}
                style={{ height: TIMING_VIZ_ROW_HEIGHT } as React.CSSProperties}
                title={fullText}
              >
                {fullText}
              </div>
              <div
                className={styles.timingVizDiffRowTrack}
                style={{ height: TIMING_VIZ_ROW_HEIGHT } as React.CSSProperties}
              >
                <div
                  className={`${styles.timingVizDiffBar} ${row.className}`}
                  style={{ left, width }}
                  title={fullText}
                />
              </div>
            </React.Fragment>
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
const NoteProvenanceDetails = observer(
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
    // Stop the container's mousedown handler so clicks on the toggle don't
    // begin a marquee selection (which would clear the surrounding note's
    // selection and immediately unmount this component).
    const stop = (e: React.MouseEvent) => e.stopPropagation();

    // Eager per-bar timings table provided once at JotView. Lets the
    // "Final position" row resolve a bar's absolute jot-time start without
    // waiting for the player's timeline to be built — pre-Play the
    // player's timeline is `EMPTY_TIMELINE`, but the math doesn't actually
    // need any playback state.
    const barTimings = React.useContext(BarTimingsContext);
    // The current rendered jot — used to read `effectiveDrumOffsetBeats`,
    // the user-applied Beat-offset slider value, as a labelled stage in
    // the detected → final timing-drift chain.
    const renderedJot = React.useContext(RenderedJotContext);
    // Grid density the jot was produced at; drives every slot readout in
    // this panel. Falls back to the default when no rendered jot is in
    // context (filtered-ghost rendering).
    const gridDivision = renderedJot ? gridDivisionFor(renderedJot) : DEFAULT_GRID_DIVISION;
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
    let originalBar: StructuralBar | undefined;
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
      const structBars = renderedJot?.structure.voices[0]?.bars;
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
      if (originalBar && originalSecPerQuarterNote !== undefined && entry.tick !== null) {
        const gridTicks = midiGridTicks(gridDivision);
        const postSnapTick = Math.round(entry.tick / gridTicks) * gridTicks;
        const snapDeltaQn = (postSnapTick - entry.tick) / TICKS_PER_BEAT;
        // qn → ts-beats: 1 ts-beat = 4/unit qn, so 1 qn = unit/4 ts-beats.
        snapBeats = snapDeltaQn * (originalBar.time.unit / 4);
        snapSec = snapDeltaQn * originalSecPerQuarterNote;
        snapMs = snapSec * 1000;
        // Post-snap position in the ORIGINAL bar, built off the detected
        // position rather than re-deriving from the post-snap tick (which
        // would require walking the structure to recover the bar's start
        // tick). Equivalent: detected + snap_delta lands at the snapped
        // position by construction.
        originalQuantizedBeat = entry.beat_in_bar + snapBeats;
        originalQuantizedSec = entry.detected_time_sec + snapSec;
      }

      // Final position uses the CURRENT bar's timing; where the note
      // plays now after the slider has applied its shift.
      if (currentBarTiming && currentSecPerQuarterNote !== undefined) {
        const intra =
          (currentQuantizedBeat - 1) * (4 / rendered.bar.time.unit) * currentSecPerQuarterNote;
        finalSec = currentBarTiming.startSec + intra + jotPlayer.drumsT0Sec;
      }

      // Bar-anchor drift: the difference between where the JOT places
      // the note's original bar in audio time and where the detector's
      // post-alignment view of beat_in_bar implies it should be. In a
      // perfect round-trip these match exactly; in practice the most
      // common source of mismatch is `transcriber/app/pipeline/onsets_midi.py`'s
      // `compute_bar_tick_grid` rounding `lead_bars = round(lead_in_secs *
      // initial_tempo_ticks_per_sec / bar0_ticks)` to zero when the
      // pre-roll is shorter than ~half a bar. The MIDI then carries no
      // lead-in, `from_midi.ts` reconstructs `drumsT0Sec = 0`, and every
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
          (entry.beat_in_bar - 1) * (4 / originalBar.time.unit) * originalSecPerQuarterNote;
        const transcriberBarAudioTime = entry.detected_time_sec - intraSecFromDetected;
        // Where the JOT places that same bar in audio time.
        const jotBarAudioTime = originalBarTiming.startSec + jotPlayer.drumsT0Sec;
        const drift = jotBarAudioTime - transcriberBarAudioTime;
        if (Math.abs(drift) > 1e-6) {
          anchorDriftSec = drift;
          anchorDriftBeats = (drift / originalSecPerQuarterNote) * (originalBar.time.unit / 4);
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
      const offsetQn = renderedJot?.effectiveDrumOffsetBeats;
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
        ? (sec / originalSecPerQuarterNote) * (originalBar.time.unit / 4)
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
        ? (beats * gridDivision) / originalBar.time.unit
        : undefined;

    // ─── Unknown drift residual ───
    //
    // `finalSec` minus everything the named-stages chain accounts for.
    // Surfaces every drift source we haven't enumerated yet (per-bar
    // BPM attribution mismatches between the transcriber and from_midi,
    // cross-bar drum-offset re-bucketing under varying tempo, etc.).
    let unknownDriftSec: number | undefined;
    let unknownDriftBeats: number | undefined;
    let unknownDriftMs: number | undefined;
    if (finalSec !== undefined && rendered) {
      let accountedSec = entry.detected_time_sec;
      if (envelopeRefineSec !== undefined) accountedSec += envelopeRefineSec;
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
      accountedSec += combinedAlignSec;
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
    // Each entry is one named shift the popup attributes to a specific
    // stage; the trailing unknown-drift residual is appended inside
    // `OnsetTimingVisualization` so it always lands at the chain's end.
    // Pipeline order: detection envelope-refine → beat-grid alignment
    // (coarse, fine) → quantise passes → bar anchor drift → MIDI snap →
    // drum offset.
    const stages: StageShift[] = [];
    if (envelopeRefineSec !== undefined && Math.abs(envelopeRefineSec) > 1e-9) {
      stages.push({
        key: 'env-refine',
        label: 'Envelope refine (onsets)',
        className: styles.timingVizDiffBarEnvRefine,
        deltaSec: envelopeRefineSec,
        deltaBeats: secToOrigBeats(envelopeRefineSec),
      });
    }
    if (hasAlignSplit) {
      if (coarseAlignSec !== null && Math.abs(coarseAlignSec) > 1e-9) {
        stages.push({
          key: 'align-coarse',
          label: 'Beat align · coarse',
          className: styles.timingVizDiffBarAlignCoarse,
          deltaSec: coarseAlignSec,
          deltaBeats: secToOrigBeats(coarseAlignSec),
        });
      }
      if (fineAlignSec !== null && Math.abs(fineAlignSec) > 1e-9) {
        stages.push({
          key: 'align-fine',
          label: 'Beat align · fine',
          className: styles.timingVizDiffBarAlignFine,
          deltaSec: fineAlignSec,
          deltaBeats: secToOrigBeats(fineAlignSec),
        });
      }
    } else if (Math.abs(combinedAlignSec) > 1e-9) {
      stages.push({
        key: 'align',
        label: 'Beat alignment',
        className: styles.timingVizDiffBarAlignCoarse,
        deltaSec: combinedAlignSec,
        deltaBeats: secToOrigBeats(combinedAlignSec),
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
          {open ? '▾' : '▸'} Debug details
        </button>
        {open && rendered && (
          <OnsetTimingVisualization
            entry={entry}
            rendered={rendered}
            gridDivision={gridDivision}
            stages={stages}
            finalSec={finalSec}
            displayedBarIndex={displayedBarIndex}
          />
        )}
        {open &&
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
      </div>
    );
  }
);

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
  originalBar: StructuralBar | undefined;
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
    drumOffsetBeats, unknownDriftSec, unknownDriftBeats, unknownDriftMs,
    gridDivision, slotsPerQuarterNote, secToOrigBeats, origBeatsToSlots,
    renderSignedMs, renderSignedBeats, renderSignedSec, formatSlots,
  } = p;
  const velocity = (
    rendered?.note.source.metadata as { midi?: { velocity?: number } } | undefined
  )?.midi?.velocity;

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
    <>
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
              timeSig: originalBar?.time,
              audioSec: entry.detected_time_sec,
            }).toString()}
          </dd>
          <dt>Strength</dt>
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
                    timeSig: originalBar?.time,
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
                  timeSig: rendered?.bar.time,
                  audioSec: finalSec,
                  offsetMs: rendered?.note.source.offset,
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
    </>
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

/**
 * Renders one rejected onset as a dashed ghost circle at its detected
 * `(bar, beat_in_bar)` position inside an `InstrumentRow`'s bars row.
 * Absolutely positioned via the same `--note-pad-px` / `--px-per-beat`
 * CSS vars the real notes use, but with `--filtered-beat` = the
 * onset's cumulative beat offset from the start of the row (lead-in +
 * prior bars + intra-bar offset) so it lands at the right absolute x
 * without needing per-bar ResolvedBar geometry.
 *
 * Click toggles a stuck-open detail popover (independent of the
 * SelectionStore — a filtered onset is not a real note); hover shows
 * the same popover transiently.
 */
export const FilteredOnsetView = observer(({
  entry,
  beatOffset,
  color,
  trackHeight,
}: {
  entry: NoteProvenanceEntry;
  /** Total beat offset from the start of the bars row (leadInBeats +
   * cumulative bar beats + (beat_in_bar - 1)). The CSS calc derives
   * the pixel `left` from this and the score-root's `--px-per-beat`. */
  beatOffset: number;
  /** Pitch lane colour. Mirrors what the real notes use; falls back to
   * a neutral grey for filtered-only pitches with no rendered notes. */
  color: string;
  trackHeight: number;
}) => {
  const store = React.useContext(JotViewStoreContext);
  const pinnedKey = `${entry.pitch}:${entry.detected_time_sec}`;
  const clicked = store?.pinnedFilteredOnsetKey === pinnedKey;
  const [hovered, setHovered] = React.useState(false);
  const show = hovered || clicked;
  const stop = (e: React.MouseEvent) => e.stopPropagation();
  const anchorRef = React.useRef<HTMLDivElement>(null);
  const labelRef = React.useRef<HTMLDivElement>(null);
  const flipAbove = usePopoverFlipAbove(anchorRef, labelRef, show);
  // Click-outside to dismiss the stuck-open popover. Without this the
  // only way to close it is clicking the (small, easy-to-miss) dashed
  // ring again. We treat clicks anywhere outside the anchor OR its
  // label as "outside" so users can interact with the popover itself
  // (e.g. expand Debug details) without dismissing it.
  //
  // Registered in capture phase so the listener runs before React's
  // root-level click delegation; calling stopPropagation prevents the
  // dismissing click from also moving the playhead via the bars-row
  // seek handler (or any other bubbling onClick further up the tree).
  React.useEffect(() => {
    if (!clicked || !store) return;
    const onClickCapture = (e: MouseEvent) => {
      const target = e.target as Node;
      if (anchorRef.current?.contains(target)) return;
      if (labelRef.current?.contains(target)) return;
      e.stopPropagation();
      store.setPinnedFilteredOnsetKey(undefined);
    };
    document.addEventListener('click', onClickCapture, true);
    return () => document.removeEventListener('click', onClickCapture, true);
  }, [clicked, store]);
  return (
    <div
      ref={anchorRef}
      // Same opt-out as real notes so a click on the ghost doesn't move
      // the playhead via the bars-row seek handler.
      data-noseek="true"
      // Mirror the real note (see NoteView): mark the row as carrying an
      // open popover so the `:has([data-note-label-open])` rule in
      // mixer.module.css lifts the whole instrument row above the sibling
      // rows this detail popover extends down into. Without it the
      // `.filteredOnsetLabel` stays trapped in `.barsRow`'s `z-index: 1`
      // stacking context and is painted under the equal-z-index bars of
      // the tracks below.
      data-note-label-open={show || undefined}
      className={classNames(styles.filteredOnset, show && styles.filteredOnsetShowingLabel)}
      style={
        {
          ['--filtered-beat' as string]: beatOffset,
          top: trackHeight / 2,
          color,
        } as React.CSSProperties
      }
      onMouseDown={stop}
      onClick={(e) => {
        stop(e);
        store?.setPinnedFilteredOnsetKey(clicked ? undefined : pinnedKey);
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      title={`Filtered onset · pitch ${entry.pitch} · bar ${entry.bar} beat ${entry.beat_in_bar.toFixed(2)}`}
    >
      {show && (
        <div
          ref={labelRef}
          className={classNames(
            styles.filteredOnsetLabel,
            flipAbove && styles.filteredOnsetLabelAbove
          )}
        >
          <NoteProvenanceDetails entry={entry} startOpen />
        </div>
      )}
    </div>
  );
});

function pickBadge(note: StructuralNote): string | undefined {
  const m = note.modifiers;
  if (m.has('c')) return 'C';
  if (m.has('o')) return 'O';
  if (m.has('h')) return 'H';
  if (m.has('f')) return 'F';
  if (m.has('s')) return 'S';
  if (m.has('r')) return 'R';
  if (m.has('z')) return 'Z';
  if (m.has('k')) return 'K';
  if (m.has('m')) return 'M';
  if (m.has('l')) return 'L';
  if (m.has('rf')) return 'Ruff';
  return undefined;
}

/**
 * Human-readable tooltip text for a note. Combines the resolved instrument
 * name with friendly modifier / sticking / roll labels.
 *
 * Examples:
 *   `s:a`       -> "Snare (accented)"
 *   `s:fl@l`    -> "Snare (flam, left hand)"
 *   `h:c`       -> "Hi-Hat (closed)"
 *   `c~_8:o`    -> "Crash (open, roll)"
 */
function describeNote(note: StructuralNote, instrument: Instrument): string {
  const name = instrument.name ?? `Pitch ${note.pitch}`;
  const qualifiers: string[] = [];
  for (const mod of note.modifiers) {
    qualifiers.push(MODIFIER_LABELS[mod] ?? mod);
  }
  if (note.roll) qualifiers.push('roll');
  if (note.sticking) qualifiers.push(STICKING_LABELS[note.sticking]);
  return qualifiers.length > 0 ? `${name} (${qualifiers.join(', ')})` : name;
}

const MODIFIER_LABELS: Partial<Record<Modifier, string>> = {
  a: 'accented',
  g: 'ghost',
  c: 'closed',
  h: 'half-open',
  o: 'open',
  f: 'foot',
  s: 'splash',
  r: 'rim shot',
  x: 'cross-stick',
  z: 'buzz',
  k: 'choke',
  m: 'mute',
  l: 'let ring',
  fl: 'flam',
  dr: 'drag',
  rf: 'ruff',
};

const STICKING_LABELS: Record<Sticking, string> = {
  r: 'right hand',
  l: 'left hand',
  rf: 'right foot',
  lf: 'left foot',
};
