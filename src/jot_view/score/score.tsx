import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { createPortal } from 'react-dom';
import { perfProbe } from 'src/perf_probe';
import { NoteProvenanceEntry } from 'src/debug_zip';
import { Instrument, Modifier, Sticking } from 'src/dsl';
import {
  RenderedJot,
  StructuralBar,
  StructuralNote,
  StructuralPatternSpan,
  StructuralTupletSpan,
  ViewConfig,
} from 'src/jot';
import { msOffsetToBeats } from 'src/tempo';
import { jotPlayer } from 'src/playback';
import sharedStyles from '../../jot_view.module.css';
import { GutterResizeHandle } from '../components/gutter_resize_handle';
import {
  BarTimingsContext,
  NoteProvenanceContext,
  ProvenancePresenterContext,
  ProvenanceStoreContext,
  SelectionContext,
  ViewportStoreContext,
} from '../contexts';
import { Playhead } from '../playback/playhead';
import styles from './score.module.css';
import { barsRowWidthSeed, intersectsBeatRange } from '../utils/windowing';
import { NoteProvenanceDetails } from './note_provenance_details';

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
/**
 * Selection popover that escapes the `.jotContainer { overflow: hidden }`
 * clip by rendering into `document.body` via `createPortal`. Without
 * the portal the popover is clipped at the score's bottom edge and
 * stops short of the minimap / playback bar below; with it the popover
 * can extend over any sibling chrome (subject only to the window's own
 * edge).
 *
 * Position is computed at render time from the anchor's
 * `getBoundingClientRect()`, applied as inline `position: fixed`
 * coords. Re-renders when the score scrolls or zooms, the wrapping
 * `observer(...)` HOC subscribes to `store.scrollX` / `store.scrollY`
 * / `store.zoom`, so MobX re-fires whenever the anchor's screen
 * position changes under the (transform-driven) scroll. The bar's
 * `bounding rect` returns the post-transform viewport coordinates, so
 * one read per render is enough; no per-frame imperative updates.
 *
 * Above-flip is reused from the previous in-DOM implementation, with
 * the only change being the bottom limit, now the window edge rather
 * than the score-scroller bottom. The popover can extend through the
 * minimap / playback area, so we flip only when it would overrun the
 * window itself.
 *
 * Reading `getBoundingClientRect` at render time is the popover-
 * anchoring exception called out in AGENTS.md §5.9: a single rect
 * read per popover-open re-render, not a per-frame layout loop.
 */
type PopoverPortalProps = {
  anchorRef: React.RefObject<HTMLElement>;
  show: boolean;
  className: string;
  /** Class added on top of `className` when the popover flipped above
   *  the anchor. Optional: positioning + transform are handled inline,
   *  so consumers only pass a flipped class when there's *visual*
   *  chrome that differs (e.g. a tail pointing the other way). */
  flippedClassName?: string;
  children: React.ReactNode;
  /** Extra `<div>` props applied to the portaled wrapper (refs, mouse
   *  handlers, etc.). The wrapper's `ref` is reserved for internal
   *  measurement; consumers that need a label ref should pass it
   *  through this prop. */
  extraProps?: React.HTMLAttributes<HTMLDivElement> & {
    ref?: React.Ref<HTMLDivElement>;
  };
};

/**
 * Hidden-state gate. There is one PopoverPortal per note (and per
 * filtered-onset ghost), so on a large score the tree holds thousands of
 * them, but at most one is ever `show`n at a time (the selected/hovered
 * note's label). This wrapper reads NO observables and runs NO hooks when
 * hidden, it just returns `null`, so a zoom / scroll tick (which mutates
 * `store.zoom` / `store.scrollX`) does not wake one observer per note. The
 * subscribing logic lives in {@link PopoverPortalShown}, which only mounts
 * for the popover that's actually open. Re-rendered by its parent (NoteView
 * etc.) when `show` flips, so it doesn't need to be an observer itself.
 *
 * This split is load-bearing for zoom performance: before it, every hidden
 * popover subscribed to `store.zoom` and re-rendered on every wheel tick,
 * turning a zoom into a multi-thousand-node synchronous reconciliation.
 */
function PopoverPortal(props: PopoverPortalProps) {
  if (!props.show) return null;
  return <PopoverPortalShown {...props} />;
}

const PopoverPortalShown = observer(function PopoverPortalShown({
  anchorRef,
  className,
  flippedClassName,
  children,
  extraProps,
}: PopoverPortalProps) {
  perfProbe('PopoverPortal');
  const viewport = React.useContext(ViewportStoreContext);
  // Read these for MobX reactivity even though we don't use the values
  // directly, the bounding-rect read in the render below picks up the
  // new post-transform position whenever the score scrolls or zooms.
  // Only the open popover is mounted, so this is one subscription, not one
  // per note (see {@link PopoverPortal}).
  void viewport?.scrollX;
  void viewport?.scrollY;
  void viewport?.zoom;

  const labelRef = React.useRef<HTMLDivElement | null>(null);
  const [flip, setFlip] = React.useState(false);

  React.useLayoutEffect(() => {
    const anchor = anchorRef.current;
    const label = labelRef.current;
    if (!anchor || !label) return;
    const aRect = anchor.getBoundingClientRect();
    const lRect = label.getBoundingClientRect();
    const SAFE = 8;
    const GAP = 16;
    // Window bounds, the popover is portaled to `document.body` and
    // sits above every app-shell sibling, so the only edge it can't
    // cross is the viewport itself.
    const overflowsBelow = aRect.bottom + GAP + lRect.height > window.innerHeight - SAFE;
    const fitsAbove = aRect.top - GAP - lRect.height > SAFE;
    setFlip(overflowsBelow && fitsAbove);
  }, [anchorRef, viewport?.scrollX, viewport?.scrollY, viewport?.zoom]);

  const anchor = anchorRef.current;
  if (!anchor) return null;
  const aRect = anchor.getBoundingClientRect();
  const GAP = 16;
  const top = flip ? aRect.top - GAP : aRect.bottom + GAP;
  const left = aRect.left + aRect.width / 2;
  const { ref: forwardedRef, style: extraStyle, ...restProps } = extraProps ?? {};
  // Merge consumer ref + our own measurement ref so the layout effect
  // can size against the same node the consumer holds onto.
  const setRef = (node: HTMLDivElement | null) => {
    labelRef.current = node;
    if (typeof forwardedRef === 'function') forwardedRef(node);
    else if (forwardedRef && typeof forwardedRef === 'object') {
      (forwardedRef as React.MutableRefObject<HTMLDivElement | null>).current = node;
    }
  };
  return createPortal(
    <div
      {...restProps}
      ref={setRef}
      className={classNames(className, flip && flippedClassName)}
      data-popover="note-label"
      style={{
        position: 'fixed',
        top,
        left,
        transform: flip ? 'translate(-50%, -100%)' : 'translateX(-50%)',
        margin: 0,
        zIndex: 1100,
        ...extraStyle,
      }}
    >
      {children}
    </div>,
    document.body,
  );
});

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

    // Full (non-windowed) walk to build the per-bar tick descriptors.
    // The time-sig / bpm "changed since the previous bar" flags depend on
    // running state across every bar, so the walk can't be windowed; but
    // it only produces plain data (no DOM / React), so it stays cheap on
    // a long song. The DOM is windowed separately in {@link WindowedTicks}.
    let cumBeats = 0;
    let prevTime: { count: number; unit: number } | undefined;
    // Tempo "carried out" of the previous bar (= its last segment's bpm).
    // Rounded so float jitter (119.97 vs 120.03) doesn't paint a change.
    let prevBpm: number | undefined;
    const ticks: TickDescriptor[] = [];
    for (let i = 0; i < voice.bars.length; i++) {
      const bar = voice.bars[i];
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
      ticks.push({
        barIndex: bar.index,
        startBeat,
        beats: bar.beats,
        timeSec,
        showTimeSig,
        timeCount: bar.time.count,
        timeUnit: bar.time.unit,
        downbeatBpm,
        midBpmChanges,
      });
    }
    return (
      <div className={styles.timelineHeader}>
        <div className={styles.timelineHeaderGutter}>
          <span className={styles.timelineHeaderLabel}>Bar / Time</span>
          <GutterResizeHandle onResizeStart={onResizeGutterStart} />
        </div>
        <div
          className={styles.timelineHeaderBarsRow}
          data-bars-row
          style={
            {
              ['--voice-beats' as string]: voiceBeats,
              ['--bars-row-width' as string]: barsRowWidthSeed(jot, voiceBeats),
            } as React.CSSProperties
          }
          onClick={(e) => seekFromClick(e, onSeek)}
        >
          <WindowedTicks ticks={ticks} />
          <Playhead showLabel onSeek={onSeek} />
        </div>
      </div>
    );
  }
);

/** One timeline-header tick's render data, precomputed by the full bar
 *  walk in {@link TimelineHeader} so {@link WindowedTicks} can window the
 *  DOM without re-deriving the running tempo / time-sig change flags. */
type TickDescriptor = {
  /** Clone-stable bar index; the React key (survives the window sliding). */
  barIndex: number;
  startBeat: number;
  beats: number;
  timeSec: number;
  showTimeSig: boolean;
  timeCount: number;
  timeUnit: number;
  downbeatBpm: number | undefined;
  midBpmChanges: Array<{ beat: number; bpm: number }>;
};

/**
 * Windowed DOM for the timeline-header ticks. Split out of {@link
 * TimelineHeader} so only this map (not the header gutter or its label)
 * re-renders on a scroll / zoom tick. Renders only ticks whose bar span
 * intersects {@link JotViewStore.visibleBeatRange}; the descriptor list
 * is precomputed and stable, so the parent doesn't re-render on scroll.
 */
const WindowedTicks = observer(function WindowedTicks({ ticks }: { ticks: TickDescriptor[] }) {
  const viewport = React.useContext(ViewportStoreContext);
  const range = viewport?.visibleBeatRange ?? null;
  return (
    <>
      {ticks.map((t) => {
        if (!intersectsBeatRange(range, t.startBeat, t.beats)) return null;
        return (
          <React.Fragment key={t.barIndex}>
            <div
              className={styles.timelineHeaderTick}
              style={{ ['--bar-start-beat' as string]: t.startBeat } as React.CSSProperties}
            >
              <div className={styles.timelineHeaderTopRow}>
                <span className={styles.timelineHeaderBar}>{t.barIndex}</span>
                {t.showTimeSig && (
                  <span className={styles.timelineHeaderTimeSig}>
                    {t.timeCount}/{t.timeUnit}
                  </span>
                )}
                {t.downbeatBpm !== undefined && (
                  <span className={styles.timelineHeaderBpm}>{t.downbeatBpm} bpm</span>
                )}
              </div>
              <span className={styles.timelineHeaderTime}>{formatTime(t.timeSec)}</span>
            </div>
            {t.midBpmChanges.map((c, j) => (
              <div
                key={`bpm-${t.barIndex}-${j}`}
                className={styles.timelineHeaderBpmAnchor}
                style={
                  { ['--bar-start-beat' as string]: t.startBeat + c.beat } as React.CSSProperties
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
    </>
  );
});

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
        // Suppress the container's mousedown handler, it begins a
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
      >
        {isFlam && <FlamGrace color={color} config={config} />}
        {isDrag && <DragGrace color={color} config={config} />}
        {badge && <span className={styles.modifierBadge}>{badge}</span>}
        {note.sticking && (
          <span className={styles.stickingBadge}>{note.sticking.toUpperCase()}</span>
        )}
        <PopoverPortal
          anchorRef={noteRef}
          show={showLabel}
          className={styles.noteLabel}
        >
          <div className={styles.noteLabelText}>{description}</div>
          {provenanceEntry && (
            <NoteProvenanceDetails
              entry={provenanceEntry}
              rendered={{ note, bar, provenance: provenance! }}
              startOpen
            />
          )}
        </PopoverPortal>
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
  const provenance = React.useContext(ProvenanceStoreContext);
  const presenter = React.useContext(ProvenancePresenterContext);
  const pinnedKey = `${entry.pitch}:${entry.detected_time_sec}`;
  const clicked = provenance?.pinnedFilteredOnsetKey === pinnedKey;
  const [hovered, setHovered] = React.useState(false);
  const show = hovered || clicked;
  const stop = (e: React.MouseEvent) => e.stopPropagation();
  const anchorRef = React.useRef<HTMLDivElement>(null);
  const labelRef = React.useRef<HTMLDivElement | null>(null);
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
    if (!clicked || !presenter) return;
    const onClickCapture = (e: MouseEvent) => {
      const target = e.target as Node;
      if (anchorRef.current?.contains(target)) return;
      // The label is portaled to `document.body`, so `target.contains`
      // would walk a different subtree; keep the same "is this click
      // inside the popover?" check by comparing against `labelRef`
      // which still holds the portaled element.
      if (labelRef.current?.contains(target)) return;
      e.stopPropagation();
      presenter.setPinnedFilteredOnsetKey(undefined);
    };
    document.addEventListener('click', onClickCapture, true);
    return () => document.removeEventListener('click', onClickCapture, true);
  }, [clicked, presenter]);
  return (
    <div
      ref={anchorRef}
      // Same opt-out as real notes so a click on the ghost doesn't move
      // the playhead via the bars-row seek handler.
      data-noseek="true"
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
        presenter?.setPinnedFilteredOnsetKey(clicked ? undefined : pinnedKey);
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      title={`Filtered onset · pitch ${entry.pitch} · bar ${entry.bar} beat ${entry.beat_in_bar.toFixed(2)}`}
    >
      <PopoverPortal
        anchorRef={anchorRef}
        show={show}
        className={styles.filteredOnsetLabel}
        extraProps={{ ref: labelRef }}
      >
        <NoteProvenanceDetails entry={entry} startOpen />
      </PopoverPortal>
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
