import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { createPortal } from 'react-dom';
import { perfProbe } from 'src/perf_probe';
import { NoteProvenanceEntry } from 'src/debug_zip';
import { RenderedJot } from 'src/jot';
import { jotPlayer } from 'src/playback';
import sharedStyles from '../../jot_view.module.css';
import { GutterResizeHandle } from '../components/gutter_resize_handle';
import {
  ProvenancePresenterContext,
  ProvenanceStoreContext,
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
export function PopoverPortal(props: PopoverPortalProps) {
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

