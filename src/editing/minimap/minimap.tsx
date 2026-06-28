/**
 * Always-visible horizontal-scroll minimap, app-shell chrome between the
 * score scroller and the playback transport. Replaces the native
 * scrollbar (hidden on `.jotContainer`) with a fatter, content-aware
 * strip: a waveform of the loaded backing audio plus colored note
 * ticks sized to the whole song, a translucent viewport rectangle that
 * mirrors the score's visible range, and a live playhead line.
 *
 * Interaction (brainstorm decision (i)):
 *   - Drag on the viewport box: scrolls the score 1:1 against the
 *     minimap-px → score-px scale.
 *   - Click off the box: centers the box on the click and then continues
 *     a drag from there as one fluid gesture.
 *   - The playhead OVERLAY only; clicks do not seek playback.
 *
 * Layout in jot-time (NOT score-px) so the minimap is fully zoom-
 * invariant: every bar's minimap-x / minimap-width derives from
 * `BarTiming.startSec` / `.durationSec`, and notes anchor via
 * `note.beat / bar.beats`.
 *
 * Subcomponents (each one its own `observer` so per-frame state changes
 * only re-render the bits that actually move):
 *   - `Minimap`            shell + canvas + scroll/box wiring
 *   - `MinimapPlayhead`    side-effect-only: tracks `jotPlayer.currentTime`,
 *                          writes `left` to the playhead ref. Same trick
 *                          as `PlayheadPosVar` in jot_editor.tsx; keeps the
 *                          main shell from reconciling 60×/s.
 */
import { reaction } from 'mobx';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { jotPlayer } from 'src/editing/playback/player';
import { trackKey } from 'src/editing/tracks/tracks';
import { type BarSlice, waveformWorker } from 'src/editing/playback/waveform_worker_client';
import styles from './minimap.module.css';
import { WAVEFORM_PAINT_COLOR } from '../utils/waveform_color';
import { JotEditorStore } from '../jot_editor_store';
import { ViewportStore } from '../viewport/viewport_store';
import { MixerStore } from '../mixer/mixer_store';
import { ViewportPresenter } from '../viewport/viewport_presenter';
import { PlaybackPresenter } from '../playback/playback_presenter';
import {
  computeBarLayouts,
  EMPTY_NOTE_MARKS,
  noteMarksEqual,
  type NoteMark,
} from './minimap_layout';

const NOTE_STRIP_H = 16;
const WAVEFORM_H = 36;
const TOTAL_CANVAS_H = NOTE_STRIP_H + WAVEFORM_H;

export const Minimap = observer(
  ({
    jotEditorStore,
    viewport,
    viewportPresenter,
    mixer,
    playbackPresenter,
  }: {
    jotEditorStore: JotEditorStore;
    viewport: ViewportStore;
    viewportPresenter: ViewportPresenter;
    mixer: MixerStore;
    playbackPresenter: PlaybackPresenter;
  }) => {
    const structural = jotEditorStore.structural;
  const jot = jotEditorStore.jot;
  // Read the derived timeline into a variable so its identity (which changes on
  // a tempo edit) is what the layout `useMemo` / peaks `useEffect` depend on,
  // not the stable `jot` reference. `jot` alone never changes across edits.
  const tempoTimeline = jot?.tempoTimeline;
  const containerRef = React.useRef<HTMLDivElement>(null);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);

  // Width measurement. ResizeObserver fires synchronously on `observe`
  // so the first non-zero width arrives before paint. The deps include
  // `structural` because the component returns `null` while no song is
  // loaded (no DOM ref attached); the effect needs to re-run when the JSX
  // first renders the container or it would stay attached to a stale
  // null ref and `width` would never leave 0.
  const [width, setWidth] = React.useState(0);
  React.useLayoutEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const measure = () => setWidth(Math.floor(el.clientWidth));
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [structural]);

  // ─── Per-bar minimap layout (jot-time → minimap px) ────────────────
  const { bars, totalDuration, firstStartSec, hasContent } = React.useMemo(
    () =>
      computeBarLayouts(
        tempoTimeline?.bars ?? [],
        structural?.layers[0]?.bars ?? [],
        width
      ),
    [structural, tempoTimeline, width]
  );

  // ─── Waveform peaks (worker-computed at minimap resolution) ─────────
  // Compute peaks for every AUDIBLE audio track in parallel and sum them
  // element-wise so the minimap waveform reflects the combined mix the
  // user actually hears, not just one track. Each track's worker returns
  // a `Float32Array` of `[min, max]` pairs per pixel column; summing the
  // pairs across tracks gives an envelope-additive view (additive on the
  // bounds is a valid upper bound on the summed signal, which is what
  // the audio graph plays). Muted / solo-excluded / master-muted tracks
  // drop out via {@link JotEditorStore.isAudioTrackAudible}, so a mute
  // toggle reflects in the waveform immediately and a fully-muted bus
  // renders empty rather than misleading the operator.
  const audibleAudioTrackIds = Array.from(jotPlayer.audioTracks.keys()).filter((id) =>
    mixer.isAudioTrackAudible(id)
  );
  const audibleAudioTrackIdsKey = audibleAudioTrackIds.join(',');
  const songLeadInSec = jotPlayer.songLeadInSec;
  const [peaks, setPeaks] = React.useState<Float32Array | null>(null);
  React.useEffect(() => {
    if (
      audibleAudioTrackIds.length === 0 ||
      !hasContent ||
      width <= 0 ||
      bars.length === 0 ||
      !tempoTimeline
    ) {
      setPeaks(null);
      return;
    }
    const timeline = tempoTimeline;
    const slices: BarSlice[] = timeline.bars.map((t, i) => ({
      x: bars[i]?.x ?? 0,
      width: bars[i]?.width ?? 0,
      startSec: t.startSec,
      durationSec: t.durationSec,
    }));
    let cancelled = false;
    Promise.all(
      audibleAudioTrackIds.map((id) =>
        waveformWorker.computePeaks(id, slices, width, songLeadInSec).catch((err) => {
          // eslint-disable-next-line no-console
          console.warn('[minimap] peaks failed for', id, err);
          return null;
        })
      )
    ).then((perTrack) => {
      if (cancelled) return;
      const valid = perTrack.filter((p): p is Float32Array => p !== null);
      if (valid.length === 0) {
        setPeaks(null);
        return;
      }
      if (valid.length === 1) {
        setPeaks(valid[0]);
        return;
      }
      // Pick the longest array as the canonical length; tracks that
      // came back short (worker hiccup, dropped mid-flight) just don't
      // contribute to the tail columns.
      let len = 0;
      for (const p of valid) if (p.length > len) len = p.length;
      const combined = new Float32Array(len);
      for (const p of valid) {
        const n = Math.min(p.length, len);
        for (let i = 0; i < n; i++) combined[i] += p[i];
      }
      setPeaks(combined);
    });
    return () => {
      cancelled = true;
    };
    // `audibleAudioTrackIds` is a fresh array every render, so depend on
    // its string key rather than the array itself; otherwise the effect
    // would refire on every Minimap render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audibleAudioTrackIdsKey, bars, width, hasContent, songLeadInSec, tempoTimeline]);

  // ─── Note marks (color-coded per lane, plotted in minimap-px) ──────
  // Driven by a MobX `reaction` rather than computed in the render body so
  // the array identity stays stable when the underlying data hasn't
  // changed. Computing in-render gave a fresh array every render, and
  // even a Minimap re-render triggered by some other observable (e.g. a
  // stem-mute toggle, a colour pick) refired the canvas paint effect
  // below because its deps array saw a new `noteMarks` reference. The
  // reaction tracks the per-lane `store.getInstrumentTrack(lane).color`
  // observable that a useMemo dep array couldn't capture, and
  // `noteMarksEqual` collapses content-identical results to the same
  // reference so the canvas paint only refires when notes actually moved
  // or recoloured.
  const [noteMarks, setNoteMarks] = React.useState<readonly NoteMark[]>(EMPTY_NOTE_MARKS);
  React.useEffect(() => {
    if (!structural || !hasContent || width <= 0 || bars.length === 0) {
      setNoteMarks(EMPTY_NOTE_MARKS);
      return;
    }
    return reaction(
      () => {
        const layer0 = structural.layers[0];
        const structBars = layer0?.bars ?? [];
        const layer0Id = layer0?.id ?? '';
        const out: NoteMark[] = [];
        for (let i = 0; i < structBars.length; i++) {
          const sb = structBars[i];
          const layout = bars[i];
          if (!layout || sb.beats <= 0) continue;
          for (const [lane, track] of Object.entries(sb.tracks)) {
            // Skip muted / solo-excluded / master-muted lanes so the
            // minimap reads as the audible mix, mirroring the audio
            // waveform path. Tracking `isLaneAudible` here subscribes
            // the reaction to the underlying mute/solo observables, so
            // a toggle updates the ticks live.
            if (!mixer.isTrackAudible(trackKey(layer0Id, lane))) continue;
            const color = mixer.getInstrumentTrack(lane).color;
            for (const note of track.notes) {
              const frac = note.beat / sb.beats;
              out.push({ x: layout.x + frac * layout.width, color });
            }
          }
        }
        return out;
      },
      (next) => setNoteMarks(next),
      { fireImmediately: true, equals: noteMarksEqual }
    );
  }, [structural, bars, width, hasContent, mixer]);

  // ─── Canvas paint (waveform + note ticks in one bitmap) ─────────────
  // Waveform and per-color note ticks are batched through a `Path2D` so
  // each colour pays one `fill()` call regardless of how many bars /
  // notes contribute to it. The pre-batch version issued a `fillRect`
  // per pixel column (1000+ for a typical minimap width) plus a
  // `fillStyle` + `fillRect` pair per note, all of which the profiler
  // surfaced as a hot `fillRect` ridge under `Minimap` even when paints
  // were rare. One `fill()` per colour collapses that ridge to a
  // handful of dispatches.
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || width <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    // Setting `.width` / `.height` clears the canvas state, so only
    // touch them when they actually change. Without the guard, every
    // effect run (jot change, peaks landing, note-color tweak) reset
    // the bitmap even when dimensions were identical, and on a long
    // song the implicit clear plus reallocation was nontrivial.
    const targetW = Math.max(1, Math.floor(width * dpr));
    const targetH = Math.max(1, Math.floor(TOTAL_CANVAS_H * dpr));
    if (canvas.width !== targetW) canvas.width = targetW;
    if (canvas.height !== targetH) canvas.height = targetH;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, TOTAL_CANVAS_H);

    const waveTop = NOTE_STRIP_H;
    const waveBottom = NOTE_STRIP_H + WAVEFORM_H;
    const mid = (waveTop + waveBottom) / 2;

    if (peaks) {
      const scale = (WAVEFORM_H / 2) * 0.9;
      const cols = Math.min(width, Math.floor(peaks.length / 2));
      const wavePath = new Path2D();
      for (let p = 0; p < cols; p++) {
        const mn = peaks[p * 2];
        const mx = peaks[p * 2 + 1];
        const y0 = mid - mx * scale;
        const y1 = mid - mn * scale;
        wavePath.rect(p, y0, 1, Math.max(1, y1 - y0));
      }
      ctx.fillStyle = WAVEFORM_PAINT_COLOR;
      ctx.fill(wavePath);
    }

    if (noteMarks.length > 0) {
      const tickH = NOTE_STRIP_H - 2;
      // Group note rects by colour so each colour becomes one Path2D +
      // one `fill()`. Typical jots use ≤ 8 distinct lane colours, so
      // this collapses hundreds of fillRect dispatches into a handful.
      const pathByColor = new Map<string, Path2D>();
      for (const mark of noteMarks) {
        let path = pathByColor.get(mark.color);
        if (!path) {
          path = new Path2D();
          pathByColor.set(mark.color, path);
        }
        path.rect(Math.floor(mark.x) - 1, 1, 2, tickH);
      }
      for (const [color, path] of pathByColor) {
        ctx.fillStyle = color;
        ctx.fill(path);
      }
    }
  }, [peaks, noteMarks, width]);

  // ─── Pointer interaction: click-to-jump + drag-to-scroll ───────────
  // rAF-coalesced: pointermove can fire 120+ Hz on modern trackpads;
  // each scroll write triggers downstream observer reactions (the
  // viewport box update via `MinimapViewportBox`, PlayheadPosVar reading
  // scrollX during auto-follow). Capping writes at one per frame caps
  // that cascade at the display refresh rate, which is the maximum
  // useful rate anyway.
  //
  // The handler reads `store.scrollX` / `_contentWidth` / `_viewportWidth`
  // FRESH at pointer-down time (not from closure-captured render values)
  // so the parent `Minimap` doesn't have to subscribe to those
  // frame-cadence observables. See `MinimapViewportBox` below for why
  // the viewport-box rendering is its own observer.
  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0 || !hasContent) return;
    const sw = viewport._contentWidth;
    const cw = viewport._viewportWidth;
    if (sw <= 0 || width <= 0) return;
    e.preventDefault();
    // Clicking or dragging the minimap is an explicit "scroll the score
    // somewhere else" intent; auto-follow would re-pin the playhead on
    // the next frame and visually undo the user's nudge.
    playbackPresenter.setFollowPlayhead(false);
    const rect = e.currentTarget.getBoundingClientRect();
    const scale = sw / width;

    const boxLeftAtDown = (viewport.scrollX / sw) * width;
    const boxWidthAtDown = Math.max(2, (cw / sw) * width);
    const originX = e.clientX;
    const localXAtDown = originX - rect.left;
    const onBox = localXAtDown >= boxLeftAtDown && localXAtDown <= boxLeftAtDown + boxWidthAtDown;
    let startScroll = viewport.scrollX;
    if (!onBox) {
      // Centre the box on the click; the drag continues from there.
      const target = (localXAtDown / width) * sw - cw / 2;
      viewportPresenter.setScrollX(target);
      startScroll = viewport.scrollX;
    }

    let pendingClientX = originX;
    let rafId = 0;
    const flush = () => {
      rafId = 0;
      const dx = pendingClientX - originX;
      viewportPresenter.setScrollX(startScroll + dx * scale);
    };
    const onMove = (ev: PointerEvent) => {
      pendingClientX = ev.clientX;
      if (rafId === 0) rafId = requestAnimationFrame(flush);
    };
    const onUp = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
      if (rafId !== 0) {
        cancelAnimationFrame(rafId);
        // One final synchronous flush so a quick click-release doesn't
        // leave the latest pointer position unapplied.
        flush();
      }
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
  };

  if (!structural || !jot) return null;

  return (
    <div
      ref={containerRef}
      className={hasContent ? styles.minimap : `${styles.minimap} ${styles.minimapEmpty}`}
      onPointerDown={onPointerDown}
      aria-label="Song minimap and horizontal scroller"
    >
      <canvas ref={canvasRef} className={styles.canvas} />
      <MinimapViewportBox viewport={viewport} width={width} hasContent={hasContent} />
      <MinimapPlayhead
        totalDuration={totalDuration}
        firstStartSec={firstStartSec}
        width={width}
        hasContent={hasContent}
      />
    </div>
  );
});

/**
 * Scroll-tracking layer of the minimap: viewport box + the two dim
 * overlays that fade the off-viewport range. Isolated from the
 * `Minimap` shell because `store.scrollX` / `_contentWidth` /
 * `_viewportWidth` update at frame cadence during auto-follow playback;
 * reading them in the shell would force a per-tick re-render of the
 * whole minimap (which then rebuilt `noteMarks` and forced the canvas
 * paint effect to refire because its deps array saw a new array
 * identity, even when nothing had actually changed).
 *
 * The values computed here only ever flow into `transform: translateX`
 * / `transform: scaleX` on the corresponding elements, never into
 * `left` / `width` per-frame. That's deliberate: this re-renders at
 * frame cadence (≥ 120 Hz during auto-follow), and `left` / `width`
 * writes trigger layout. The dim strips are laid out at full minimap
 * width with `transform-origin` at their outer edge so
 * `scaleX(fraction)` collapses each to the off-viewport range without
 * a layout pass; the box only needs `translateX` because its `width`
 * is resize-cadence, not scroll-cadence. See the per-frame perf block
 * at the top of `minimap.module.css` for the full rationale; keep
 * these two in sync if either changes.
 */
const MinimapViewportBox = observer(
  ({
    viewport,
    width,
    hasContent,
  }: {
    viewport: ViewportStore;
    width: number;
    hasContent: boolean;
  }) => {
    const scrollX = viewport.scrollX;
    const sw = viewport._contentWidth;
    const cw = viewport._viewportWidth;
    let boxLeft = 0;
    let boxWidth = 0;
    let dimLeftScaleX = 0;
    let dimRightScaleX = 0;
    if (hasContent && width > 0 && sw > 0) {
      boxLeft = (scrollX / sw) * width;
      boxWidth = Math.max(2, (cw / sw) * width);
      const rightStart = boxLeft + boxWidth;
      dimLeftScaleX = Math.max(0, boxLeft) / width;
      dimRightScaleX = Math.max(0, width - rightStart) / width;
    }
    return (
      <>
        <div
          className={styles.dimLeft}
          aria-hidden="true"
          style={{ transform: `scaleX(${dimLeftScaleX})` }}
        />
        <div
          className={styles.dimRight}
          aria-hidden="true"
          style={{ transform: `scaleX(${dimRightScaleX})` }}
        />
        <div
          className={styles.viewportBox}
          aria-hidden="true"
          style={{ width: boxWidth, transform: `translateX(${boxLeft}px)` }}
        />
      </>
    );
  }
);

/**
 * Per-frame minimap playhead. Isolated as its own observer so reading
 * `jotPlayer.currentTime` (a frame-cadence observable) doesn't pull the
 * surrounding `Minimap` into the per-tick render path.
 *
 * Per-frame perf: position flows through `transform: translateX(...)`
 * - compositor-only and subpixel-precise - paired with
 * `will-change: transform` on `.playhead` so the element sits on its
 * own GPU layer and the per-tick write doesn't repaint siblings. Do
 * NOT replace with `style={{ left: ... }}` per frame: `left` triggers
 * layout, which at 120 Hz cadence across a long score costs more than
 * the frame budget. See the `.playhead` rule in `minimap.module.css`
 * for the matching CSS-side rationale.
 *
 * Inactive (`!active`) returns `null` so there's no element to paint
 * at all - cleaner than a `display: none` toggle and avoids the
 * between-mount-and-first-tick flash the old imperative version had to
 * suppress.
 */
const MinimapPlayhead = observer(
  ({
    totalDuration,
    firstStartSec,
    width,
    hasContent,
  }: {
    totalDuration: number;
    firstStartSec: number;
    width: number;
    hasContent: boolean;
  }) => {
    const t = jotPlayer.currentTime;
    const state = jotPlayer.state;
    const cued = jotPlayer.cued;
    const timeline = jotPlayer.timeline;
    const active =
      (state === 'playing' || state === 'paused' || cued) &&
      timeline.bars.length > 0 &&
      hasContent &&
      totalDuration > 0 &&
      width > 0;
    if (!active) return null;
    const frac = Math.max(0, Math.min(1, (t - firstStartSec) / totalDuration));
    const x = frac * width;
    return (
      <div
        className={styles.playhead}
        aria-hidden="true"
        style={{ transform: `translateX(${x}px)` }}
      />
    );
  }
);
