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
 *                          as `PlayheadPosVar` in jot_view.tsx; keeps the
 *                          main shell from reconciling 60×/s.
 */
import { observer } from 'mobx-react-lite';
import React from 'react';
import { buildTimeline, jotPlayer } from 'src/playback';
import { type BarSlice, waveformWorker } from 'src/playback/waveform_worker_client';
import styles from './minimap.module.css';
import { WAVEFORM_PAINT_COLOR } from './score';
import { JotViewStore } from './store';

const NOTE_STRIP_H = 16;
const WAVEFORM_H = 36;
const TOTAL_CANVAS_H = NOTE_STRIP_H + WAVEFORM_H;

type BarLayout = {
  /** Minimap-px x of the bar's left edge. */
  x: number;
  /** Minimap-px width of the bar. */
  width: number;
  /** Bar length in DSL beats (== `StructuralBar.beats`); used to map
   *  per-note `beat` positions onto the bar's minimap pixel range. */
  beats: number;
};

type NoteMark = {
  x: number;
  color: string;
};

/**
 * Pick the audio track to render as the minimap's waveform. Priority:
 * a backing track (no `pitch`; typically `no_drums` or an ad-hoc music
 * file), else the first loaded track. Per-pitch drum stems are sparse
 * and the note ticks above already convey those hits, so they're a
 * fallback rather than the preferred surface.
 */
function pickWaveformTrack(
  audioTracks: ReadonlyMap<string, { id: string; pitch?: string }>,
): { id: string } | undefined {
  let backing: { id: string } | undefined;
  let first: { id: string } | undefined;
  for (const t of audioTracks.values()) {
    if (!first) first = t;
    if (!t.pitch && !backing) backing = t;
  }
  return backing ?? first;
}

export const Minimap = observer(({ store }: { store: JotViewStore }) => {
  const jot = store.currentJot;
  const containerRef = React.useRef<HTMLDivElement>(null);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const viewportBoxRef = React.useRef<HTMLDivElement>(null);
  const dimLeftRef = React.useRef<HTMLDivElement>(null);
  const dimRightRef = React.useRef<HTMLDivElement>(null);
  const playheadRef = React.useRef<HTMLDivElement>(null);

  // Width measurement. ResizeObserver fires synchronously on `observe`
  // so the first non-zero width arrives before paint. The deps include
  // `jot` because the component returns `null` while `jot` is undefined
  // (no DOM ref attached); the effect needs to re-run when the JSX
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
  }, [jot]);

  // ─── Per-bar minimap layout (jot-time → minimap px) ────────────────
  const { bars, totalDuration, firstStartSec, hasContent } = React.useMemo(() => {
    const empty = {
      bars: [] as BarLayout[],
      totalDuration: 0,
      firstStartSec: 0,
      hasContent: false,
    };
    if (!jot || width <= 0) return empty;
    const timeline = buildTimeline(jot);
    const structBars = jot.structure.voices[0]?.bars ?? [];
    if (timeline.bars.length === 0 || structBars.length === 0) return empty;
    const first = timeline.bars[0].startSec;
    const last = timeline.bars[timeline.bars.length - 1];
    const total = last.startSec + last.durationSec - first;
    if (total <= 0) return empty;
    const layouts: BarLayout[] = new Array(timeline.bars.length);
    for (let i = 0; i < timeline.bars.length; i++) {
      const t = timeline.bars[i];
      layouts[i] = {
        x: ((t.startSec - first) / total) * width,
        width: (t.durationSec / total) * width,
        beats: structBars[i]?.beats ?? 0,
      };
    }
    return { bars: layouts, totalDuration: total, firstStartSec: first, hasContent: true };
  }, [jot, width]);

  // ─── Waveform peaks (worker-computed at minimap resolution) ─────────
  const audioTrack = pickWaveformTrack(jotPlayer.audioTracks);
  const drumsT0Sec = jotPlayer.drumsT0Sec;
  const [peaks, setPeaks] = React.useState<Float32Array | null>(null);
  React.useEffect(() => {
    if (!audioTrack || !hasContent || width <= 0 || bars.length === 0 || !jot) {
      setPeaks(null);
      return;
    }
    const timeline = buildTimeline(jot);
    const slices: BarSlice[] = timeline.bars.map((t, i) => ({
      x: bars[i]?.x ?? 0,
      width: bars[i]?.width ?? 0,
      startSec: t.startSec,
      durationSec: t.durationSec,
    }));
    let cancelled = false;
    waveformWorker
      .computePeaks(audioTrack.id, slices, width, drumsT0Sec)
      .then((p) => {
        if (!cancelled) setPeaks(p);
      })
      .catch((err) => {
        if (!cancelled) {
          // eslint-disable-next-line no-console
          console.warn('[minimap] peaks failed:', err);
          setPeaks(null);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [audioTrack?.id, bars, width, hasContent, drumsT0Sec, jot]);

  // ─── Note marks (color-coded per pitch, plotted in minimap-px) ──────
  const noteMarks = React.useMemo<NoteMark[]>(() => {
    if (!jot || !hasContent || width <= 0 || bars.length === 0) return [];
    const structBars = jot.structure.voices[0]?.bars ?? [];
    const out: NoteMark[] = [];
    for (let i = 0; i < structBars.length; i++) {
      const sb = structBars[i];
      const layout = bars[i];
      if (!layout || sb.beats <= 0) continue;
      for (const track of Object.values(sb.tracks)) {
        const color = track.color;
        for (const note of track.notes) {
          const frac = note.beat / sb.beats;
          out.push({ x: layout.x + frac * layout.width, color });
        }
      }
    }
    return out;
  }, [jot, bars, hasContent, width]);

  // ─── Canvas paint (waveform + note ticks in one bitmap) ─────────────
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || width <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(width * dpr));
    canvas.height = Math.max(1, Math.floor(TOTAL_CANVAS_H * dpr));
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, TOTAL_CANVAS_H);

    const waveTop = NOTE_STRIP_H;
    const waveBottom = NOTE_STRIP_H + WAVEFORM_H;
    const mid = (waveTop + waveBottom) / 2;

    if (peaks) {
      const scale = (WAVEFORM_H / 2) * 0.9;
      ctx.fillStyle = WAVEFORM_PAINT_COLOR;
      const cols = Math.min(width, Math.floor(peaks.length / 2));
      for (let p = 0; p < cols; p++) {
        const mn = peaks[p * 2];
        const mx = peaks[p * 2 + 1];
        const y0 = mid - mx * scale;
        const y1 = mid - mn * scale;
        ctx.fillRect(p, y0, 1, Math.max(1, y1 - y0));
      }
    }

    if (noteMarks.length > 0) {
      const tickH = NOTE_STRIP_H - 2;
      for (const mark of noteMarks) {
        ctx.fillStyle = mark.color;
        ctx.fillRect(Math.floor(mark.x) - 1, 1, 2, tickH);
      }
    }
  }, [peaks, noteMarks, width]);

  // ─── Score scroll → viewport box + dim overlay positions ──────────
  // The dim overlays sit flush against either side of the viewport box,
  // covering everything outside it with a subtle fade so the in-view
  // range reads as the prominent slice (IDE-minimap convention).
  // Inputs (`scrollX`, `_viewportWidth`, `_contentWidth`) are MobX
  // observables on the store; the component re-renders whenever any of
  // them changes and a useLayoutEffect writes the derived px values to
  // the imperative refs to avoid touching JSX for a high-frequency style.
  const scrollX = store.scrollX;
  const sw = store._contentWidth;
  const cw = store._viewportWidth;
  React.useLayoutEffect(() => {
    if (!hasContent || width <= 0) return;
    const box = viewportBoxRef.current;
    const dimLeft = dimLeftRef.current;
    const dimRight = dimRightRef.current;
    if (!box || !dimLeft || !dimRight) return;
    if (sw <= 0) {
      box.style.width = '0px';
      dimLeft.style.width = '0px';
      dimRight.style.width = '0px';
      return;
    }
    const boxLeft = (scrollX / sw) * width;
    const boxWidth = Math.max(2, (cw / sw) * width);
    box.style.left = `${boxLeft}px`;
    box.style.width = `${boxWidth}px`;
    dimLeft.style.left = '0px';
    dimLeft.style.width = `${Math.max(0, boxLeft)}px`;
    const rightStart = boxLeft + boxWidth;
    dimRight.style.left = `${rightStart}px`;
    dimRight.style.width = `${Math.max(0, width - rightStart)}px`;
  }, [scrollX, sw, cw, width, hasContent]);

  // ─── Pointer interaction: click-to-jump + drag-to-scroll ───────────
  // rAF-coalesced: pointermove can fire 120+ Hz on modern trackpads;
  // each scroll write triggers downstream observer reactions (the
  // viewport box update above, PlayheadPosVar reading scrollX during
  // auto-follow). Capping writes at one per frame caps that cascade at
  // the display refresh rate, which is the maximum useful rate anyway.
  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0 || !hasContent) return;
    if (sw <= 0 || width <= 0) return;
    e.preventDefault();
    const rect = e.currentTarget.getBoundingClientRect();
    const scale = sw / width;

    const boxLeftAtDown = (store.scrollX / sw) * width;
    const boxWidthAtDown = Math.max(2, (cw / sw) * width);
    const originX = e.clientX;
    const localXAtDown = originX - rect.left;
    const onBox =
      localXAtDown >= boxLeftAtDown && localXAtDown <= boxLeftAtDown + boxWidthAtDown;
    let startScroll = store.scrollX;
    if (!onBox) {
      // Centre the box on the click; the drag continues from there.
      const target = (localXAtDown / width) * sw - cw / 2;
      store.setScrollX(target);
      startScroll = store.scrollX;
    }

    let pendingClientX = originX;
    let rafId = 0;
    const flush = () => {
      rafId = 0;
      const dx = pendingClientX - originX;
      store.setScrollX(startScroll + dx * scale);
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

  if (!jot) return null;

  return (
    <div
      ref={containerRef}
      className={hasContent ? styles.minimap : `${styles.minimap} ${styles.minimapEmpty}`}
      onPointerDown={onPointerDown}
      aria-label="Song minimap and horizontal scroller"
    >
      <canvas ref={canvasRef} className={styles.canvas} />
      <div ref={dimLeftRef} className={styles.dimLeft} aria-hidden="true" />
      <div ref={dimRightRef} className={styles.dimRight} aria-hidden="true" />
      <div ref={viewportBoxRef} className={styles.viewportBox} aria-hidden="true" />
      <MinimapPlayhead
        playheadRef={playheadRef}
        totalDuration={totalDuration}
        firstStartSec={firstStartSec}
        width={width}
        hasContent={hasContent}
      />
      <div ref={playheadRef} className={styles.playhead} aria-hidden="true" />
    </div>
  );
});

/**
 * Side-effect-only observer that tracks `jotPlayer.currentTime` and
 * writes `left` (in CSS px) onto the playhead ref. Renders nothing; * isolating the per-frame observable read here keeps the rest of the
 * minimap shell out of MobX's tick path. Direct mirror of the score's
 * `PlayheadPosVar` (see jot_view.tsx).
 */
const MinimapPlayhead = observer(
  ({
    playheadRef,
    totalDuration,
    firstStartSec,
    width,
    hasContent,
  }: {
    playheadRef: React.RefObject<HTMLDivElement>;
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
    React.useLayoutEffect(() => {
      const el = playheadRef.current;
      if (!el) return;
      if (!active) {
        el.style.display = 'none';
        return;
      }
      el.style.display = 'block';
      const frac = Math.max(0, Math.min(1, (t - firstStartSec) / totalDuration));
      el.style.left = `${frac * width}px`;
    }, [t, active, totalDuration, firstStartSec, width, playheadRef]);
    return null;
  }
);
