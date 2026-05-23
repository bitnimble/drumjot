import { untracked } from 'mobx';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { Point } from 'src/geom';
import { RenderedJot } from 'src/jot';
import { BarTiming, buildTimeline, jotPlayer, timeToX } from 'src/playback';
import { SelectionStore } from 'src/selection';
import styles from './jot_view.module.css';
import {
  BarTimingsContext,
  NoteProvenanceContext,
  NoteProvenanceContextValue,
  SelectionContext,
} from './jot_view/contexts';
import {
  AudioTrackControls,
  MixerView,
  VoiceControls,
} from './jot_view/mixer';
import { PlaybackBar } from './jot_view/playback';
import { Legend, TimelineHeader, formatSubtitle } from './jot_view/score';
import { JotViewStore, TrackKey } from './jot_view/store';
import { DebugPanel, Toolbar } from './jot_view/toolbar';
import { ExampleJot } from 'src/fakes';

export { JotViewStore } from './jot_view/store';
export type { TrackKey, TranscribeOptions, TranscribeStatus } from './jot_view/store';

type CreateJotViewOptions = {
  examples?: readonly ExampleJot[];
};

type CreateJotViewResult = {
  store: JotViewStore;
  View: React.FC;
};

export function createJotView(options: CreateJotViewOptions = {}): CreateJotViewResult {
  const store = new JotViewStore();
  if (options.examples) store.setExamples(options.examples);
  const selection = new SelectionStore(store);

  // The marquee div is `position: absolute` inside `.jotContainer`
  // (the scroll surface, `position: relative`), so its `top`/`left`
  // need to be in that container's content coordinate space — viewport
  // `clientX`/`Y` would offset the rectangle by everything between the
  // viewport edge and the container's content origin (toolbar + the
  // Audio/Drums master rows + whatever's scrolled out of view above).
  const containerPoint = (e: React.MouseEvent<HTMLDivElement>): Point => {
    const el = e.currentTarget;
    const rect = el.getBoundingClientRect();
    return new Point(
      e.clientX - rect.left + el.scrollLeft,
      e.clientY - rect.top + el.scrollTop,
    );
  };
  const onMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    selection.beginSelection(containerPoint(e));
  };
  const onMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    selection.moveSelection(containerPoint(e));
  };

  const View: React.FC = observer(() => {
    const jot = store.currentJot;

    // Spacebar = play / pause / resume, from anywhere on the page. Skip
    // only when a text-entry control has focus (the user is typing) or
    // a SELECT is focused (let space/arrows drive the native picker).
    // A focused BUTTON deliberately falls through: preventDefault both
    // stops the browser's space-to-scroll and suppresses the button's
    // space-activation, so spacebar *always* toggles transport. A
    // focused range slider (e.g. Zoom) also falls through — space has
    // no native slider function, so swallowing it here would silently
    // break play/pause until the user clicked elsewhere.
    React.useEffect(() => {
      // INPUT types where space is meaningful text/native input and the
      // shortcut must yield. A range/checkbox/etc. input is not listed,
      // so spacebar still toggles transport while it has focus.
      const TEXT_ENTRY_INPUT_TYPES = new Set([
        'text',
        'search',
        'email',
        'url',
        'tel',
        'password',
        'number',
      ]);
      const onKeyDown = (e: KeyboardEvent) => {
        if (e.code !== 'Space' && e.key !== ' ') return;
        const el = e.target as HTMLElement | null;
        const tag = el?.tagName;
        const isTextEntryInput =
          tag === 'INPUT' &&
          TEXT_ENTRY_INPUT_TYPES.has((el as HTMLInputElement).type);
        if (
          isTextEntryInput ||
          tag === 'TEXTAREA' ||
          tag === 'SELECT' ||
          el?.isContentEditable
        ) {
          return;
        }
        e.preventDefault();
        void store.togglePlayPause();
      };
      window.addEventListener('keydown', onKeyDown);
      return () => window.removeEventListener('keydown', onKeyDown);
    }, []);

    const provenanceContextValue: NoteProvenanceContextValue | null = store.noteProvenance
      ? {
          byTick: store.noteProvenanceByTick,
          rejectedByPitch: store.filteredOnsetsByPitch,
          leadBars: store.noteProvenance.lead_bars ?? 0,
          showFiltered: store.showFilteredOnsets,
          beatAlignmentOffsetSec:
            store.noteProvenance.beat_alignment_offset_sec ?? null,
          // Bundle manifest mapping is `Record<string, string>`; rebuild
          // it as a Map for ergonomic .get() lookups inside the per-onset
          // timing visualization. Empty when the current bundle didn't
          // ship a manifest (hand-authored jots, legacy bundles).
          audioFilenameByPitch: new Map(
            Object.entries(store.lastDebugBundle?.mapping ?? {}),
          ),
        }
      : null;

    return (
      <SelectionContext.Provider value={selection}>
      <NoteProvenanceContext.Provider value={provenanceContextValue}>
      <div className={styles.appContainer}>
        <Toolbar
          examples={store.examples}
          currentId={store.currentExampleId}
          onSelect={(id) => store.loadExample(id)}
          transcribeStatus={store.transcribeStatus}
          transcribeOptions={store.transcribeOptions}
          onTranscribe={(file) => store.transcribeAudio(file)}
          onResumeTranscribe={(folder, stage) =>
            store.resumeTranscribe(folder, stage)
          }
          onLoadJot={(file) => store.loadJotFile(file)}
          onLoadMidi={(file) => store.loadMidiFile(file)}
          onLoadParadb={(file) => store.loadParadbMap(file)}
          onLoadDebugBundle={(file) => store.loadDebugBundleFile(file)}
          onLoadAudioTrack={(file) => store.loadAudioTrack(file)}
          onCancelTranscribe={() => store.cancelTranscribe()}
          onClearTranscribeStatus={() => store.clearTranscribeStatus()}
          onSetBeatInput={(b) => store.setBeatInput(b)}
          zoom={store.zoom}
          onSetZoom={(z) => store.setZoom(z)}
          hasNoteProvenance={store.noteProvenance !== undefined}
          showFilteredOnsets={store.showFilteredOnsets}
          onSetShowFilteredOnsets={(v) => store.setShowFilteredOnsets(v)}
          recentTranscriptions={store.recentTranscriptions}
          selectedResumeFolder={store.selectedResumeFolder}
          selectedResumeStage={store.selectedResumeStage}
          onSetSelectedResumeFolder={(f) => store.setSelectedResumeFolder(f)}
          onSetSelectedResumeStage={(s) => store.setSelectedResumeStage(s)}
          onRefreshRecentTranscriptions={() =>
            store.refreshRecentTranscriptions()
          }
        />
        {jot ? (
          <JotView
            jot={jot}
            highlightedPattern={selection.selectedPattern}
            onPatternClick={(name) => selection.togglePattern(name)}
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={selection.endSelection}
            onSeek={(x) => store.seekToX(x)}
            onZoomBy={(factor) => store.setZoom(store.zoom * factor)}
            trackOrder={store.trackOrder}
            onMoveTrack={(from, to) => store.moveTrack(from, to)}
            voiceControls={{
              mutedPitches: store.mutedPitches,
              soloedPitches: store.soloedPitches,
              isPitchAudible: (pitch) => store.isPitchAudible(pitch),
              volumeFor: (pitch) => store.pitchVolume(pitch),
              onSetVolume: (pitch, v) => store.setPitchVolume(pitch, v),
              onToggleMute: (pitch) => store.toggleMute(pitch),
              onToggleSolo: (pitch) => store.toggleSolo(pitch),
            }}
            audioTrackControls={{
              mutedAudioTracks: store.mutedAudioTracks,
              soloedAudioTracks: store.soloedAudioTracks,
              isAudioTrackAudible: (id) => store.isAudioTrackAudible(id),
              volumeFor: (id) => store.audioTrackVolume(id),
              onSetVolume: (id, v) => store.setAudioTrackVolume(id, v),
              onToggleMute: (id) => store.toggleAudioTrackMute(id),
              onToggleSolo: (id) => store.toggleAudioTrackSolo(id),
              onClear: (id) => store.clearAudioTrack(id),
            }}
            getGutterWidth={() => store.gutterWidth}
            onSetGutterWidth={(px) => store.setGutterWidth(px)}
          />
        ) : (
          <div className={styles.empty}>No jot loaded</div>
        )}
        <PlaybackBar store={store} />
        <DebugPanel store={store} />
        <LoadingOverlay store={store} />
      </div>
      </NoteProvenanceContext.Provider>
      </SelectionContext.Provider>
    );
  });

  return { store, View };
}

type JotViewProps = {
  jot: RenderedJot;
  highlightedPattern: string | undefined;
  onPatternClick: (name: string) => void;
  onMouseDown: (e: React.MouseEvent<HTMLDivElement>) => void;
  onMouseMove: (e: React.MouseEvent<HTMLDivElement>) => void;
  onMouseUp: () => void;
  /** Click-to-seek with a bars-row-local pixel x. */
  onSeek: (x: number) => void;
  /**
   * Multiply the current score zoom by `factor` (Cmd/Ctrl + wheel).
   * The store clamps to the slider's range.
   */
  onZoomBy: (factor: number) => void;
  /**
   * User-customizable mixer ordering — drum-instrument rows and audio
   * tracks freely interleaved. Drives both row order and which
   * drum-pitch row hosts the pattern/tuplet bracket overlay (the
   * topmost drum row in this list).
   */
  trackOrder: readonly TrackKey[];
  /** Move the row at `from` to position `to` (drag-and-drop / Alt+arrow). */
  onMoveTrack: (from: number, to: number) => void;
  voiceControls: VoiceControls;
  audioTrackControls: AudioTrackControls;
  /** Read the current sticky-gutter width (px). A getter (not a value)
   * so the parent View doesn't reactively re-render on every resize
   * tick — the value flows into the DOM via `GutterWidthVar` as a
   * side-effect-only CSS-var update, and is also read once at drag
   * start to snapshot the starting width. */
  getGutterWidth: () => number;
  /** Apply a new sticky-gutter width (px); called by the gutter resize
   * handle's pointer-move stream. */
  onSetGutterWidth: (px: number) => void;
};

const JotView = observer((props: JotViewProps) => {
  const {
    jot,
    highlightedPattern,
    onPatternClick,
    onMouseDown,
    onMouseMove,
    onMouseUp,
    onSeek,
    onZoomBy,
    trackOrder,
    onMoveTrack,
    voiceControls,
    audioTrackControls,
    getGutterWidth,
    onSetGutterWidth,
  } = props;
  // Intentionally NOT reading `jot.resolved` here — every observable
  // touched in this body triggers a JotView re-render on zoom, and the
  // title / subtitle / Legend / mixer subtree all derive from zoom-
  // invariant data via `jot.structure` / `jot.title` /
  // `jot.globalMetadata`. JotView itself is then stable across zoom
  // (ScoreZoomVar updates the one CSS variable that propagates the
  // new scale to every descendant via calc()).
  const config = jot.config;
  const containerRef = React.useRef<HTMLDivElement>(null);

  // Wheel zooms the score (mirrors the Zoom slider) — no modifier
  // required, and Ctrl/Cmd + wheel still works (also covers the macOS
  // trackpad pinch gesture, which Chrome/Safari deliver as a synthetic
  // Ctrl + wheel). The listener is registered natively with
  // `{ passive: false }` because React's synthetic `onWheel` is passive
  // — `preventDefault` there is a no-op, and we must cancel both the
  // native page scroll and the browser's own page zoom on Ctrl/Cmd +
  // wheel. Wheel events are coalesced per animation frame: a 120 Hz
  // trackpad fires ~8 events per frame, but only the final composite
  // zoom is visible, so summing deltas and applying once skips ~7×
  // wasted layout/render passes per frame.
  const onZoomByRef = React.useRef(onZoomBy);
  onZoomByRef.current = onZoomBy;
  React.useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    let pendingDelta = 0;
    let rafId = 0;
    const flush = () => {
      rafId = 0;
      const delta = pendingDelta;
      pendingDelta = 0;
      if (delta === 0) return;
      onZoomByRef.current(Math.exp(-delta * 0.0015));
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      // deltaMode 1 = lines (typically a notched mouse wheel); scale
      // it up so a single notch zooms a comparable amount to a
      // pixel-mode trackpad swipe. Scrolling up (deltaY < 0) zooms in.
      const unit = e.deltaMode === 1 ? 16 : 1;
      pendingDelta += e.deltaY * unit;
      if (rafId === 0) rafId = requestAnimationFrame(flush);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => {
      el.removeEventListener('wheel', onWheel);
      if (rafId !== 0) cancelAnimationFrame(rafId);
    };
  }, []);

  // Middle-mouse + drag pans the scroller in both axes. The mousedown
  // listener is on the container so preventDefault can suppress the
  // Windows/Linux autoscroll cursor (and X11 middle-click paste); the
  // mousemove/up listeners go on window so a drag that wanders out of
  // the container still tracks and releases cleanly.
  React.useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    let panning = false;
    let lastX = 0;
    let lastY = 0;
    let prevCursor = '';
    const onMouseDown = (e: MouseEvent) => {
      if (e.button !== 1) return;
      e.preventDefault();
      panning = true;
      lastX = e.clientX;
      lastY = e.clientY;
      prevCursor = el.style.cursor;
      el.style.cursor = 'grabbing';
    };
    const onMouseMove = (e: MouseEvent) => {
      if (!panning) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      el.scrollLeft -= dx;
      el.scrollTop -= dy;
    };
    const stop = () => {
      if (!panning) return;
      panning = false;
      el.style.cursor = prevCursor;
    };
    el.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', stop);
    window.addEventListener('blur', stop);
    return () => {
      el.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', stop);
      window.removeEventListener('blur', stop);
    };
  }, []);

  // `--note-pad-px` is the engraving inset every note's CSS `left`
  // calc() reads; it never changes at runtime. `--px-per-beat` and
  // `--gutter-width` are the two values that get mutated at runtime —
  // both live on the same root so every descendant calc() chain reads
  // from a single ancestor. The pad var goes in inline style (set
  // once). `--px-per-beat` and `--gutter-width` are updated by
  // `ScoreZoomVar` / `GutterWidthVar`, side-effect-only observers
  // that write via `setProperty` on `containerRef.current` so the
  // tick doesn't re-render JotView — only mutates one DOM attribute.
  // With many tracks loaded a JotView re-render is expensive enough
  // to make a 120 Hz resize drag visibly laggy, so we keep both reads
  // off the render path.
  //
  // We do, however, seed the *initial* `--gutter-width` value inline
  // so the very first paint has the right value — otherwise the var
  // would be undefined for one frame and every gutter's
  // `width: var(--gutter-width)` would fall back to `auto`, sizing
  // each row to its own content (a visible staggered-edge flash
  // until `GutterWidthVar`'s `useLayoutEffect` could update the
  // children). The seed is read with mobx's `untracked` so this
  // render doesn't subscribe — `GutterWidthVar` is still the only
  // reactive reader.
  const initialGutterPx = React.useMemo(
    () => untracked(() => getGutterWidth()),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- seed only.
    [],
  );
  const containerStyle = {
    ['--note-pad-px' as string]: `${config.barNotePaddingLeft}px`,
    ['--gutter-width' as string]: `${initialGutterPx}px`,
  } as React.CSSProperties;
  // Eager bar-timings table for `BarTimingsContext`. Built once per jot
  // identity here so deep consumers (NoteProvenanceDetails' "Final
  // position" row) don't need the player's timeline to have been
  // initialized — pre-Play, `jotPlayer.timeline` is `EMPTY_TIMELINE`.
  // `buildTimeline` now reads zoom-invariant fields only, so calling
  // it here doesn't bind JotView's render to the zoom variable.
  const barTimings = React.useMemo<ReadonlyMap<number, BarTiming>>(() => {
    const timeline = buildTimeline(jot);
    const structBars = jot.structure.voices[0]?.bars ?? [];
    const map = new Map<number, BarTiming>();
    for (let i = 0; i < structBars.length; i++) {
      const timing = timeline.bars[i];
      if (timing) map.set(structBars[i].index, timing);
    }
    return map;
  }, [jot]);
  // The starting width is captured at the start of each drag (the
  // pointermove deltas then read against that snapshot) so an in-
  // flight resize stays anchored to where the user grabbed even
  // though the live `gutterWidth` observable is being updated every
  // frame. Read via the getter so JotView itself doesn't subscribe.
  const onResizeGutterStart = (e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    const startX = e.clientX;
    const startWidth = getGutterWidth();
    const target = e.currentTarget;
    target.setPointerCapture(e.pointerId);
    const onMove = (ev: PointerEvent) => {
      onSetGutterWidth(startWidth + (ev.clientX - startX));
    };
    const onUp = (ev: PointerEvent) => {
      target.releasePointerCapture(ev.pointerId);
      target.removeEventListener('pointermove', onMove);
      target.removeEventListener('pointerup', onUp);
      target.removeEventListener('pointercancel', onUp);
    };
    target.addEventListener('pointermove', onMove);
    target.addEventListener('pointerup', onUp);
    target.addEventListener('pointercancel', onUp);
  };
  return (
    <BarTimingsContext.Provider value={barTimings}>
      <div
        ref={containerRef}
        className={styles.jotContainer}
        style={containerStyle}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
      >
        <ScoreZoomVar jot={jot} containerRef={containerRef} />
        <GutterWidthVar getGutterWidth={getGutterWidth} containerRef={containerRef} />
        <h2 className={styles.title}>{jot.title || 'Untitled jot'}</h2>
        <p className={styles.subtitle}>{formatSubtitle(jot)}</p>
        <Legend jot={jot} />
        <TimelineHeader jot={jot} onSeek={onSeek} onResizeGutterStart={onResizeGutterStart} />
        <MixerView
          jot={jot}
          config={config}
          trackOrder={trackOrder}
          highlightedPattern={highlightedPattern}
          onPatternClick={onPatternClick}
          onSeek={onSeek}
          onMoveTrack={onMoveTrack}
          voiceControls={voiceControls}
          audioTrackControls={audioTrackControls}
          onResizeGutterStart={onResizeGutterStart}
        />
        <PlayheadAutoScroller containerRef={containerRef} />
        <MarqueeOverlay />
      </div>
    </BarTimingsContext.Provider>
  );
});

/**
 * Side-effect-only observer that writes `--px-per-beat` onto the score
 * container whenever the zoom-derived pixel-per-beat changes. Isolated
 * so reading `jot.pxPerBeat` (a zoom-dependent observable) doesn't
 * re-render JotView — the variable update happens via DOM
 * `setProperty` on the ref instead, then CSS `calc()` propagates the
 * new value to every bar / note / bracket without React touching the
 * subtree.
 */
const ScoreZoomVar = observer(
  ({
    jot,
    containerRef,
  }: {
    jot: RenderedJot;
    containerRef: React.RefObject<HTMLDivElement>;
  }) => {
    const pxPerBeat = jot.pxPerBeat;
    React.useEffect(() => {
      const el = containerRef.current;
      if (!el) return;
      // Set as a unitless number, not a CSS length, so it can be
      // divided in calc() (CSS forbids dividing length / length).
      // Layout calcs multiply by `1px` to recover a length; the
      // waveform's `transform: scaleX(...)` reads it directly against
      // a unitless `--rendered-px-per-beat` for a clean number result.
      el.style.setProperty('--px-per-beat', String(pxPerBeat));
    }, [pxPerBeat, containerRef]);
    return null;
  }
);

/**
 * Side-effect-only observer that writes `--gutter-width` onto the
 * JotView container whenever the store's gutter width changes. Same
 * trick as `ScoreZoomVar`: a resize tick mutates one CSS variable on
 * the root and CSS propagates the new width to every sticky gutter
 * (score header, mixer rows) without React touching the subtree —
 * which matters on a debug bundle with lots of tracks, where a full
 * JotView re-render at 120 Hz pointermove rate is visibly laggy.
 * `useLayoutEffect` (not `useEffect`) so the var is set before paint
 * — avoids a one-frame flash of zero-width gutters on initial mount.
 */
const GutterWidthVar = observer(
  ({
    getGutterWidth,
    containerRef,
  }: {
    getGutterWidth: () => number;
    containerRef: React.RefObject<HTMLDivElement>;
  }) => {
    const px = getGutterWidth();
    React.useLayoutEffect(() => {
      containerRef.current?.style.setProperty('--gutter-width', `${px}px`);
    }, [px, containerRef]);
    return null;
  }
);

/**
 * Isolated observer for the in-flight marquee rectangle so a mousemove
 * (which fires many times per second and mutates `selection.marquee`)
 * only re-renders this 4-style div instead of the whole JotView tree —
 * `JotView`/`MixerView`/per-row waveforms etc. are expensive enough that
 * reading `marquee` in any of their ancestors made the drag visibly laggy.
 */
const MarqueeOverlay = observer(() => {
  const selection = React.useContext(SelectionContext);
  const marquee = selection?.marquee;
  if (!marquee) return null;
  return (
    <div
      className={styles.marquee}
      style={{
        top: marquee.y,
        left: marquee.x,
        width: marquee.width,
        height: marquee.height,
      }}
    />
  );
});

/**
 * Side-effect-only component: keeps the playhead pinned to the
 * horizontal centre of the viewport during playback by tracking
 * `scrollLeft` to it every frame. `scrollLeft` is auto-clamped by the
 * browser, so near the start / end of the score — where there isn't
 * enough content on one side to centre — the playhead simply rides
 * toward that edge instead of snapping. Renders nothing.
 *
 * Wrapped with `observer` so MobX reactivity drives re-renders on every
 * rAF-driven `currentTime` update; the body just reads observables and
 * runs the side effect.
 */
const PlayheadAutoScroller = observer(
  ({ containerRef }: { containerRef: React.RefObject<HTMLDivElement> }) => {
    const t = jotPlayer.currentTime;
    const state = jotPlayer.state;
    const timeline = jotPlayer.timeline;

    React.useEffect(() => {
      if (state !== 'playing' || timeline.bars.length === 0) return;
      const container = containerRef.current;
      if (!container) return;
      // Anchor x via any bars-row inside the container — they all share
      // the same left edge because rows stack vertically. The
      // `data-bars-row` attribute is stamped by mixer rows specifically
      // for this query so the shell doesn't depend on a CSS-module
      // class name from another file.
      const barsRow = container.querySelector<HTMLDivElement>('[data-bars-row]');
      if (!barsRow) return;

      const containerRect = container.getBoundingClientRect();
      const barsRect = barsRow.getBoundingClientRect();
      const playheadViewportX = barsRect.left + timeToX(timeline, t);
      // Pin the playhead to the viewport's horizontal centre. Assigning
      // an out-of-range scrollLeft is clamped by the browser, so the
      // first/last screenful (not enough content to centre) degrades
      // gracefully — the playhead rides toward that edge instead.
      const viewportCenter = containerRect.left + containerRect.width / 2;
      container.scrollLeft += playheadViewportX - viewportCenter;
    }, [t, state, timeline, containerRef]);

    return null;
  }
);

/**
 * Full-app modal overlay shown while a file is loading (jot, midi, paradb
 * map, debug bundle, audio track). Lightly transparent so the user can
 * still see the underlying UI freeze in place, and `pointer-events: auto`
 * blocks all clicks underneath until the load resolves — protects against
 * double-clicks racing a long debug-bundle import. Driven by the store's
 * `withLoading` counter, so nested loads (debug bundle → many audio
 * tracks) read as one continuous spinner.
 */
const LoadingOverlay = observer(({ store }: { store: JotViewStore }) => {
  if (!store.isLoading) return null;
  return (
    <div
      className={styles.loadingOverlay}
      role="status"
      aria-live="polite"
      onMouseDown={(e) => e.stopPropagation()}
    >
      <div className={styles.loadingSpinner} aria-hidden="true" />
      {store.loadingLabel && (
        <div className={styles.loadingLabel}>{store.loadingLabel}</div>
      )}
    </div>
  );
});
