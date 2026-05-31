import { untracked } from 'mobx';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { Point } from 'src/geom';
import { RenderedJot } from 'src/jot';
import { BarTiming, buildTimeline, jotPlayer, timeToX } from 'src/playback';
import { SelectionStore } from 'src/selection';
import styles from './jot_view.module.css';
import {
  AudioWorkletWarningModal,
  detectAudioWorkletState,
} from './jot_view/audio_worklet_warning_modal';
import { LyricsSearchModal } from './jot_view/lyrics_search_modal';
import { LyricsTextLoadModal } from './jot_view/lyrics_text_modal';
import {
  BarTimingsContext,
  FollowPlayheadContext,
  GridLineSettingsContext,
  JotViewStoreContext,
  NoteProvenanceContext,
  RenderedJotContext,
  SelectionContext,
  UniformWaveformsContext,
} from './jot_view/contexts';
import { AudioTrackControls, MixerView, VoiceControls } from './jot_view/mixer';
import { Logo } from './jot_view/components/logo';
import { Minimap } from './jot_view/minimap';
import { VerticalScrollbar } from './jot_view/vertical_scrollbar';
import { PlaybackBar } from './jot_view/playback';
import {
  Legend,
  TimelineHeader,
  extractArtist,
  formatDisplayTitle,
  formatSubtitle,
} from './jot_view/score';
import { GridLineSettings, JotViewStore, TrackKey, snapToDevicePx } from './jot_view/store';
import { RecentTranscriptionsPicker } from './jot_view/recent_transcriptions';
import { ToastContainer } from './jot_view/toast_container';
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

  // Translate a click on `.jotContainer` into the marquee's coordinate
  // space (the inner `.scrollViewport` wrapper, which is where the
  // marquee div lives and where its `top` / `left` are interpreted).
  // The container's `getBoundingClientRect` reflects its visual rect
  // (unaffected by our virtual scroll, since the transform is on the
  // inner wrapper, not the container itself), so adding `store.scrollX`
  // / `store.scrollY` re-derives the wrapper-local position the marquee
  // needs. Reading the observables outside any observer/render path is
  // intentional: this fires only on pointer events, not per-frame, so
  // we don't want to subscribe createJotView to scroll motion.
  const containerPoint = (e: React.MouseEvent<HTMLDivElement>): Point => {
    const el = e.currentTarget;
    const rect = el.getBoundingClientRect();
    return new Point(e.clientX - rect.left + store.scrollX, e.clientY - rect.top + store.scrollY);
  };
  const onMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    selection.beginSelection(containerPoint(e));
  };
  const onMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    selection.moveSelection(containerPoint(e));
  };

  /**
   * Zoom-slider handler. Mirrors the cursor-anchored math the wheel
   * zoom uses (see the `flush` closure in JotView), but anchors at the
   * viewport's horizontal centre so dragging the slider keeps the
   * musical content under the centre of the score pinned. The wheel
   * path still anchors at the cursor; this is only for the slider /
   * any other absolute-zoom caller that lacks a pointer origin.
   */
  const setZoomCentered = (z: number) => {
    const scroller = document.querySelector<HTMLElement>('[data-jot-scroller]');
    const barsRow = scroller?.querySelector<HTMLElement>('[data-bars-row]');
    const j = store.currentJot;
    if (!scroller || !barsRow || !j) {
      store.setZoom(z);
      return;
    }
    const pxPerBeatBefore = j.pxPerBeat;
    const padLeft = j.config.barNotePaddingBeats * pxPerBeatBefore;
    const scrollerRect = scroller.getBoundingClientRect();
    const barsRowRect = barsRow.getBoundingClientRect();
    const anchorBarsRowX = scrollerRect.left + scrollerRect.width / 2 - barsRowRect.left;
    store.setZoom(z);
    if (pxPerBeatBefore <= 0) return;
    const factor = j.pxPerBeat / pxPerBeatBefore;
    if (factor === 1) return;
    store.setScrollBy((anchorBarsRowX - padLeft) * (factor - 1), 0);
  };

  // Computed once per page load; AudioWorklet availability doesn't
  // change after boot, so capturing this outside the component (and
  // outside any observer) keeps it stable across renders without
  // entering React's dep tracking.
  const audioWorkletState = detectAudioWorkletState();

  const View: React.FC = observer(() => {
    const jot = store.currentJot;
    // Modal opens on first mount whenever AudioWorklet is unavailable
    // and closes for the rest of the session once dismissed. A reload
    // re-shows it intentionally; the limitation is real and ongoing,
    // and the user might just have forgotten between sessions.
    const [audioWorkletWarningOpen, setAudioWorkletWarningOpen] = React.useState(
      audioWorkletState !== 'available'
    );

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
          tag === 'INPUT' && TEXT_ENTRY_INPUT_TYPES.has((el as HTMLInputElement).type);
        if (isTextEntryInput || tag === 'TEXTAREA' || tag === 'SELECT' || el?.isContentEditable) {
          return;
        }
        e.preventDefault();
        void store.togglePlayPause();
      };
      window.addEventListener('keydown', onKeyDown);
      return () => window.removeEventListener('keydown', onKeyDown);
    }, []);

    const provenanceContextValue = store.provenanceContextValue;

    // Lyrics modal visibility lives on the store so any TS consumer can
    // observe / drive it; the seeded title/artist fields are still local
    // (re-derived from the current jot on open).
    const lyricsInitialTitle = jot?.title.trim() ?? '';
    const lyricsInitialArtist = jot ? (extractArtist(jot) ?? '') : '';

    const followPlayheadContextValue = React.useMemo(
      () => ({
        follow: store.followPlayhead,
        toggle: () => store.toggleFollowPlayhead(),
      }),
      // eslint-disable-next-line react-hooks/exhaustive-deps -- observable read; observer wrapper rebuilds the memo when followPlayhead flips.
      [store.followPlayhead]
    );

    return (
      <JotViewStoreContext.Provider value={store}>
        <SelectionContext.Provider value={selection}>
          <NoteProvenanceContext.Provider value={provenanceContextValue}>
            <GridLineSettingsContext.Provider value={store.gridLines}>
              <UniformWaveformsContext.Provider value={store.uniformWaveforms}>
                <FollowPlayheadContext.Provider value={followPlayheadContextValue}>
                  <div className={styles.appContainer}>
                    <Toolbar
                      examples={store.examples}
                      currentId={store.currentExampleId}
                      onSelect={(id) => store.loadExample(id)}
                      transcribeStatus={store.transcribeStatus}
                      transcribeOptions={store.transcribeOptions}
                      onTranscribe={(file) => store.transcribeAudio(file)}
                      onResumeTranscribe={(folder, stage) => store.resumeTranscribe(folder, stage)}
                      onLoadJot={(file) => store.loadJotFile(file)}
                      onLoadMidi={(file) => store.loadMidiFile(file)}
                      onLoadParadb={(file) => store.loadParadbMap(file)}
                      onScoreParadb={(file) => store.scoreParadbMap(file)}
                      onLoadDebugBundle={(file) => store.loadDebugBundleFile(file)}
                      onLoadAudioTrack={(file) => store.loadAudioTrack(file)}
                      onLoadLyricsFile={(file) => store.loadLyricsFile(file)}
                      onOpenLyricsTextLoad={() => store.setLyricsTextOpen(true)}
                      onOpenLyricsSearch={() => store.setLyricsSearchOpen(true)}
                      onCancelTranscribe={() => store.cancelTranscribe()}
                      lyricsAlignBusyPhase={store.lyricsAlignBusyPhase}
                      onSetBeatInput={(b) => store.setBeatInput(b)}
                      onSetLlmModel={(m) => store.setLlmModel(m)}
                      zoom={store.zoom}
                      onSetZoom={setZoomCentered}
                      hasNoteProvenance={store.noteProvenance !== undefined}
                      showFilteredOnsets={store.showFilteredOnsets}
                      onSetShowFilteredOnsets={(v) => store.setShowFilteredOnsets(v)}
                      gridLines={store.gridLines}
                      onToggleGridLine={(k) => store.toggleGridLine(k)}
                      uniformWaveforms={store.uniformWaveforms}
                      onSetUniformWaveforms={(v) => store.setUniformWaveforms(v)}
                      autoFollowOnPlay={store.autoFollowOnPlay}
                      onSetAutoFollowOnPlay={(v) => store.setAutoFollowOnPlay(v)}
                      recentTranscriptions={store.recentTranscriptions}
                      recentTranscriptionsLoaded={store.recentTranscriptionsLoaded}
                      recentTranscriptionsLoading={store.recentTranscriptionsLoading}
                      selectedResumeFolder={store.selectedResumeFolder}
                      selectedResumeStage={store.selectedResumeStage}
                      onSetSelectedResumeFolder={(f) => store.setSelectedResumeFolder(f)}
                      onSetSelectedResumeStage={(s) => store.setSelectedResumeStage(s)}
                      onRefreshRecentTranscriptions={() => store.refreshRecentTranscriptions()}
                      onLoadRecentTranscription={(folder) => store.loadRecentTranscription(folder)}
                      transcribeMode={store.transcribeMode}
                      onSetTranscribeMode={(m) => store.setTranscribeMode(m)}
                    />
                    {jot ? (
                      <JotView
                        store={store}
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
                          isPitchAudible: store.isPitchAudible,
                          volumeFor: (pitch) => store.pitchVolume(pitch),
                          onSetVolume: (pitch, v) => store.setPitchVolume(pitch, v),
                          onToggleMute: (pitch) => store.toggleMute(pitch),
                          onToggleSolo: (pitch) => store.toggleSolo(pitch),
                          masterMuted: store.drumMasterMuted,
                          masterSoloed: store.drumMasterSoloed,
                          masterAudible: store.isDrumSectionAudible,
                          onToggleMasterMute: () => store.toggleDrumMasterMute(),
                          onToggleMasterSolo: () => store.toggleDrumMasterSolo(),
                        }}
                        audioTrackControls={{
                          mutedAudioTracks: store.mutedAudioTracks,
                          soloedAudioTracks: store.soloedAudioTracks,
                          isAudioTrackAudible: store.isAudioTrackAudible,
                          volumeFor: (id) => store.audioTrackVolume(id),
                          onSetVolume: (id, v) => store.setAudioTrackVolume(id, v),
                          onToggleMute: (id) => store.toggleAudioTrackMute(id),
                          onToggleSolo: (id) => store.toggleAudioTrackSolo(id),
                          onClear: (id) => store.clearAudioTrack(id),
                          onSplitFromMix: (id) => store.splitAudioTrackFromMix(id),
                          onSplitDrumPieces: (id) => store.splitAudioTrackDrumPieces(id),
                          masterMuted: store.audioMasterMuted,
                          masterSoloed: store.audioMasterSoloed,
                          masterAudible: store.isAudioSectionAudible,
                          onToggleMasterMute: () => store.toggleAudioMasterMute(),
                          onToggleMasterSolo: () => store.toggleAudioMasterSolo(),
                        }}
                        getGutterWidth={() => store.gutterWidth}
                        onSetGutterWidth={(px) => store.setGutterWidth(px)}
                      />
                    ) : (
                      <EmptyState store={store} />
                    )}
                    <Minimap store={store} />
                    {jot && <PlaybackBar store={store} />}
                    <DebugPanel store={store} />
                    <LyricsSearchModal
                      open={store.lyricsSearchOpen}
                      initialTitle={lyricsInitialTitle}
                      initialArtist={lyricsInitialArtist}
                      onClose={() => store.setLyricsSearchOpen(false)}
                      store={store}
                    />
                    <LyricsTextLoadModal
                      open={store.lyricsTextOpen}
                      onClose={() => store.setLyricsTextOpen(false)}
                      store={store}
                    />
                    <AudioWorkletWarningModal
                      state={audioWorkletState}
                      open={audioWorkletWarningOpen}
                      onClose={() => setAudioWorkletWarningOpen(false)}
                    />
                    <LoadingOverlay store={store} />
                    <ToastContainer />
                  </div>
                </FollowPlayheadContext.Provider>
              </UniformWaveformsContext.Provider>
            </GridLineSettingsContext.Provider>
          </NoteProvenanceContext.Provider>
        </SelectionContext.Provider>
      </JotViewStoreContext.Provider>
    );
  });

  return { store, View };
}

type JotViewProps = {
  /**
   * View-state store. Threaded down because the scroll viewport's
   * native `overflow: auto` has been replaced by a CSS `transform` on
   * `.scrollViewport`, and `store.scrollX` / `store.scrollY` are the
   * canonical source of scroll position. JotView's ResizeObservers feed
   * `store.setViewportSize` / `setContentSize`; auto-follow / zoom-
   * anchor / middle-click pan / Stop reset all drive `store.setScrollX`
   * / `setScrollY` / `setScrollBy` / `resetScroll`.
   */
  store: JotViewStore;
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
   * instrument row hosts the pattern/tuplet bracket overlay (the
   * topmost drum row in this list).
   */
  trackOrder: readonly TrackKey[];
  /** Move the row at `from` to position `to` (drag-and-drop / Alt+arrow). */
  onMoveTrack: (from: number, to: number) => void;
  voiceControls: VoiceControls;
  audioTrackControls: AudioTrackControls;
  /** Read the current sticky-gutter width (px). A getter (not a value)
   * so the parent View doesn't reactively re-render on every resize
   * tick, the value flows into the DOM via `GutterWidthVar` as a
   * side-effect-only CSS-var update, and is also read once at drag
   * start to snapshot the starting width. */
  getGutterWidth: () => number;
  /** Apply a new sticky-gutter width (px); called by the gutter resize
   * handle's pointer-move stream. */
  onSetGutterWidth: (px: number) => void;
};

const JotView = observer((props: JotViewProps) => {
  const {
    store,
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
  // Ref to the inner `.scrollViewport` wrapper. Its `offsetWidth` /
  // `offsetHeight` is the scroll-content's natural size (the analogue
  // of `scrollWidth` / `scrollHeight` in the previous native-overflow
  // model); fed into `store.setContentSize` so `setScrollX` /
  // `setScrollY` clamp to `[0, content - viewport]`.
  const viewportRef = React.useRef<HTMLDivElement>(null);
  // Shared DOM-target cache for the per-frame animation vars. Owned by
  // JotView so its lifetime matches the container; ScrollVar and
  // PlayheadPosVar read through it instead of querying the DOM on every
  // tick.
  const cacheRef = React.useRef<DomTargetCache | null>(null);
  React.useLayoutEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const cache = new DomTargetCache();
    cache.attach(container, styles.scrollStickyHorizontal, store);
    cacheRef.current = cache;
    return () => {
      cache.detach();
      cacheRef.current = null;
    };
  }, [store]);
  React.useEffect(() => {
    const container = containerRef.current;
    const viewport = viewportRef.current;
    if (!container || !viewport) return;
    const updateContainer = () => {
      store.setViewportSize(container.clientWidth, container.clientHeight);
    };
    const updateViewport = () => {
      store.setContentSize(viewport.offsetWidth, viewport.offsetHeight);
    };
    updateContainer();
    updateViewport();
    const containerRo = new ResizeObserver(updateContainer);
    const viewportRo = new ResizeObserver(updateViewport);
    containerRo.observe(container);
    viewportRo.observe(viewport);
    return () => {
      containerRo.disconnect();
      viewportRo.disconnect();
    };
  }, [store]);

  // Wheel zooms the score (mirrors the Zoom slider), no modifier
  // required, and Ctrl/Cmd + wheel still works (also covers the macOS
  // trackpad pinch gesture, which Chrome/Safari deliver as a synthetic
  // Ctrl + wheel). The listener is registered natively with
  // `{ passive: false }` because React's synthetic `onWheel` is passive
  //; `preventDefault` there is a no-op, and we must cancel both the
  // native page scroll and the browser's own page zoom on Ctrl/Cmd +
  // wheel. Wheel events are coalesced per animation frame: a 120 Hz
  // trackpad fires ~8 events per frame, but only the final composite
  // zoom is visible, so summing deltas and applying once skips ~7×
  // wasted layout/render passes per frame.
  //
  // Zoom anchors at the playhead so scaling in/out keeps the currently-
  // playing position pinned to its screen X (musician's eyes don't have
  // to chase the playhead). When no playhead is rendered (idle, no cue)
  // we fall back to the viewport's horizontal centre.
  const onZoomByRef = React.useRef(onZoomBy);
  onZoomByRef.current = onZoomBy;
  const jotRef = React.useRef(jot);
  jotRef.current = jot;
  React.useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    let pendingDelta = 0;
    let pendingScrollX = 0;
    // Latest pointer x (viewport coords) captured from wheel events;
    // used as the zoom anchor. Coalesced wheel events take the most
    // recent position as a natural follow-the-cursor behaviour. When no
    // wheel event has fired yet (cleared after flush), we fall back to
    // the viewport centre so the first paint doesn't snap.
    let pendingClientX: number | undefined;
    let rafId = 0;
    const flush = () => {
      rafId = 0;
      const delta = pendingDelta;
      const scrollDx = pendingScrollX;
      const clientX = pendingClientX;
      pendingDelta = 0;
      pendingScrollX = 0;
      pendingClientX = undefined;
      if (scrollDx !== 0) store.setScrollBy(scrollDx, 0);
      if (delta === 0) return;
      const factor = Math.exp(-delta * 0.0015);

      // Bars-row coords of the anchor point (cursor position when the
      // wheel event fired; viewport centre as fallback). `bar.x` /
      // `bar.width` are linear in `pxPerBeat`, so the new x of the same
      // musical point is `padLeft + (anchorX - padLeft) * actualFactor`;
      // adjusting scrollLeft by the delta keeps the on-screen position
      // pinned without waiting for layout.
      const currentJot = jotRef.current;
      const pxPerBeatBefore = currentJot.pxPerBeat;
      // `padLeft` scales with `pxPerBeat` (see ViewConfig.barNotePaddingBeats);
      // anchor math reads the live pre-zoom value so the on-screen
      // musical point stays pinned across the zoom step.
      const padLeft = currentJot.config.barNotePaddingBeats * pxPerBeatBefore;
      const barsRow = el.querySelector<HTMLElement>('[data-bars-row]');

      let anchorBarsRowX: number | undefined;
      if (barsRow) {
        const barsRowRect = barsRow.getBoundingClientRect();
        if (clientX !== undefined) {
          // Cursor-anchored zoom; clamp at the bars-row's left edge so a
          // wheel event with the cursor over the sticky gutter still
          // anchors at the leftmost visible musical content rather than
          // off-content negative x.
          anchorBarsRowX = Math.max(0, clientX - barsRowRect.left);
        } else {
          const containerRect = el.getBoundingClientRect();
          anchorBarsRowX = containerRect.left + containerRect.width / 2 - barsRowRect.left;
        }
      }

      onZoomByRef.current(factor);

      if (anchorBarsRowX !== undefined && pxPerBeatBefore > 0) {
        const actualFactor = currentJot.pxPerBeat / pxPerBeatBefore;
        if (actualFactor !== 1) {
          const delta = (anchorBarsRowX - padLeft) * (actualFactor - 1);
          // Virtual scroll: write through the store instead of native
          // `el.scrollLeft +=`. The store's clamp uses the pre-zoom
          // content width (the ResizeObserver on `.scrollViewport` is
          // async); for zoom-in this is conservative-safe (smaller
          // max), for zoom-out the next observer tick re-clamps within
          // a frame, so a one-frame overshoot is the worst case.
          store.setScrollBy(delta, 0);
        }
      }
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      // deltaMode 1 = lines (typically a notched mouse wheel); scale
      // it up so a single notch zooms a comparable amount to a
      // pixel-mode trackpad swipe. Scrolling up (deltaY < 0) zooms in.
      const unit = e.deltaMode === 1 ? 16 : 1;
      // On touchpads (deltaMode 0) a horizontal two-finger swipe often
      // carries 1-3 px of incidental deltaY per event, which would
      // otherwise accumulate into a visible zoom drift. Dead-zone tiny
      // deltaY on pixel-mode events only; mouse wheels (deltaMode 1)
      // deliver large discrete steps and bypass this.
      const effectiveDeltaY = e.deltaMode === 0 && Math.abs(e.deltaY) < 4 ? 0 : e.deltaY;
      // Horizontal-dominant events (two-finger horizontal swipe on a
      // touchpad, shift + wheel on a mouse) pan the timeline instead of
      // zooming. Pinch (delivered as ctrlKey + deltaY by Chrome/Safari)
      // always falls through to zoom regardless of deltaX, so a
      // diagonal pinch still scales.
      if (!e.ctrlKey && !e.metaKey && Math.abs(e.deltaX) > Math.abs(effectiveDeltaY)) {
        pendingScrollX += e.deltaX * unit;
      } else {
        pendingDelta += effectiveDeltaY * unit;
        pendingClientX = e.clientX;
      }
      if (rafId === 0) rafId = requestAnimationFrame(flush);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => {
      el.removeEventListener('wheel', onWheel);
      if (rafId !== 0) cancelAnimationFrame(rafId);
    };
    // `store` is a stable singleton from `createJotView`; included for
    // exhaustive-deps correctness even though it never changes in practice.
  }, [store]);

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
      // Middle-mouse pan is an explicit "I want to look somewhere else"
      // gesture; auto-follow would just fight it on the next frame.
      store.setFollowPlayhead(false);
    };
    const onMouseMove = (e: MouseEvent) => {
      if (!panning) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      // Virtual scroll: drag-pan writes the inverse cursor delta into
      // both axes of the store. Replaces the previous native
      // `el.scrollLeft -= dx` / `el.scrollTop -= dy` since the
      // container is now `overflow: hidden`.
      store.setScrollBy(-dx, -dy);
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
  }, [store]);

  // Touch gestures on the score: one finger pans (analogue of the
  // middle-mouse drag above), two fingers pinch the score-zoom slider
  // (analogue of Ctrl + wheel). The native `touchstart` listener uses
  // `{ passive: false }` because we call `preventDefault` to:
  //   - suppress the browser's own pan/zoom (also blocked at the CSS
  //     layer via `touch-action: none` on `.jotContainer`, but a JS
  //     preventDefault is the belt to the CSS suspenders), and
  //   - suppress the synthesized `mousedown` that would otherwise fire
  //     the parent's marquee-selection handler on every touch.
  // Touches that land on interactive controls (buttons, range inputs,
  // selects, labels wrapping inputs) bail before preventDefault so the
  // native input range thumb-drag, button taps, and slider scrubs still
  // work. Pinch anchors at the midpoint of the two fingers in bars-row
  // content space and re-derives scrollX from the initial-frame snapshot
  // each move so the musical point under the midpoint stays pinned (no
  // integration drift, same math as the wheel-zoom anchor).
  React.useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    type Gesture =
      | { kind: 'none' }
      | { kind: 'pan'; lastX: number; lastY: number }
      | {
          kind: 'pinch';
          initDistance: number;
          initZoom: number;
          initScrollX: number;
          initPxPerBeat: number;
          initPadLeft: number;
          anchorBarsRowX: number;
        };
    let gesture: Gesture = { kind: 'none' };

    // Tap-to-seek bookkeeping. Touchstart's `preventDefault` (needed to
    // suppress browser pan/zoom and the synthesized mousedown that
    // would fire marquee selection) also kills the browser's
    // synthesized `click` on touchend, so the bars-row onClick
    // handlers (seekFromClick) never fire on tap. We detect taps
    // ourselves: a single-finger touchstart records a candidate, any
    // significant movement during touchmove invalidates it (and
    // promotes to a real pan), and touchend within the tap budget
    // dispatches a synthetic click on the original target so the
    // existing onClick path runs unchanged.
    const TAP_MOVE_PX = 8;
    const TAP_MAX_DURATION_MS = 250;
    let tapCandidate: { startX: number; startY: number; t: number; target: Element } | null = null;

    const isInteractiveControl = (target: EventTarget | null): boolean => {
      if (!(target instanceof Element)) return false;
      return !!target.closest('input, button, select, textarea, a, label, [role="slider"]');
    };

    const touchDistance = (a: Touch, b: Touch): number =>
      Math.hypot(b.clientX - a.clientX, b.clientY - a.clientY);

    const disengageFollow = () => {
      // Touch-pan / pinch are explicit "I want to look somewhere else"
      // gestures; auto-follow would just fight them on the next frame.
      // A tap-only touch should NOT disengage follow, so this is only
      // called once the gesture has been confirmed as a pan/pinch
      // (movement threshold crossed, or a second finger landed).
      store.setFollowPlayhead(false);
    };

    // Start tracking pan state without yet committing to "this is a
    // pan"; follow-toggle is deferred until movement crosses
    // TAP_MOVE_PX. Used by the single-finger touchstart path so a
    // static tap doesn't disengage follow.
    const startPanState = (t: Touch) => {
      gesture = { kind: 'pan', lastX: t.clientX, lastY: t.clientY };
    };

    const beginPan = (t: Touch) => {
      startPanState(t);
      disengageFollow();
    };

    const beginPinch = (a: Touch, b: Touch) => {
      const j = jotRef.current;
      const initPxPerBeat = j.pxPerBeat;
      const initPadLeft = j.config.barNotePaddingBeats * initPxPerBeat;
      const midClientX = (a.clientX + b.clientX) / 2;
      const barsRow = el.querySelector<HTMLElement>('[data-bars-row]');
      let anchorBarsRowX: number;
      if (barsRow) {
        const rect = barsRow.getBoundingClientRect();
        anchorBarsRowX = Math.max(0, midClientX - rect.left);
      } else {
        const rect = el.getBoundingClientRect();
        anchorBarsRowX = midClientX - rect.left;
      }
      gesture = {
        kind: 'pinch',
        initDistance: touchDistance(a, b),
        initZoom: store.zoom,
        initScrollX: store.scrollX,
        initPxPerBeat,
        initPadLeft,
        anchorBarsRowX,
      };
      store.setFollowPlayhead(false);
    };

    const onTouchStart = (e: TouchEvent) => {
      if (isInteractiveControl(e.target)) {
        gesture = { kind: 'none' };
        tapCandidate = null;
        return;
      }
      if (e.touches.length === 1) {
        const t = e.touches[0];
        // Set up pan state but DON'T disengage follow yet, wait until
        // movement crosses TAP_MOVE_PX in onTouchMove. Until then the
        // touch could still resolve as a tap.
        startPanState(t);
        tapCandidate =
          e.target instanceof Element
            ? { startX: t.clientX, startY: t.clientY, t: performance.now(), target: e.target }
            : null;
        e.preventDefault();
      } else if (e.touches.length >= 2) {
        // A second finger landing is unambiguous: this is a pinch, not
        // a tap. Drop any in-flight tap candidate.
        tapCandidate = null;
        beginPinch(e.touches[0], e.touches[1]);
        e.preventDefault();
      }
    };

    const onTouchMove = (e: TouchEvent) => {
      if (gesture.kind === 'pan' && e.touches.length === 1) {
        const t = e.touches[0];
        if (tapCandidate) {
          const adx = t.clientX - tapCandidate.startX;
          const ady = t.clientY - tapCandidate.startY;
          if (Math.hypot(adx, ady) > TAP_MOVE_PX) {
            // Promote tap candidate to a real pan: disengage follow
            // (mirrors the mouse-pan onMouseDown side effect) and
            // invalidate the tap.
            tapCandidate = null;
            disengageFollow();
          } else {
            // Still inside the tap budget; don't scroll yet, but
            // suppress the browser's native pan so a long-but-still
            // gesture doesn't end up scrolling the page.
            e.preventDefault();
            return;
          }
        }
        const dx = t.clientX - gesture.lastX;
        const dy = t.clientY - gesture.lastY;
        gesture.lastX = t.clientX;
        gesture.lastY = t.clientY;
        store.setScrollBy(-dx, -dy);
        e.preventDefault();
      } else if (gesture.kind === 'pinch' && e.touches.length >= 2) {
        const a = e.touches[0];
        const b = e.touches[1];
        const dist = touchDistance(a, b);
        if (gesture.initDistance > 0 && gesture.initPxPerBeat > 0) {
          store.setZoom(gesture.initZoom * (dist / gesture.initDistance));
          const actualFactor = jotRef.current.pxPerBeat / gesture.initPxPerBeat;
          if (actualFactor !== 1) {
            store.setScrollX(
              gesture.initScrollX +
                (gesture.anchorBarsRowX - gesture.initPadLeft) * (actualFactor - 1)
            );
          }
        }
        e.preventDefault();
      }
    };

    const onTouchEnd = (e: TouchEvent) => {
      // Re-initialise based on the remaining touches so a 2 → 1 finger
      // transition cleanly switches from pinch to pan (and 1 → 0 ends
      // the gesture). Skips re-init if the remaining touch is on an
      // interactive control (defensive; shouldn't normally happen since
      // start would have bailed).
      if (e.touches.length === 0) {
        // Tap-to-seek: if the touch stayed within the move budget and
        // finished within the duration budget, synthesize a click on
        // the original target. It bubbles through React's delegated
        // root listener and reaches the bars-row onClick (seekFromClick)
        // via the normal handler path; data-noseek targets (notes,
        // pattern labels, playhead) keep their own behaviour because
        // seekFromClick already bails on them.
        if (tapCandidate && performance.now() - tapCandidate.t <= TAP_MAX_DURATION_MS) {
          const touch = e.changedTouches[0];
          if (touch) {
            tapCandidate.target.dispatchEvent(
              new MouseEvent('click', {
                bubbles: true,
                cancelable: true,
                clientX: touch.clientX,
                clientY: touch.clientY,
                button: 0,
              })
            );
          }
        }
        tapCandidate = null;
        gesture = { kind: 'none' };
        return;
      }
      // From here on the user still has at least one finger down, so
      // we're committing to a continued pan/pinch, drop any tap
      // candidate.
      tapCandidate = null;
      if (e.touches.length === 1) {
        if (isInteractiveControl(e.touches[0].target)) {
          gesture = { kind: 'none' };
          return;
        }
        beginPan(e.touches[0]);
      } else {
        beginPinch(e.touches[0], e.touches[1]);
      }
    };

    const onTouchCancel = () => {
      tapCandidate = null;
      gesture = { kind: 'none' };
    };

    el.addEventListener('touchstart', onTouchStart, { passive: false });
    el.addEventListener('touchmove', onTouchMove, { passive: false });
    el.addEventListener('touchend', onTouchEnd);
    el.addEventListener('touchcancel', onTouchCancel);
    return () => {
      el.removeEventListener('touchstart', onTouchStart);
      el.removeEventListener('touchmove', onTouchMove);
      el.removeEventListener('touchend', onTouchEnd);
      el.removeEventListener('touchcancel', onTouchCancel);
    };
  }, [store]);

  // `--note-pad-px` is the engraving inset every note's CSS `left`
  // calc() reads. Its value is derived from `--note-pad-beats` (the
  // zoom-invariant fraction-of-a-beat config) and `--px-per-beat`
  // (zoom-driven), so the inset scales with the bar width; without
  // this, zooming out leaves the inset fixed at 14px while a beat
  // shrinks, and notes at the end of a bar overshoot into the next
  // bar's space. `--note-pad-beats` is set inline once; `--px-per-beat`
  // is updated at runtime by `ScoreZoomVar`. Because `--note-pad-px`
  // is itself a calc reading `--px-per-beat`, every consumer reads the
  // live scaled value without us touching the var on each zoom tick.
  // `--gutter-width` is mutated at runtime too; both live on the same
  // root so every descendant calc() chain reads from a single ancestor.
  // `ScoreZoomVar` / `GutterWidthVar` are side-effect-only observers
  // that write via `setProperty` on `containerRef.current` so the tick
  // doesn't re-render JotView; only mutates one DOM attribute. With
  // many tracks loaded a JotView re-render is expensive enough to make
  // a 120 Hz resize drag visibly laggy, so we keep both reads off the
  // render path.
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
    []
  );
  const containerStyle = {
    ['--note-pad-beats' as string]: String(config.barNotePaddingBeats),
    ['--note-pad-px' as string]: 'calc(var(--note-pad-beats) * var(--px-per-beat) * 1px)',
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
    <RenderedJotContext.Provider value={jot}>
      <BarTimingsContext.Provider value={barTimings}>
        <div
          ref={containerRef}
          className={styles.jotContainer}
          // Stable hook for descendant popovers (note + filtered-onset
          // labels) to find the scroll viewport's bottom edge and flip
          // upward when the natural below-placement would overflow into
          // the playback bar / debug panel below.
          data-jot-scroller
          style={containerStyle}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
        >
          <ScoreZoomVar jot={jot} containerRef={containerRef} />
          <GutterWidthVar getGutterWidth={getGutterWidth} containerRef={containerRef} />
          <GridLineVars containerRef={containerRef} />
          <ScrollVar cacheRef={cacheRef} store={store} />
          <PlayheadPosVar cacheRef={cacheRef} getGutterWidth={getGutterWidth} store={store} />
          <div
            ref={viewportRef}
            className={styles.scrollViewport}
            // Stable hook used by JotView's content ResizeObserver and by
            // the minimap (offsetWidth is the scroll-content's `scrollWidth`
            // analogue in this no-native-scroll model).
            data-jot-scroll-content
          >
            <h2 className={styles.title}>{formatDisplayTitle(jot) || 'Untitled jot'}</h2>
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
            <MarqueeOverlay />
          </div>
          <VerticalScrollbar store={store} />
        </div>
      </BarTimingsContext.Provider>
    </RenderedJotContext.Provider>
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
  ({ jot, containerRef }: { jot: RenderedJot; containerRef: React.RefObject<HTMLDivElement> }) => {
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
 * Beat-grid visibility CSS vars. Each grid family (main beat / 16ths /
 * triplets / 48ths) has a permanently-mounted overlay div per bar (see
 * `.gridLayer*` in score.module.css) whose `display` is read from a
 * matching `--grid-display-*` custom property; `block` when the
 * toggle is on, falling back to `none` (the var-fallback) when off.
 * Toggling a grid is therefore a single `setProperty` (or
 * `removeProperty`) on the score root, with zero per-bar React work
 * and zero per-bar DOM mutations; the cascade flips visibility for
 * every overlay at once.
 *
 * Replaces the previous per-divider `<div>` chrome (up to ~44 dotted
 * divs per bar when 48ths were on, which on a long score scaled to
 * thousands of nodes per row and turned per-frame playhead repaints
 * into a paint storm).
 */
const GRID_LINE_DISPLAY_VARS: Record<keyof GridLineSettings, string> = {
  mainBeat: '--grid-display-main',
  subBeat16: '--grid-display-16',
  subBeatQuarterTriplet: '--grid-display-quarter-triplet',
  subBeatTriplet: '--grid-display-triplet',
  subBeat48: '--grid-display-48',
};

const GridLineVars = observer(
  ({ containerRef }: { containerRef: React.RefObject<HTMLDivElement> }) => {
    const gridLines = React.useContext(GridLineSettingsContext);
    React.useEffect(() => {
      const el = containerRef.current;
      if (!el) return;
      for (const key of Object.keys(GRID_LINE_DISPLAY_VARS) as (keyof GridLineSettings)[]) {
        const cssVar = GRID_LINE_DISPLAY_VARS[key];
        if (gridLines[key]) el.style.setProperty(cssVar, 'block');
        else el.style.removeProperty(cssVar);
      }
    }, [gridLines, containerRef]);
    return null;
  }
);

/**
 * Cached DOM targets for the per-frame animation vars. Each playback tick
 * writes `--scroll-x` / `--scroll-y` to the inner `.scrollViewport` plus
 * every `.scrollStickyHorizontal` element, and `--playhead-x` to every
 * `[data-playhead="1"]` element. With many tracks loaded, querySelectorAll
 * on every tick accounted for ~2 ms of frame budget by itself; we instead
 * scan once on mount, then maintain the cache via a single
 * `MutationObserver` so per-frame writes iterate plain arrays.
 *
 * New sticky elements get their `--scroll-x` seeded at mutation time so a
 * gutter mounted after the user scrolled doesn't snap to 0 for one frame.
 * `--playhead-x` is `<length>`-typed with `initial-value: 0px`, so a freshly
 * mounted playhead's first frame at 0 is harmless; the next tick writes the
 * live value.
 */
class DomTargetCache {
  viewport: HTMLElement | null = null;
  sticky: Set<HTMLElement> = new Set();
  playheads: Set<HTMLElement> = new Set();
  private observer?: MutationObserver;

  attach(root: HTMLElement, stickyClass: string, store: JotViewStore): void {
    this.viewport = root.querySelector<HTMLElement>('[data-jot-scroll-content]');
    for (const el of root.querySelectorAll<HTMLElement>('.' + stickyClass)) {
      this.sticky.add(el);
    }
    for (const el of root.querySelectorAll<HTMLElement>('[data-playhead="1"]')) {
      this.playheads.add(el);
    }
    this.observer = new MutationObserver((records) => {
      const seedX = String(store.scrollX);
      for (const r of records) {
        for (const node of r.addedNodes) {
          if (!(node instanceof HTMLElement)) continue;
          if (node.classList.contains(stickyClass)) {
            this.sticky.add(node);
            node.style.setProperty('--scroll-x', seedX);
          }
          if (node.dataset.playhead === '1') this.playheads.add(node);
          for (const sub of node.querySelectorAll<HTMLElement>('.' + stickyClass)) {
            this.sticky.add(sub);
            sub.style.setProperty('--scroll-x', seedX);
          }
          for (const sub of node.querySelectorAll<HTMLElement>('[data-playhead="1"]')) {
            this.playheads.add(sub);
          }
        }
        for (const node of r.removedNodes) {
          if (!(node instanceof HTMLElement)) continue;
          this.sticky.delete(node);
          this.playheads.delete(node);
          for (const sub of node.querySelectorAll<HTMLElement>('.' + stickyClass)) {
            this.sticky.delete(sub);
          }
          for (const sub of node.querySelectorAll<HTMLElement>('[data-playhead="1"]')) {
            this.playheads.delete(sub);
          }
        }
      }
    });
    this.observer.observe(root, { childList: true, subtree: true });
  }

  detach(): void {
    this.observer?.disconnect();
    this.observer = undefined;
    this.viewport = null;
    this.sticky.clear();
    this.playheads.clear();
  }
}

/**
 * Side-effect-only observer that writes `--scroll-x` / `--scroll-y` on
 * each consumer (the inner `.scrollViewport` plus every
 * `.scrollStickyHorizontal` element) whenever the store's virtual
 * scroll offsets change. Mirrors the `ScoreZoomVar` / `GutterWidthVar`
 * pattern: read the observable, write the var, no React re-render in
 * the subtree.
 *
 * Vars are registered `inherits: false` (see `design_tokens.css`) so
 * the per-tick `setProperty` only invalidates style on the elements
 * that actually consume them. Letting `--scroll-x` cascade across the
 * entire score subtree costs ~21 ms per playback tick on a long song,
 * blowing the 8.3 ms frame budget.
 *
 * The wrapper `.scrollViewport` reads these vars via
 * `transform: translate3d(calc(var(--scroll-x) * -1px), ...)`, and the
 * `.scrollStickyHorizontal` class reads the same `--scroll-x` to
 * counter-transform formerly `position: sticky; left: 0` elements
 * (gutters, title/subtitle/legend); both consumers stay subpixel-locked
 * because the writes happen in the same effect tick.
 */
const ScrollVar = observer(
  ({
    cacheRef,
    store,
  }: {
    cacheRef: React.RefObject<DomTargetCache | null>;
    store: JotViewStore;
  }) => {
    const x = store.scrollX;
    const y = store.scrollY;
    React.useLayoutEffect(() => {
      const cache = cacheRef.current;
      if (!cache) return;
      setScrollX(cache, x);
      setScrollY(cache, y);
    }, [x, y, cacheRef]);
    return null;
  }
);

/**
 * Per-frame animation var writers. All four target a {@link DomTargetCache}
 * rather than walking the DOM each call. `--scroll-x` / `--scroll-y` /
 * `--playhead-x` are all registered `inherits: false` (see
 * `design_tokens.css`); per-element writes keep the per-tick style
 * invalidation scoped to the few elements that actually consume each var.
 */
function setPlayheadVar(cache: DomTargetCache, x: number): void {
  const px = `${x}px`;
  for (const ph of cache.playheads) ph.style.setProperty('--playhead-x', px);
}

function clearPlayheadVar(cache: DomTargetCache): void {
  for (const ph of cache.playheads) ph.style.removeProperty('--playhead-x');
}

function setScrollX(cache: DomTargetCache, x: number): void {
  const xStr = String(x);
  if (cache.viewport) cache.viewport.style.setProperty('--scroll-x', xStr);
  for (const el of cache.sticky) el.style.setProperty('--scroll-x', xStr);
}

function setScrollY(cache: DomTargetCache, y: number): void {
  if (cache.viewport) cache.viewport.style.setProperty('--scroll-y', String(y));
}

/**
 * Side-effect-only observer that writes `--playhead-x` (in px) onto
 * every `[data-playhead="1"]` element on every player tick AND, when
 * playback follow is engaged, pins the score's virtual scrollX to keep
 * the playhead at the viewport's horizontal centre. Both writes happen
 * in the same `useLayoutEffect` (pre-paint) so the playhead position
 * and the score's scroll update in the same frame. See `setPlayheadVar`
 * for why the var is written per-element instead of on the container.
 *
 * Per-frame cost: one `timeToX` walk, one CSS-var write, and (during
 * playback follow) one `store.setScrollX` call. The target scrollX is
 * computed algebraically from the gutter width and the container's
 * `clientWidth` rather than via `getBoundingClientRect()` /
 * `querySelector()` every frame; both of those force a style-layout
 * flush. The math: the bars-row's content starts at content-x = gutter
 * width (sticky-left gutter occupies the leading `gutter-width` px of
 * the scroll content), so the playhead's content-x is `gw +
 * timeToX(t)`. Centring it in the viewport gives `scrollX = gw +
 * timeToX(t) - clientWidth/2`. `setScrollX` clamps to `[0, content -
 * viewport]`, so the first / last screenful degrades gracefully to the
 * playhead riding toward that edge instead of snapping.
 *
 * Subpixel: unlike the native `scrollLeft` setter (browser snaps to
 * integer px), `setScrollX` writes a fractional value that the inner
 * `.scrollViewport` then applies via CSS `transform`; transforms on
 * composited layers render at subpixel precision. The previous code
 * had to round `scrollLeft` and back-derive `--playhead-x` to keep the
 * playhead from wobbling ±0.5px against the integer-snapped scroll;
 * with virtual scroll, `rawX` is written directly into `--playhead-x`
 * and the two stay locked because they share the same fractional
 * coordinate system.
 */
const PlayheadPosVar = observer(
  ({
    cacheRef,
    getGutterWidth,
    store,
  }: {
    cacheRef: React.RefObject<DomTargetCache | null>;
    getGutterWidth: () => number;
    store: JotViewStore;
  }) => {
    const t = jotPlayer.currentTime;
    const state = jotPlayer.state;
    const cued = jotPlayer.cued;
    const timeline = jotPlayer.timeline;
    const follow = React.useContext(FollowPlayheadContext).follow;
    // Read `pxPerBeat` so this observer re-runs whenever zoom changes;
    // `timeToX` resolves bar widths from `pxPerBeat * structure.bars[i].beats`,
    // but the effect deps wouldn't otherwise know to re-fire when the user
    // zooms while paused or cued, leaving `--playhead-x` stale until
    // the next `currentTime` tick. During playback `t` updates every
    // frame so this read is a no-op; while paused/idle/cued it's the
    // signal that keeps the playhead pinned to the right bar.
    const pxPerBeat = timeline.rendered?.pxPerBeat ?? 0;
    const prevStateRef = React.useRef(state);
    React.useLayoutEffect(() => {
      const cache = cacheRef.current;
      if (!cache) return;
      const wasActive = prevStateRef.current === 'playing' || prevStateRef.current === 'paused';
      prevStateRef.current = state;

      // Stop (active → idle): snap the score back to its start so the
      // reset playhead position is visible. Bypasses the follow flag;
      // Stop is a reset, not a follow. Pause stays at 'paused' and
      // keeps its scroll; initial mount is idle→idle so this is a
      // no-op until the first play has happened.
      if (wasActive && state === 'idle') {
        store.resetScrollX();
        clearPlayheadVar(cache);
        return;
      }

      const active = state === 'playing' || state === 'paused' || cued;
      if (!active || timeline.bars.length === 0) {
        clearPlayheadVar(cache);
        return;
      }

      // Snap the playhead's pixel position to the device-pixel grid
      // BEFORE deriving the auto-follow scrollX target so both CSS
      // vars share the same snapped value. `setScrollX` also snaps
      // inside the store, so if we passed the unsnapped `x` here AND
      // again to `setPlayheadVar` below, the two would round
      // independently; the difference (always < 1/dpr CSS px) would
      // drift between frames as the targets cross snap boundaries,
      // leaving the centred playhead visibly wobbling sub-pixel
      // against the scrolling bars. Using one snapped `x` for both
      // keeps the relationship exact.
      const x = snapToDevicePx(timeToX(timeline, t));

      // Auto-scroll only while playing AND when follow is engaged. A
      // paused or cued playhead still updates `--playhead-x` below (so
      // the parked playhead reflects its bar) but doesn't pull the
      // score under the user; manual scrolls during playback with
      // follow off are no longer fought either.
      //
      // After `store.setScrollX(...)` the canonical `--scroll-x` write
      // still goes through `ScrollVar` (an observer of the store), but
      // we also write the CSS var directly here so the wrapper's
      // transform is updated in the SAME useLayoutEffect as
      // `--playhead-x`. Without that, ScrollVar's update could land in
      // a later commit (mobx-react-lite schedules via React's setState;
      // useLayoutEffects flush before paint but a re-render triggered
      // from inside one isn't guaranteed to). Setting both vars here
      // eliminates a possible one-frame lag of the score behind the
      // playhead during auto-follow.
      if (follow && state === 'playing') {
        const clientWidth = store._viewportWidth;
        if (clientWidth > 0) {
          store.setScrollX(getGutterWidth() + x - clientWidth / 2);
          setScrollX(cache, store.scrollX);
        }
      }
      setPlayheadVar(cache, x);
    }, [t, state, cued, timeline, pxPerBeat, follow, cacheRef, getGutterWidth, store]);
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
 * First-load welcome screen rendered when no jot is loaded. Surfaces the
 * primary entry points (.jot file, ParaDB map, recent transcriptions)
 * directly and lists the built-in example jots as one-click shortcuts;
 * other formats (MIDI, debug bundle, audio tracks, transcribe) stay in
 * the toolbar's File / Transcribe menus to avoid duplicating that whole
 * surface here.
 */
const EmptyState = observer(({ store }: { store: JotViewStore }) => {
  const jotInputRef = React.useRef<HTMLInputElement>(null);
  const paradbInputRef = React.useRef<HTMLInputElement>(null);
  const handleJotFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) store.loadJotFile(file);
    e.target.value = '';
  };
  const handleParadbFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) store.loadParadbMap(file);
    e.target.value = '';
  };
  return (
    <div className={styles.emptyState}>
      <div className={styles.emptyStateCard}>
        <div className={styles.emptyStateIcon} aria-hidden="true">
          <Logo size={56} />
        </div>
        <h2 className={styles.emptyStateTitle}>Open a file to get started</h2>
        <p className={styles.emptyStateBody}>
          Load a Drumjot <code>.jot</code>, a ParaDB map, or a recent transcription, or try one of
          the examples below.
        </p>
        <div className={styles.emptyStateActions}>
          <button
            type="button"
            className={styles.emptyStatePrimary}
            onClick={() => jotInputRef.current?.click()}
          >
            Open .jot file
          </button>
          <div className={styles.emptyStateAltActions}>
            <button
              type="button"
              className={styles.emptyStateSecondary}
              onClick={() => paradbInputRef.current?.click()}
              title="Load a ParaDB / Paradiddle map pack (.zip). The chart is converted to a score and its audio tracks are loaded automatically for play-along practice."
            >
              Open ParaDB map
            </button>
            <RecentTranscriptionsPicker
              variant="cta"
              triggerLabel="Open recent"
              triggerTitle="Open a previously transcribed audio file from the server's recent runs."
              items={store.recentTranscriptions}
              loaded={store.recentTranscriptionsLoaded}
              loading={store.recentTranscriptionsLoading}
              onRefresh={() => store.refreshRecentTranscriptions()}
              onPick={(folder) => store.loadRecentTranscription(folder)}
            />
          </div>
        </div>
        <p className={styles.emptyStateHint}>
          For other formats, use the <b>File</b> or <b>Transcribe</b> menus in the toolbar above.
        </p>
        {store.examples.length > 0 && (
          <div className={styles.emptyStateExamples}>
            <span className={styles.emptyStateExamplesLabel}>Or try an example</span>
            <div className={styles.emptyStateExampleRow}>
              {store.examples.map((ex) => (
                <button
                  key={ex.id}
                  type="button"
                  className={styles.emptyStateExampleButton}
                  onClick={() => store.loadExample(ex.id)}
                >
                  {ex.label}
                </button>
              ))}
            </div>
          </div>
        )}
        <input
          ref={jotInputRef}
          type="file"
          accept=".jot,.txt,text/plain"
          className={styles.emptyStateFileInput}
          onChange={handleJotFileChange}
        />
        <input
          ref={paradbInputRef}
          type="file"
          accept=".zip,application/zip"
          className={styles.emptyStateFileInput}
          onChange={handleParadbFileChange}
        />
      </div>
    </div>
  );
});

/**
 * Full-app modal overlay shown while a file is loading (jot, midi, paradb
 * map, debug bundle, audio track). Lightly transparent so the user can
 * still see the underlying UI freeze in place, and `pointer-events: auto`
 * blocks all clicks underneath until the load resolves; protects against
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
      {store.loadingLabel && <div className={styles.loadingLabel}>{store.loadingLabel}</div>}
    </div>
  );
});
