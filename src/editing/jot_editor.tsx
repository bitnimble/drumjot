import { untracked } from 'mobx';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { Box, Point } from 'src/utils/geom';
import { Jot } from 'src/schema/dsl/dsl';
import type { StructuralPresenter } from 'src/editing/structure/structural_presenter';
import type { StructNote } from 'src/editing/structure/structure_store';
import { boundingBoxOfNotes, notesById, notesInBox } from 'src/editing/score/note_geometry';
import type { TempoPresenter } from 'src/editing/playback/tempo_presenter';
import type { PaletteStore } from 'src/editing/palette/palette_store';
import { perfProbe } from 'src/utils/perf_probe';
import { jotPlayer } from 'src/editing/playback/player';
import { BarTiming, timeToX } from 'src/editing/playback/timeline';
import { SelectionStore } from 'src/editing/selection/selection';
import {
  SelectionPresenter,
  SelectionPresenterContext,
  orderedNotes,
} from 'src/editing/selection/selection_presenter';
import styles from './jot_editor.module.css';
import {
  AudioWorkletWarningModal,
  detectAudioWorkletState,
} from './playback/audio_worklet_warning_modal';
import { LyricsSearchModal } from './lyrics/lyrics_search_modal';
import { LyricsTextLoadModal } from './lyrics/lyrics_text_modal';
import { SelectionContext } from 'src/editing/selection/selection';
import {
  BarTimingsContext,
  StructuralContext,
  TempoContext,
  PaletteContext,
} from './jot_editor_contexts';
import { GridLineSettingsContext } from '../settings/settings_contexts';
import { MixerStoreContext, UniformWaveformsContext } from './mixer/mixer_contexts';
import { ViewportStoreContext } from './viewport/viewport_contexts';
import {
  LyricsAlignStoreContext,
  LyricsPresenterContext,
} from './lyrics/lyrics_contexts';
import {
  NoteProvenanceContext,
  ProvenancePresenterContext,
  ProvenanceStoreContext,
} from './provenance/provenance_contexts';
import { FollowPlayheadContext } from './playback/playback_contexts';
import { AudioTrackControls, MixerView, LayerControls } from './mixer/mixer';
import { Logo } from 'src/ui/logo/logo';
import { Minimap } from './minimap/minimap';
import { EditingStore } from './editing_store';
import { EditingPresenter } from './editing_presenter';
import { EditingStoreContext, EditingPresenterContext } from './editing_contexts';
import { EditingToolbar } from './editing_toolbar';
import { useEditorKeymap } from './keyboard/keymap';
import { VerticalScrollbar } from './viewport/vertical_scrollbar';
import { PlaybackBar } from './playback/playback';
import {
  Legend,
  extractArtist,
  formatDisplayTitle,
  formatSubtitle,
} from './score/score_header';
import { TimelineHeader } from './score/timeline_header';
import type { TrackKey } from 'src/editing/tracks/tracks';
import { SettingsStore, type GridLineSettings } from '../settings/settings_store';
import { JotEditorStore } from './jot_editor_store';
import { TranscribeStore } from './transcribe/transcribe_store';
import { ProvenanceStore } from './provenance/provenance_store';
import { LyricsAlignStore } from './lyrics/lyrics_align_store';
import { PlaybackStore } from './playback/playback_store';
import { ViewportStore, snapToDevicePx } from './viewport/viewport_store';
import { MixerStore } from './mixer/mixer_store';
import { SettingsPresenter } from '../settings/settings_presenter';
import { ViewportPresenter } from './viewport/viewport_presenter';
import { MixerPresenter } from './mixer/mixer_presenter';
import { PlaybackPresenter } from './playback/playback_presenter';
import { ProvenancePresenter } from './provenance/provenance_presenter';
import { LyricsPresenter } from './lyrics/lyrics_presenter';
import { JotEditorPresenter } from './jot_editor_presenter';
import { TranscribePresenter } from './transcribe/transcribe_presenter';
import { RecentTranscriptionsPicker } from './transcribe/recent_transcriptions';
import { ToastContainer } from '../ui/toasts/toast_container';
import { Toolbar } from '../toolbar/toolbar';
import { Sidebar } from '../sidebar/sidebar';
import { SidebarStore } from '../sidebar/sidebar_store';
import { SidebarPresenter } from '../sidebar/sidebar_presenter';
import { SidebarStoreContext, SidebarPresenterContext } from '../sidebar/sidebar_contexts';
import { DebugPanel } from './provenance/debug_panel';
import { ExampleJot } from 'src/fakes/fakes';

export { TranscribePresenter } from './transcribe/transcribe_presenter';
export type { TrackKey } from 'src/editing/tracks/tracks';
export type { TranscribeOptions, TranscribeStatus } from './transcribe/transcribe_store';

type CreateJotEditorOptions = {
  examples?: readonly ExampleJot[];
};

type CreateJotEditorResult = {
  /** Data-only stores + the presenter. Exposed so the app shell (and
   *  e2e) can reach each peer directly; there is no single top-level
   *  store. */
  jotEditorStore: JotEditorStore;
  settings: SettingsStore;
  transcribe: TranscribeStore;
  provenance: ProvenanceStore;
  lyricsAlign: LyricsAlignStore;
  playback: PlaybackStore;
  viewport: ViewportStore;
  mixer: MixerStore;
  /** Selection + editing peers, exposed for console / e2e. */
  selection: SelectionStore;
  selectionPresenter: SelectionPresenter;
  editingStore: EditingStore;
  editingPresenter: EditingPresenter;
  /** Right-sidebar peers. */
  sidebar: SidebarStore;
  sidebarPresenter: SidebarPresenter;
  /** Per-domain presenters split out of the catch-all. Exposed for
   *  console / e2e. */
  viewportPresenter: ViewportPresenter;
  mixerPresenter: MixerPresenter;
  provenancePresenter: ProvenancePresenter;
  playbackPresenter: PlaybackPresenter;
  lyricsPresenter: LyricsPresenter;
  jotEditorPresenter: JotEditorPresenter;
  /** Transcribe presenter (`/transcribe` + `/resume` flows, progress
   *  pill, recent-runs picker, form options). */
  transcribePresenter: TranscribePresenter;
  View: React.FC;
};

export function createJotEditor(options: CreateJotEditorOptions = {}): CreateJotEditorResult {
  const jotEditorStore = new JotEditorStore();
  const settings = new SettingsStore();
  const transcribe = new TranscribeStore();
  const provenance = new ProvenanceStore();
  const lyricsAlign = new LyricsAlignStore();
  const viewport = new ViewportStore(jotEditorStore);
  const mixer = new MixerStore(jotEditorStore);
  // PlaybackStore reads the mixer for the engine-facing filter computeds
  // the player pulls; construct the mixer first.
  const playback = new PlaybackStore(jotEditorStore, mixer);
  const settingsPresenter = new SettingsPresenter(settings);
  const viewportPresenter = new ViewportPresenter(viewport, jotEditorStore);
  const mixerPresenter = new MixerPresenter(mixer, jotEditorStore);
  const provenancePresenter = new ProvenancePresenter(provenance, viewport);
  const playbackPresenter = new PlaybackPresenter(playback, jotEditorStore);
  const lyricsPresenter = new LyricsPresenter(lyricsAlign, jotEditorStore);
  const jotEditorPresenter = new JotEditorPresenter(
    jotEditorStore,
    settingsPresenter,
    mixerPresenter,
    provenancePresenter,
    lyricsPresenter
  );
  const transcribePresenter = new TranscribePresenter({ transcribe, jotEditorPresenter });
  if (options.examples) jotEditorPresenter.setExamples(options.examples);
  const selection = new SelectionStore();
  const selectionPresenter = new SelectionPresenter(selection, () =>
    orderedNotes(jotEditorStore.structural?.layers ?? [])
  );
  const editingStore = new EditingStore();
  const editingPresenter = new EditingPresenter(
    editingStore,
    jotEditorStore,
    settings,
    selection,
    selectionPresenter
  );
  const sidebar = new SidebarStore();
  const sidebarPresenter = new SidebarPresenter(sidebar);

  // Marquee hit-test: which notes a rubber-band box (scroll-content coords)
  // encloses, resolved to the current StructNotes. Reads the DOM, so it only
  // runs from the pointer handlers below (never a render path).
  const marqueeHitTest = (box: Box): StructNote[] => {
    const layers = jotEditorStore.structural?.musicalLayers;
    if (!layers) return [];
    return notesInBox(box, notesById(layers));
  };
  // Origin of the in-flight marquee drag (scroll-content coords). Closure-local
  // transient interaction state, not persisted, not observed.
  let marqueeOrigin: Point | undefined;

  // Translate a click on `.jotContainer` into the marquee's coordinate
  // space (the inner `.scrollViewport` wrapper, which is where the
  // marquee div lives and where its `top` / `left` are interpreted).
  // The container's `getBoundingClientRect` reflects its visual rect
  // (unaffected by our virtual scroll, since the transform is on the
  // inner wrapper, not the container itself), so adding `store.scrollX`
  // / `store.scrollY` re-derives the wrapper-local position the marquee
  // needs. Reading the observables outside any observer/render path is
  // intentional: this fires only on pointer events, not per-frame, so
  // we don't want to subscribe createJotEditor to scroll motion.
  const containerPoint = (e: React.MouseEvent<HTMLDivElement>): Point => {
    const el = e.currentTarget;
    const rect = el.getBoundingClientRect();
    return new Point(
      e.clientX - rect.left + viewport.scrollX,
      e.clientY - rect.top + viewport.scrollY
    );
  };
  const onMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    marqueeOrigin = containerPoint(e);
    selectionPresenter.beginMarquee(marqueeOrigin);
  };
  const onMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!marqueeOrigin) return;
    const p = containerPoint(e);
    selectionPresenter.updateMarquee(marqueeOrigin, p, marqueeHitTest(Box.create(marqueeOrigin, p)));
  };
  const onMouseUp = () => {
    if (!marqueeOrigin) return;
    marqueeOrigin = undefined;
    const box = selection.marquee;
    selectionPresenter.endMarquee(box ? marqueeHitTest(box) : []);
  };

  /**
   * Zoom-slider handler. Mirrors the cursor-anchored math the wheel
   * zoom uses (see the `flush` closure in JotEditor), but anchors at the
   * viewport's horizontal centre so dragging the slider keeps the
   * musical content under the centre of the score pinned. The wheel
   * path still anchors at the cursor; this is only for the slider /
   * any other absolute-zoom caller that lacks a pointer origin.
   */
  const setZoomCentered = (z: number) => {
    const scroller = document.querySelector<HTMLElement>('[data-jot-scroller]');
    const barsRow = scroller?.querySelector<HTMLElement>('[data-bars-row]');
    const s = jotEditorStore.structural;
    if (!scroller || !barsRow || !s) {
      viewportPresenter.setZoom(z);
      return;
    }
    const pxPerBeatBefore = s.pxPerBeat;
    const padLeft = s.config.barNotePaddingBeats * pxPerBeatBefore;
    const scrollerRect = scroller.getBoundingClientRect();
    const barsRowRect = barsRow.getBoundingClientRect();
    const anchorBarsRowX = scrollerRect.left + scrollerRect.width / 2 - barsRowRect.left;
    viewportPresenter.setZoom(z);
    if (pxPerBeatBefore <= 0) return;
    const factor = s.pxPerBeat / pxPerBeatBefore;
    if (factor === 1) return;
    viewportPresenter.setScrollBy((anchorBarsRowX - padLeft) * (factor - 1), 0);
    // Same-tick scale + scroll write (cache-free; this runs outside the
    // component) so the slider never paints the post-zoom scroll offset at
    // the pre-zoom scale; see applyZoomVarsSync.
    applyZoomVarsSync(scroller, null, s.pxPerBeat, s.layerBeats, viewport.scrollX);
  };

  // Stable JotEditor callback identities. Each only ever delegates to the
  // (stable) `store` / `selection` instances, so defining them once here
  // (rather than as inline arrows in `View`'s render) keeps their
  // reference identity constant across `View` re-renders. That's load-
  // bearing: `JotEditor` is `observer`-wrapped (React.memo), and on a long
  // song its subtree (mixer → every InstrumentTrackView → BarViews → NoteViews)
  // is the expensive reconciliation. A fresh closure prop on any `View`
  // re-render (e.g. the zoom slider writing `store.zoom`) would defeat
  // that memo and reconcile the whole score; stable props let the memo
  // hold so `View` can re-render without touching `JotEditor`. The
  // observable VALUES JotEditor needs (jot, trackOrder, highlightedPattern,
  // the mute/solo snapshots) still flow through props/memo below and stay
  // reactive. Reads of `store.*` inside these bodies run at call time, not
  // render time, so they don't subscribe createJotEditor to anything.
  const onPatternClick = (name: string) => selectionPresenter.togglePattern(name);
  const onSeek = (x: number) => playbackPresenter.seekToX(x);
  const onZoomBy = (factor: number) => viewportPresenter.setZoom(viewport.zoom * factor);
  const onMoveTrack = (from: number, to: number) => mixerPresenter.moveTrack(from, to);
  const getGutterWidth = () => viewport.gutterWidth;
  const onSetGutterWidth = (px: number) => viewportPresenter.setGutterWidth(px);

  // Computed once per page load; AudioWorklet availability doesn't
  // change after boot, so capturing this outside the component (and
  // outside any observer) keeps it stable across renders without
  // entering React's dep tracking.
  const audioWorkletState = detectAudioWorkletState();

  const View: React.FC = observer(() => {
    // The loaded song's peer domains (all set/cleared together; `structural`
    // present implies the rest are). `structural` is the "is a song loaded"
    // signal that gates JotEditor / the playback bar.
    const structural = jotEditorStore.structural;
    const source = jotEditorStore.source;
    // Modal opens on first mount whenever AudioWorklet is unavailable
    // and closes for the rest of the session once dismissed. A reload
    // re-shows it intentionally; the limitation is real and ongoing,
    // and the user might just have forgotten between sessions.
    const [audioWorkletWarningOpen, setAudioWorkletWarningOpen] = React.useState(
      audioWorkletState !== 'available'
    );

    // Global editor keyboard shortcuts, dispatched through the keymap →
    // command layer (Space = play/pause from anywhere; Delete/Backspace =
    // delete the selection). The dispatcher skips text-entry targets but lets
    // a focused BUTTON / range slider fall through, so Space `preventDefault`
    // both stops page scroll and the button's space-activation and always
    // toggles transport. Keys are remappable by swapping the keymap.
    useEditorKeymap({ editingPresenter, playbackPresenter });

    const provenanceContextValue = provenance.provenanceContextValue;

    // Lyrics modal visibility lives on the store so any TS consumer can
    // observe / drive it; the seeded title/artist fields are still local
    // (re-derived from the current jot on open).
    const lyricsInitialTitle = source?.title.trim() ?? '';
    const lyricsInitialArtist = source ? (extractArtist(source) ?? '') : '';

    const followPlayheadContextValue = React.useMemo(
      () => ({
        follow: playback.followPlayhead,
        toggle: () => playbackPresenter.toggleFollowPlayhead(),
      }),
      // eslint-disable-next-line react-hooks/exhaustive-deps -- observable read; observer wrapper rebuilds the memo when followPlayhead flips.
      [playback.followPlayhead]
    );

    // The mixer control bundles embed observable SNAPSHOT values (the
    // master mute/solo/audible booleans) plus stable delegations + live
    // Set references. Memoising each on exactly those snapshot inputs
    // keeps the object's identity constant when none of them changed.
    // Crucially that holds across a zoom tick, which re-renders `View` (it reads
    // `store.zoom` for the toolbar slider) but must NOT churn `JotEditor`'s
    // props. The mute/solo toggles still rebuild the bundle (deps change)
    // so the mixer updates; per-row volume / audibility stays reactive
    // because the consumer rows call `isLaneAudible` / `volumeFor`
    // (which read the store) inside their own `observer` bodies, so they
    // re-render regardless of the bundle's identity.
    const layerControls: LayerControls = React.useMemo(
      () => ({
        mutedLanes: mixer.mutedLanes,
        soloedLanes: mixer.soloedLanes,
        isLaneAudible: mixer.isLaneAudible,
        volumeFor: (lane) => mixer.laneVolume(lane),
        onSetVolume: (lane, v) => mixerPresenter.setLaneVolume(lane, v),
        onToggleMute: (lane) => mixerPresenter.toggleMute(lane),
        onToggleSolo: (lane) => mixerPresenter.toggleSolo(lane),
        masterMuted: mixer.drumMasterMuted,
        masterSoloed: mixer.drumMasterSoloed,
        masterAudible: mixer.isDrumSectionAudible,
        onToggleMasterMute: () => mixerPresenter.toggleDrumMasterMute(),
        onToggleMasterSolo: () => mixerPresenter.toggleDrumMasterSolo(),
      }),
      // eslint-disable-next-line react-hooks/exhaustive-deps -- observable snapshots; observer wrapper rebuilds when any of these change.
      [
        mixer.mutedLanes,
        mixer.soloedLanes,
        mixer.drumMasterMuted,
        mixer.drumMasterSoloed,
        mixer.isDrumSectionAudible,
      ]
    );
    const audioTrackControls: AudioTrackControls = React.useMemo(
      () => ({
        mutedAudioTracks: mixer.mutedAudioTracks,
        soloedAudioTracks: mixer.soloedAudioTracks,
        isAudioTrackAudible: mixer.isAudioTrackAudible,
        volumeFor: (id) => mixer.audioTrackVolume(id),
        onSetVolume: (id, v) => mixerPresenter.setAudioTrackVolume(id, v),
        onToggleMute: (id) => mixerPresenter.toggleAudioTrackMute(id),
        onToggleSolo: (id) => mixerPresenter.toggleAudioTrackSolo(id),
        onClear: (id) => mixerPresenter.clearAudioTrack(id),
        onSplitFromMix: (id) => mixerPresenter.splitAudioTrackFromMix(id),
        onSplitDrumPieces: (id) => mixerPresenter.splitAudioTrackDrumPieces(id),
        masterMuted: mixer.audioMasterMuted,
        masterSoloed: mixer.audioMasterSoloed,
        masterAudible: mixer.isAudioSectionAudible,
        onToggleMasterMute: () => mixerPresenter.toggleAudioMasterMute(),
        onToggleMasterSolo: () => mixerPresenter.toggleAudioMasterSolo(),
      }),
      // eslint-disable-next-line react-hooks/exhaustive-deps -- observable snapshots; observer wrapper rebuilds when any of these change.
      [
        mixer.mutedAudioTracks,
        mixer.soloedAudioTracks,
        mixer.audioMasterMuted,
        mixer.audioMasterSoloed,
        mixer.isAudioSectionAudible,
      ]
    );

    return (
        <SidebarStoreContext.Provider value={sidebar}>
        <SidebarPresenterContext.Provider value={sidebarPresenter}>
        <LyricsPresenterContext.Provider value={lyricsPresenter}>
        <ProvenancePresenterContext.Provider value={provenancePresenter}>
        <ProvenanceStoreContext.Provider value={provenance}>
        <LyricsAlignStoreContext.Provider value={lyricsAlign}>
        <ViewportStoreContext.Provider value={viewport}>
        <MixerStoreContext.Provider value={mixer}>
        <SelectionContext.Provider value={selection}>
        <SelectionPresenterContext.Provider value={selectionPresenter}>
        <EditingStoreContext.Provider value={editingStore}>
        <EditingPresenterContext.Provider value={editingPresenter}>
          <NoteProvenanceContext.Provider value={provenanceContextValue}>
            <GridLineSettingsContext.Provider value={settings.gridLines}>
              <UniformWaveformsContext.Provider value={settings.uniformWaveforms}>
                <FollowPlayheadContext.Provider value={followPlayheadContextValue}>
                  <div className={styles.appContainer}>
                    <div className={styles.mainColumn}>
                    <Toolbar
                      examples={jotEditorStore.examples}
                      currentId={jotEditorStore.currentExampleId}
                      onSelect={(id) => jotEditorPresenter.loadExample(id)}
                      transcribeStatus={transcribe.transcribeStatus}
                      transcribeOptions={transcribe.transcribeOptions}
                      onTranscribe={(file) => transcribePresenter.transcribeAudio(file)}
                      onResumeTranscribe={(folder, stage) => transcribePresenter.resumeTranscribe(folder, stage)}
                      onLoadJot={(file) => jotEditorPresenter.loadJotFile(file)}
                      onLoadMidi={(file) => jotEditorPresenter.loadMidiFile(file)}
                      onLoadParadb={(file) => jotEditorPresenter.loadParadbMap(file)}
                      onScoreParadb={(file) => jotEditorPresenter.scoreParadbMap(file)}
                      onLoadDebugBundle={(file) => jotEditorPresenter.loadDebugBundleFile(file)}
                      onLoadAudioTrack={(file) => jotEditorPresenter.loadAudioTrack(file)}
                      onLoadLyricsFile={(file) => jotEditorPresenter.loadLyricsFile(file)}
                      onOpenLyricsTextLoad={() => lyricsPresenter.setLyricsTextOpen(true)}
                      onOpenLyricsSearch={() => lyricsPresenter.setLyricsSearchOpen(true)}
                      onCancelTranscribe={() => transcribePresenter.cancelTranscribe()}
                      lyricsAlignBusyPhase={lyricsAlign.lyricsAlignBusyPhase}
                      onSetBeatInput={(b) => transcribePresenter.setBeatInput(b)}
                      onSetDrumSeparator={(s) => transcribePresenter.setDrumSeparator(s)}
                      onSetLlmModel={(m) => transcribePresenter.setLlmModel(m)}
                      onSetQuantise={(v) => transcribePresenter.setQuantise(v)}
                      onSetQuantiseUseLlm={(v) => transcribePresenter.setQuantiseUseLlm(v)}
                      onSetZoom={setZoomCentered}
                      hasNoteProvenance={provenance.noteProvenance !== undefined}
                      showFilteredOnsets={provenance.showFilteredOnsets}
                      onSetShowFilteredOnsets={(v) => provenancePresenter.setShowFilteredOnsets(v)}
                      gridLines={settings.gridLines}
                      onToggleGridLine={(k) => settingsPresenter.toggleGridLine(k)}
                      uniformWaveforms={settings.uniformWaveforms}
                      onSetUniformWaveforms={(v) => settingsPresenter.setUniformWaveforms(v)}
                      autoFollowOnPlay={playback.autoFollowOnPlay}
                      onSetAutoFollowOnPlay={(v) => playbackPresenter.setAutoFollowOnPlay(v)}
                      recentTranscriptions={transcribe.recentTranscriptions}
                      recentTranscriptionsLoaded={transcribe.recentTranscriptionsLoaded}
                      recentTranscriptionsLoading={transcribe.recentTranscriptionsLoading}
                      selectedResumeFolder={transcribe.selectedResumeFolder}
                      selectedResumeStage={transcribe.selectedResumeStage}
                      onSetSelectedResumeFolder={(f) => transcribePresenter.setSelectedResumeFolder(f)}
                      onSetSelectedResumeStage={(s) => transcribePresenter.setSelectedResumeStage(s)}
                      onRefreshRecentTranscriptions={() => transcribePresenter.refreshRecentTranscriptions()}
                      onLoadRecentTranscription={(folder) => transcribePresenter.loadRecentTranscription(folder)}
                      transcribeMode={transcribe.transcribeMode}
                      onSetTranscribeMode={(m) => transcribePresenter.setTranscribeMode(m)}
                    />
                    {structural ? (
                      <JotEditor
                        viewport={viewport}
                        viewportPresenter={viewportPresenter}
                        playbackPresenter={playbackPresenter}
                        structural={structural}
                        tempo={jotEditorStore.tempo!}
                        palette={jotEditorStore.palette!}
                        source={jotEditorStore.source!}
                        highlightedPattern={selection.selectedPattern}
                        onPatternClick={onPatternClick}
                        onMouseDown={onMouseDown}
                        onMouseMove={onMouseMove}
                        onMouseUp={onMouseUp}
                        onSeek={onSeek}
                        onZoomBy={onZoomBy}
                        trackOrder={mixer.trackOrder}
                        onMoveTrack={onMoveTrack}
                        layerControls={layerControls}
                        audioTrackControls={audioTrackControls}
                        getGutterWidth={getGutterWidth}
                        onSetGutterWidth={onSetGutterWidth}
                      />
                    ) : (
                      <EmptyState
                        transcribePresenter={transcribePresenter}
                        jotEditorPresenter={jotEditorPresenter}
                        jotEditorStore={jotEditorStore}
                        transcribe={transcribe}
                      />
                    )}
                    <Minimap
                      jotEditorStore={jotEditorStore}
                      viewport={viewport}
                      viewportPresenter={viewportPresenter}
                      mixer={mixer}
                      playbackPresenter={playbackPresenter}
                    />
                    {structural && (
                      <PlaybackBar
                        jotEditorStore={jotEditorStore}
                        playback={playback}
                        presenter={playbackPresenter}
                      />
                    )}
                    {structural && <EditingToolbar />}
                    <DebugPanel provenance={provenance} presenter={provenancePresenter} />
                    </div>
                    <Sidebar />
                    <LyricsSearchModal
                      open={lyricsAlign.lyricsSearchOpen}
                      initialTitle={lyricsInitialTitle}
                      initialArtist={lyricsInitialArtist}
                      onClose={() => lyricsPresenter.setLyricsSearchOpen(false)}
                      presenter={lyricsPresenter}
                    />
                    <LyricsTextLoadModal
                      open={lyricsAlign.lyricsTextOpen}
                      onClose={() => lyricsPresenter.setLyricsTextOpen(false)}
                      presenter={lyricsPresenter}
                    />
                    <AudioWorkletWarningModal
                      state={audioWorkletState}
                      open={audioWorkletWarningOpen}
                      onClose={() => setAudioWorkletWarningOpen(false)}
                    />
                    <LoadingOverlay jotEditorStore={jotEditorStore} />
                    <ToastContainer />
                  </div>
                </FollowPlayheadContext.Provider>
              </UniformWaveformsContext.Provider>
            </GridLineSettingsContext.Provider>
          </NoteProvenanceContext.Provider>
        </EditingPresenterContext.Provider>
        </EditingStoreContext.Provider>
        </SelectionPresenterContext.Provider>
        </SelectionContext.Provider>
        </MixerStoreContext.Provider>
        </ViewportStoreContext.Provider>
        </LyricsAlignStoreContext.Provider>
        </ProvenanceStoreContext.Provider>
        </ProvenancePresenterContext.Provider>
        </LyricsPresenterContext.Provider>
        </SidebarPresenterContext.Provider>
        </SidebarStoreContext.Provider>
    );
  });

  return {
    jotEditorStore,
    settings,
    transcribe,
    provenance,
    lyricsAlign,
    playback,
    viewport,
    mixer,
    selection,
    selectionPresenter,
    editingStore,
    editingPresenter,
    sidebar,
    sidebarPresenter,
    viewportPresenter,
    mixerPresenter,
    provenancePresenter,
    playbackPresenter,
    lyricsPresenter,
    jotEditorPresenter,
    transcribePresenter,
    View,
  };
}

type JotEditorProps = {
  /** Score viewport state (scroll offsets, extents, zoom, gutter). The
   * scroll viewport's native `overflow: auto` is replaced by a CSS
   * `transform` on `.scrollViewport`; `viewport.scrollX`/`scrollY` are
   * the canonical source of scroll position, fed by JotEditor's
   * ResizeObservers (via `viewportPresenter.setViewportSize`/`setContentSize`)
   * and driven by auto-follow / zoom-anchor / pan / Stop (via
   * `viewportPresenter.setScrollX`/`setScrollBy`/`resetScrollX`). */
  viewport: ViewportStore;
  /** Viewport mutations (scroll / zoom / size / gutter). */
  viewportPresenter: ViewportPresenter;
  /** Transport / follow-playhead mutations. JotEditor disengages follow on
   * pan / pinch gestures. */
  playbackPresenter: PlaybackPresenter;
  /** The loaded song's peer domains (provided to descendants via context). */
  structural: StructuralPresenter;
  tempo: TempoPresenter;
  palette: PaletteStore;
  source: Jot;
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
  layerControls: LayerControls;
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

const JotEditor = observer((props: JotEditorProps) => {
  const {
    viewport,
    viewportPresenter,
    playbackPresenter,
    structural,
    tempo,
    palette,
    source,
    highlightedPattern,
    onPatternClick,
    onMouseDown,
    onMouseMove,
    onMouseUp,
    onSeek,
    onZoomBy,
    trackOrder,
    onMoveTrack,
    layerControls,
    audioTrackControls,
    getGutterWidth,
    onSetGutterWidth,
  } = props;
  // Render-counter hook for `e2e/zoom-rerender.spec.ts` (no-op unless
  // `window.__perf` is set). Guards the invariant that zoom never
  // re-renders JotEditor; see `src/perf_probe.ts`.
  perfProbe('JotEditor');
  // Intentionally NOT reading any zoom-dependent (pixel) observable here.
  // Every observable touched in this body triggers a JotEditor re-render on
  // zoom, and the title / subtitle / Legend / mixer subtree all derive from
  // zoom-invariant data via `structural.layers` / `source.title` /
  // `source.globalMetadata`. JotEditor itself is then stable across zoom
  // (ScoreZoomVar updates the one CSS variable that propagates the
  // new scale to every descendant via calc()).
  const config = structural.config;
  const containerRef = React.useRef<HTMLDivElement>(null);
  // Ref to the inner `.scrollViewport` wrapper. Its `offsetWidth` /
  // `offsetHeight` is the scroll-content's natural size (the analogue
  // of `scrollWidth` / `scrollHeight` in the previous native-overflow
  // model); fed into `store.setContentSize` so `setScrollX` /
  // `setScrollY` clamp to `[0, content - viewport]`.
  const viewportRef = React.useRef<HTMLDivElement>(null);
  // Shared DOM-target cache for the per-frame animation vars. Owned by
  // JotEditor so its lifetime matches the container; ScrollVar and
  // PlayheadPosVar read through it instead of querying the DOM on every
  // tick.
  const cacheRef = React.useRef<DomTargetCache | null>(null);
  React.useLayoutEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const cache = new DomTargetCache();
    cache.attach(container, styles.scrollStickyHorizontal, viewport);
    cacheRef.current = cache;
    return () => {
      cache.detach();
      cacheRef.current = null;
    };
  }, [playbackPresenter]);
  React.useEffect(() => {
    const container = containerRef.current;
    const viewport = viewportRef.current;
    if (!container || !viewport) return;
    const updateContainer = () => {
      viewportPresenter.setViewportSize(container.clientWidth, container.clientHeight);
    };
    const updateViewport = () => {
      viewportPresenter.setContentSize(viewport.offsetWidth, viewport.offsetHeight);
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
  }, [playbackPresenter]);

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
  // Stable ref to the live structural presenter for the non-React rAF
  // closures (wheel / pinch zoom) that read `pxPerBeat` / `layerBeats` /
  // `config` outside the render path.
  const structuralRef = React.useRef(structural);
  structuralRef.current = structural;
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
      if (scrollDx !== 0) viewportPresenter.setScrollBy(scrollDx, 0);
      if (delta === 0) return;
      const factor = Math.exp(-delta * 0.0015);

      // Bars-row coords of the anchor point (cursor position when the
      // wheel event fired; viewport centre as fallback). `bar.x` /
      // `bar.width` are linear in `pxPerBeat`, so the new x of the same
      // musical point is `padLeft + (anchorX - padLeft) * actualFactor`;
      // adjusting scrollLeft by the delta keeps the on-screen position
      // pinned without waiting for layout.
      const currentStructural = structuralRef.current;
      const pxPerBeatBefore = currentStructural.pxPerBeat;
      // `padLeft` scales with `pxPerBeat` (see ViewConfig.barNotePaddingBeats);
      // anchor math reads the live pre-zoom value so the on-screen
      // musical point stays pinned across the zoom step.
      const padLeft = currentStructural.config.barNotePaddingBeats * pxPerBeatBefore;
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
        const actualFactor = currentStructural.pxPerBeat / pxPerBeatBefore;
        if (actualFactor !== 1) {
          const delta = (anchorBarsRowX - padLeft) * (actualFactor - 1);
          // Virtual scroll: write through the store instead of native
          // `el.scrollLeft +=`. The store's clamp uses the pre-zoom
          // content width (the ResizeObserver on `.scrollViewport` is
          // async); for zoom-in this is conservative-safe (smaller
          // max), for zoom-out the next observer tick re-clamps within
          // a frame, so a one-frame overshoot is the worst case.
          viewportPresenter.setScrollBy(delta, 0);
        }
      }

      // Apply the new scale + scroll to the DOM in the same tick so no
      // painted frame shows the post-zoom scroll offset at the pre-zoom
      // scale; see applyZoomVarsSync.
      applyZoomVarsSync(
        el,
        cacheRef.current,
        currentStructural.pxPerBeat,
        currentStructural.layerBeats,
        viewport.scrollX
      );
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
    // `store` is a stable singleton from `createJotEditor`; included for
    // exhaustive-deps correctness even though it never changes in practice.
  }, [playbackPresenter]);

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
      playbackPresenter.setFollowPlayhead(false);
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
      viewportPresenter.setScrollBy(-dx, -dy);
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
  }, [playbackPresenter]);

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
      playbackPresenter.setFollowPlayhead(false);
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
      const s = structuralRef.current;
      const initPxPerBeat = s.pxPerBeat;
      const initPadLeft = s.config.barNotePaddingBeats * initPxPerBeat;
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
        initZoom: viewport.zoom,
        initScrollX: viewport.scrollX,
        initPxPerBeat,
        initPadLeft,
        anchorBarsRowX,
      };
      playbackPresenter.setFollowPlayhead(false);
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
        viewportPresenter.setScrollBy(-dx, -dy);
        e.preventDefault();
      } else if (gesture.kind === 'pinch' && e.touches.length >= 2) {
        const a = e.touches[0];
        const b = e.touches[1];
        const dist = touchDistance(a, b);
        if (gesture.initDistance > 0 && gesture.initPxPerBeat > 0) {
          viewportPresenter.setZoom(gesture.initZoom * (dist / gesture.initDistance));
          const actualFactor = structuralRef.current.pxPerBeat / gesture.initPxPerBeat;
          if (actualFactor !== 1) {
            viewportPresenter.setScrollX(
              gesture.initScrollX +
                (gesture.anchorBarsRowX - gesture.initPadLeft) * (actualFactor - 1)
            );
          }
          // Same-tick scale + scroll write so the pinch never paints the
          // post-zoom scroll offset at the pre-zoom scale; see
          // applyZoomVarsSync.
          applyZoomVarsSync(
            el,
            cacheRef.current,
            structuralRef.current.pxPerBeat,
            structuralRef.current.layerBeats,
            viewport.scrollX
          );
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
  }, [playbackPresenter]);

  // `--note-pad-beats` is the engraving inset (a zoom-invariant
  // fraction of a beat) every note / grid / bracket reads in its
  // percentage-of-bar `left` calc; set inline once here and inherited
  // down the score. The bar width that those percentages resolve
  // against is driven per row by `ScoreZoomVar` (see `setBarsRowVars`),
  // so the inset scales with zoom without any var write on each tick.
  // `--gutter-width` is mutated at runtime by `GutterWidthVar`, a
  // side-effect-only observer that writes via `setProperty` on
  // `containerRef.current` so the tick doesn't re-render JotEditor. With
  // many tracks loaded a JotEditor re-render is expensive enough to make
  // a 120 Hz resize drag visibly laggy, so we keep that read off the
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
    ['--gutter-width' as string]: `${initialGutterPx}px`,
  } as React.CSSProperties;
  // Eager bar-timings table for `BarTimingsContext`. Built once per jot
  // identity here so deep consumers (NoteProvenanceDetails' "Final
  // position" row) don't need the player's timeline to have been
  // initialized — pre-Play, `jotPlayer.timeline` is `EMPTY_TIMELINE`.
  // `buildTimeline` now reads zoom-invariant fields only, so calling
  // it here doesn't bind JotEditor's render to the zoom variable.
  const barTimings = React.useMemo<ReadonlyMap<number, BarTiming>>(() => {
    const timeline = tempo.timeline;
    const structBars = structural.layers[0]?.bars ?? [];
    const map = new Map<number, BarTiming>();
    for (let i = 0; i < structBars.length; i++) {
      const timing = timeline.bars[i];
      if (timing) map.set(structBars[i].index, timing);
    }
    return map;
  }, [structural, tempo]);
  // The starting width is captured at the start of each drag (the
  // pointermove deltas then read against that snapshot) so an in-
  // flight resize stays anchored to where the user grabbed even
  // though the live `gutterWidth` observable is being updated every
  // frame. Read via the getter so JotEditor itself doesn't subscribe.
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
    <StructuralContext.Provider value={structural}>
      <TempoContext.Provider value={tempo}>
        <PaletteContext.Provider value={palette}>
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
              <ScoreZoomVar structural={structural} containerRef={containerRef} />
              <GutterWidthVar getGutterWidth={getGutterWidth} containerRef={containerRef} />
              <GridLineVars containerRef={containerRef} />
              <ScrollVar cacheRef={cacheRef} viewport={viewport} />
              <PlayheadPosVar
                cacheRef={cacheRef}
                getGutterWidth={getGutterWidth}
                viewport={viewport}
                viewportPresenter={viewportPresenter}
              />
              <div
                ref={viewportRef}
                className={styles.scrollViewport}
                // Stable hook used by JotEditor's content ResizeObserver and by
                // the minimap (offsetWidth is the scroll-content's `scrollWidth`
                // analogue in this no-native-scroll model).
                data-jot-scroll-content
              >
                <h2 className={styles.title}>{formatDisplayTitle(source) || 'Untitled jot'}</h2>
                <p className={styles.subtitle}>{formatSubtitle(source, tempo)}</p>
                <Legend palette={palette} />
                <TimelineHeader onSeek={onSeek} onResizeGutterStart={onResizeGutterStart} />
                <MixerView
                  config={config}
                  trackOrder={trackOrder}
                  highlightedPattern={highlightedPattern}
                  onPatternClick={onPatternClick}
                  onSeek={onSeek}
                  onMoveTrack={onMoveTrack}
                  layerControls={layerControls}
                  audioTrackControls={audioTrackControls}
                  onResizeGutterStart={onResizeGutterStart}
                />
                <MarqueeOverlay />
                <SelectionFrame />
              </div>
              <VerticalScrollbar viewport={viewport} viewportPresenter={viewportPresenter} />
            </div>
          </BarTimingsContext.Provider>
        </PaletteContext.Provider>
      </TempoContext.Provider>
    </StructuralContext.Provider>
  );
});

/**
 * Side-effect-only observer that writes each bars-row's pixel width (the
 * one quantity zoom mutates) whenever the zoom-derived pixel-per-beat or
 * the layer length changes. Isolated so reading `jot.pxPerBeat` (a
 * zoom-dependent observable) doesn't re-render JotEditor, the writes go
 * out via `setBarsRowVars` (DOM `setProperty` on the few `[data-bars-row]`
 * elements), then CSS percentages reposition every bar / note / bracket
 * within each row without React touching the subtree and without a
 * `--px-per-beat` style-recalc cascade.
 *
 * `useLayoutEffect` (not `useEffect`) so the widths are set before paint:
 * the rows' percentage children would otherwise resolve against the
 * `--bars-row-width` initial value (0px) for one frame on mount. It runs
 * with `containerRef.current` populated and the bars-row DOM committed
 * (effects fire after the whole tree mounts), so the query finds every
 * row on the first pass.
 */
const ScoreZoomVar = observer(
  ({
    structural,
    containerRef,
  }: {
    structural: StructuralPresenter;
    containerRef: React.RefObject<HTMLDivElement>;
  }) => {
    const pxPerBeat = structural.pxPerBeat;
    const layerBeats = structural.layerBeats;
    React.useLayoutEffect(() => {
      // On the FIRST layout effect `containerRef.current` is still null:
      // a child's layout effect (this) runs before the parent fiber
      // (`.jotContainer`) attaches its ref. The committed DOM exists
      // though, so fall back to resolving the scroller from it, that
      // keeps the very first paint from rendering 0-width bars-rows
      // (which also collapses the ResizeObserver's content-width read).
      const el = containerRef.current ?? document.querySelector<HTMLElement>('[data-jot-scroller]');
      if (!el) return;
      setBarsRowVars(el, pxPerBeat, layerBeats);
    }, [pxPerBeat, layerBeats, containerRef]);
    return null;
  }
);

/**
 * Side-effect-only observer that writes `--gutter-width` onto the
 * JotEditor container whenever the store's gutter width changes. Same
 * trick as `ScoreZoomVar`: a resize tick mutates one CSS variable on
 * the root and CSS propagates the new width to every sticky gutter
 * (score header, mixer rows) without React touching the subtree —
 * which matters on a debug bundle with lots of tracks, where a full
 * JotEditor re-render at 120 Hz pointermove rate is visibly laggy.
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

  attach(root: HTMLElement, stickyClass: string, viewportStore: ViewportStore): void {
    this.viewport = root.querySelector<HTMLElement>('[data-jot-scroll-content]');
    for (const el of root.querySelectorAll<HTMLElement>('.' + stickyClass)) {
      this.sticky.add(el);
    }
    for (const el of root.querySelectorAll<HTMLElement>('[data-playhead="1"]')) {
      this.playheads.add(el);
    }
    this.observer = new MutationObserver((records) => {
      const seedX = String(viewportStore.scrollX);
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
    viewport,
  }: {
    cacheRef: React.RefObject<DomTargetCache | null>;
    viewport: ViewportStore;
  }) => {
    const x = viewport.scrollX;
    const y = viewport.scrollY;
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
 * Write the one quantity zoom mutates - each bars-row's pixel width
 * (`layerBeats × pxPerBeat`) - onto every `[data-bars-row]` under `root`.
 * Replaces the old single inherited `--px-per-beat` on the score
 * container: every beat-anchored element now sizes/positions itself as a
 * PERCENTAGE of its bars-row, so only the ~handful of row elements need
 * touching (registered `inherits: false`, so each write invalidates only
 * that row, not its subtree), and the percentage children relayout
 * without a style-recalc cascade.
 *
 * Lyrics rows additionally get a scoped `--px-per-beat` (their font
 * metrics can't be a percentage); only `[data-lyrics-bars-row]` elements
 * receive it, keeping that small cascade off every other row.
 *
 * Queried per write rather than cached: zoom changes are gesture-rate
 * (not the idle 120 fps that `--scroll-x` runs at), and the row count is
 * tiny, so a `querySelectorAll` is cheap and sidesteps the
 * mount-ordering between this and `DomTargetCache`.
 */
function setBarsRowVars(root: HTMLElement, pxPerBeat: number, layerBeats: number): void {
  const width = `${pxPerBeat * layerBeats}px`;
  const ppb = String(pxPerBeat);
  for (const el of root.querySelectorAll<HTMLElement>('[data-bars-row]')) {
    el.style.setProperty('--bars-row-width', width);
    if (el.dataset.lyricsBarsRow === '1') el.style.setProperty('--px-per-beat', ppb);
  }
}

/**
 * Apply a zoom step's new SCALE (bars-row widths) and SCROLL
 * (`--scroll-x`) to the DOM together, synchronously, from inside a zoom
 * handler.
 *
 * The two otherwise land via separate mobx observers, ScoreZoomVar's
 * `useLayoutEffect` (scale) and ScrollVar's `useLayoutEffect` (scroll),
 * which mobx-react-lite can commit in different React passes. For ≥1
 * painted frame the wrapper would then sit at the post-zoom scroll offset
 * while the bars are still at the pre-zoom scale. Cursor anchoring keeps
 * `scrollΔ ≈ scrollX·(factor−1)`, which is many screens deep in a long
 * song, so that stale frame paints the viewport multiple screens off (a
 * zoom-out renders an earlier section) and reads as a random jump.
 * Writing both here closes that window; the observers re-apply identical
 * values, so it's idempotent. Mirrors the `--scroll-x` + `--playhead-x`
 * pairing in PlayheadPosVar.
 *
 * `cache` is passed by the in-component wheel/pinch handlers that own one;
 * the zoom slider runs outside the component (no cache) and passes null,
 * so we fall back to querying the same targets the cache would hold.
 */
function applyZoomVarsSync(
  scroller: HTMLElement,
  cache: DomTargetCache | null,
  pxPerBeat: number,
  layerBeats: number,
  scrollX: number
): void {
  setBarsRowVars(scroller, pxPerBeat, layerBeats);
  if (cache) {
    setScrollX(cache, scrollX);
    return;
  }
  const xStr = String(scrollX);
  scroller.querySelector<HTMLElement>('[data-jot-scroll-content]')?.style.setProperty('--scroll-x', xStr);
  for (const el of scroller.querySelectorAll<HTMLElement>(`.${styles.scrollStickyHorizontal}`)) {
    el.style.setProperty('--scroll-x', xStr);
  }
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
    viewport,
    viewportPresenter,
  }: {
    cacheRef: React.RefObject<DomTargetCache | null>;
    getGutterWidth: () => number;
    viewport: ViewportStore;
    viewportPresenter: ViewportPresenter;
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
        viewportPresenter.resetScrollX();
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
        const clientWidth = viewport._viewportWidth;
        if (clientWidth > 0) {
          viewportPresenter.setScrollX(getGutterWidth() + x - clientWidth / 2);
          setScrollX(cache, viewport.scrollX);
        }
      }
      setPlayheadVar(cache, x);
    }, [
      t,
      state,
      cued,
      timeline,
      pxPerBeat,
      follow,
      cacheRef,
      getGutterWidth,
      viewport,
      viewportPresenter,
    ]);
    return null;
  }
);

/**
 * Isolated observer for the in-flight marquee rectangle so a mousemove
 * (which fires many times per second and mutates `selection.marquee`)
 * only re-renders this 4-style div instead of the whole JotEditor tree —
 * `JotEditor`/`MixerView`/per-row waveforms etc. are expensive enough that
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
 * Subtle bounding box drawn around a multi-note selection (the "selection
 * frame"). Lives inside the scroll-content wrapper so it scrolls with the
 * notes for free; its pixel extents are read from the selected glyphs' DOM
 * rects in a layout effect that re-runs when the selection or the zoom
 * (`pxPerBeat`) changes, not on scroll. Hidden for 0 or 1 selected notes.
 */
const SelectionFrame = observer(() => {
  const selection = React.useContext(SelectionContext);
  const ids = selection?.effectiveIds;
  // Bail BEFORE reading any zoom observable when there's no multi-selection,
  // so this component never re-renders on a zoom tick in the common (no
  // selection) case, keeping the 120fps zoom path free of a per-frame render.
  if (!ids || ids.size < 2) return null;
  return <SelectionFrameBox ids={ids} />;
});

/** The frame box itself, mounted only for a ≥2-note selection. Reads
 *  `pxPerBeat` so the box re-measures on zoom; that subscription is scoped to
 *  the (rare) selected state by {@link SelectionFrame}'s early return. */
const SelectionFrameBox = observer(({ ids }: { ids: ReadonlySet<string> }) => {
  const structural = React.useContext(StructuralContext);
  const pxPerBeat = structural?.pxPerBeat ?? 0;
  const [box, setBox] = React.useState<Box | null>(null);
  React.useLayoutEffect(() => {
    setBox(boundingBoxOfNotes(ids));
  }, [ids, pxPerBeat]);
  if (!box) return null;
  return (
    <div
      className={styles.selectionFrame}
      data-testid="selection-frame"
      aria-hidden="true"
      style={{ left: box.x, top: box.y, width: box.width, height: box.height }}
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
const EmptyState = observer(
  ({
    transcribePresenter,
    jotEditorPresenter,
    jotEditorStore,
    transcribe,
  }: {
    transcribePresenter: TranscribePresenter;
    jotEditorPresenter: JotEditorPresenter;
    jotEditorStore: JotEditorStore;
    transcribe: TranscribeStore;
  }) => {
  const jotInputRef = React.useRef<HTMLInputElement>(null);
  const paradbInputRef = React.useRef<HTMLInputElement>(null);
  const handleJotFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) jotEditorPresenter.loadJotFile(file);
    e.target.value = '';
  };
  const handleParadbFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) jotEditorPresenter.loadParadbMap(file);
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
              items={transcribe.recentTranscriptions}
              loaded={transcribe.recentTranscriptionsLoaded}
              loading={transcribe.recentTranscriptionsLoading}
              onRefresh={() => transcribePresenter.refreshRecentTranscriptions()}
              onPick={(folder) => transcribePresenter.loadRecentTranscription(folder)}
            />
          </div>
        </div>
        <p className={styles.emptyStateHint}>
          For other formats, use the <b>File</b> or <b>Transcribe</b> menus in the toolbar above.
        </p>
        {jotEditorStore.examples.length > 0 && (
          <div className={styles.emptyStateExamples}>
            <span className={styles.emptyStateExamplesLabel}>Or try an example</span>
            <div className={styles.emptyStateExampleRow}>
              {jotEditorStore.examples.map((ex) => (
                <button
                  key={ex.id}
                  type="button"
                  className={styles.emptyStateExampleButton}
                  onClick={() => jotEditorPresenter.loadExample(ex.id)}
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
const LoadingOverlay = observer(({ jotEditorStore }: { jotEditorStore: JotEditorStore }) => {
  if (!jotEditorStore.isLoading) return null;
  return (
    <div
      className={styles.loadingOverlay}
      role="status"
      aria-live="polite"
      onMouseDown={(e) => e.stopPropagation()}
    >
      <div className={styles.loadingSpinner} aria-hidden="true" />
      {jotEditorStore.loadingLabel && (
        <div className={styles.loadingLabel}>{jotEditorStore.loadingLabel}</div>
      )}
    </div>
  );
});
