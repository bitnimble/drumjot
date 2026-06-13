import { makeAutoObservable } from 'mobx';
import { px } from 'src/jot';
import { BeatInput, DrumSeparator, LlmModel, TranscribeStage } from 'src/transcriber';
import { DocumentStore } from './stores/document_store';
import { GridLineSettings, SettingsStore } from './stores/settings_store';
import { TranscribeStore } from './stores/transcribe_store';
import { ProvenanceStore } from './stores/provenance_store';
import { LyricsAlignStore } from './stores/lyrics_align_store';
import { PlaybackStore } from './stores/playback_store';
import {
  BASE_BAR_WIDTH,
  MAX_GUTTER_WIDTH,
  MAX_ZOOM,
  MIN_GUTTER_WIDTH,
  MIN_ZOOM,
  snapToDevicePx,
  ViewportStore,
} from './stores/viewport_store';

/**
 * Dependencies the presenter orchestrates over. Every store is a plain
 * data container; the presenter is the single place that mutates them.
 *
 * This grows one entry per extracted store as the `JotViewStore` carve-up
 * proceeds. It is a TEMPORARY catch-all for all orchestration that used
 * to (incorrectly) live on `JotViewStore`; once the carve-up is complete
 * the methods here get split into per-feature presenters, each owning the
 * subset of stores its feature touches.
 */
export type JotViewerPresenterDeps = {
  settings: SettingsStore;
  transcribe: TranscribeStore;
  provenance: ProvenanceStore;
  lyricsAlign: LyricsAlignStore;
  playback: PlaybackStore;
  document: DocumentStore;
  viewport: ViewportStore;
};

/**
 * Catch-all presenter for the jot viewer. Holds the actions, reactions,
 * and orchestration that mutate the data-only stores; React components
 * bind its methods to UI callbacks and read store state for rendering.
 *
 * The split exists so business logic can be unit-tested with mocked
 * stores (e.g. `presenter.setUniformWaveforms(true)` and assert the
 * mocked `SettingsStore` was updated) without standing up React or the
 * full store graph.
 *
 * Methods are grouped by the feature/domain they'll eventually move to;
 * each group is fronted by a `// --- <domain> ---` banner.
 */
export class JotViewerPresenter {
  // Store dependencies. `makeAutoObservable` is told to leave these
  // non-observable (they're already-observable stores; the presenter
  // only holds references, it doesn't own their reactivity).
  readonly settings: SettingsStore;
  readonly transcribe: TranscribeStore;
  readonly provenance: ProvenanceStore;
  readonly lyricsAlign: LyricsAlignStore;
  readonly playback: PlaybackStore;
  readonly document: DocumentStore;
  readonly viewport: ViewportStore;

  constructor(deps: JotViewerPresenterDeps) {
    this.settings = deps.settings;
    this.transcribe = deps.transcribe;
    this.provenance = deps.provenance;
    this.lyricsAlign = deps.lyricsAlign;
    this.playback = deps.playback;
    this.document = deps.document;
    this.viewport = deps.viewport;
    makeAutoObservable(
      this,
      {
        settings: false,
        transcribe: false,
        provenance: false,
        lyricsAlign: false,
        playback: false,
        document: false,
        viewport: false,
      },
      { autoBind: true }
    );
  }

  // --- settings ---

  toggleGridLine(key: keyof GridLineSettings) {
    this.settings.gridLines = {
      ...this.settings.gridLines,
      [key]: !this.settings.gridLines[key],
    };
  }

  setUniformWaveforms(on: boolean) {
    this.settings.uniformWaveforms = on;
  }

  // --- transcribe (form options + resume picker) ---

  setDebug(enabled: boolean) {
    this.transcribe.transcribeOptions.debug = enabled;
  }

  setBeatInput(input: BeatInput) {
    this.transcribe.transcribeOptions.beatInput = input;
  }

  setDrumSeparator(separator: DrumSeparator) {
    this.transcribe.transcribeOptions.drumSeparator = separator;
  }

  setLlmModel(model: LlmModel) {
    this.transcribe.transcribeOptions.llmModel = model;
  }

  setQuantise(enabled: boolean) {
    this.transcribe.transcribeOptions.quantise = enabled;
  }

  setQuantiseUseLlm(enabled: boolean) {
    this.transcribe.transcribeOptions.quantiseUseLlm = enabled;
  }

  setSelectedResumeFolder(folder: string | undefined) {
    this.transcribe.selectedResumeFolder = folder;
    // Clearing the folder (or picking a different one) invalidates any
    // stage selection, different folders have different `resumable_stages`,
    // so a stale pick could land on a stage missing its prerequisites.
    this.transcribe.selectedResumeStage = undefined;
  }

  setSelectedResumeStage(stage: TranscribeStage | undefined) {
    this.transcribe.selectedResumeStage = stage;
  }

  setTranscribeMode(mode: 'new' | 'resume') {
    this.transcribe.transcribeMode = mode;
  }

  // --- provenance / debug panel ---

  /** Replace the toolbar's `Show filtered` checkbox state. */
  setShowFilteredOnsets(show: boolean) {
    this.provenance.showFilteredOnsets = show;
  }

  setPinnedFilteredOnsetKey(key: string | undefined) {
    this.provenance.pinnedFilteredOnsetKey = key;
  }

  /** Toggle the DebugPanel's open state without forgetting the bundle. */
  toggleDebugPanel() {
    this.provenance.debugPanelOpen = !this.provenance.debugPanelOpen;
  }

  // --- lyrics (modal visibility) ---

  setLyricsSearchOpen(open: boolean) {
    this.lyricsAlign.lyricsSearchOpen = open;
  }

  setLyricsTextOpen(open: boolean) {
    this.lyricsAlign.lyricsTextOpen = open;
  }

  // --- playback / transport ---

  setAutoFollowOnPlay(on: boolean) {
    this.playback.autoFollowOnPlay = on;
  }

  // --- viewport (zoom / scroll / gutter) ---

  setZoom(z: number) {
    const clamped = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z));
    this.viewport.zoom = clamped;
    this.document.viewConfig.barWidth = px(BASE_BAR_WIDTH * clamped);
  }

  /** Cache the score viewport's pixel dimensions. Fed by a ResizeObserver
   * on `.jotContainer`. Re-clamps scroll so a resize that shrinks the
   * viewport (or grows it past the content) doesn't leave scroll parked
   * off the new end. */
  setViewportSize(width: number, height: number): void {
    this.viewport._viewportWidth = width;
    this.viewport._viewportHeight = height;
    this.viewport.scrollX = this.clampScrollX(this.viewport.scrollX);
    this.viewport.scrollY = this.clampScrollY(this.viewport.scrollY);
  }

  /** Cache the scroll-content's pixel dimensions (the inner
   * `.scrollViewport` wrapper's offset size). Re-clamps as above. */
  setContentSize(width: number, height: number): void {
    this.viewport._contentWidth = width;
    this.viewport._contentHeight = height;
    this.viewport.scrollX = this.clampScrollX(this.viewport.scrollX);
    this.viewport.scrollY = this.clampScrollY(this.viewport.scrollY);
  }

  setScrollX(x: number): void {
    this.viewport.scrollX = this.clampScrollX(snapToDevicePx(x));
  }

  setScrollY(y: number): void {
    this.viewport.scrollY = this.clampScrollY(snapToDevicePx(y));
  }

  setScrollBy(dx: number, dy: number): void {
    this.viewport.scrollX = this.clampScrollX(snapToDevicePx(this.viewport.scrollX + dx));
    this.viewport.scrollY = this.clampScrollY(snapToDevicePx(this.viewport.scrollY + dy));
  }

  /** Reset the horizontal scroll to the score's start (Stop transitions).
   * Deliberately does NOT touch scrollY, the user's vertical view
   * shouldn't snap back on Stop, only the playhead-tracking axis. */
  resetScrollX(): void {
    this.viewport.scrollX = 0;
  }

  /** Clamp a tentative target to `[0, contentSize - viewportSize]`. */
  clampScrollX(x: number): number {
    const max = Math.max(0, this.viewport._contentWidth - this.viewport._viewportWidth);
    if (!(x > 0)) return 0;
    if (x > max) return max;
    return x;
  }

  clampScrollY(y: number): number {
    const max = Math.max(0, this.viewport._contentHeight - this.viewport._viewportHeight);
    if (!(y > 0)) return 0;
    if (y > max) return max;
    return y;
  }

  /** Resize the sticky gutter column, clamped to a sensible range so a
   * runaway drag can't collapse the controls or push the bars row off
   * screen. */
  setGutterWidth(width: number): void {
    if (!Number.isFinite(width)) return;
    this.viewport.gutterWidth = Math.min(MAX_GUTTER_WIDTH, Math.max(MIN_GUTTER_WIDTH, width));
  }
}
