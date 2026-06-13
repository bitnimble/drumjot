import { makeAutoObservable } from 'mobx';
import { BeatInput, DrumSeparator, LlmModel, TranscribeStage } from 'src/transcriber';
import { GridLineSettings, SettingsStore } from './stores/settings_store';
import { TranscribeStore } from './stores/transcribe_store';
import { ProvenanceStore } from './stores/provenance_store';
import { LyricsAlignStore } from './stores/lyrics_align_store';

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

  constructor(deps: JotViewerPresenterDeps) {
    this.settings = deps.settings;
    this.transcribe = deps.transcribe;
    this.provenance = deps.provenance;
    this.lyricsAlign = deps.lyricsAlign;
    makeAutoObservable(
      this,
      { settings: false, transcribe: false, provenance: false, lyricsAlign: false },
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
}
