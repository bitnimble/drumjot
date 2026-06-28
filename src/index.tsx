import React from 'react';
import { createRoot } from 'react-dom/client';
import 'src/design_tokens.css';
import { Jot } from 'src/schema/dsl/dsl';
import { EXAMPLE_JOTS, ExampleJot, rockJot, tripletJot } from 'src/fakes/fakes';
import { createJotEditor } from 'src/editing/jot_editor';
import { TranscribePresenter } from 'src/editing/transcribe/transcribe_presenter';
import { ViewportPresenter } from 'src/editing/viewport/viewport_presenter';
import { MixerPresenter } from 'src/editing/mixer/mixer_presenter';
import { ProvenancePresenter } from 'src/editing/provenance/provenance_presenter';
import { PlaybackPresenter } from 'src/editing/playback/playback_presenter';
import { LyricsPresenter } from 'src/editing/lyrics/lyrics_presenter';
import { JotEditorPresenter } from 'src/editing/jot_editor_presenter';
import { JotEditorStore } from 'src/editing/jot_editor_store';
import { SelectionStore } from 'src/editing/selection/selection';
import { SelectionPresenter } from 'src/editing/selection/selection_presenter';
import { EditingStore } from 'src/editing/editing_store';
import { EditingPresenter } from 'src/editing/editing_presenter';
import { SidebarStore } from 'src/sidebar/sidebar_store';
import { SidebarPresenter } from 'src/sidebar/sidebar_presenter';
import { SettingsStore } from 'src/settings/settings_store';
import { TranscribeStore } from 'src/editing/transcribe/transcribe_store';
import { ProvenanceStore } from 'src/editing/provenance/provenance_store';
import { LyricsAlignStore } from 'src/editing/lyrics/lyrics_align_store';
import { PlaybackStore } from 'src/editing/playback/playback_store';
import { ViewportStore } from 'src/editing/viewport/viewport_store';
import { MixerStore } from 'src/editing/mixer/mixer_store';
import { parse } from 'src/schema/dsl/parser/parser';
import { writeDsl } from 'src/schema/dsl/writer';
import { mutableToDsl } from 'src/schema/dsl/to_dsl';
import { jotPlayer } from 'src/editing/playback/player';
// Side-effect import: instantiates the theme controller so the
// `<html data-theme>` attribute is in sync with the user's saved choice
// (or the live OS preference in `system` mode) before React mounts.
// index.html runs a synchronous boot script that sets the attribute for
// the very first paint; this import then takes over for live updates.
import 'src/settings/theme';

class Drumjot {
  // Data-only stores + presenter. Exposed (via `window.drumjot`) so
  // console / e2e can reach each peer directly; there is no single
  // top-level store.
  readonly jotEditorStore: JotEditorStore;
  readonly settings: SettingsStore;
  readonly transcribe: TranscribeStore;
  readonly provenance: ProvenanceStore;
  readonly lyricsAlign: LyricsAlignStore;
  readonly playback: PlaybackStore;
  readonly viewport: ViewportStore;
  readonly mixer: MixerStore;
  readonly selection: SelectionStore;
  readonly selectionPresenter: SelectionPresenter;
  readonly editingStore: EditingStore;
  readonly editingPresenter: EditingPresenter;
  readonly sidebar: SidebarStore;
  readonly sidebarPresenter: SidebarPresenter;
  readonly viewportPresenter: ViewportPresenter;
  readonly mixerPresenter: MixerPresenter;
  readonly provenancePresenter: ProvenancePresenter;
  readonly playbackPresenter: PlaybackPresenter;
  readonly lyricsPresenter: LyricsPresenter;
  readonly jotEditorPresenter: JotEditorPresenter;
  readonly transcribePresenter: TranscribePresenter;

  constructor(root: HTMLElement, examples: readonly ExampleJot[] = EXAMPLE_JOTS) {
    const {
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
    } = createJotEditor({ examples });
    this.jotEditorStore = jotEditorStore;
    this.settings = settings;
    this.transcribe = transcribe;
    this.provenance = provenance;
    this.lyricsAlign = lyricsAlign;
    this.playback = playback;
    this.viewport = viewport;
    this.mixer = mixer;
    this.selection = selection;
    this.selectionPresenter = selectionPresenter;
    this.editingStore = editingStore;
    this.editingPresenter = editingPresenter;
    this.sidebar = sidebar;
    this.sidebarPresenter = sidebarPresenter;
    this.viewportPresenter = viewportPresenter;
    this.mixerPresenter = mixerPresenter;
    this.provenancePresenter = provenancePresenter;
    this.playbackPresenter = playbackPresenter;
    this.lyricsPresenter = lyricsPresenter;
    this.jotEditorPresenter = jotEditorPresenter;
    this.transcribePresenter = transcribePresenter;
    createRoot(root).render(<View />);
  }

  load(jot: Jot) {
    // The presenter builds the song's peers against the shared ViewConfig
    // (so `setZoom`, which mutates `viewConfig.barWidth`, drives this jot's
    // `pxPerBeat`/layout), matching every other loader.
    this.jotEditorPresenter.setJot(jot);
  }

  /** Parse a DSL source string (SPEC.md syntax) and load the resulting jot. */
  loadDsl(source: string) {
    this.load(parse(source));
  }

  /** Serialize the currently-loaded song back to DSL (.jot) source text,
   *  the inverse of {@link loadDsl}. Empty string when nothing is loaded.
   *  Exports the CURRENT reactive document (`mutableToDsl` -> `writeDsl`), so
   *  edits made since load are reflected.
   *
   *  Still the *subset* format: the DSL can't carry the editor metadata
   *  (mixer, palette, display settings) a mutable `.jot` does, so use
   *  {@link saveMutable} for a lossless save. */
  toDsl(): string {
    const jot = this.jotEditorStore.jot;
    return jot ? writeDsl(mutableToDsl(jot)) : '';
  }

  /** Save the current session to a mutable `.jot` file (browser download).
   *  The lossless *superset* format: the edited document plus editor metadata
   *  (mixer, display settings, palette) the DSL can't carry. */
  saveMutable(): Promise<void> {
    return this.jotEditorPresenter.saveMutableFile();
  }

  /** Encode the current session to mutable `.jot` bytes in memory (no
   *  download), or `undefined` when nothing is loaded. For programmatic use
   *  and e2e round-tripping without the OS file picker. */
  toMutableBytes(): Promise<Uint8Array | undefined> {
    return this.jotEditorPresenter.toMutableBytes();
  }

  /** Load mutable `.jot` bytes (the inverse of {@link toMutableBytes}). */
  loadMutableBytes(bytes: Uint8Array): Promise<void> {
    const file = new File([new Uint8Array(bytes)], 'session.jot');
    return this.jotEditorPresenter.loadMutableFile(file);
  }

  /** Load one of the registered example jots by id. */
  loadExample(id: string) {
    this.jotEditorPresenter.loadExample(id);
  }

  loadTestJot() {
    this.load(rockJot);
  }

  loadTripletJot() {
    this.load(tripletJot);
  }
}

// Exposed for the browser console and e2e probing. `Drumjot` is the
// class; `drumjot` is the live instance (set on bootstrap below);
// `jotPlayer` is the playback singleton — the canonical surface for
// asserting playback / audio-track state from tests rather than scraping the
// DOM for things that only exist in JS (decoded AudioBuffers, etc.).
type DrumjotGlobals = {
  Drumjot: typeof Drumjot;
  drumjot?: Drumjot;
  jotPlayer: typeof jotPlayer;
};
const globals = window as unknown as DrumjotGlobals;
globals.Drumjot = Drumjot;
globals.jotPlayer = jotPlayer;

// Audio-track playback runs through an AudioWorklet (Signalsmith Stretch).
// Guard at boot so the failure is surfaced up front instead of as a
// cryptic library trace the first time a user presses Play. Two distinct
// failure modes worth distinguishing:
//   - Browser exposes no AudioWorklet at all (very old Firefox / Safari):
//     audio-track playback simply won't work, no fix available client-side.
//   - AudioWorklet exists in principle but the page is not in a secure
//     context (LAN IP over plain HTTP is the common one with `vite --host`):
//     fixable by switching to localhost / 127.0.0.1 / HTTPS.
// `isSecureContext` is the deciding factor; the constructor presence is
// only checked as a secondary signal for the generic-unsupported case.
if (typeof window !== 'undefined') {
  if (!window.isSecureContext) {
    console.warn(
      '[drumjot] This page is not running in a secure context, so the ' +
        'browser will not expose AudioWorklet. Audio-track playback ' +
        'will not work. Drum (MIDI) playback is unaffected. Open ' +
        'the page via localhost / 127.0.0.1 / HTTPS instead of a LAN IP ' +
        'over plain HTTP.'
    );
  } else if (typeof AudioWorkletNode === 'undefined') {
    console.warn(
      '[drumjot] AudioWorklet is not available in this browser. ' +
        'Audio-track playback ' +
        'will not work. Drum (MIDI) playback is unaffected.'
    );
  }
}

// Auto-bootstrap when loaded as the Vite entry. The store starts with no
// jot loaded; the View renders an empty-state welcome screen with file-load
// and example-picker shortcuts until the user picks something.
const mount = document.getElementById('app');
if (mount) {
  const app = new Drumjot(mount);
  globals.drumjot = app;
  // Guard tab close / reload / external navigation while a transcribe is
  // in flight. Browsers no longer honour custom messages here; setting
  // returnValue just triggers the native "Leave site?" confirm.
  window.addEventListener('beforeunload', (event) => {
    const t = app.transcribe;
    if (t.trackStatuses.size > 0 || t.replaceStatus.phase === 'uploading') {
      event.preventDefault();
      event.returnValue = '';
    }
  });
}
