import React from 'react';
import { createRoot } from 'react-dom/client';
import 'src/design_tokens.css';
import { Jot } from 'src/dsl';
import { EXAMPLE_JOTS, ExampleJot, rockJot, tripletJot } from 'src/fakes';
import { RenderedJot } from 'src/jot';
import { JotViewStore, createJotView } from 'src/jot_view';
import { parse } from 'src/parser';
import { jotPlayer } from 'src/playback';
// Side-effect import: instantiates the theme controller so the
// `<html data-theme>` attribute is in sync with the user's saved choice
// (or the live OS preference in `system` mode) before React mounts.
// index.html runs a synchronous boot script that sets the attribute for
// the very first paint; this import then takes over for live updates.
import 'src/theme';

class Drumjot {
  readonly store: JotViewStore;

  constructor(root: HTMLElement, examples: readonly ExampleJot[] = EXAMPLE_JOTS) {
    const { store, View } = createJotView({ examples });
    this.store = store;
    createRoot(root).render(<View />);
  }

  load(jot: Jot) {
    // Pass the store's shared ViewConfig so `store.setZoom` (which mutates
    // `viewConfig.barWidth`) actually drives this jot's `pxPerBeat`/layout.
    // Every in-store loader (loadExample/transcribe/file) does the same;
    // omitting it here left zoom a no-op for `window.drumjot.load`/`loadDsl`.
    this.store.setJot(new RenderedJot(jot, this.store.viewConfig));
  }

  /** Parse a DSL source string (SPEC.md syntax) and load the resulting jot. */
  loadDsl(source: string) {
    this.load(parse(source));
  }

  /** Load one of the registered example jots by id. */
  loadExample(id: string) {
    this.store.loadExample(id);
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
export default Drumjot;

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
    if (app.store.transcribeStatus.phase === 'uploading') {
      event.preventDefault();
      event.returnValue = '';
    }
  });
}
