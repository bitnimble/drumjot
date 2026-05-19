import React from 'react';
import { createRoot } from 'react-dom/client';
import { Jot } from 'src/dsl';
import { EXAMPLE_JOTS, ExampleJot, rockJot, tripletJot } from 'src/fakes';
import { RenderedJot } from 'src/jot';
import { JotViewStore, createJotView } from 'src/jot_view';
import { parse } from 'src/parser';
import { jotPlayer } from 'src/playback';

class Drumjot {
  readonly store: JotViewStore;

  constructor(root: HTMLElement, examples: readonly ExampleJot[] = EXAMPLE_JOTS) {
    const { store, View } = createJotView({ examples });
    this.store = store;
    createRoot(root).render(<View />);
  }

  load(jot: Jot) {
    this.store.setJot(new RenderedJot(jot));
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

// Auto-bootstrap when loaded as the Vite entry. Load the first registered
// example so the picker reflects what is on screen.
const mount = document.getElementById('app');
if (mount) {
  const app = new Drumjot(mount);
  globals.drumjot = app;
  if (EXAMPLE_JOTS.length > 0) {
    app.loadExample(EXAMPLE_JOTS[0].id);
  }
}
