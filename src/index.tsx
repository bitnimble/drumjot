import React from 'react';
import { createRoot } from 'react-dom/client';
import { Jot } from 'src/dsl';
import { rockJot, tripletJot } from 'src/fakes';
import { RenderedJot } from 'src/jot';
import { JotViewStore, createJotView } from 'src/jot_view';
import { parse } from 'src/parser';

class Drumjot {
  readonly store: JotViewStore;

  constructor(root: HTMLElement) {
    const { store, View } = createJotView();
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

  loadTestJot() {
    this.load(rockJot);
  }

  loadTripletJot() {
    this.load(tripletJot);
  }
}

(window as unknown as { Drumjot: typeof Drumjot }).Drumjot = Drumjot;
export default Drumjot;

// Auto-bootstrap when loaded as the Vite entry.
const mount = document.getElementById('app');
if (mount) {
  const app = new Drumjot(mount);
  app.loadTestJot();
}
