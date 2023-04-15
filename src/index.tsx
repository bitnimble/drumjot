import React from 'react';
import { createRoot } from 'react-dom/client';
import { rockJot } from 'src/fakes';
import { RenderedJot } from 'src/jot';
import { createJotView, JotViewStore } from 'src/jot_view';

class Drumjot {
  private jotStore: JotViewStore;

  constructor(root: HTMLElement) {
    const { store, View } = createJotView();
    this.jotStore = store;

    const _root = createRoot(root);
    _root.render(<View />);
  }

  load(jot: RenderedJot<string>) {
    // TODO: confirm if currentJot is saved
    this.jotStore.currentJot = jot;
  }

  loadTestJot() {
    this.load(new RenderedJot(rockJot));
  }
}

(window as any).Drumjot = Drumjot;
export default Drumjot;
