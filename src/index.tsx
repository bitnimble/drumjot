import React from 'react';
import { createRoot } from 'react-dom/client';
import { createJotView, JotStore } from 'src/jot_view';
import { Jot, rockJot } from 'src/schema';

class Drumjot {
  private jotStore: JotStore;

  constructor(root: HTMLElement) {
    const { store, View } = createJotView();
    this.jotStore = store;

    const _root = createRoot(root);
    _root.render(<View />);
  }

  load(jot: Jot<string>) {
    // TODO: confirm if currentJot is saved
    this.jotStore.currentJot = jot;
  }

  loadTestJot() {
    this.load(rockJot);
  }
}

(window as any).Drumjot = Drumjot;
export default Drumjot;
