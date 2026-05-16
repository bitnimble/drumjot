import { makeAutoObservable, observable } from 'mobx';
import { Box, Point } from 'src/geom';
import { ResolvedBar, ResolvedNote, ResolvedVoice } from 'src/jot';
import { JotViewStore } from 'src/jot_view';

export type SelectionState =
  | { type: 'notes'; notes: Set<ResolvedNote> }
  | { type: 'bars'; bars: ResolvedBar[] }
  | { type: 'voice'; voice: ResolvedVoice }
  /**
   * A pattern is "selected" when its definition + all usages should be
   * visually highlighted. Created by clicking a pattern bracket label.
   */
  | { type: 'pattern'; name: string };

/**
 * Selection store: holds the committed `state`, an in-progress marquee
 * `transientState`, and exposes mutators for both marquee drags and
 * discrete selection events (pattern label clicks, etc.).
 *
 * UX rule: any mouse-down on empty container space clears the existing
 * selection. If that mouse-down turns into a drag, the resulting marquee
 * commits a new selection on mouse-up. If it stays a plain click, the
 * selection ends up cleared. Selections produced by clicking on a specific
 * element (e.g. a pattern bracket) bypass this by stopping mouse-down
 * propagation before the container handler fires.
 */
export class SelectionStore {
  state?: SelectionState = undefined;
  transientState?: SelectionState = undefined;

  private mousedownPoint: Point | undefined = undefined;
  marquee: Box | undefined = undefined;

  constructor(private readonly jotStore: JotViewStore) {
    makeAutoObservable(
      this,
      { state: observable.deep },
      { autoBind: true }
    );
  }

  beginSelection(p: Point) {
    // Mouse-down on empty space clears the existing selection. A subsequent
    // drag will overwrite it with a marquee result; a plain click will leave
    // the store empty.
    this.state = undefined;
    this.mousedownPoint = p;
  }

  moveSelection(p: Point) {
    if (!this.mousedownPoint) return;
    this.marquee = Box.create(this.mousedownPoint, p);
    this.transientState = this.getSelectionForMarquee();
  }

  endSelection() {
    if (this.transientState) {
      this.state = this.transientState;
    }
    this.mousedownPoint = undefined;
    this.transientState = undefined;
    this.marquee = undefined;
  }

  clear() {
    this.state = undefined;
  }

  /** Toggle the selection of a pattern by name. */
  togglePattern(name: string) {
    if (this.state?.type === 'pattern' && this.state.name === name) {
      this.state = undefined;
    } else {
      this.state = { type: 'pattern', name };
    }
  }

  /** Convenience: returns the currently-selected pattern name, if any. */
  get selectedPattern(): string | undefined {
    return this.state?.type === 'pattern' ? this.state.name : undefined;
  }

  private getSelectionForMarquee(): SelectionState | undefined {
    // TODO: project the marquee Box into the resolved jot's coordinate space
    // and pick notes / bars / voices it encloses. Returning undefined is fine
    // until that is implemented; the marquee still renders.
    void this.jotStore;
    return undefined;
  }
}
