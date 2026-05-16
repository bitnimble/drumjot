import { makeAutoObservable, observable } from 'mobx';
import { Box, Point } from 'src/geom';
import { ResolvedBar, ResolvedNote, ResolvedVoice } from 'src/jot';
import { JotViewStore } from 'src/jot_view';

export type SelectionState =
  | { type: 'notes'; notes: Set<ResolvedNote> }
  | { type: 'bars'; bars: ResolvedBar[] }
  | { type: 'voice'; voice: ResolvedVoice };

/**
 * Marquee selection store. Tracks an in-progress drag and exposes both a
 * committed `state` and an in-progress `transientState`. The intersection
 * logic that turns a marquee Box into a SelectionState is intentionally
 * left as a TODO so the surrounding plumbing can be exercised first.
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

  private getSelectionForMarquee(): SelectionState | undefined {
    // TODO: project the marquee Box into the resolved jot's coordinate space
    // and pick notes / bars / voices it encloses. Returning undefined is fine
    // until that is implemented; the marquee still renders.
    void this.jotStore;
    return undefined;
  }
}
