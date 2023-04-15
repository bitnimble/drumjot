import { makeAutoObservable, observable, runInAction } from 'mobx';
import { Box, Point } from 'src/geom';
import { Bar, Loop, NoteState } from 'src/jot';
import { JotViewStore } from 'src/jot_view';

type SelectionState =
  | {
      type: 'notes';
      notes: Set<NoteState>;
    }
  | {
      type: 'bar';
      bars: Bar[];
    }
  | {
      type: 'loop';
      loop: Loop<string>;
    };

export class SelectionStore {
  state?: SelectionState = undefined;

  private mousedownPoint: Point | undefined = undefined;
  marquee: Box | undefined = undefined;

  constructor(private readonly jotStore: JotViewStore) {
    makeAutoObservable(
      this,
      {
        state: observable.deep,
      },
      { autoBind: true }
    );
  }

  beginSelection(p: Point) {
    this.mousedownPoint = p;
  }

  moveSelection(p: Point) {
    if (!this.mousedownPoint) {
      return;
    }
    this.marquee = Box.create(this.mousedownPoint!, p);
  }

  clearSelection() {
    this.mousedownPoint = undefined;
    this.marquee = undefined;
  }
}
