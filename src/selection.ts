import { makeAutoObservable, observable } from 'mobx';
import { Bar, Loop, NoteState } from 'src/jot';

type SelectionState = {
  type: 'notes',
  notes: Set<NoteState>,
} | {
  type: 'bar',
  bars: Bar[];
} | {
  type: 'loop',
  loop: Loop<string>,
}

class SelectionStore {
  state?: SelectionState;

  constructor() {
    makeAutoObservable(this, {
      state: observable.deep,
    });
  }
}
