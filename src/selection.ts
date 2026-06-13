import { makeAutoObservable, observable } from 'mobx';
import { Box, Point } from 'src/geom';
import { StructuralBar, StructuralNote, StructuralVoice } from 'src/jot';
import { DocumentStore } from 'src/jot_view/document/document_store';

export type SelectionState =
  | { type: 'notes'; notes: Set<StructuralNote> }
  | { type: 'bars'; bars: StructuralBar[] }
  | { type: 'voice'; voice: StructuralVoice }
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

  constructor(private readonly documentStore: DocumentStore) {
    // `observable.ref` (not `.deep`) for state: every transition replaces
    // the whole state object — we never mutate the inner Set/array — so a
    // ref-equality reaction is enough, and crucially it stops MobX from
    // wrapping inner values (like `ResolvedNote` instances in a `notes`
    // Set) in observable proxies. Reference identity is what `NoteView`
    // checks against `selectedNote`, so proxy-wrapped values would
    // silently never match the prop and the selected-state label would
    // never appear. (`Set<StructuralNote>` now, but the same reasoning
    // — selection compares by reference identity.)
    makeAutoObservable(
      this,
      { state: observable.ref },
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

  /** Replace the selection with exactly one note. */
  selectNote(note: StructuralNote) {
    this.state = { type: 'notes', notes: new Set([note]) };
  }

  /**
   * The currently-selected note when exactly one is selected; otherwise
   * undefined. Drives the inline-label rendering — multi-note selections
   * (marquee result) deliberately suppress the label.
   */
  get selectedNote(): StructuralNote | undefined {
    if (this.state?.type !== 'notes') return undefined;
    if (this.state.notes.size !== 1) return undefined;
    return this.state.notes.values().next().value;
  }

  private getSelectionForMarquee(): SelectionState | undefined {
    // TODO: project the marquee Box into the resolved jot's coordinate space
    // and pick notes / bars / voices it encloses. Returning undefined is fine
    // until that is implemented; the marquee still renders.
    void this.documentStore;
    return undefined;
  }
}
