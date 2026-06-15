import { makeAutoObservable, observable } from 'mobx';
import React from 'react';
import { Box } from 'src/utils/geom';
import type {
  StructBar,
  StructNote,
  StructLayer,
} from 'src/editing/structure/structure_store';

export type SelectionState =
  | { type: 'notes'; notes: Set<StructNote> }
  | { type: 'bars'; bars: StructBar[] }
  | { type: 'layer'; layer: StructLayer }
  /**
   * A pattern is "selected" when its definition + all usages should be
   * visually highlighted. Created by clicking a pattern bracket label.
   */
  | { type: 'pattern'; name: string };

const EMPTY_NOTES: ReadonlySet<StructNote> = Object.freeze(new Set<StructNote>());

/**
 * Selection store, DATA ONLY (observables + computeds). Every mutation
 * lives on {@link SelectionPresenter}. Holds the committed `state`, an
 * in-progress marquee `transientState` + `marquee` box, and the
 * Explorer-style selection bookkeeping (`anchor` pivot + `base` set) the
 * presenter reads to compute shift-range extensions.
 *
 * UX rule: a mouse-down on empty container space clears the selection. If it
 * becomes a drag, the marquee commits a new selection on mouse-up; if it
 * stays a plain click, the selection ends up cleared. Selections from
 * clicking a specific element (note, pattern bracket) bypass this by stopping
 * mouse-down propagation before the container handler fires.
 */
export class SelectionStore {
  state?: SelectionState = undefined;
  transientState?: SelectionState = undefined;
  marquee: Box | undefined = undefined;

  /** Pivot for shift-range extension (Windows Explorer semantics): set by a
   *  plain click or ctrl-click, read by `extendTo`. */
  anchor: StructNote | undefined = undefined;
  /** The selection as it stood when `anchor` was last set, i.e. excluding the
   *  current shift-range. A shift-extend yields `base ∪ range(anchor,target)`,
   *  so re-shift-clicking recomputes the range without losing ctrl-toggled
   *  items. */
  base: ReadonlySet<StructNote> = EMPTY_NOTES;

  constructor() {
    // `observable.ref` for the state objects: every transition replaces the
    // whole object (we never mutate the inner Set), so ref-equality reactions
    // suffice AND, crucially, MobX won't wrap the inner `StructNote`s in
    // observable proxies. `NoteView` checks reference identity against
    // `selectedNote`/`selectedNotes`, so proxied values would never match.
    makeAutoObservable(
      this,
      {
        state: observable.ref,
        transientState: observable.ref,
        anchor: observable.ref,
        base: observable.ref,
      },
      { autoBind: true }
    );
  }

  /** The committed set of selected notes (empty unless a notes-selection is
   *  active). The transient marquee result, while dragging, is reflected via
   *  `transientState` and overrides this in `effectiveNotes`. */
  get selectedNotes(): ReadonlySet<StructNote> {
    return this.state?.type === 'notes' ? this.state.notes : EMPTY_NOTES;
  }

  /** Selected notes including the in-flight marquee preview, for rendering. */
  get effectiveNotes(): ReadonlySet<StructNote> {
    const s = this.transientState ?? this.state;
    return s?.type === 'notes' ? s.notes : EMPTY_NOTES;
  }

  /** Selected note ids (incl. marquee preview). Matching is by id, not object
   *  identity, so a selection survives a structural recompute, the derived
   *  `StructNote` objects are rebuilt on every edit (e.g. after a move), but
   *  their ids are stable. */
  get effectiveIds(): ReadonlySet<string> {
    const out = new Set<string>();
    for (const n of this.effectiveNotes) out.add(n.id);
    return out;
  }

  /**
   * The currently-selected note when exactly one is selected; otherwise
   * undefined. Drives the inline-label rendering, multi-note selections
   * deliberately suppress the label.
   */
  get selectedNote(): StructNote | undefined {
    const notes = this.effectiveNotes;
    if (notes.size !== 1) return undefined;
    return notes.values().next().value;
  }

  /** Whether a note is part of the current (or in-flight) selection. Matches
   *  by id so it holds across structural recomputes. */
  isSelected(note: StructNote): boolean {
    return this.effectiveIds.has(note.id);
  }

  /** Convenience: the currently-selected pattern name, if any. */
  get selectedPattern(): string | undefined {
    return this.state?.type === 'pattern' ? this.state.name : undefined;
  }

  /** A multi-note selection shows the bounding "selection frame"; the overlay
   *  component computes its pixel extents from the notes' geometry. */
  get hasFrame(): boolean {
    return this.effectiveNotes.size >= 2;
  }
}

/**
 * Routes the active {@link SelectionStore} to deep score chrome (today:
 * `NoteView`) without threading props through `JotEditor → MixerView →
 * InstrumentTrackView → BarView`. `null` outside the view so a `NoteView`
 * rendered in isolation just no-ops the click-to-select interaction.
 */
export const SelectionContext = React.createContext<SelectionStore | null>(null);
