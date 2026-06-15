import { makeAutoObservable } from 'mobx';
import React from 'react';
import { Box, Point } from 'src/utils/geom';
import type { StructLayer, StructNote } from 'src/editing/structure/structure_store';
import type { SelectionStore } from './selection';

/**
 * All notes in stable document order: layers, then bars in render order, then
 * within a bar by `(beat, lane)`. This is the order shift-range selection
 * walks. A plain, geometry-free derivation so it stays cheap and testable.
 */
export function orderedNotes(layers: readonly StructLayer[]): StructNote[] {
  const out: StructNote[] = [];
  for (const layer of layers) {
    for (const bar of layer.bars) {
      const inBar: StructNote[] = [];
      for (const lane of Object.keys(bar.tracks)) inBar.push(...bar.tracks[lane].notes);
      inBar.sort((a, b) => a.beat - b.beat || a.lane.localeCompare(b.lane));
      out.push(...inBar);
    }
  }
  return out;
}

/**
 * The only writer of {@link SelectionStore}. Owns discrete selection events
 * (click / ctrl-click / shift-click on a note, pattern-label clicks) and the
 * marquee drag lifecycle.
 *
 * Shift-range follows **Windows File Explorer** semantics via the store's
 * `anchor` (pivot) + `base` (selection when the pivot was set): a shift-extend
 * yields `base ∪ range(anchor, target)`, so re-shift-clicking recomputes the
 * range from the same pivot without discarding ctrl-toggled items, and a
 * ctrl-click in the middle of a range moves the pivot + folds the new
 * selection into `base`.
 *
 * The document order used for ranges is supplied by `getOrderedNotes` (decoupled
 * from geometry for testability); the marquee hit-test is supplied by the
 * caller (it needs layout), so this presenter never reads pixels.
 */
export class SelectionPresenter {
  constructor(
    private readonly store: SelectionStore,
    /** All notes in stable document order (bar, then beat, then lane). */
    private readonly getOrderedNotes: () => readonly StructNote[]
  ) {
    makeAutoObservable<this, 'store' | 'getOrderedNotes'>(this, {
      store: false,
      getOrderedNotes: false,
    });
  }

  /** Replace the selection with exactly `note` (plain click). New pivot. */
  replace(note: StructNote): void {
    this.commit(new Set([note]), note);
  }

  /** Toggle `note` in/out of the selection (ctrl/cmd-click). Moves the pivot
   *  to `note` and folds the result into `base` (Explorer: a ctrl-click
   *  commits the running selection). */
  toggle(note: StructNote): void {
    const next = new Set(this.store.selectedNotes);
    if (next.has(note)) next.delete(note);
    else next.add(note);
    this.commit(next, note);
  }

  /** Extend the selection to `note` (shift-click): `base ∪ range(anchor, note)`.
   *  Falls back to a plain replace when there's no pivot yet. The pivot and
   *  `base` are left unchanged so a subsequent shift-click re-ranges. */
  extendTo(note: StructNote): void {
    const anchor = this.store.anchor;
    if (!anchor) {
      this.replace(note);
      return;
    }
    const next = new Set(this.store.base);
    for (const n of this.range(anchor, note)) next.add(n);
    this.store.state = { type: 'notes', notes: next };
    this.store.transientState = undefined;
  }

  /** Replace the selection with an explicit set (e.g. select-all of a lane, or
   *  a committed marquee). Resets the pivot to one member. */
  setNotes(notes: Iterable<StructNote>): void {
    const set = new Set(notes);
    const pivot = set.size > 0 ? set.values().next().value : undefined;
    this.commit(set, pivot);
  }

  clear(): void {
    this.commit(new Set(), undefined);
  }

  /** Toggle pattern-name highlight (clears any note selection). */
  togglePattern(name: string): void {
    if (this.store.selectedPattern === name) {
      this.clear();
    } else {
      this.store.state = { type: 'pattern', name };
      this.store.transientState = undefined;
      this.store.anchor = undefined;
      this.store.base = new Set();
    }
  }

  // ---------- Marquee drag ----------

  /** Begin a marquee at `p`; clears the existing selection (a plain click that
   *  never drags thus leaves it cleared). */
  beginMarquee(p: Point): void {
    this.store.state = undefined;
    this.store.transientState = undefined;
    this.store.anchor = undefined;
    this.store.base = new Set();
    this.store.marquee = Box.create(p, p);
  }

  /** Update the in-flight marquee to span `from..p`, previewing `enclosed`
   *  (computed by the caller from layout) as the transient selection. */
  updateMarquee(from: Point, p: Point, enclosed: Iterable<StructNote>): void {
    this.store.marquee = Box.create(from, p);
    const set = new Set(enclosed);
    this.store.transientState = set.size > 0 ? { type: 'notes', notes: set } : undefined;
  }

  /** Commit the marquee preview as the selection and clear the rubber band. */
  endMarquee(): void {
    if (this.store.transientState?.type === 'notes') {
      this.store.state = this.store.transientState;
      const pivot = this.store.state.notes.values().next().value;
      this.store.anchor = pivot;
      this.store.base = new Set(this.store.state.notes);
    }
    this.store.transientState = undefined;
    this.store.marquee = undefined;
  }

  // ---------- internals ----------

  /** Commit `notes` as the selection and set the pivot + base bookkeeping. */
  private commit(notes: Set<StructNote>, pivot: StructNote | undefined): void {
    this.store.state = notes.size > 0 ? { type: 'notes', notes } : undefined;
    this.store.transientState = undefined;
    this.store.anchor = pivot;
    this.store.base = new Set(notes);
  }

  /** Contiguous run of notes between `a` and `b` (inclusive) in document
   *  order. Empty if either isn't found. */
  private range(a: StructNote, b: StructNote): StructNote[] {
    const ordered = this.getOrderedNotes();
    const ia = ordered.indexOf(a);
    const ib = ordered.indexOf(b);
    if (ia < 0 || ib < 0) return [];
    const [lo, hi] = ia <= ib ? [ia, ib] : [ib, ia];
    return ordered.slice(lo, hi + 1);
  }
}

/** Routes the {@link SelectionPresenter} to deep score chrome (NoteView) for
 *  click/ctrl/shift selection. `null` outside the editor view. */
export const SelectionPresenterContext = React.createContext<SelectionPresenter | null>(null);
