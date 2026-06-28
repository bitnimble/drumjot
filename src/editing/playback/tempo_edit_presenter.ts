/**
 * The tempo-editing domain: the single writer for sticky tempo changes and the
 * song's initial tempo. Reads the reactive document for the editable BPM
 * "markers" the timeline header paints (each flat `tempoEvent`, plus an initial
 * pill for the song-start tempo) and mutates `jot.tempoEvents` through one
 * atomic CRDT op each (so undo/redo treats a tempo change as a single step).
 * Tempo lives entirely in `tempoEvents`: the initial tempo is just the event at
 * the first source bar's downbeat, so the initial pill upserts/edits that
 * event (there is no separate `jot.bpm`). Gradual `BpmTransition` ramps are NOT
 * editable here, they carry no flat marker and stay read-only (authored via DSL).
 *
 * Counterpart to the read-only {@link TempoPresenter}: that derives the
 * per-bar tempo segments / timeline the renderer consumes; this is the write
 * surface the header's context menu + inline-edit pills drive.
 */
import { runInAction } from 'mobx';
import { computedFn } from 'mobx-utils';
import { computed, makeObservable } from 'mobx';
import type { JotEditorStore } from 'src/editing/jot_editor_store';
import { LEAD_IN_BAR_ID } from 'src/editing/structure/structure_store';
import { initialBpm, tempoAt } from 'src/schema/dsl/tempo';

/** The lowest / highest BPM an edit may set. Integers only. */
export const MIN_BPM = 20;
export const MAX_BPM = 400;

/** Where an editable BPM pill is anchored + what backs it. `initial` is the
 *  song-start placeholder shown when no flat event sits on the first bar's
 *  downbeat; editing it upserts that event (so it then renders as an `event`
 *  pill). `event` edits/deletes a flat `tempoEvent` by its stable id. */
export type BpmMarkerSource = { kind: 'initial' } | { kind: 'event'; id: string };

export type BpmMarker = {
  /** Lead-in-inclusive global beat (same space as the header `--bar-start-beat`). */
  globalBeat: number;
  /** Displayed (rounded) bpm. */
  bpm: number;
  source: BpmMarkerSource;
};

/** A resolved anchor for a new tempo change: a real source bar + snapped beat. */
type Anchor = { barId: string; barIndex: number; beat: number };

export class TempoEditPresenter {
  constructor(private readonly jotEditorStore: JotEditorStore) {
    makeObservable(this, { bpmMarkers: computed });
  }

  /**
   * The editable flat BPM pills, in lead-in-inclusive global-beat order: one
   * per flat `tempoEvent` (carrying its id) plus the initial-tempo pill at the
   * very start (unless a flat event already sits on the first bar's downbeat,
   * in which case that event IS the leading pill). Transition ramps are
   * skipped. Mirrors `TempoPresenter.tempoRamps`'s bar->global-beat anchoring.
   */
  get bpmMarkers(): BpmMarker[] {
    const structural = this.jotEditorStore.structural;
    const jot = this.jotEditorStore.jot;
    const bars = structural?.layers[0]?.bars;
    if (!structural || !jot || !bars) return [];

    const { sourceBarStart, barIndexById } = this.barIndex(bars);
    // The drums-enter bar's source index (lead-in bars sit before it); the
    // initial tempo event lives on that bar's downbeat, matching `tempoSource` /
    // `initialBpm`. 0 for a jot with no pre-roll.
    const leadBars = jot.leadBars ?? 0;
    const markers: BpmMarker[] = [];
    let hasLeadingEvent = false;
    for (const ev of jot.tempoEvents.values()) {
      if (typeof ev.bpm !== 'number') continue; // ramps stay read-only
      const barIndex = barIndexById.get(ev.barId);
      if (barIndex === undefined) continue;
      markers.push({
        globalBeat: sourceBarStart[barIndex] + ev.beat,
        bpm: Math.round(ev.bpm),
        source: { kind: 'event', id: ev.id },
      });
      if (barIndex === leadBars && ev.beat === 0) hasLeadingEvent = true;
    }
    if (!hasLeadingEvent) {
      // No event on the drums-enter downbeat: show the song-start tempo
      // (`initialBpm`, = DEFAULT_BPM when nothing's anchored there yet) as a
      // placeholder pill at the very start (before any lead-in); editing it
      // upserts the leading event on the drums-enter bar.
      markers.push({
        globalBeat: 0,
        bpm: Math.round(initialBpm(structural.tempoSource)),
        source: { kind: 'initial' },
      });
    }
    markers.sort((a, b) => a.globalBeat - b.globalBeat);
    return markers;
  }

  /** Whether a marker can be deleted (every flat event; not the initial pill). */
  canDelete = computedFn((source: BpmMarkerSource): boolean => source.kind === 'event');

  /**
   * Create a sticky tempo change at the bars-row pixel `x` the user invoked the
   * context menu on: snap to the nearest whole beat of the bar there, seed it
   * with the tempo currently in force (so the pill appears as a no-op the user
   * then types over), and return the new event's id so the caller can focus it
   * for editing. Returns the existing event's id instead when one already sits
   * on that anchor (edit it rather than duplicate). `undefined` if nothing's
   * loaded or the click can't be placed.
   */
  createTempoChangeAtX(x: number): string | undefined {
    const structural = this.jotEditorStore.structural;
    const jot = this.jotEditorStore.jot;
    if (!structural || !jot) return undefined;
    const pxPerBeat = structural.pxPerBeat;
    if (!(pxPerBeat > 0)) return undefined;
    const anchor = this.resolveAnchor(x / pxPerBeat);
    if (!anchor) return undefined;

    const existing = this.flatEventAt(anchor.barId, anchor.beat);
    if (existing) return existing;

    const bpm = Math.round(tempoAt(structural.tempoSource, anchor.barIndex, anchor.beat));
    const id = `t_${crypto.randomUUID()}`;
    runInAction(() => {
      jot.tempoEvents.set(id, { id, barId: anchor.barId, beat: anchor.beat, bpm });
    });
    return id;
  }

  /** Commit a pill's edited text: empty -> delete the event (or, for the
   *  initial pill, no-op); a valid integer -> clamp to [MIN_BPM, MAX_BPM] and
   *  save in place; anything else -> no-op (the caller reverts the display). */
  commitMarker(source: BpmMarkerSource, raw: string): void {
    const text = raw.trim();
    if (text === '') {
      if (source.kind === 'event') this.deleteEvent(source.id);
      return;
    }
    const value = Number.parseInt(text, 10);
    if (!Number.isFinite(value)) return;
    const bpm = clampBpm(value);
    if (source.kind === 'initial') this.setInitialBpm(bpm);
    else this.setEventBpm(source.id, bpm);
  }

  /** Mutate an existing tempo event's bpm in place (stable id; one CRDT op). */
  setEventBpm(id: string, bpm: number): void {
    const jot = this.jotEditorStore.jot;
    const ev = jot?.tempoEvents.get(id);
    if (!ev) return;
    runInAction(() => {
      ev.bpm = clampBpm(bpm);
    });
  }

  /** Set the song's initial tempo by upserting the flat event on the
   *  drums-enter bar's downbeat (the first rendered bar, `index >= 1`; the
   *  initial tempo IS that event). Skips any pre-roll lead-in bars (negative
   *  index), where the event would be invisible to `initialBpm`. One CRDT op. */
  setInitialBpm(bpm: number): void {
    const jot = this.jotEditorStore.jot;
    const bars = this.jotEditorStore.structural?.layers[0]?.bars;
    if (!jot || !bars) return;
    const first = bars.find((b) => b.index >= 1);
    if (!first) return;
    const clamped = clampBpm(bpm);
    runInAction(() => {
      const existingId = this.flatEventAt(first.id, 0);
      const existing = existingId ? jot.tempoEvents.get(existingId) : undefined;
      if (existing) {
        existing.bpm = clamped;
      } else {
        const id = `t_${crypto.randomUUID()}`;
        jot.tempoEvents.set(id, { id, barId: first.id, beat: 0, bpm: clamped });
      }
    });
  }

  /** Remove a sticky tempo change. */
  deleteEvent(id: string): void {
    const jot = this.jotEditorStore.jot;
    if (!jot) return;
    runInAction(() => {
      jot.tempoEvents.delete(id);
    });
  }

  /** The id of a flat tempo event already anchored at (`barId`, `beat`), if any. */
  private flatEventAt(barId: string, beat: number): string | undefined {
    const jot = this.jotEditorStore.jot;
    if (!jot) return undefined;
    for (const ev of jot.tempoEvents.values()) {
      if (typeof ev.bpm === 'number' && ev.barId === barId && Math.abs(ev.beat - beat) < 1e-6) {
        return ev.id;
      }
    }
    return undefined;
  }

  /** Map a global beat to a real source bar + snapped whole beat. Clicks in the
   *  view-only lead-in or an anacrusis snap to the first real bar's downbeat. */
  private resolveAnchor(globalBeat: number): Anchor | undefined {
    const bars = this.jotEditorStore.structural?.layers[0]?.bars;
    if (!bars || bars.length === 0) return undefined;
    const { barIndexById } = this.barIndex(bars);

    let cum = 0;
    let target: { id: string; start: number; beats: number; isSource: boolean } | undefined;
    for (const bar of bars) {
      const isSource = bar.id !== LEAD_IN_BAR_ID && !bar.anacrusis;
      if (globalBeat < cum + bar.beats || bar === bars[bars.length - 1]) {
        target = { id: bar.id, start: cum, beats: bar.beats, isSource };
        break;
      }
      cum += bar.beats;
    }
    if (!target) return undefined;

    if (target.isSource) {
      const barIndex = barIndexById.get(target.id);
      if (barIndex === undefined) return undefined;
      return { barId: target.id, barIndex, beat: snapBeat(globalBeat - target.start, target.beats) };
    }
    // Lead-in / anacrusis: anchor the first real source bar's downbeat.
    const first = bars.find((b) => b.id !== LEAD_IN_BAR_ID && !b.anacrusis);
    if (!first) return undefined;
    const barIndex = barIndexById.get(first.id);
    if (barIndex === undefined) return undefined;
    return { barId: first.id, barIndex, beat: 0 };
  }

  /** Source-bar index space (skips the synthetic lead-in + any anacrusis, so
   *  it matches `StructuralPresenter.tempoSource`), with each source bar's
   *  lead-in-inclusive global start beat. */
  private barIndex(bars: readonly { id: string; beats: number; anacrusis: boolean }[]): {
    sourceBarStart: number[];
    barIndexById: Map<string, number>;
  } {
    const sourceBarStart: number[] = [];
    const barIndexById = new Map<string, number>();
    let cum = 0;
    let idx = 0;
    for (const bar of bars) {
      if (bar.id !== LEAD_IN_BAR_ID && !bar.anacrusis) {
        barIndexById.set(bar.id, idx);
        sourceBarStart[idx] = cum;
        idx++;
      }
      cum += bar.beats;
    }
    return { sourceBarStart, barIndexById };
  }
}

/** Clamp to an integer in `[MIN_BPM, MAX_BPM]`. */
export function clampBpm(bpm: number): number {
  const n = Math.round(bpm);
  return n < MIN_BPM ? MIN_BPM : n > MAX_BPM ? MAX_BPM : n;
}

/** Snap a beat-within-bar to the nearest whole beat strictly inside the bar. */
function snapBeat(beat: number, barBeats: number): number {
  const maxBeat = Math.max(0, Math.ceil(barBeats) - 1);
  const snapped = Math.round(beat);
  return snapped < 0 ? 0 : snapped > maxBeat ? maxBeat : snapped;
}
