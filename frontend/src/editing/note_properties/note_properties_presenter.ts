import { makeAutoObservable } from 'mobx';
import type { SelectionStore } from 'src/editing/selection/selection';
import type { JotEditorStore } from 'src/editing/jot_editor_store';
import type { LayersPresenter } from 'src/editing/layers/layers_presenter';
import type { Modifier, NoteElement, Sticking } from 'src/schema/schema';
import { layerIdOfTrack } from 'src/schema/ordering';
import {
  BEAT_STEP,
  MICRO_TIMING_STEP_MS,
  ROLL_DISABLED_MODIFIERS,
  uiToVelocity,
  velocityToUi,
} from './note_properties_fields';

type Bar = { id: string; index: number; beats: number };

const EPS = 1e-6;

/**
 * The only writer of the selected notes' editable fields. Edits apply to the
 * whole committed selection at once (one Loro commit via `elements.setAll`).
 * Numeric "step" actions nudge each note independently (so a mixed selection
 * keeps its spread); "set" actions assign a shared value to all.
 */
export class NotePropertiesPresenter {
  constructor(
    private readonly jotEditorStore: JotEditorStore,
    private readonly selection: SelectionStore,
    private readonly layersPresenter: LayersPresenter
  ) {
    makeAutoObservable<this, 'jotEditorStore' | 'selection' | 'layersPresenter'>(this, {
      jotEditorStore: false,
      selection: false,
      layersPresenter: false,
    });
  }

  // ---------- lane ----------

  /** Re-home every selected note onto `lane`, minting/﻿finding the instrument
   *  track for the note's layer (mirrors a cross-lane drag). */
  setLane(lane: string): void {
    const jot = this.jotEditorStore.jot;
    const structural = this.jotEditorStore.structural;
    if (!jot) return;
    this.commit((el) => {
      if (el.lane === lane) return undefined;
      const curLayer = el.trackId !== undefined ? layerIdOfTrack(jot, el.trackId) : undefined;
      const targetLayer = structural?.ownerLayerFor(lane) ?? curLayer;
      const trackId =
        targetLayer !== undefined
          ? this.layersPresenter.ensureInstrumentTrack(targetLayer, lane)
          : undefined;
      return { lane, ...(trackId !== undefined ? { trackId } : {}) };
    });
  }

  // ---------- bar / beat ----------

  /** Nudge every note's beat by one step, carrying overflow into the adjacent
   *  bar (past the last beat -> next bar's downbeat, and vice versa). */
  stepBeat(dir: 1 | -1): void {
    const bars = this.bars();
    this.commit((el) => this.movedBeat(el, dir * BEAT_STEP, bars));
  }

  /** Set every note's beat to `displayBeat` (1-based), clamped within its bar. */
  setBeat(displayBeat: number): void {
    const bars = this.bars();
    this.commit((el) => {
      const bar = el.barId !== undefined ? bars.find((b) => b.id === el.barId) : undefined;
      if (!bar) return undefined;
      const beat = clamp(roundStep(displayBeat - 1), 0, bar.beats - BEAT_STEP);
      return { beat };
    });
  }

  /** Move every note one bar earlier/later, keeping its beat (clamped). */
  stepBar(dir: 1 | -1): void {
    const bars = this.bars();
    this.commit((el) => this.movedBar(el, dir, bars));
  }

  /** Move every note to bar `displayIndex` (1-based), keeping its beat. */
  setBar(displayIndex: number): void {
    const bars = this.bars();
    const dest = bars.find((b) => b.index === displayIndex);
    if (!dest) return;
    this.commit((el) => ({ barId: dest.id, beat: clamp(el.beat, 0, dest.beats - BEAT_STEP) }));
  }

  // ---------- volume ----------

  /** Assign a shared 0-10 volume (stored as a 0-127 velocity) to all. */
  setVolume(ui: number): void {
    const velocity = uiToVelocity(ui);
    this.commit(() => ({ velocity }));
  }

  /** Nudge each note's volume by one UI step, independently. */
  stepVolume(dir: 1 | -1): void {
    this.commit((el) => ({ velocity: uiToVelocity(velocityToUi(el.velocity) + dir) }));
  }

  // ---------- micro timing ----------

  setMicroTiming(ms: number): void {
    const offsetMs = Math.round(ms);
    this.commit(() => ({ offsetMs: offsetMs === 0 ? undefined : offsetMs }));
  }

  stepMicroTiming(dir: 1 | -1): void {
    this.commit((el) => {
      const next = (el.offsetMs ?? 0) + dir * MICRO_TIMING_STEP_MS;
      return { offsetMs: next === 0 ? undefined : next };
    });
  }

  // ---------- roll ----------

  /** Tri-state Roll: all-on -> all-off, otherwise -> all-on. Enabling Roll
   *  drops the modifiers it's incompatible with. */
  toggleRoll(): void {
    const turnOn = !this.allRoll();
    this.commit((el) => ({
      roll: turnOn ? true : undefined,
      ...(turnOn
        ? { modifiers: el.modifiers.filter((m) => !ROLL_DISABLED_MODIFIERS.has(m)) }
        : {}),
    }));
  }

  // ---------- modifiers ----------

  /** Tri-state a modifier across the selection: if every (eligible) note has
   *  it -> remove from all, otherwise -> add to all where it's valid. */
  toggleModifier(mod: Modifier): void {
    const allOn = this.selectedElements().every((el) => el.modifiers.includes(mod));
    this.commit((el) => {
      const has = el.modifiers.includes(mod);
      if (allOn) {
        return has ? { modifiers: el.modifiers.filter((m) => m !== mod) } : undefined;
      }
      return has ? undefined : { modifiers: [...el.modifiers, mod] };
    });
  }

  // ---------- sticking ----------

  setSticking(value: Sticking | 'none'): void {
    this.commit(() => ({ sticking: value === 'none' ? undefined : value }));
  }

  // ---------- internals ----------

  private bars(): Bar[] {
    return (this.jotEditorStore.structural?.layers[0]?.bars ?? []).map((b) => ({
      id: b.id,
      index: b.index,
      beats: b.beats,
    }));
  }

  private movedBeat(el: NoteElement, delta: number, bars: Bar[]): Partial<NoteElement> | undefined {
    const idx = bars.findIndex((b) => b.id === el.barId);
    if (idx < 0) return undefined;
    const cur = bars[idx];
    const next = roundStep(el.beat + delta);
    if (next >= cur.beats - EPS) {
      const after = bars[idx + 1];
      if (!after) return { beat: roundStep(cur.beats - BEAT_STEP) }; // clamp at end
      return { barId: after.id, beat: roundStep(next - cur.beats) };
    }
    if (next < -EPS) {
      const before = bars[idx - 1];
      if (!before) return { beat: 0 }; // clamp at start
      return { barId: before.id, beat: roundStep(before.beats - BEAT_STEP) };
    }
    return { beat: next };
  }

  private movedBar(el: NoteElement, dir: 1 | -1, bars: Bar[]): Partial<NoteElement> | undefined {
    const idx = bars.findIndex((b) => b.id === el.barId);
    const dest = idx >= 0 ? bars[idx + dir] : undefined;
    if (!dest) return undefined;
    return { barId: dest.id, beat: clamp(el.beat, 0, dest.beats - BEAT_STEP) };
  }

  private allRoll(): boolean {
    const els = this.selectedElements();
    return els.length > 0 && els.every((el) => el.roll === true);
  }

  private selectedElements(): NoteElement[] {
    const jot = this.jotEditorStore.jot;
    if (!jot) return [];
    const out: NoteElement[] = [];
    for (const n of this.selection.selectedNotes) {
      const el = jot.elements.get(n.id);
      if (el && el.kind === 'note') out.push(el as NoteElement);
    }
    return out;
  }

  /** Apply `patch(el)` to every selected note in one commit. A `patch` of
   *  `undefined` skips that note; `undefined` patch fields delete the key. */
  private commit(patch: (el: NoteElement) => Partial<NoteElement> | undefined): void {
    const jot = this.jotEditorStore.jot;
    if (!jot) return;
    const updates: [string, Record<string, unknown>][] = [];
    for (const n of this.selection.selectedNotes) {
      const el = jot.elements.get(n.id);
      if (!el || el.kind !== 'note') continue;
      const p = patch(el as NoteElement);
      if (!p) continue;
      const next: Record<string, unknown> = { ...el, ...p };
      for (const k of Object.keys(next)) if (next[k] === undefined) delete next[k];
      updates.push([n.id, next]);
    }
    if (updates.length > 0) jot.elements.setAll(updates);
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(Math.max(v, lo), Math.max(lo, hi));
}

function roundStep(beat: number): number {
  return Math.round(beat / BEAT_STEP) * BEAT_STEP;
}
