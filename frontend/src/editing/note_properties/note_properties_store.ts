import { computed, makeObservable } from 'mobx';
import type { SelectionStore } from 'src/editing/selection/selection';
import type { JotEditorStore } from 'src/editing/jot_editor_store';
import type { Element, Modifier, NoteElement, Sticking } from 'src/schema/schema';
import { defaultKindForLane, isValidModifier } from 'src/instruments/instruments';
import {
  MODIFIER_FIELDS,
  ROLL_DISABLED_MODIFIERS,
  velocityToUi,
  VOLUME_UI_LABELS,
} from './note_properties_fields';

/** A field's aggregate state across the selection: a shared value, or `mixed`
 *  when members disagree. `null` value carries through (e.g. an unresolved
 *  bar). The view renders `mixed` as `--`. */
export type Mixed = 'mixed';
export const MIXED: Mixed = 'mixed';

export type TristateValue = boolean | Mixed;

export type ModifierRow = {
  mod: Modifier;
  label: string;
  /** Whether all / none / some selected notes carry the modifier. */
  state: TristateValue;
  /** False when the modifier is irrelevant to a selected note's lane, or
   *  disabled because Roll is active. */
  enabled: boolean;
};

/**
 * Reactive read-model for the Note properties editor: the current selection's
 * editable fields, aggregated across one OR many notes. A field whose members
 * disagree reports {@link MIXED} so the view can show `--`/an indeterminate
 * control. Reads the selection + jot; writes nothing (every
 * mutation is on {@link NotePropertiesPresenter}).
 */
export class NotePropertiesStore {
  constructor(
    private readonly selection: SelectionStore,
    private readonly jotEditorStore: JotEditorStore
  ) {
    makeObservable(this, {
      selectedElements: computed,
      count: computed,
      noteIdLabel: computed,
      lane: computed,
      availableLanes: computed,
      bar: computed,
      beat: computed,
      volumeUi: computed,
      volumeLabel: computed,
      microTiming: computed,
      roll: computed,
      rollActive: computed,
      stickingValues: computed,
      modifierRows: computed,
      articulationSummary: computed,
    });
  }

  /** The selected NoteElements (the editable truth, carrying `barId`/`beat`),
   *  resolved from the committed selection's stable ids. */
  get selectedElements(): NoteElement[] {
    const jot = this.jotEditorStore.jot;
    if (!jot) return [];
    const out: NoteElement[] = [];
    for (const n of this.selection.selectedNotes) {
      const el = jot.elements.get(n.id) as Element | undefined;
      if (el && el.kind === 'note') out.push(el);
    }
    return out;
  }

  get count(): number {
    return this.selectedElements.length;
  }

  /** The id line under the panel header: the note's id for one, a count hint
   *  for many, undefined when nothing is selected. */
  get noteIdLabel(): string | undefined {
    const els = this.selectedElements;
    if (els.length === 0) return undefined;
    if (els.length === 1) return `id: ${els[0].id}`;
    return '(multiple notes selected)';
  }

  /** The shared lane id, or {@link MIXED}. */
  get lane(): string | Mixed | undefined {
    return this.common(this.selectedElements.map((e) => e.lane));
  }

  /** Lanes available to assign to, from the jot's instrument mapping. */
  get availableLanes(): { lane: string; name: string }[] {
    const jot = this.jotEditorStore.jot;
    if (!jot) return [];
    const out: { lane: string; name: string }[] = [];
    for (const [lane, inst] of jot.instruments.entries()) {
      out.push({ lane, name: inst.name ?? `Lane ${lane.toUpperCase()}` });
    }
    return out;
  }

  /** Shared 1-based bar number, or {@link MIXED}. */
  get bar(): number | Mixed | undefined {
    const info = this.barInfo;
    const els = this.selectedElements;
    if (els.length === 0) return undefined;
    const idxs = els.map((e) => (e.barId !== undefined ? info.get(e.barId)?.index : undefined));
    if (idxs.some((i) => i === undefined)) return MIXED;
    return this.common(idxs as number[]);
  }

  /** Shared 1-based beat within the bar, or {@link MIXED}. */
  get beat(): number | Mixed | undefined {
    return this.common(this.selectedElements.map((e) => round(e.beat + 1)));
  }

  /** Shared volume on the 0-10 UI scale, or {@link MIXED}. */
  get volumeUi(): number | Mixed | undefined {
    return this.common(this.selectedElements.map((e) => velocityToUi(e.velocity)));
  }

  /** Dynamic marker (pp..ff) for the shared volume step, if it lands on one. */
  get volumeLabel(): string | undefined {
    const v = this.volumeUi;
    return typeof v === 'number' ? VOLUME_UI_LABELS[v] : undefined;
  }

  /** Shared micro-timing offset (ms), or {@link MIXED}. */
  get microTiming(): number | Mixed | undefined {
    return this.common(this.selectedElements.map((e) => e.offsetMs ?? 0));
  }

  /** Roll: all on / all off / {@link MIXED} (off when nothing is selected). */
  get roll(): TristateValue {
    return this.tristate(this.selectedElements.map((e) => e.roll === true));
  }

  /** Whether every selected note has Roll on (drives modifier disabling). */
  get rollActive(): boolean {
    const els = this.selectedElements;
    return els.length > 0 && els.every((e) => e.roll === true);
  }

  /** The set of sticking values present across the selection (`none` for an
   *  unset note), so the radio can light up every value in a mixed selection. */
  get stickingValues(): ReadonlySet<Sticking | 'none'> {
    const out = new Set<Sticking | 'none'>();
    for (const e of this.selectedElements) out.add(e.sticking ?? 'none');
    return out;
  }

  /** Per-modifier checkbox state + whether it applies to the selection. */
  get modifierRows(): ModifierRow[] {
    const els = this.selectedElements;
    const kinds = els.map((e) => this.kindOf(e.lane));
    const rollOn = this.rollActive;
    return MODIFIER_FIELDS.map(({ mod, label }) => {
      const validForAll = els.length > 0 && kinds.every((k) => isValidModifier(k, mod));
      const enabled = validForAll && !(rollOn && ROLL_DISABLED_MODIFIERS.has(mod));
      return {
        mod,
        label,
        state: this.tristate(els.map((e) => e.modifiers.includes(mod))),
        enabled,
      };
    });
  }

  /** Comma-joined labels of the articulations that are ON across the whole
   *  selection (Roll first, then modifiers in display order). Mixed/off ones
   *  are omitted. Drives the collapsed articulation dropdown's text. */
  get articulationSummary(): string {
    const parts: string[] = [];
    if (this.roll === true) parts.push('Roll');
    for (const row of this.modifierRows) if (row.state === true) parts.push(row.label);
    return parts.join(', ');
  }

  // ---------- internals ----------

  /** barId -> { 1-based index, length in beats }, from the shared bar grid. */
  private get barInfo(): Map<string, { index: number; beats: number }> {
    const bars = this.jotEditorStore.jot?.renderedLayers[0]?.bars ?? [];
    const map = new Map<string, { index: number; beats: number }>();
    for (const b of bars) map.set(b.id, { index: b.index, beats: b.beats });
    return map;
  }

  private kindOf(lane: string) {
    return this.jotEditorStore.jot?.instruments.get(lane)?.kind ?? defaultKindForLane(lane);
  }

  /** Shared value across a non-empty list, else {@link MIXED}; undefined when
   *  the list is empty. */
  private common<T>(values: T[]): T | Mixed | undefined {
    if (values.length === 0) return undefined;
    const first = values[0];
    return values.every((v) => Object.is(v, first)) ? first : MIXED;
  }

  /** Reduce booleans to all-true / all-false / {@link MIXED}. */
  private tristate(values: boolean[]): TristateValue {
    if (values.length === 0) return false;
    if (values.every((v) => v)) return true;
    if (values.every((v) => !v)) return false;
    return MIXED;
  }
}

/** Round to a clean grid multiple, killing float drift from beat arithmetic. */
function round(beat: number): number {
  return Math.round(beat * 1e6) / 1e6;
}
