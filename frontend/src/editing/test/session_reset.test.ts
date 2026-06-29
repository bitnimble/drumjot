import { describe, expect, test } from 'bun:test';
import { SessionReset, type Resettable } from 'src/editing/session_reset';
import { EditingStore, type EditMode } from 'src/editing/editing_store';
import { SelectionStore } from 'src/editing/selection/selection';
import { ViewportStore } from 'src/editing/viewport/viewport_store';
import { JotEditorStore } from 'src/editing/jot_editor_store';
import { SettingsStore, DEFAULT_GRID_LINES } from 'src/settings/settings_store';

/** A {@link Resettable} whose `reset` records its call count + order. */
function spyResettable(order: string[], label: string): Resettable & { calls: number } {
  return {
    calls: 0,
    reset() {
      this.calls++;
      order.push(label);
    },
  };
}

describe('SessionReset registry', () => {
  test('fires every registered target exactly once per reset', () => {
    const order: string[] = [];
    const a = spyResettable(order, 'a');
    const b = spyResettable(order, 'b');
    const c = spyResettable(order, 'c');
    const registry = new SessionReset([a, b, c]);

    registry.reset();

    // Not 0, not more than 1, exactly one call apiece (the load-time guarantee).
    expect(a.calls).toBe(1);
    expect(b.calls).toBe(1);
    expect(c.calls).toBe(1);
    // In registration order.
    expect(order).toEqual(['a', 'b', 'c']);
  });

  test('a second load resets each target again (once more apiece)', () => {
    const order: string[] = [];
    const a = spyResettable(order, 'a');
    const registry = new SessionReset([a]);
    registry.reset();
    registry.reset();
    expect(a.calls).toBe(2);
  });

  test('an empty registry is a no-op', () => {
    expect(() => new SessionReset([]).reset()).not.toThrow();
  });
});

describe('pure-state store resets', () => {
  test('EditingStore.reset clears transient interaction state, keeps snapping pref', () => {
    const store = new EditingStore();
    store.mode = 'insert';
    store.placeholder = { lane: 'c', barId: 'b1', beat: 1, absBeat: 1, barBeats: 4 };
    store.dragActive = true;
    store.dragPreview = [{ id: 'e1', lane: 'c', absBeat: 2 }];
    store.snappingEnabled = false;

    store.reset();

    // `store.mode = 'insert'` narrows the property to the literal, which TS
    // keeps across the opaque `reset()` call; the cast widens it back to
    // EditMode so the post-reset 'select' assertion typechecks.
    expect(store.mode as EditMode).toBe('select');
    expect(store.placeholder).toBeUndefined();
    expect(store.dragActive).toBe(false);
    expect(store.dragPreview).toEqual([]);
    // Snapping is a workflow preference, not per-song state: it survives.
    expect(store.snappingEnabled).toBe(false);
  });

  test('SelectionStore.reset drops the selection + marquee bookkeeping', () => {
    const store = new SelectionStore();
    store.state = { type: 'pattern', name: 'verse' };
    store.reset();
    expect(store.state).toBeUndefined();
    expect(store.transientState).toBeUndefined();
    expect(store.marquee).toBeUndefined();
    expect(store.selectedPattern).toBeUndefined();
  });

  test('ViewportStore.reset scrolls to the origin but keeps zoom + gutter', () => {
    const store = new ViewportStore(new JotEditorStore());
    store.scrollX = 500;
    store.scrollY = 300;
    store.zoom = 2;
    store.gutterWidth = 200;

    store.reset();

    expect(store.scrollX).toBe(0);
    expect(store.scrollY).toBe(0);
    // Zoom + gutter are global view preferences: untouched by a load.
    expect(store.zoom).toBe(2);
    expect(store.gutterWidth).toBe(200);
  });

  test('SettingsStore.reset restores fresh-load display defaults', () => {
    const store = new SettingsStore();
    store.gridLines = {
      mainBeat: false,
      subBeat16: false,
      subBeatQuarterTriplet: true,
      subBeatTriplet: true,
      subBeat48: true,
    };
    store.uniformWaveforms = false;
    store.mergeLayers = true;

    store.reset();

    expect(store.gridLines).toEqual(DEFAULT_GRID_LINES);
    expect(store.uniformWaveforms).toBe(true);
    expect(store.mergeLayers).toBe(false);
    // A reset clone, not the shared default object.
    expect(store.gridLines).not.toBe(DEFAULT_GRID_LINES);
  });
});
