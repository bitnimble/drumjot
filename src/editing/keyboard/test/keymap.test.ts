import { describe, expect, it } from 'bun:test';
import { COMMANDS_BY_ID, EDITOR_COMMANDS, type CommandContext } from 'src/editing/keyboard/commands';
import {
  DEFAULT_KEYMAP,
  eventCombo,
  formatCombo,
  shortcutForCommand,
} from 'src/editing/keyboard/keymap';

/** A keydown-shaped object with no modifiers held; spread to override. */
const KEY = { ctrlKey: false, metaKey: false, altKey: false, shiftKey: false };

describe('eventCombo', () => {
  it('spells the space key as "Space"', () => {
    expect(eventCombo({ ...KEY, key: ' ', code: 'Space' })).toBe('Space');
  });

  it('uses the key name for unmodified keys', () => {
    expect(eventCombo({ ...KEY, key: 'Delete', code: 'Delete' })).toBe('Delete');
    expect(eventCombo({ ...KEY, key: 'Backspace', code: 'Backspace' })).toBe('Backspace');
  });

  it('prefixes Ctrl and Cmd alike as "Mod" so one binding serves both platforms', () => {
    expect(eventCombo({ ...KEY, key: 'z', code: 'KeyZ', ctrlKey: true })).toBe('Mod+z');
    expect(eventCombo({ ...KEY, key: 'z', code: 'KeyZ', metaKey: true })).toBe('Mod+z');
  });

  it('carries Shift as an explicit token and lower-cases the letter', () => {
    // Shift+Z reports key "Z"; the combo must still be Mod+Shift+z, not Mod+Z.
    expect(eventCombo({ ...KEY, key: 'Z', code: 'KeyZ', ctrlKey: true, shiftKey: true })).toBe(
      'Mod+Shift+z'
    );
  });

  it('orders modifiers Mod, Alt, Shift', () => {
    expect(
      eventCombo({ ...KEY, key: 'a', code: 'KeyA', metaKey: true, altKey: true, shiftKey: true })
    ).toBe('Mod+Alt+Shift+a');
  });
});

describe('formatCombo (menu shortcut display)', () => {
  it('renders an unmodified key as-is', () => {
    expect(formatCombo('Delete')).toBe('Delete');
  });

  it('uppercases the letter and renders the Mod prefix (platform-specific glyph)', () => {
    // Ctrl (⌘ on macOS) varies by platform, but the letter is always uppercased
    // and present, never the bare combo's lowercase `z`.
    const undo = formatCombo('Mod+z');
    expect(undo).toContain('Z');
    expect(undo).not.toContain('z');
  });

  it('includes a Shift token for a shifted combo', () => {
    const redo = formatCombo('Mod+Shift+z');
    expect(redo).toContain('Z');
    // ⇧ on macOS, "Shift" elsewhere, one or the other is present.
    expect(redo === '⌘⇧Z' || redo.includes('Shift')).toBe(true);
  });
});

describe('shortcutForCommand (from the registry, not hardcoded)', () => {
  it('returns the bound shortcut for a command id', () => {
    expect(shortcutForCommand('undo')).toContain('Z');
  });

  it('uses the FIRST binding when a command has several (redo: Shift+Z before Y)', () => {
    const redo = shortcutForCommand('redo')!;
    expect(redo).toContain('Z');
    expect(redo).not.toContain('Y');
  });

  it('reflects a custom keymap (a rebind shows the new key)', () => {
    expect(shortcutForCommand('undo', { 'Mod+u': 'undo' })).toContain('U');
  });

  it('returns undefined for an unbound command', () => {
    expect(shortcutForCommand('nonexistent')).toBeUndefined();
  });
});

describe('default keymap', () => {
  it('binds Delete and Backspace to deleteSelection', () => {
    expect(DEFAULT_KEYMAP.Delete).toBe('deleteSelection');
    expect(DEFAULT_KEYMAP.Backspace).toBe('deleteSelection');
  });

  it('binds Space to togglePlayPause', () => {
    expect(DEFAULT_KEYMAP.Space).toBe('togglePlayPause');
  });

  it('binds Mod+Z to undo and Mod+Shift+Z / Mod+Y to redo', () => {
    expect(DEFAULT_KEYMAP['Mod+z']).toBe('undo');
    expect(DEFAULT_KEYMAP['Mod+Shift+z']).toBe('redo');
    expect(DEFAULT_KEYMAP['Mod+y']).toBe('redo');
  });

  it('binds Mod+G to group and Mod+Shift+G to ungroup', () => {
    expect(DEFAULT_KEYMAP['Mod+g']).toBe('group');
    expect(DEFAULT_KEYMAP['Mod+Shift+g']).toBe('ungroup');
  });

  it('every bound command id resolves to a registered command', () => {
    for (const id of Object.values(DEFAULT_KEYMAP)) {
      expect(COMMANDS_BY_ID.has(id)).toBe(true);
    }
  });
});

describe('commands', () => {
  it('deleteSelection invokes the editing presenter', () => {
    let called = false;
    const ctx = {
      editingPresenter: { deleteSelection: () => (called = true) },
      playbackPresenter: { togglePlayPause: () => Promise.resolve() },
    } as unknown as CommandContext;
    COMMANDS_BY_ID.get('deleteSelection')!.run(ctx);
    expect(called).toBe(true);
  });

  it('togglePlayPause invokes the playback presenter', () => {
    let called = false;
    const ctx = {
      editingPresenter: { deleteSelection: () => {} },
      playbackPresenter: {
        togglePlayPause: () => {
          called = true;
          return Promise.resolve();
        },
      },
    } as unknown as CommandContext;
    COMMANDS_BY_ID.get('togglePlayPause')!.run(ctx);
    expect(called).toBe(true);
  });

  it('undo and redo invoke the history presenter', () => {
    const calls: string[] = [];
    const ctx = {
      historyPresenter: { undo: () => calls.push('undo'), redo: () => calls.push('redo') },
    } as unknown as CommandContext;
    COMMANDS_BY_ID.get('undo')!.run(ctx);
    COMMANDS_BY_ID.get('redo')!.run(ctx);
    expect(calls).toEqual(['undo', 'redo']);
  });

  it('group / ungroup invoke the editing presenter', () => {
    const calls: string[] = [];
    const ctx = {
      editingPresenter: {
        groupSelection: () => calls.push('group'),
        ungroupSelection: () => calls.push('ungroup'),
      },
      playbackPresenter: { togglePlayPause: () => Promise.resolve() },
    } as unknown as CommandContext;
    COMMANDS_BY_ID.get('group')!.run(ctx);
    COMMANDS_BY_ID.get('ungroup')!.run(ctx);
    expect(calls).toEqual(['group', 'ungroup']);
  });

  it('exposes commands as an enumerable list for a future remap UI', () => {
    expect(EDITOR_COMMANDS.map((c) => c.id).sort()).toEqual([
      'deleteSelection',
      'group',
      'redo',
      'togglePlayPause',
      'undo',
      'ungroup',
    ]);
  });
});
