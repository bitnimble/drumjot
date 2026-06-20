import { describe, expect, it } from 'bun:test';
import { COMMANDS_BY_ID, EDITOR_COMMANDS, type CommandContext } from 'src/editing/keyboard/commands';
import { DEFAULT_KEYMAP, eventCombo } from 'src/editing/keyboard/keymap';

const NO_MODS = { ctrlKey: false, metaKey: false, shiftKey: false } as const;

describe('eventCombo', () => {
  it('spells the space key as "Space"', () => {
    expect(eventCombo({ key: ' ', code: 'Space', ...NO_MODS })).toBe('Space');
  });

  it('uses the key name for other keys', () => {
    expect(eventCombo({ key: 'Delete', code: 'Delete', ...NO_MODS })).toBe('Delete');
    expect(eventCombo({ key: 'Backspace', code: 'Backspace', ...NO_MODS })).toBe('Backspace');
  });

  it('spells a Ctrl/⌘ letter chord as Ctrl+<lowercase>, regardless of platform/case', () => {
    expect(eventCombo({ key: 'g', code: 'KeyG', ...NO_MODS, ctrlKey: true })).toBe('Ctrl+g');
    expect(eventCombo({ key: 'g', code: 'KeyG', ...NO_MODS, metaKey: true })).toBe('Ctrl+g');
    // Shift uppercases `key` to 'G'; the chord still normalizes to lowercase.
    expect(eventCombo({ key: 'G', code: 'KeyG', ...NO_MODS, ctrlKey: true, shiftKey: true })).toBe(
      'Ctrl+Shift+g'
    );
  });

  it('does not prefix a bare key when only Shift is held', () => {
    expect(eventCombo({ key: 'Delete', code: 'Delete', ...NO_MODS, shiftKey: true })).toBe('Delete');
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

  it('binds Ctrl+G to group and Ctrl+Shift+G to ungroup', () => {
    expect(DEFAULT_KEYMAP['Ctrl+g']).toBe('group');
    expect(DEFAULT_KEYMAP['Ctrl+Shift+g']).toBe('ungroup');
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
      'togglePlayPause',
      'ungroup',
    ]);
  });
});
