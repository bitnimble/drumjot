import { describe, expect, it } from 'bun:test';
import { COMMANDS_BY_ID, EDITOR_COMMANDS, type CommandContext } from 'src/editing/keyboard/commands';
import { DEFAULT_KEYMAP, eventCombo } from 'src/editing/keyboard/keymap';

describe('eventCombo', () => {
  it('spells the space key as "Space"', () => {
    expect(eventCombo({ key: ' ', code: 'Space' })).toBe('Space');
  });

  it('uses the key name for other keys', () => {
    expect(eventCombo({ key: 'Delete', code: 'Delete' })).toBe('Delete');
    expect(eventCombo({ key: 'Backspace', code: 'Backspace' })).toBe('Backspace');
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

  it('exposes commands as an enumerable list for a future remap UI', () => {
    expect(EDITOR_COMMANDS.map((c) => c.id).sort()).toEqual([
      'deleteSelection',
      'togglePlayPause',
    ]);
  });
});
