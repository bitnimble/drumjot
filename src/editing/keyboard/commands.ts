import type { EditingPresenter } from 'src/editing/editing_presenter';
import type { PlaybackPresenter } from 'src/editing/playback/playback_presenter';

/**
 * Dependencies a command needs to run. The keymap dispatcher builds this from
 * the live presenters and hands it to the matched command, keeping commands
 * decoupled from how they're constructed.
 */
export type CommandContext = {
  editingPresenter: EditingPresenter;
  playbackPresenter: PlaybackPresenter;
};

/**
 * A named, enumerable editing action. The keyboard layer is deliberately split
 * from the action layer: keys map to command *ids* (see {@link keymap}), and
 * the command's `run` performs the action. This makes the set of actions
 * introspectable for a future "remap keys" settings UI and lets any binding
 * (key, context menu, button) invoke the same command.
 */
export type EditorCommand = {
  id: string;
  /** Human-readable name for a future keybinding settings UI. */
  label: string;
  run: (ctx: CommandContext) => void;
};

export const EDITOR_COMMANDS: readonly EditorCommand[] = [
  {
    id: 'deleteSelection',
    label: 'Delete selected notes',
    run: (ctx) => ctx.editingPresenter.deleteSelection(),
  },
  {
    id: 'togglePlayPause',
    label: 'Play / pause',
    run: (ctx) => void ctx.playbackPresenter.togglePlayPause(),
  },
];

export const COMMANDS_BY_ID: ReadonlyMap<string, EditorCommand> = new Map(
  EDITOR_COMMANDS.map((c) => [c.id, c])
);
