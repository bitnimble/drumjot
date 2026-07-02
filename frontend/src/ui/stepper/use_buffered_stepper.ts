import React from 'react';

/**
 * Shared buffered-input + keyboard plumbing for the two stepper controls
 * ({@link Stepper}, {@link NumberStepper}). Both hold a local text buffer
 * so a partial keystroke (a lone `-`, an empty field mid-edit) isn't
 * clobbered by the owner's value formatting/clamp, and both wire the same
 * bindings: Enter commits + blurs, Escape reverts + blurs, blur commits,
 * Arrow Up/Down nudges.
 *
 * The buffer is `string | null`: `null` means "reflect the committed
 * value" (the input shows {@link BufferedStepperConfig.deriveDisplay}(null));
 * a string is the in-progress edit. The hook owns only this generic
 * skeleton; each caller supplies its own value semantics via callbacks:
 *
 * - `commit(text)` turns the input text into a committed value (parse +
 *   clamp + `onChange`). Runs once per blur when the buffer was actually
 *   edited (Enter blurs to reach it); an untouched focus+blur commits
 *   nothing.
 * - `deriveDisplay(text)` renders the string shown in the input for a
 *   buffer state (formatting the committed value when `text` is `null`).
 * - `step(dir)` applies a +1/-1 nudge (absolute-and-clamped, or a
 *   direction callback). Fired by Arrow Up/Down when `handleArrowKeys` is
 *   set; a native `<input type="number">` handles arrows itself, so
 *   NumberStepper leaves it off.
 * - `commitOnChange` commits every keystroke (NumberStepper) vs buffering
 *   until Enter/blur (Stepper).
 *
 * Escape reverts (buffer → `null`, display falls back to the committed
 * value) and blurs WITHOUT committing the abandoned text: the blur it
 * triggers is flagged to skip its commit. A commit-on-keystroke caller
 * has already committed the last keystroke, so skipping here is a no-op
 * for it; a buffer-only caller correctly drops the uncommitted edit.
 */
export type BufferedStepperConfig = {
  commit: (text: string) => void;
  deriveDisplay: (text: string | null) => string;
  step: (dir: 1 | -1) => void;
  handleArrowKeys: boolean;
  commitOnChange: boolean;
};

export type BufferedStepper = {
  display: string;
  onChange: (raw: string) => void;
  onBlur: (e: React.FocusEvent<HTMLInputElement>) => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLInputElement>) => void;
};

export function useBufferedStepper(config: BufferedStepperConfig): BufferedStepper {
  const { commit, deriveDisplay, step, handleArrowKeys, commitOnChange } = config;
  const [text, setText] = React.useState<string | null>(null);
  const skipNextCommit = React.useRef(false);

  return {
    display: deriveDisplay(text),
    onChange: (raw) => {
      setText(raw);
      if (commitOnChange) commit(raw);
    },
    onBlur: (e) => {
      // Commit only a real edit: an untouched buffer (`text === null`,
      // e.g. focus then blur without typing) commits nothing, and Escape
      // flags the blur it triggers to skip even a touched buffer.
      const skip = skipNextCommit.current;
      skipNextCommit.current = false;
      if (!skip && text !== null) commit(e.currentTarget.value);
      setText(null);
    },
    onKeyDown: (e) => {
      if (e.key === 'Enter') {
        // Blur drives the single commit (via `onBlur`); committing here
        // too would double-fire it.
        e.preventDefault();
        e.currentTarget.blur();
      } else if (e.key === 'Escape') {
        e.preventDefault();
        skipNextCommit.current = true;
        setText(null);
        e.currentTarget.blur();
      } else if (handleArrowKeys && e.key === 'ArrowUp') {
        e.preventDefault();
        step(1);
      } else if (handleArrowKeys && e.key === 'ArrowDown') {
        e.preventDefault();
        step(-1);
      }
    },
  };
}
