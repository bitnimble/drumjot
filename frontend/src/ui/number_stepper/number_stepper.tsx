import classNames from 'classnames';
import React from 'react';
import { useBufferedStepper } from 'src/ui/stepper/use_buffered_stepper';
import styles from './number_stepper.module.css';

/**
 * Compact numeric up/down control. Replaces the native browser spinner
 * (inconsistent across engines, doesn't match the warm-paper design
 * tokens) with explicit [−] [input] [+] children inside one pill-shaped
 * border. The input remains a real `<input type="number">` so keyboard
 * up/down arrows and mousewheel-while-focused still nudge the value;
 * only the native spinner UI is hidden via CSS.
 *
 * State semantics mirror the original buffered inputs in
 * `playback.tsx::OffsetControl` and `lyrics_track_view.tsx::OffsetInput`: a
 * local text buffer (in {@link useBufferedStepper}) lets the user clear the
 * field or type a leading `-` without the parent's value clamp snapping
 * mid-keystroke, and the buffer reflects `value` whenever the input isn't
 * being edited (so a fresh jot resetting the offset shows immediately).
 *
 * Stepping (+/− buttons, keyboard up/down) operates on the current
 * committed `value`, not the text buffer, so the button always moves
 * by exactly one `step` from the truth. The result is clamped to
 * `[min, max]` if either bound is set.
 */
export const NumberStepper = ({
  value,
  onChange,
  step,
  min,
  max,
  precision = 2,
  ariaLabel,
  title,
  disabled,
  stopPropagation = true,
  testId,
  className,
}: {
  value: number;
  onChange: (v: number) => void;
  step: number;
  min?: number;
  max?: number;
  /** Decimal places for the displayed text. Default 2. */
  precision?: number;
  ariaLabel: string;
  title?: string;
  disabled?: boolean;
  /** Stop click/mousedown propagation. Needed when the stepper lives
   *  inside a marquee- or seek-on-click listening container. */
  stopPropagation?: boolean;
  testId?: string;
  className?: string;
}) => {
  const [editing, setEditing] = React.useState(false);
  const inputRef = React.useRef<HTMLInputElement>(null);
  const wrapperRef = React.useRef<HTMLDivElement>(null);

  // Defocus on mousedown outside the stepper. Native focus shift would
  // normally do this for free, but several parent containers
  // (score selection, mixer/minimap drag, etc.) call `preventDefault()`
  // on mousedown, which suppresses the browser's automatic blur. We
  // listen in the capture phase and call `.blur()` programmatically so
  // those `preventDefault` calls don't stop us; the side effect being
  // that spacebar-to-play resumes working as soon as the user clicks
  // anywhere outside the input.
  React.useEffect(() => {
    if (!editing) return;
    const onDocMouseDown = (e: MouseEvent) => {
      const wrapper = wrapperRef.current;
      if (!wrapper) return;
      if (wrapper.contains(e.target as Node)) return;
      inputRef.current?.blur();
    };
    document.addEventListener('mousedown', onDocMouseDown, true);
    return () => document.removeEventListener('mousedown', onDocMouseDown, true);
  }, [editing]);

  const clamp = (n: number): number => {
    let out = n;
    if (typeof min === 'number' && out < min) out = min;
    if (typeof max === 'number' && out > max) out = max;
    return out;
  };

  const stepBy = (sign: 1 | -1) => {
    if (disabled) return;
    const next = clamp(value + sign * step);
    if (next === value) return;
    onChange(next);
  };

  const buffered = useBufferedStepper({
    commit: (raw) => {
      const n = parseFloat(raw);
      if (!Number.isFinite(n)) return;
      onChange(clamp(n));
    },
    deriveDisplay: (text) => (text !== null ? text : value.toFixed(precision)),
    step: stepBy,
    // The native `<input type="number">` handles arrow keys itself.
    handleArrowKeys: false,
    commitOnChange: true,
  });

  const atMin = typeof min === 'number' && value <= min;
  const atMax = typeof max === 'number' && value >= max;

  const stopIfNeeded = (e: React.SyntheticEvent) => {
    if (stopPropagation) e.stopPropagation();
  };

  return (
    <div
      ref={wrapperRef}
      className={classNames(styles.stepper, className)}
      data-disabled={disabled ? 'true' : undefined}
      onClick={stopIfNeeded}
      onMouseDown={stopIfNeeded}
      title={title}
    >
      <button
        type="button"
        className={classNames(styles.stepperButton, styles.stepperButtonMinus)}
        onClick={() => stepBy(-1)}
        disabled={disabled || atMin}
        tabIndex={-1}
        aria-label={`Decrease ${ariaLabel}`}
        data-testid={testId ? `${testId}-minus` : undefined}
      >
        −
      </button>
      <input
        ref={inputRef}
        type="number"
        className={styles.stepperInput}
        min={min}
        max={max}
        step={step}
        value={buffered.display}
        disabled={disabled}
        onFocus={() => setEditing(true)}
        onBlur={(e) => {
          setEditing(false);
          buffered.onBlur(e);
        }}
        onChange={(e) => buffered.onChange(e.target.value)}
        onKeyDown={buffered.onKeyDown}
        aria-label={ariaLabel}
        data-testid={testId}
      />
      <button
        type="button"
        className={classNames(styles.stepperButton, styles.stepperButtonPlus)}
        onClick={() => stepBy(1)}
        disabled={disabled || atMax}
        tabIndex={-1}
        aria-label={`Increase ${ariaLabel}`}
        data-testid={testId ? `${testId}-plus` : undefined}
      >
        +
      </button>
    </div>
  );
};
