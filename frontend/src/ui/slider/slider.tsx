import React from 'react';

/**
 * Horizontal range slider. Wraps `<input type="range">` and folds in the two
 * things every fader in the app repeated by hand: the `--value` custom
 * property (0..1, normalised from value/min/max) that drives the "filled to
 * the thumb" track paint, and swallowing mouse events so a drag doesn't bubble
 * into the page-level marquee selection or the seek-on-click handler.
 *
 * Visual chrome (track height, thumb) comes from the global
 * `input[type='range']` rules; pass `className` for per-site geometry (width).
 */
export const Slider = ({
  value,
  onChange,
  min = 0,
  max = 1,
  step,
  className,
  ariaLabel,
  title,
  disabled,
  /** Stop mouse events from bubbling (needed inside the mixer / seek areas). */
  stopPropagation = true,
  testId,
}: {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  className?: string;
  ariaLabel?: string;
  title?: string;
  disabled?: boolean;
  stopPropagation?: boolean;
  testId?: string;
}) => {
  const stop = stopPropagation ? (e: React.MouseEvent) => e.stopPropagation() : undefined;
  const fraction = max > min ? (value - min) / (max - min) : 0;
  return (
    <input
      type="range"
      className={className}
      min={min}
      max={max}
      step={step}
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      onClick={stop}
      onMouseDown={stop}
      onMouseUp={stop}
      aria-label={ariaLabel}
      title={title}
      data-testid={testId}
      style={{ ['--value' as string]: fraction } as React.CSSProperties}
    />
  );
};
