import { ChevronDown, ChevronUp } from 'lucide-react';
import React from 'react';
import styles from './bar_beat_field.module.css';

type Seg = 'bar' | 'beat';

type BarBeatFieldProps = {
  /** 1-based bar number, or `null` when a multi-selection's bars disagree. */
  bar: number | null;
  /** 1-based beat within the bar, or `null` when they disagree. */
  beat: number | null;
  /** Step the focused segment. Direction only, the owner applies the delta
   *  (beat by a musical step, bar by 1) and carries overflow across bars. */
  onStepBar: (dir: 1 | -1) => void;
  onStepBeat: (dir: 1 | -1) => void;
  /** Commit a typed absolute value. Omit to make the field step-only. */
  onSetBar?: (v: number) => void;
  onSetBeat?: (v: number) => void;
  disabled?: boolean;
  testId?: string;
};

function fmtBeat(beat: number): string {
  return beat % 1 === 0 ? String(beat) : String(Number(beat.toFixed(2)));
}

/**
 * A single field holding two numeric segments. Bar and Beat, like an
 * `hh:mm` time input, with the labels aligned above each segment and one
 * up/down pair that steps whichever segment is focused (also via ArrowUp/Down).
 * Either segment shows `--` when its value is `null` (a mixed multi-selection).
 * Overflow (beat past the bar's length → next bar) is the owner's job; this
 * control only reports step direction and typed values.
 */
export const BarBeatField = ({
  bar,
  beat,
  onStepBar,
  onStepBeat,
  onSetBar,
  onSetBeat,
  disabled = false,
  testId,
}: BarBeatFieldProps) => {
  const [focused, setFocused] = React.useState<Seg>('beat');
  const [barText, setBarText] = React.useState<string | null>(null);
  const [beatText, setBeatText] = React.useState<string | null>(null);

  const step = (seg: Seg, dir: 1 | -1) => (seg === 'bar' ? onStepBar(dir) : onStepBeat(dir));

  const barDisplay = barText !== null ? barText : bar === null ? '' : String(bar);
  const beatDisplay = beatText !== null ? beatText : beat === null ? '' : fmtBeat(beat);

  const commit = (seg: Seg) => {
    const text = seg === 'bar' ? barText : beatText;
    const set = seg === 'bar' ? onSetBar : onSetBeat;
    if (text !== null) {
      const trimmed = text.trim();
      const n = Number(trimmed);
      if (trimmed !== '' && Number.isFinite(n)) set?.(n);
    }
    if (seg === 'bar') setBarText(null);
    else setBeatText(null);
  };

  const segInput = (seg: Seg) => {
    const editable = seg === 'bar' ? !!onSetBar : !!onSetBeat;
    const setText = seg === 'bar' ? setBarText : setBeatText;
    return {
      className: styles.segment,
      value: seg === 'bar' ? barDisplay : beatDisplay,
      placeholder: '--',
      inputMode: 'decimal' as const,
      disabled,
      readOnly: !editable,
      'aria-label': seg === 'bar' ? 'Bar' : 'Beat',
      'data-testid': testId ? `${testId}-${seg}` : undefined,
      onFocus: (e: React.FocusEvent<HTMLInputElement>) => {
        setFocused(seg);
        e.currentTarget.select();
      },
      onChange: (e: React.ChangeEvent<HTMLInputElement>) => editable && setText(e.target.value),
      onBlur: () => commit(seg),
      onKeyDown: (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter') {
          commit(seg);
          e.currentTarget.blur();
        } else if (e.key === 'Escape') {
          setText(null);
          e.currentTarget.blur();
        } else if (e.key === 'ArrowUp') {
          e.preventDefault();
          step(seg, 1);
        } else if (e.key === 'ArrowDown') {
          e.preventDefault();
          step(seg, -1);
        }
      },
    };
  };

  return (
    <div className={styles.wrap} data-disabled={disabled || undefined} data-testid={testId}>
      <span className={styles.label} style={{ gridArea: '1 / 1' }}>
        Bar
      </span>
      <span className={styles.label} style={{ gridArea: '1 / 3' }}>
        Beat
      </span>
      <div className={styles.fieldBg} />
      <input {...segInput('bar')} />
      <span className={styles.divider} />
      <input {...segInput('beat')} />
      <div className={styles.steppers}>
        <button
          type="button"
          className={styles.stepBtn}
          aria-label="Increase"
          disabled={disabled}
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => step(focused, 1)}
        >
          <ChevronUp size={12} />
        </button>
        <button
          type="button"
          className={styles.stepBtn}
          aria-label="Decrease"
          disabled={disabled}
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => step(focused, -1)}
        >
          <ChevronDown size={12} />
        </button>
      </div>
    </div>
  );
};
