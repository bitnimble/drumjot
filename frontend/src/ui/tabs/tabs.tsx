import classNames from 'classnames';
import React from 'react';
import styles from './tabs.module.css';

export type TabOption<T extends string> = {
  value: T;
  label: React.ReactNode;
  disabled?: boolean;
  title?: string;
  /** Optional data-testid stamped on the tab button so e2e can scope
   *  assertions without depending on the label text. */
  testId?: string;
};

type TabsProps<T extends string> = {
  options: ReadonlyArray<TabOption<T>>;
  value: T;
  onChange: (v: T) => void;
  ariaLabel: string;
};

/**
 * Generic tab strip used inside DropdownButton panels (today: the
 * Transcribe dropdown's New ↔ Resume switch). Renders a WAI-ARIA
 * tablist; left/right arrows move selection across enabled tabs
 * (wrapping at the ends), Home/End jump to the first/last enabled tab.
 *
 * Activation is automatic on focus change (the simpler of the two
 * patterns the WAI-ARIA tabs spec permits) so a keyboard user doesn't
 * need a separate Enter/Space press to commit; for the two-tab Transcribe
 * case this matches what mouse users get and avoids a "selected but not
 * active" intermediate state.
 */
export function Tabs<T extends string>({ options, value, onChange, ariaLabel }: TabsProps<T>) {
  const refs = React.useRef<Array<HTMLButtonElement | null>>([]);
  refs.current.length = options.length;

  const enabledIndices = options
    .map((opt, i) => (opt.disabled ? -1 : i))
    .filter((i) => i >= 0);

  const focusAt = (idx: number) => {
    const opt = options[idx];
    if (!opt || opt.disabled) return;
    onChange(opt.value);
    // Move focus to the newly-active tab so subsequent arrows continue
    // from the right starting point.
    refs.current[idx]?.focus();
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLButtonElement>, currentIdx: number) => {
    if (enabledIndices.length === 0) return;
    const pos = enabledIndices.indexOf(currentIdx);
    if (e.key === 'ArrowRight') {
      e.preventDefault();
      const next = enabledIndices[(pos + 1) % enabledIndices.length];
      focusAt(next);
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      const prev = enabledIndices[(pos - 1 + enabledIndices.length) % enabledIndices.length];
      focusAt(prev);
    } else if (e.key === 'Home') {
      e.preventDefault();
      focusAt(enabledIndices[0]);
    } else if (e.key === 'End') {
      e.preventDefault();
      focusAt(enabledIndices[enabledIndices.length - 1]);
    }
  };

  return (
    <div className={styles.tabs} role="tablist" aria-label={ariaLabel}>
      {options.map((opt, i) => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            ref={(el) => {
              refs.current[i] = el;
            }}
            type="button"
            role="tab"
            aria-selected={active}
            aria-disabled={opt.disabled || undefined}
            tabIndex={active ? 0 : -1}
            disabled={opt.disabled}
            title={opt.title}
            data-testid={opt.testId}
            className={classNames(styles.tab, active && styles.tabActive)}
            onClick={() => {
              if (opt.disabled) return;
              onChange(opt.value);
            }}
            onKeyDown={(e) => onKeyDown(e, i)}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
