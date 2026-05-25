import { makeAutoObservable, runInAction } from 'mobx';

export type ThemeMode = 'system' | 'light' | 'dark';
export type ResolvedTheme = 'light' | 'dark';

const STORAGE_KEY = 'drumjot.theme';

function readSavedMode(): ThemeMode {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v === 'light' || v === 'dark' || v === 'system') return v;
  } catch {
    // localStorage unavailable (private mode, sandboxed iframe); fall through.
  }
  return 'system';
}

function systemPrefersDark(): boolean {
  if (typeof window === 'undefined') return false;
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}

/**
 * Singleton theme controller. Reads the user's last explicit choice from
 * localStorage; in `system` mode it tracks the OS `prefers-color-scheme`
 * media query live, so a system-level light↔dark switch (macOS auto
 * appearance, GNOME night light, etc.) flips the app's data-theme
 * attribute without a reload.
 *
 * The resolved theme is exposed as a `data-theme` attribute on
 * `<html>`. `src/design_tokens.css` defines a `:root[data-theme='dark']`
 * block that re-maps every color token; module CSS files reference
 * those tokens, so no per-component dark-mode CSS is needed.
 *
 * To avoid a flash-of-wrong-theme on first paint, `index.html` carries
 * a tiny synchronous bootstrap script that reads the same localStorage
 * key and sets `data-theme` before the JS bundle loads. This module
 * then takes over once it's evaluated and keeps the attribute in sync
 * with the live media-query state.
 */
class ThemeStore {
  mode: ThemeMode = readSavedMode();
  private systemIsDark: boolean = systemPrefersDark();

  constructor() {
    makeAutoObservable(this);
    if (typeof window !== 'undefined' && typeof window.matchMedia === 'function') {
      const mq = window.matchMedia('(prefers-color-scheme: dark)');
      mq.addEventListener('change', (e) => {
        runInAction(() => {
          this.systemIsDark = e.matches;
        });
        this.apply();
      });
    }
    this.apply();
  }

  get resolved(): ResolvedTheme {
    if (this.mode === 'system') return this.systemIsDark ? 'dark' : 'light';
    return this.mode;
  }

  setMode(mode: ThemeMode) {
    this.mode = mode;
    try {
      if (mode === 'system') localStorage.removeItem(STORAGE_KEY);
      else localStorage.setItem(STORAGE_KEY, mode);
    } catch {
      // Persistence is best-effort; the in-memory choice still works.
    }
    this.apply();
  }

  private apply() {
    if (typeof document === 'undefined') return;
    document.documentElement.dataset.theme = this.resolved;
  }
}

export const themeStore = new ThemeStore();
