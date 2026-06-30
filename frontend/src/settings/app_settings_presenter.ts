import { autorun, makeAutoObservable } from 'mobx';
import { AppSettingsStore, type BackendMode } from './app_settings_store';

const STORAGE_KEY = 'drumjot.settings';

/**
 * Sole writer for {@link AppSettingsStore}, and the seam that loads the saved
 * settings on boot and persists every change to localStorage. Mirrors the
 * theme controller's persistence approach; kept as a store + presenter pair so
 * the settings logic is unit-testable against a mocked store.
 */
export class AppSettingsPresenter {
  readonly settings: AppSettingsStore;

  constructor(settings: AppSettingsStore) {
    this.settings = settings;
    makeAutoObservable(this, { settings: false });
    this.load();
    // Persist on any change. Best-effort: localStorage may be unavailable
    // (private mode, sandboxed context), in which case the in-memory value
    // still works for the session.
    autorun(() => {
      const snapshot = JSON.stringify({
        backendMode: this.settings.backendMode,
        transcriberUrl: this.settings.transcriberUrl,
      });
      try {
        localStorage.setItem(STORAGE_KEY, snapshot);
      } catch {
        // ignore
      }
    });
  }

  setBackendMode(mode: BackendMode): void {
    this.settings.backendMode = mode;
  }

  setTranscriberUrl(url: string): void {
    this.settings.transcriberUrl = url;
  }

  private load(): void {
    let raw: string | null = null;
    try {
      raw = localStorage.getItem(STORAGE_KEY);
    } catch {
      return;
    }
    if (raw == null) return;
    try {
      const parsed: unknown = JSON.parse(raw);
      if (typeof parsed !== 'object' || parsed == null) return;
      const obj = parsed as Record<string, unknown>;
      if (obj.backendMode === 'local' || obj.backendMode === 'hosted') {
        this.settings.backendMode = obj.backendMode;
      }
      if (typeof obj.transcriberUrl === 'string') {
        this.settings.transcriberUrl = obj.transcriberUrl;
      }
    } catch {
      // corrupt JSON; keep defaults
    }
  }
}

/** The one device-global settings store + presenter for the app. Reads in hot
 *  paths (`backendClient()`, `transcriberBase`) hit the store; UI mutations go
 *  through the presenter. */
export const appSettingsStore = new AppSettingsStore();
export const appSettingsPresenter = new AppSettingsPresenter(appSettingsStore);
