import { makeAutoObservable } from 'mobx';

export type BackendMode = 'local' | 'hosted';

/** Compiled-in fallback for {@link AppSettingsStore.transcriberUrl}. Only the
 *  default is baked; the live value is user-editable + persisted. Dev / docker
 *  / e2e set `VITE_TRANSCRIBER_URL` (often empty → origin-relative `/api`, the
 *  edge-proxy path); production falls back to the hosted instance. */
const envDefault = import.meta.env.VITE_TRANSCRIBER_URL;
export const DEFAULT_TRANSCRIBER_URL =
  typeof envDefault === 'string' ? envDefault : 'https://drumjot.kumo.dev';

/**
 * Device-global app settings that persist across songs and app launches: the
 * transcription backend mode + the hosted transcriber's URL today, with room
 * for future global preferences. Distinct from the per-song {@link
 * import('./settings_store').SettingsStore}, which resets on every load and
 * serialises into the `.jot` file, these do neither.
 *
 * Data only: observables + read accessors. Writes and localStorage persistence
 * live on {@link import('./app_settings_presenter').AppSettingsPresenter}.
 */
export class AppSettingsStore {
  /**
   * Desktop transcription transport: `local` runs the bundled Python sidecar,
   * `hosted` calls the remote HTTP backend (useful on a desktop without GPU
   * acceleration). Forced to `hosted` on web + mobile, which have no sidecar
   * (the selection lives in `backendClient()`), so this field only steers the
   * desktop choice.
   */
  backendMode: BackendMode = 'local';

  /**
   * Origin of the hosted transcriber (scheme + host, no trailing `/api`). An
   * empty string means origin-relative, i.e. the `/api` the dev / docker edge
   * proxy serves on the app's own origin.
   */
  transcriberUrl: string = DEFAULT_TRANSCRIBER_URL;

  constructor() {
    makeAutoObservable(this);
  }

  /** `<origin>/api`, the base every backend request composes against. An empty
   *  origin collapses to the origin-relative `/api`. */
  get apiBase(): string {
    const origin = this.transcriberUrl.trim().replace(/\/+$/, '');
    return `${origin}/api`;
  }
}
