import { isTauri } from 'src/desktop/is_tauri';
import { isMobile } from 'src/desktop/platform';
import { appSettingsStore } from 'src/settings/app_settings_presenter';
import { type BackendClient } from './backend';
import { HttpBackendClient } from './http_backend';
import { SidecarBackendClient } from './sidecar_backend';

let httpCached: HttpBackendClient | null = null;
let sidecarCached: SidecarBackendClient | null = null;

/**
 * The backend client for the current deployment and the user's backend choice:
 *
 * - web + mobile: always HTTP (there is no local Python sidecar there).
 * - desktop: the bundled sidecar by default, or the hosted HTTP backend when
 *   the user selects it in Settings → Advanced (e.g. a machine with no GPU).
 *
 * Resolved per call (the two clients are cached, not the choice) so toggling
 * Local ↔ Hosted takes effect immediately, without an app restart. Both impls
 * are web-safe to construct (the sidecar client loads its Tauri-only deps
 * lazily), so picking here pulls nothing extra into the web boot path.
 */
export function backendClient(): BackendClient {
  if (!isTauri() || isMobile) return http();
  return appSettingsStore.backendMode === 'hosted' ? http() : sidecar();
}

function http(): HttpBackendClient {
  return (httpCached ??= new HttpBackendClient());
}

function sidecar(): SidecarBackendClient {
  return (sidecarCached ??= new SidecarBackendClient());
}
