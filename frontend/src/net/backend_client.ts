import { isTauri } from 'src/desktop/is_tauri';
import { type BackendClient } from './backend';
import { HttpBackendClient } from './http_backend';
import { SidecarBackendClient } from './sidecar_backend';

let cached: BackendClient | null = null;

/**
 * The backend client for this deployment: the Tauri sidecar in the desktop app,
 * HTTP otherwise. One instance, cached. Both impls are web-safe to construct
 * (the sidecar client loads its Tauri-only deps lazily), so picking here doesn't
 * pull plugin-fs into the web boot path.
 */
export function backendClient(): BackendClient {
  if (cached == null) {
    cached = isTauri() ? new SidecarBackendClient() : new HttpBackendClient();
  }
  return cached;
}
