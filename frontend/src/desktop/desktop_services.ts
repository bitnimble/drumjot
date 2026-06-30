import { CapabilityPresenter } from './capability_presenter';
import { CapabilityStore } from './capability_store';
import { TauriBridge } from './desktop_bridge';
import { isDesktopShell } from './platform';

let cached: { store: CapabilityStore; presenter: CapabilityPresenter } | null = null;

/**
 * The one shared capability store + presenter for the desktop shell, so the
 * first-run panel, the point-of-use install prompt, and the transcribe gate all
 * see the same install state. Returns null in the web build and on mobile (no
 * local sidecar / capability install there; transcription is HTTP-only).
 * Web-safe to import (only touches `@tauri-apps/api/core`); the sidecar backend
 * client loads its plugin-fs deps lazily.
 */
export function desktopCapabilities(): {
  store: CapabilityStore;
  presenter: CapabilityPresenter;
} | null {
  if (!isDesktopShell()) {
    return null;
  }
  if (cached == null) {
    const store = new CapabilityStore();
    cached = { store, presenter: new CapabilityPresenter({ store, bridge: new TauriBridge() }) };
    void cached.presenter.refresh();
  }
  return cached;
}
