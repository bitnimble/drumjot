import { isTauri as tauriIsTauri } from '@tauri-apps/api/core';

/** True when running inside the Tauri desktop shell (vs the plain web build).
 *  Guards every desktop-only call path so the web bundle degrades cleanly. */
export function isTauri(): boolean {
  try {
    return tauriIsTauri();
  } catch {
    return false;
  }
}
