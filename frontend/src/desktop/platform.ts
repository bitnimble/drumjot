import { isTauri } from './is_tauri';

// Vite replaces `__IS_MOBILE__` at build time (see vite.config.ts), set from
// Tauri's `TAURI_ENV_PLATFORM`. The `typeof` guard keeps it safe under `bun
// test`, where there is no Vite define and the identifier is unbound.
declare const __IS_MOBILE__: boolean;

/** True when this build targets a mobile platform (Android / iOS). On mobile
 *  there is no Python sidecar and no capability install: transcription runs
 *  over the HTTP backend, same as the web build. */
export const isMobile: boolean =
  typeof __IS_MOBILE__ !== 'undefined' ? __IS_MOBILE__ : false;

/** True only inside the desktop Tauri shell (not mobile, not the web build),
 *  i.e. where the local sidecar + capability system are available. */
export function isDesktopShell(): boolean {
  return isTauri() && !isMobile;
}
