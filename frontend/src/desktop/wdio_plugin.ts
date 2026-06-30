// The WebdriverIO frontend plugin: installs `window.wdioTauri` + invoke
// interception so `browser.tauri.execute()` / `.mock()` work. Loaded ONLY in the
// e2e build (`__WDIO__`, a Vite define set by scripts/build-wdio-app.ts); the
// `if (false)` branch is dead-code-eliminated from every normal build, so the
// package never ships. The `typeof` guard keeps it inert under bun (no define).
declare const __WDIO__: boolean;

if (typeof __WDIO__ !== 'undefined' && __WDIO__) {
  void import('@wdio/tauri-plugin');
}
