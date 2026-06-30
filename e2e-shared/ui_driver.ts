// A runner-agnostic seam over the bits of a browser-automation API the shared
// cross-engine flows need. Implemented twice -- once over Playwright's `page`
// (Chromium, the web build) and once over WebdriverIO's `browser`/`$` (the real
// WebKitGTK Tauri webview) -- so one flow body runs on both engines. Kept tiny
// on purpose: only CSS selectors + page-eval, never file-chooser or network
// stubbing (a native dialog can't be driven over WebDriver, and the desktop
// backend isn't HTTP), which is also why the shared subset is render/interaction
// flows, not backend-dependent ones.
export interface UiDriver {
  /** Load the app at its root (empty state). */
  open(): Promise<void>;
  /** Click the first element matching the CSS selector (waits for it first). */
  click(selector: string): Promise<void>;
  /** Current number of elements matching the selector (point-in-time, no wait).
   *  Retrying is done by the shared `waitForCount` helper. */
  count(selector: string): Promise<number>;
  /** Text content of the first match (waits for it to exist). */
  text(selector: string): Promise<string>;
  /** Run a self-contained function in the page; returns its JSON-able result.
   *  The function CANNOT close over test-process variables (it's serialised). */
  evalJs<T>(fn: () => T): Promise<T>;
}

export type SharedFlow = {
  name: string;
  run: (ui: UiDriver) => Promise<void>;
};
