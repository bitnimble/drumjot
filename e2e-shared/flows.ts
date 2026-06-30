// Curated web-editor flows that run on BOTH engines: Chromium via Playwright
// (the web build) and the real WebKitGTK webview via WebdriverIO (the desktop
// shell). These are the highest cross-engine value -- the full render path and
// core load -- exercised on the engine Playwright can't reach, catching WebKit
// divergence (canvas/@property/color-mix/AudioWorklet) the Chromium suite can't.
// Each runner imports `sharedFlows`, wraps its API in a `UiDriver`, and registers
// the bodies as its own tests (see cross_engine.e2e.ts / cross_engine.wdio.ts).
import assert from 'node:assert/strict';
import { type SharedFlow, type UiDriver } from './ui_driver';

const INSTRUMENT_ROW = '[data-testid^="instrument-track-"]';
const NOTE = '[data-noseek="true"]';

/** Poll `ui.count(selector)` until `predicate` holds (the runner-agnostic
 *  stand-in for Playwright's auto-retrying locators / WDIO's waitUntil). */
async function waitForCount(
  ui: UiDriver,
  selector: string,
  predicate: (n: number) => boolean,
  timeoutMs = 15_000,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const n = await ui.count(selector);
    if (predicate(n)) return;
    if (Date.now() > deadline) {
      throw new Error(`waitForCount('${selector}') unsatisfied after ${timeoutMs}ms (last: ${n})`);
    }
    await new Promise((r) => setTimeout(r, 100));
  }
}

/** Poll a page-side predicate until it holds (e.g. the app's async bootstrap
 *  has installed `window.drumjot`); the runner-agnostic "wait for JS state". */
async function waitForJs(ui: UiDriver, fn: () => boolean, timeoutMs = 15_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    if (await ui.evalJs(fn)) return;
    if (Date.now() > deadline) throw new Error('waitForJs predicate unsatisfied');
    await new Promise((r) => setTimeout(r, 100));
  }
}

/** A loaded score renders one row per kit piece plus note glyphs; on a WebKit
 *  render regression either count stays at zero. */
async function assertScoreRendered(ui: UiDriver): Promise<void> {
  await waitForCount(ui, INSTRUMENT_ROW, (n) => n >= 1);
  await waitForCount(ui, NOTE, (n) => n >= 1);
}

export const sharedFlows: SharedFlow[] = [
  {
    name: 'boots to the empty state with the probe surface wired',
    run: async (ui) => {
      await ui.open();
      assert.ok(
        (await ui.text('h2')).includes('Start a new jot'),
        'empty-state title not shown',
      );
      // The probe globals are installed at the end of the async bootstrap, so
      // wait for them rather than reading once (else a slow boot races us).
      await waitForJs(
        ui,
        () =>
          typeof (window as { jotPlayer?: unknown }).jotPlayer === 'object' &&
          Boolean((window as { drumjot?: unknown }).drumjot),
      );
    },
  },
  {
    name: 'loads a built-in example from the empty-state picker and renders it',
    run: async (ui) => {
      await ui.open();
      await ui.click('[data-testid^="empty-state-example-"]');
      await assertScoreRendered(ui);
    },
  },
  {
    name: 'loads a jot through the window.drumjot JS API and renders it',
    run: async (ui) => {
      // The file-chooser-free load path (the one the desktop shell relies on).
      await ui.open();
      await waitForJs(ui, () => Boolean((window as { drumjot?: unknown }).drumjot));
      await ui.evalJs(() =>
        (window as unknown as { drumjot: { loadTripletJot: () => void } }).drumjot.loadTripletJot(),
      );
      await assertScoreRendered(ui);
    },
  },
];
