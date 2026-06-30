// Runs the shared cross-engine flows on the REAL WebKitGTK webview via the Tauri
// binary. Same bodies as the Chromium/Playwright run (frontend/src/editing/test/
// cross_engine.e2e.ts), so a WebKit-only render/JS divergence shows up as a
// desktop-only failure. Globals (`browser`, `$`, `$$`) come from @wdio/globals.
import { sharedFlows } from '../e2e-shared/flows';
import { type UiDriver } from '../e2e-shared/ui_driver';

class WdioDriver implements UiDriver {
  async open(): Promise<void> {
    // One app instance serves the whole session, so reset to the empty state
    // (Playwright gets a fresh page per test; here a reload is the equivalent).
    await browser.refresh();
  }

  async click(selector: string): Promise<void> {
    const el = await $(selector);
    await el.waitForExist({ timeout: 15_000 });
    await el.click();
  }

  async count(selector: string): Promise<number> {
    return (await $$(selector)).length;
  }

  async text(selector: string): Promise<string> {
    const el = await $(selector);
    await el.waitForExist({ timeout: 15_000 });
    return el.getText();
  }

  async evalJs<T>(fn: () => T): Promise<T> {
    return browser.execute(fn);
  }
}

describe('Cross-engine web flows (real WebKitGTK webview)', () => {
  for (const flow of sharedFlows) {
    it(flow.name, async () => {
      await flow.run(new WdioDriver());
    });
  }
});
