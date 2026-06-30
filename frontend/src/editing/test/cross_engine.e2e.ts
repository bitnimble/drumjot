// Runs the shared cross-engine flows on Chromium via Playwright (the web build).
// The same bodies run on the real WebKitGTK webview via e2e-tauri/cross_engine.wdio.ts.
import { test, type Page } from '@playwright/test';
import { sharedFlows } from '../../../../e2e-shared/flows';
import { type UiDriver } from '../../../../e2e-shared/ui_driver';

class PlaywrightDriver implements UiDriver {
  constructor(private readonly page: Page) {}

  async open(): Promise<void> {
    await this.page.goto('/');
  }

  async click(selector: string): Promise<void> {
    await this.page.locator(selector).first().click();
  }

  async count(selector: string): Promise<number> {
    return this.page.locator(selector).count();
  }

  async text(selector: string): Promise<string> {
    const loc = this.page.locator(selector).first();
    await loc.waitFor();
    return (await loc.textContent()) ?? '';
  }

  async evalJs<T>(fn: () => T): Promise<T> {
    return this.page.evaluate(fn);
  }
}

for (const flow of sharedFlows) {
  test(`cross-engine: ${flow.name}`, async ({ page }) => {
    await flow.run(new PlaywrightDriver(page));
  });
}
