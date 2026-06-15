import { expect, test } from '@playwright/test';

// With the backend unreachable, a user-initiated backend action surfaces the
// generic "Server is down" toast (rather than a confusing per-request error
// or a silent failure). Simulated by aborting every /api request at the
// network layer, the same shape as a fully static deploy with no backend.
test('shows a "Server is down" toast when a backend request fails', async ({ page }) => {
  await page.route('**/api/**', (route) => route.abort());
  await page.goto('/');
  await page.waitForFunction(() => !!(window as any).drumjot);

  // Fire a backend action (score) that goes through backendFetch. A tiny
  // synthetic file is enough, the request never reaches a server.
  await page.evaluate(() => {
    const file = new File([new Uint8Array([1, 2, 3])], 'probe.zip', {
      type: 'application/zip',
    });
    void (window as any).drumjot.jotEditorPresenter.scoreParadbMap(file);
  });

  await expect(page.getByTestId('server-down-toast')).toBeVisible();
});
