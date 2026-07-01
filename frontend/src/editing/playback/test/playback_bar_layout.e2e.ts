import { expect, test } from '@playwright/test';
import { loadRockJot } from '../../test/editing_e2e_utils';

/**
 * Mobile styling guard: on a narrow (mobile-portrait) viewport the transport
 * bar keeps its buttons and the master volume side by side, NOT stacked into a
 * single column (the <=640px rule in playback.module.css).
 */

const play = (page: import('@playwright/test').Page) =>
  page.getByRole('button', { name: 'Play', exact: true });
const master = (page: import('@playwright/test').Page) => page.getByLabel('Master master volume');

test('the playback bar keeps transport and volume on one row on a narrow viewport', async ({
  page,
}) => {
  await loadRockJot(page);

  // Baseline: at a wide desktop viewport the transport group is centred, so the
  // Play button sits well right of the bar's left edge.
  await page.setViewportSize({ width: 1280, height: 720 });
  await expect(play(page)).toBeVisible();
  const wide = (await play(page).boundingBox())!;
  expect(wide.x).toBeGreaterThan(300);

  // Narrow (mobile portrait): the centring spacer is dropped, the transport is
  // left-anchored and the master volume right-anchored on the SAME row.
  await page.setViewportSize({ width: 390, height: 800 });
  await expect(play(page)).toBeVisible();
  await expect(master(page)).toBeVisible();
  const p = (await play(page).boundingBox())!;
  const m = (await master(page).boundingBox())!;

  // Left-anchored now (moved left of where it sat centred on the wide bar).
  expect(p.x).toBeLessThan(150);
  expect(p.x).toBeLessThan(wide.x);
  // Master volume is to the right of the transport...
  expect(m.x).toBeGreaterThan(p.x + p.width);
  // ...and on the same row: their vertical extents overlap (not stacked).
  const overlap = Math.min(p.y + p.height, m.y + m.height) - Math.max(p.y, m.y);
  expect(overlap).toBeGreaterThan(0);
});
