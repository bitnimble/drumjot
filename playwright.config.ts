import { defineConfig, devices } from '@playwright/test';

// e2e runs its own Vite dev server on a dedicated port so it never
// collides with a hand-run `bun run dev` (which stays on vite.config's
// default 5173). `--strictPort` makes Vite fail loudly instead of
// hopping to a random free port — a silent hop is how the suite once
// ended up driving an unrelated server. Override with E2E_PORT if 5273
// is taken too.
const E2E_PORT = Number(process.env.E2E_PORT ?? 5273);
const E2E_URL = `http://localhost:${E2E_PORT}`;

/**
 * Playwright e2e config for the Drumjot web app.
 *
 * Runs headless Chromium against the Vite dev server. The dev box is a
 * headless container, so:
 *   - `--no-sandbox` is required (standard in containers).
 *   - `--disable-dev-shm-usage` is deliberately NOT set: the container's
 *     /dev/shm is sized to 2GB, and forcing shm through /tmp instead is
 *     a measurable perf hit on DOM-heavy pages. If you ever see opaque
 *     "Target closed" crashes under parallelism, check shm size before
 *     reaching for that flag.
 *   - Debugging is trace-viewer driven (no display for `--headed` /
 *     Inspector). `npm run e2e:report` serves the HTML report on
 *     0.0.0.0:9323 — port-forward it to view from your machine.
 *
 * Unit tests stay on `bun test` (scoped to `src/` via bunfig.toml, which
 * matches `*.test.ts`); this runner only owns the co-located
 * `src/<feature>/tests/*.e2e.ts` specs.
 */
export default defineConfig({
  // E2E specs are co-located with the feature they cover, under
  // `src/<feature>/tests/*.e2e.ts`. The `.e2e.ts` suffix (not `.spec.ts`)
  // keeps them out of `bun test`'s auto-discovery, which matches
  // `.test.ts` / `.spec.ts` and has no ignore config, while `testMatch`
  // here keeps Playwright off the `.test.ts` unit tests living alongside.
  testDir: './src',
  testMatch: '**/*.e2e.ts',
  // One worker == one Chromium. Default locally; pinned low in CI since
  // the container's memory budget is shared with the transcriber image.
  fullyParallel: true,
  workers: process.env.CI ? 2 : undefined,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: E2E_URL,
    trace: 'on-first-retry',
    video: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          args: ['--no-sandbox'],
        },
      },
    },
  ],
  webServer: {
    command: `bun run dev -- --port ${E2E_PORT} --strictPort`,
    // The docker dev frontend runs Vite as root over a bind-mount, so the
    // default `node_modules/.vite` cache ends up root-owned and a
    // host-spawned Vite can't rewrite it (EACCES on startup). Point the
    // e2e server at a writable host-owned cache dir instead; vite.config
    // reads VITE_CACHE_DIR. Spread process.env so PATH etc. survive.
    env: { ...process.env, VITE_CACHE_DIR: process.env.VITE_CACHE_DIR ?? '/tmp/drumjot-vite-e2e' },
    url: E2E_URL,
    reuseExistingServer: !process.env.CI,
    // Cold Vite start is ~250ms, but a fresh container may need to warm
    // the dependency optimiser; give it generous headroom.
    timeout: 60_000,
    stdout: 'ignore',
    stderr: 'pipe',
  },
});
