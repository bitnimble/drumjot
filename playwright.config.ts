import { defineConfig, devices } from '@playwright/test';

// e2e runs its own Vite dev server on a dedicated port so it never
// collides with a hand-run `bun run dev` (which stays on vite.config's
// default 5173). `--strictPort` makes Vite fail loudly instead of
// hopping to a random free port — a silent hop is how the suite once
// ended up driving an unrelated server. Override with E2E_PORT if 5273
// is taken too.
const E2E_PORT = Number(process.env.E2E_PORT ?? 5273);
const E2E_URL = `http://localhost:${E2E_PORT}`;

// Headless Chromium defaults to SwiftShader (CPU raster), whose per-frame
// timings are contention-sensitive: software raster competes for CPU cores
// with the parallel test workers (and any ML jobs on the box), which inflates
// the worst frames and flakes the 120fps perf budget. We composite on the idle
// AMD iGPU instead (Mesa radv, via ANGLE's Vulkan backend), taking raster off
// the CPU entirely. The iGPU is the deliberate choice over the NVIDIA card:
// it's never under ML load, so compositing never contends with training.
//
// Container prereqs: the AMD render node (/dev/dri/renderD128) must be passed
// in, with the host 'render' group via `group_add` so the browser can open it
// (see the compose config). The radeon Vulkan ICD ships in the image; no
// vendor-JSON bind-mounts are needed.
//
// GPU is ON BY DEFAULT for the perf project (its frame-budget medians are only
// meaningful on a real GPU). It's opt-in via E2E_GPU for the other projects:
// they test correctness, not timing, run fully parallel, and many workers
// sharing one render node would only add GPU contention for no benefit. Force
// it off anywhere with E2E_GPU=0. `gpu_renderer.e2e.ts` (gated on E2E_GPU)
// reports the active backend and fails if it's still software.
const GPU_OFF = /^(0|false|off|no)$/i.test(process.env.E2E_GPU ?? '');
// ANGLE's Vulkan backend on Mesa's radv driver. (gl-egl can't get a hardware
// context from Mesa's libEGL headless here; radv enumerates the iGPU fine once
// the render node is present.) --disable-gpu-sandbox: the GPU-process sandbox
// otherwise blocks the driver from opening its device node in a container,
// dropping ANGLE back to software.
const AMD_GPU_ARGS = [
  '--use-gl=angle',
  '--use-angle=vulkan',
  '--enable-features=Vulkan',
  '--enable-gpu',
  '--ignore-gpu-blocklist',
  '--disable-gpu-sandbox',
];
// Pin the Vulkan loader to Mesa's radeon ICD so ANGLE can't fall through to
// llvmpipe (software Vulkan) or any other ICD.
const AMD_GPU_ENV = {
  VK_DRIVER_FILES: '/usr/share/vulkan/icd.d/radeon_icd.json',
  VK_ICD_FILENAMES: '/usr/share/vulkan/icd.d/radeon_icd.json',
};
// launchOptions.env replaces (not merges) the browser environment, so spread
// process.env to keep PATH etc. `gpu` adds the iGPU args/env on top.
const chromiumLaunch = (gpu: boolean) => ({
  args: ['--no-sandbox', ...(gpu ? AMD_GPU_ARGS : [])],
  env: { ...process.env, ...(gpu ? AMD_GPU_ENV : {}) },
});
const GPU_PERF = !GPU_OFF; // perf: default on
const GPU_OTHER = !GPU_OFF && !!process.env.E2E_GPU; // functional/heavy: opt-in

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
 * `src/<feature>/test/*.e2e.ts` specs.
 */
export default defineConfig({
  // E2E specs are co-located with the feature they cover, under
  // `src/<feature>/test/*.e2e.ts`. The `.e2e.ts` suffix (not `.spec.ts`)
  // keeps them out of `bun test`'s auto-discovery, which matches
  // `.test.ts` / `.spec.ts` and has no ignore config, while `testMatch`
  // here keeps Playwright off the `.test.ts` unit tests living alongside.
  testDir: './frontend/src',
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
    // Functional specs: everything except the per-frame perf suite and the
    // decode-heavy real-song specs, run fully in parallel across the pool.
    {
      name: 'functional',
      testIgnore: ['**/perf.e2e.ts', '**/*.perf.e2e.ts', '**/*.heavy.e2e.ts'],
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: chromiumLaunch(GPU_OTHER),
      },
    },
    // Perf specs measure per-frame timing against a tight 120fps budget, so
    // they must NOT run alongside the parallel functional workers, their CPU
    // contention inflates the medians and flakes the budget. `dependencies:
    // ['functional']` runs this right after the functional pool frees (its
    // validated, low-noise slot); the specs are already `mode: 'serial'` within
    // the file, and `fullyParallel: false` keeps them on a single worker.
    // (Caveat: a functional failure skips this dependent project; fix
    // functional first, or run `bun run e2e:perf` to measure in isolation.)
    {
      name: 'perf',
      testMatch: ['**/perf.e2e.ts', '**/*.perf.e2e.ts'],
      fullyParallel: false,
      dependencies: ['functional'],
      use: {
        ...devices['Desktop Chrome'],
        // GPU on by default here: the frame-budget medians need a real GPU.
        launchOptions: chromiumLaunch(GPU_PERF),
      },
    },
    // Heavy specs load a real, full-length song (30 MB+ zip unpack + parallel
    // multi-track audio decode), which pegs the CPU and leaves GC/memory
    // pressure that would both starve the parallel functional tests and
    // perturb perf's per-frame medians. So it runs DEAD LAST, alone:
    // `dependencies: ['perf']` chains functional -> perf -> heavy, so the heavy
    // decode can't disturb either timing-sensitive phase before it. Opt a spec
    // in with the `.heavy.e2e.ts` suffix. (Caveat: an earlier-phase failure
    // skips this; run a `.heavy.e2e.ts` spec directly with `bun run e2e <spec>`
    // to exercise it in isolation.)
    {
      name: 'heavy',
      testMatch: '**/*.heavy.e2e.ts',
      fullyParallel: false,
      dependencies: ['perf'],
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: chromiumLaunch(GPU_OTHER),
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
    // VITE_TRANSCRIBER_URL='' keeps the transcriber base origin-relative
    // (`/api`) so the per-spec `**/api/**` route stubs intercept it, rather
    // than the prod default (https://drumjot.kumo.dev) reaching the network.
    env: {
      ...process.env,
      VITE_CACHE_DIR: process.env.VITE_CACHE_DIR ?? '/tmp/drumjot-vite-e2e',
      VITE_TRANSCRIBER_URL: '',
    },
    url: E2E_URL,
    reuseExistingServer: !process.env.CI,
    // Cold Vite start is ~250ms, but a fresh container may need to warm
    // the dependency optimiser; give it generous headroom.
    timeout: 60_000,
    stdout: 'ignore',
    stderr: 'pipe',
  },
});
