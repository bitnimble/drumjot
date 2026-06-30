import { join } from 'node:path';
import { buildOutputEnv } from './scripts/build_env';

// Drives the REAL desktop binary (WebKitGTK webview + real Tauri IPC), the one
// thing the Chromium Playwright suite can't cover. The embedded driver provider
// runs a W3C WebDriver server inside the app (tauri-plugin-wdio-webdriver), so
// no system WebKitWebDriver/tauri-driver is needed. On a headless Linux box the
// webview still needs a display: `xvfb-run -a bun run e2e:tauri`.
//
// Build the binary first with `bun run e2e:tauri:build` (or `bun run e2e:tauri`,
// which chains both). Specs live in `e2e-tauri/*.wdio.ts` (a suffix neither the
// Playwright `**/*.e2e.ts` nor the bun `*.test.ts` runners pick up).
//
// Resolve the binary from the SAME cargo target dir `build-wdio-app.ts` builds
// into, so the two agree when `DRUMJOT_BUILD_DIR` relocates `CARGO_TARGET_DIR`
// (else the build lands in `$DRUMJOT_BUILD_DIR/cargo-target` and this lookup
// misses it); falls back to the in-repo `src-tauri/target` when it's unset.
const TARGET_DIR = buildOutputEnv().CARGO_TARGET_DIR ?? join(import.meta.dirname, 'src-tauri', 'target');
const APP_BINARY = join(TARGET_DIR, 'debug', 'app');

export const config: WebdriverIO.Config = {
  runner: 'local',
  specs: ['./e2e-tauri/*.wdio.ts'],
  maxInstances: 1,
  capabilities: [
    {
      browserName: 'tauri',
      'tauri:options': { application: APP_BINARY },
    },
  ],
  services: [['@wdio/tauri-service', { driverProvider: 'embedded' }]],
  framework: 'mocha',
  reporters: ['spec'],
  mochaOpts: { ui: 'bdd', timeout: 60_000 },
  logLevel: 'warn',
};
