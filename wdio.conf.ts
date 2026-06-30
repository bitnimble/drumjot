import { join } from 'node:path';

// Drives the REAL desktop binary (WebKitGTK webview + real Tauri IPC), the one
// thing the Chromium Playwright suite can't cover. The embedded driver provider
// runs a W3C WebDriver server inside the app (tauri-plugin-wdio-webdriver), so
// no system WebKitWebDriver/tauri-driver is needed. On a headless Linux box the
// webview still needs a display: `xvfb-run -a bun run e2e:tauri`.
//
// Build the binary first with `bun run e2e:tauri:build` (or `bun run e2e:tauri`,
// which chains both). Specs live in `e2e-tauri/*.wdio.ts` (a suffix neither the
// Playwright `**/*.e2e.ts` nor the bun `*.test.ts` runners pick up).
const APP_BINARY = join(import.meta.dirname, 'src-tauri', 'target', 'debug', 'app');

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
