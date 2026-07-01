import { spawn, type ChildProcess } from 'node:child_process'
import { existsSync } from 'node:fs'
import { resolve } from 'node:path'

// The raw release binary tauri-driver launches. The crate name is `app`
// (src-tauri/Cargo.toml); the "Drumjot" productName only renames the packaged
// AppImage/deb/rpm, not the release executable.
const APP_BINARY = resolve(__dirname, '../src-tauri/target/release/app')

// tauri-driver proxies WebDriver to the platform driver (WebKitWebDriver on
// Linux). We spawn it per session and tear it down after.
let tauriDriver: ChildProcess | undefined

export const config: WebdriverIO.Config = {
  runner: 'local',
  specs: ['./specs/**/*.e2e.ts'],
  maxInstances: 1,
  capabilities: [
    {
      // tauri:options is consumed by tauri-driver, not part of the wdio types.
      // @ts-expect-error injected capability
      'tauri:options': { application: APP_BINARY },
    },
  ],
  hostname: '127.0.0.1',
  port: 4444,
  path: '/',
  framework: 'mocha',
  reporters: ['spec'],
  mochaOpts: { ui: 'bdd', timeout: 120_000 },
  logLevel: 'warn',

  onPrepare: () => {
    if (!existsSync(APP_BINARY)) {
      throw new Error(
        `Tauri release binary not found at ${APP_BINARY}.\n` +
          'Build it first: `bun run tauri build` (or `cargo build --release ' +
          '--manifest-path src-tauri/Cargo.toml`).',
      )
    }
  },

  beforeSession: () => {
    tauriDriver = spawn('tauri-driver', [], {
      stdio: [null, process.stdout, process.stderr],
    })
    tauriDriver.on('error', (err) => {
      console.error(
        'tauri-driver failed to start. Install it with `cargo install tauri-driver` ' +
          'and ensure WebKitWebDriver is on PATH (apt: webkit2gtk-driver).',
        err,
      )
      process.exit(1)
    })
  },

  afterSession: () => {
    tauriDriver?.kill()
  },
}
