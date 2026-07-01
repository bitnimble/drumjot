import { spawn, type ChildProcess } from 'node:child_process'
import { existsSync, mkdirSync, writeFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

// The raw release binary tauri-driver launches. The crate name is `app`
// (src-tauri/Cargo.toml); the "Drumjot" productName only renames the packaged
// AppImage/deb/rpm, not the release executable.
const here = fileURLToPath(new URL('.', import.meta.url))
const APP_BINARY = resolve(here, '../src-tauri/target/release/app')

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
    // The full-transcribe e2e (opt-in) drives desktopTranscribe, which gates on
    // the 'transcription' capability being installed -- otherwise it shows an
    // install prompt that would hang the run. Seed the app's capability-state
    // file as installed BEFORE the window boots so the boot refresh reads it as
    // ready and the gate passes with no prompt. The real sidecar interpreter is
    // still DRUMJOT_SIDECAR_PYTHON (the dev venv), not an app-managed venv; this
    // only flips the UI gate. Scoped to the opt-in run so other specs are
    // untouched. Path = Tauri's Linux app_local_data_dir for the identifier.
    if (process.env.DRUMJOT_E2E_TRANSCRIBE != null) {
      const dataRoot = join(homedir(), '.local/share/dev.drumjot.studio')
      mkdirSync(dataRoot, { recursive: true })
      writeFileSync(
        join(dataRoot, 'capabilities.json'),
        JSON.stringify({ separation: { installed: true }, transcription: { installed: true } }, null, 2),
      )
    }
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
