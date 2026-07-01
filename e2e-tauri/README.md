# Tauri desktop e2e (WebDriver)

Drives the **built desktop app** through WebDriver (via `tauri-driver`), the
window actually launches and its webview is inspected. This is separate from the
in-browser Playwright suite (`src/**/test/*.e2e.ts`, run with `bun run e2e`),
which exercises the frontend in a plain headless Chromium and never boots the
Tauri shell / Rust broker / Python sidecar.

Scope today: an app-launch smoke test (`specs/app_launch.e2e.ts`), the window
opens, the frontend renders into `#app`, and the Tauri IPC bridge is present.
The ONNX inference path is validated headlessly and doesn't need the GUI, see
`transcriber/tests/test_onnx_model_e2e.py`.

## One-time tooling (not installed by default)

WebDriver-driving a GUI app needs system + cargo tools the repo doesn't vendor:

```sh
# 1. tauri-driver (WebDriver <-> platform-driver proxy)
cargo install tauri-driver --locked

# 2. Linux platform driver + a virtual display for headless boxes
sudo apt install webkit2gtk-driver xvfb

# 3. wdio client deps (adds to package.json devDependencies -- run once)
bun add -d @wdio/cli @wdio/local-runner @wdio/mocha-framework \
           @wdio/spec-reporter @wdio/globals expect-webdriverio
```

macOS uses the built-in WebKit driver; Windows needs `msedgedriver`. See
<https://v2.tauri.app/develop/tests/webdriver/>.

## Run

Build the app first (produces `src-tauri/target/release/app`):

```sh
bun run tauri build          # or: cargo build --release --manifest-path src-tauri/Cargo.toml
```

Then, from the repo root:

```sh
bun run e2e:tauri            # headless box: xvfb-run -a bun run e2e:tauri
```

`e2e:tauri` runs `wdio run e2e-tauri/wdio.conf.ts`, which spawns/kills
`tauri-driver` around the session and points it at the release binary.
