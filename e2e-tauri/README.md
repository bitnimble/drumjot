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

## Tooling

The wdio client deps (`@wdio/*`, `expect-webdriverio`) are in `package.json`
devDependencies, so `bun install` provides them. WebDriver-driving a GUI app
additionally needs two host tools the repo can't vendor, a cargo binary and an
apt package, plus a virtual display on a headless box:

```sh
# tauri-driver: the WebDriver <-> platform-driver proxy
cargo install tauri-driver --locked

# Linux platform driver (WebKitWebDriver) + a virtual display for headless
sudo apt-get install -y webkit2gtk-driver xvfb
```

macOS uses the built-in WebKit driver; Windows needs `msedgedriver`. See
<https://v2.tauri.app/develop/tests/webdriver/>.

### Dev-container provisioning

These host tools live outside the repo, so a container that wipes them on reboot
must reinstall on boot. Drop into the provisioning script:

```sh
apt-get update && apt-get install -y webkit2gtk-driver xvfb
cargo install tauri-driver --locked   # idempotent: no-op if already current
```

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
