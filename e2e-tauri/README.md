# Tauri desktop e2e (WebDriver)

Drives the **built desktop app** through WebDriver (via `tauri-driver`), the
window actually launches and its webview is inspected. This is separate from the
in-browser Playwright suite (`src/**/test/*.e2e.ts`, run with `bun run e2e`),
which exercises the frontend in a plain headless Chromium and never boots the
Tauri shell / Rust broker / Python sidecar.

Specs:

- `specs/app_launch.e2e.ts`, smoke: the window opens, the frontend renders into
  `#app`, and the Tauri IPC bridge is present. No models / env needed.
- `specs/beat_detection.e2e.ts`, **true ML e2e**: invokes the `beats` sidecar op
  through the app on a generated click track and asserts the ONNX Beat This! model
  ran (`engine === 'onnx'`) and returned a 120 BPM grid. Exercises webview
  `invoke('run_job')` → Rust broker → Python sidecar → ONNX. Needs the env below;
  self-skips if `MODELS_DIR/beat_this.fp16.onnx` is absent.

(A headless twin of the beat e2e, same op through the real `StdioAdapter`
registry, no GUI, lives at `transcriber/tests/test_onnx_model_e2e.py`.)

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

### Env for the beat-detection e2e

The app inherits these from the process env; the Rust broker + Python sidecar
read them at runtime:

```sh
MODELS_DIR=/dir/with/beat_this.fp16.onnx \
DRUMJOT_SIDECAR_PYTHON="$PWD/transcriber/.venv/bin/python3" \
DRUMJOT_BEAT_ONNX=1 \
  xvfb-run -a bun run e2e:tauri
```

- `MODELS_DIR`, where the sidecar finds `beat_this.fp16.onnx` (provision it, or
  point at a dir that already has it). Absent → the beat spec self-skips.
- `DRUMJOT_SIDECAR_PYTHON`, abs path to the interpreter the broker spawns
  (`resolve_python` prefers this; the dev fallback is a CWD-relative path that
  won't resolve when tauri-driver launches the binary).
- `DRUMJOT_BEAT_ONNX=1`, default; runs Beat This! on onnxruntime.

Rebuild the app (`bun run tauri build`) after changing `withGlobalTauri` or any
frontend/Rust source, the config + frontend bundle are baked into the binary.
`withGlobalTauri: true` (tauri.conf.json) is what exposes `window.__TAURI__` so
the spec can `invoke('run_job', …)` from the webview.
