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
- `specs/transcribe.e2e.ts`, **full black-box ML e2e**: loads an audio fixture and
  triggers a full transcribe via the real frontend client
  (`window.drumjot.desktopTranscribe(path, {filter:false, quantise:false})` →
  `backendClient()` → sidecar → the whole ONNX pipeline: separation → onsets →
  beats → MIDI → loaded into the editor), then asserts the frontend rendered a
  bar/beat structure (`jotEditorStore.structural`). Exercises every ONNX model
  end-to-end. Hermetic (no LLM, no API key). **Opt-in + heavy**: needs a GPU (fp16
  models are GPU-only) + the full model set; gated on `DRUMJOT_E2E_TRANSCRIBE` so
  normal runs skip it (see the env block below).

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

### Dev-container provisioning (build + test)

Everything below lives outside the repo, so a container that wipes it on reboot
must reinstall on boot. This is the full set to **build and WebDriver-test** the
desktop app from clean, the Rust toolchain, Tauri's Linux system libs, and the
cargo/apt test tooling. (The Rust *crate* deps in `src-tauri/Cargo.lock` are NOT
listed: `cargo build` / `bun run tauri build` fetch them automatically into the
cargo registry cache.)

```sh
# 1. Rust toolchain (rustup installs the stable cargo/rustc). Skip if ~/.cargo
#    survives reboot; re-running is a no-op.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"

# 2. Tauri v2 Linux system libs + build tools + the WebDriver test tooling
#    (webkit2gtk-driver + xvfb). One apt transaction.
apt-get update && apt-get install -y \
  build-essential curl wget file patchelf \
  libssl-dev libxdo-dev librsvg2-dev \
  libgtk-3-dev libsoup-3.0-dev \
  libwebkit2gtk-4.1-dev libjavascriptcoregtk-4.1-dev \
  libayatana-appindicator3-dev \
  webkit2gtk-driver xvfb

# 3. cargo-installed binaries (the WebDriver proxy). Idempotent.
cargo install tauri-driver --locked
```

The wdio client deps (`@wdio/*`) come with `bun install` (they're in
`package.json`), so no extra step for those.

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

### Env for the full-transcribe e2e

Everything above, PLUS `DRUMJOT_E2E_TRANSCRIBE=1` to opt in (it's off by default
because the run is heavy and needs a GPU). `MODELS_DIR` must hold the full set
(`python -m app.pipeline.provision transcription`), not just `beat_this`:

```sh
DRUMJOT_E2E_TRANSCRIBE=1 \
MODELS_DIR=/dir/with/full/model/set \
DRUMJOT_SIDECAR_PYTHON="$PWD/transcriber/.venv/bin/python3" \
  xvfb-run -a bun run e2e:tauri
```

The `transcription` capability gate is auto-seeded (wdio.conf `beforeSession`,
same env guard) so `desktopTranscribe` doesn't show an install prompt. The fp16
models are GPU-only, so this needs a working CUDA/accelerator; on CPU the fp16
GRU onset model can't run.

Rebuild the app (`bun run tauri build`) after changing `withGlobalTauri` or any
frontend/Rust source, the config + frontend bundle are baked into the binary.
`withGlobalTauri: true` (tauri.conf.json) is what exposes `window.__TAURI__` so
the spec can `invoke('run_job', …)` from the webview.
