# Desktop app (Tauri) + capability-based dependency install, design spec

Status: research / design (brainstorm 2026-06-29). Not approved for
implementation; captures the converged direction so a later session can pick
it up. Nothing here is built yet.

## Goal

Ship Drumjot (React/MobX frontend + the pure-Python transcriber backend) as a
desktop app. Two hard constraints shape every decision:

- **Tiny app download + tiny updates.** The heavy ML stack must NOT ride in the
  installer or in every Drumjot update.
- **Pay for what you use.** The core editor (load/edit `.jot`, playback,
  MIDI→Jot) must work with **zero** heavy deps installed; ML/audio features
  pull their dependencies **lazily, at point of use**, as opt-in
  "capabilities".

Mobile (React Native + Expo, or Tauri-mobile) is out of scope here; see
"Mobile, later".

## Decision: Tauri (not Electron)

Tauri 2, with the Python backend as a sidecar process. Rationale specific to
this project:

- The "huge .exe" worry is about the **Python/PyTorch/CUDA stack, not the
  shell**, measured at **7.9 GB** on disk for the full GPU venv (see sizes
  below). Once that stack is deferred-downloaded and never bundled, Tauri's
  ~3-15 MB shell (vs Electron's ~150 MB Chromium) is a real, clean win.
- Tauri also targets mobile with the same web frontend, keeping the
  one-codebase option open later.

**Accepted caveat:** Tauri uses the OS webview. WebView2/Chromium on Windows,
**WebKit on macOS, WebKitGTK on Linux**. Our Playwright e2e and 120 fps / Web
Audio tuning are Chromium-only, so macOS/Linux run a *different engine than we
test*. Mitigation: keep the e2e suite as-is for the Chromium path and add
manual WebKit smoke passes on the perf/audio surfaces before shipping those
platforms. If WebKit perf/audio regressions prove unacceptable, Electron stays
the fallback (the frontend code is unchanged either way).

## Process topology & IPC

Three processes, with **Rust as the broker** so the Python backend is never
network-exposed and the webview never sees a port:

```
webview (frontend)  <-- Tauri IPC (invoke + ipc::Channel) -->  Rust core
Rust core           <-- stdio, JSON-lines control protocol  -->  Python sidecar
```

- **No bound TCP port.** The only native Tauri↔sidecar channel is stdin/stdout
  pipes; we use them. No port conflicts, nothing other local apps can poke.
- **stdout = thin JSON-lines control channel** (request · progress · result ·
  error). It carries *no binary* (see data flow), so plain JSON, no msgpack.
- **stdout is protocol-only.** torch/transformers/madmom print to stdout/C
  stdout, redirect ALL logging to **stderr** (Rust captures it for
  diagnostics) and dup the real stdout fd at startup so library noise can't
  corrupt frames.
- **Streaming** (long transcription with progress) maps onto repeated progress
  frames then a terminal frame, re-emitted to the webview via Tauri `Channel`.
- **If a socket is ever needed** (multiple clients, decoupled sidecar
  lifecycle): UDS on macOS/Linux, ephemeral `127.0.0.1`+token on Windows.
  **Not Windows named pipes**, no ASGI server supports them and async Python
  has no good library. Drop them from the requirements.

## Data flow: results (backend → frontend)

Backend and frontend share a disk, so **pass references, not bytes**. Backend
writes artifacts to appdata; stdout reports paths only.

- **Large media** (separated stems, backing audio → Web Audio / `<audio>`):
  frontend loads via `convertFileSrc(path)` → `asset://` URL, with range
  requests for seeking. Requires `app.security.assetProtocol.enable = true`,
  the scope locked to the appdata output dir, and `asset:` in the CSP. Tens of
  MB stream straight off disk, never through IPC.
- **MIDI** (→ `src/midi/from_midi.ts`): needs the bytes to parse, but it's KB, read via `fs` plugin `readFile(path)`. (Inline-in-JSON would also be fine at
  that size.)
- **Full control option:** a custom Rust URI scheme
  (`register_asynchronous_uri_scheme_protocol`) if the asset scope isn't
  enough.

Two obligations this introduces:

- **Lifecycle / cleanup.** We own files now. Key each job's output dir by a
  **content hash of input audio + params** → free re-transcribe cache; evict an
  LRU cache dir on a size budget. Weights live in the HF cache, evicted
  separately by repo-id.
- **Deployment-agnostic references.** The protocol expresses a result as a
  **URI the frontend resolves**, not a hardcoded path: local → `asset://`,
  remote → `https://` (or inline for tiny payloads). One frontend code path for
  both local and remote backends.

## Data flow: inputs / file "uploads" (frontend → backend)

Same pass-by-path logic; in local mode "upload" disappears, the backend reads
the user's file straight off disk.

- **File picker:** `plugin-dialog` `open()` returns the **absolute path**; hand
  it to the backend, which `open()`s it.
- **Drag-drop:** use Tauri's **native** `onDragDropEvent` →
  `event.payload.paths` (absolute paths). **Footgun:** under Tauri the
  webview's HTML5 `ondrop` does NOT fire for OS file drops; only the native
  event does. `dragDropEnabled` defaults `true`; flipping it off to get HTML5
  DnD silently breaks OS file drops. It's a window-level event (paths + coords;
  do your own drop-zone hit-testing).
- **Backend needs no Tauri permission to read inputs**, it's a separate OS
  process at user privileges; Tauri fs scope only constrains the webview.
- **Fallback for path-less inputs** (clipboard paste, in-app `Blob`, web
  build): write bytes to an appdata temp file, pass that path.
- **Remote mode:** reverts to a real multipart upload. Keep input in the
  protocol as an **"audio source" reference** (path local / upload remote) so
  the core never knows which, symmetric with results.

## Comms abstraction

The seam sits **below the web framework**, because the gap is in the transport,
not the app. ASGI separates app from server, but no server spans HTTP +
Windows-IPC, so:

```
functional core   (transport-agnostic; async service; yields progress,
                    returns artifact references; imports no web framework)
   |  small message protocol (request · progress · result · error)
transport adapters (thin, ~30-60 lines each)
   |- stdio   -> local Tauri sidecar (default desktop path)
   |- HTTP/WS -> remote / dev (ASGI: keep FastAPI here if convenient)
   |- UDS     -> optional local socket
```

The core produces artifacts; the **adapter** decides whether a reference is a
local path or a remote URL/upload. This is what keeps the local/remote fork
open with one frontend.

## Dependency model: tiers vs capabilities

Two layers, related as a **DAG, not a flat list**:

- **Capabilities** = user-facing features shown on the first-run screen.
- **Tiers** = shared infra the resolver pulls behind capabilities (not
  separately chosen). User toggles features; the resolver computes the
  transitive closure, dedups shared tiers, and shows the **incremental** cost.

### Capabilities (user-facing)

- **Local transcription**, headline feature. Bundles stem separation +
  beat/downbeat + the learned onset model (one pipeline; don't surface those as
  separate choices). Pulls torch + transformers + onnxruntime + accelerator +
  MERT-330M (~1.3 GB) + separation models.
- **Lyrics alignment**, ctc-forced-aligner + MMS-300m (~1.2 GB), with a
  **Japanese sub-option** (unidic-lite ~250 MB + cutlet/fugashi) only JP users
  pull.
- **AI assist (LLM)**, a *different kind*: no download, gated on "API key +
  network configured", not "bytes installed".

### Tiers (shared, resolver-managed)

- **Python runtime base**, standalone Python + uv + numpy/scipy/librosa/
  soundfile.
- **Accelerator variant**, orthogonal cross-cutting axis (below), selected
  once by hardware detection (user-overridable), shared by every torch-needing
  capability.

Core editor (`.jot` load/edit, playback, MIDI→Jot, all TS) needs **zero**
capabilities and ideally **no Python runtime at all**; the sidecar itself sits
behind the first capability installed.

### Measured size breakdown (full GPU venv = 7.9 GB on disk)

| Component | Size | Bucket |
|---|---:|---|
| `nvidia/*` (cuDNN, cuBLAS, cuFFT, …) | 4.3 GB | CUDA stack |
| `torch` | 1.6 GB | CUDA stack |
| `triton` | 641 MB | CUDA stack (Linux) |
| `onnxruntime` (GPU) | 426 MB | CUDA stack |
| `unidic_lite` | 249 MB | JP lyrics |
| `llvmlite` | 161 MB | sci-Python |
| scipy/numpy/transformers/madmom/sklearn/… | ~600 MB | sci-Python |
| matplotlib/fontTools/PIL/Cython/ruff/pytest | ~110 MB | dev/build, don't ship |
| your code + fastapi/pydantic/httpx | ~30 MB | yours |

Plus lazy HF weights (not in the venv): MERT-330M ~1.3 GB, MMS-300m ~1.2 GB,
separation models ~hundreds of MB. On-disk figures overstate download (wheels
are compressed; ~3.5-5 GB transfer for the full GPU set). The CUDA stack alone
is ~88% of bytes and rarely changes → the cacheable bulk.

## Accelerator variants

Selected by hardware detection (user-overridable). Only CUDA and ROCm carry the
multi-GB library tax; the rest are comparatively tiny.

| Variant | Target | Notes |
|---|---|---|
| `cu128` | Windows/Linux NVIDIA | Needs driver 570+. Driver provides ONLY `nvcuda.dll`; cuDNN/cuBLAS are bundled by the wheels (downloaded), NOT by the driver. |
| `rocm` | Linux AMD | Upstream PyTorch ROCm is Linux-only; CUDA-sized wheel set. |
| `directml` | Windows AMD/Intel | Covers any DX12 GPU; smaller. Near-term Windows-AMD path (native Windows ROCm is preview-only). |
| `mps` | Apple Silicon | Ships in the macOS torch wheel; no nvidia download. **Uses the GPU, NOT the Neural Engine**, ANE needs a separate CoreML path, future work. |
| `cpu` | fallback | ~400-800 MB total; slow for separation/MERT. |

## Capability mechanism (uv)

- Each capability = a **dependency-group** (PEP 735) / extra; weights = a
  separate manifest of HF repo-ids.
- "Install capability" = `uv sync` the **union of enabled groups** into **one
  shared venv**. uv resolves the closure, downloads only wheels not in its
  content-addressed cache (shared tiers download once), and the **incremental**
  size shown = diff vs current venv state. First ML capability shows the big
  accelerator number; the next shows a small delta.
- **One shared venv keyed by the union of groups**. NOT per-capability venvs
  (those duplicate the 7 GB torch stack). **Uninstall = re-resolve to the
  smaller group set** (uv prunes). Weights evict separately by repo-id.
- Beats PyInstaller for this goal: PyInstaller is a monolithic blob with no
  cross-version dedup, so any bump re-downloads everything.
- uv is **Apache-2.0 OR MIT**; fine to bundle in a closed-source commercial
  app; only obligation is including the license notice.

## Capability check (probe semantics)

"Installed" means **usable**, verified in layers (all must pass):

1. venv has the group synced (lockfile state),
2. import actually initializes (e.g. `torch.cuda` inits → driver present & new
   enough),
3. required weights present in HF cache.

- **Cache the probe result**; invalidate on app/driver/version change (don't
  re-init torch per click).
- **Self-test on install**, run a tiny inference so "downloaded but broken"
  surfaces immediately, not at first real use.
- **Driver prereqs aren't downloadable.** CUDA needs driver ≥ X; if too old,
  tell the user to update (with a link), a distinct failure path from "deps
  missing". Same for "no compatible GPU → offer CPU or remote".

## UX flows

**First run:** detect hardware FIRST (show sizes for the *right* variant, never
quote 7 GB CUDA to a Mac user). List capabilities with plain-language "what
this does" + per-variant size, and a prominent **"Skip, just edit jots"** that
lands in a fully working editor with nothing downloaded.

**Point of use:** trigger an ML action → capability check fails → modal
("Transcription needs the ML runtime, ~X GB for your hardware; downloads once,
reused") → install with live progress → **auto-resume the original action**
(queue the triggering intent; don't make them re-click).

## Install robustness / gotchas

- **Incremental, not total, sizing** everywhere (vs current venv state).
- **Atomic + resumable.** Multi-GB downloads WILL be interrupted (uv/HF
  resume). Sync into staging; mark ready only after sync AND self-test pass, a
  half-synced venv must never read as installed.
- **App-update top-ups.** Version the capability manifest. On a torch/model
  bump, mark affected capabilities "update available" and re-sync lazily, never
  force a 7 GB re-download on a routine app update; uv pulls only changed
  wheels.
- **Disk preflight + offline handling** before any large install.
- **Remote fulfillment** keeps the cloud fork open: a capability can be
  satisfied remotely ("use cloud transcription"), same check, different
  fulfillment. Make local-vs-cloud a per-capability choice.

## Frontend integration (store / presenter / component)

- `CapabilityStore`, observables only: per-capability status
  (`not-installed | installing(progress) | ready | update-available | error`),
  detected hardware + chosen accelerator.
- `CapabilityPresenter`, sole writer: orchestrates install (drives Rust, which
  spawns uv and streams progress), runs probes, gates point-of-use actions,
  queues the resume intent.
- Components read the store to enable/annotate ML buttons and render the
  first-run + install modals. Install work itself lives in Rust.

## WebKit compatibility audit

Tauri's macOS/Linux webview is WebKit, so the app must run on WebKit. We want
WebKit for the **web build regardless**, so these fallbacks are needed
independent of Tauri; Tauri only makes them mandatory. The real variable is
**WebKitGTK on Linux** (lags Safari) plus Tauri's secure-context/origin
handling. Audit of `src/**` (CSS + TS) against recent Safari / WebKitGTK:

**Clean, no concern:** no Web MIDI (`requestMIDIAccess`), MIDI is file-parsed
via `from_midi.ts`, so Safari's missing Web MIDI is a non-issue; no File System
Access API (Tauri native dialogs instead); no scroll-driven animations / anchor
positioning / subgrid / `field-sizing` / WebGPU / `SharedArrayBuffer` / native
`popover` API (popovers are manual portals). `:has()` is Baseline 2023.

**Flags (verify / add fallback):**

1. **`backdrop-filter` is unprefixed and the build does NOT autoprefix.** Vite
   here runs `patchCssModules` only, no autoprefixer, no lightningcss, and
   `src/**` has zero `-webkit-backdrop-filter`. Used in `playback`, `toolbar`,
   `jot_editor` CSS. Unprefixed `backdrop-filter` is Safari 18+ only; older
   Safari/WebKitGTK needs `-webkit-backdrop-filter`. **Fix:** add the prefix or
   wire autoprefixer/lightningcss to the browserslist. Most concrete item.
2. **`@property` registered custom props on the perf hot path** (`--playhead-x`,
   `--scroll-x/y`, `--bars-row-width` in `design_tokens.css`). Safari 16.4+ /
   recent WebKitGTK. If the shipped WebKitGTK predates it, typed interpolation
   silently degrades on the playhead/scroll path. Not feature-detected (pure
   CSS). **Action:** pin a WebKitGTK floor that supports `@property`.
3. **AudioWorklet** (Signalsmith Stretch = audio-track playback). Supported in
   WebKit but **secure-context-gated** and historically buggy. Already
   feature-detected with graceful fallback (`detectAudioWorkletState`).
   **Action:** verify Tauri's app origin counts as a secure context so
   `audioWorklet` is exposed, and ear-check stretch quality on macOS WebKit +
   WebKitGTK.
4. **OffscreenCanvas** (waveform worker tiles + lyrics text measurement).
   Safari 16.4+ / recent WebKitGTK; already feature-detected (waveforms
   blank-fallback). **Action:** confirm the target WebKitGTK ships
   OffscreenCanvas 2D-in-worker.
5. **`color-mix()`** (score / lyrics). Safari 16.2+. Fine on recent WebKit;
   flag only if the WebKitGTK floor is older.

Net: no architectural blockers. One concrete fix (backdrop-filter prefix), the
rest is "pin a minimum WebKitGTK version for the Linux target and verify".
Items 2 + 1 are pure CSS that degrade silently, so they matter most.

## Capability manifest schema

The single data source both the Rust installer and `CapabilityStore` read. One
entry per **user-facing capability**; tiers are derived by the resolver, not
authored.

Per-capability fields:

- `id`, stable key (`transcription`, `lyrics`, `lyrics.japanese`, `ai-assist`).
- `name`, `description`, first-run UI copy ("what this does").
- `kind`, `deps` (downloads) | `credentials` (API key, e.g. LLM) | `system`
  (non-downloadable prereq).
- `groups`, uv dependency-group names this capability adds.
- `weights`, `[{repoId, revision, approxBytes}]` HF artifacts (lazy,
  content-addressed, downloaded on first use not at install unless prefetched).
- `requires`, other capability ids (the prereq DAG; e.g. `lyrics.japanese`
  requires `lyrics`).
- `accelerator`, `required | optional | none` (whether it pulls the torch /
  accel tier).
- `probe`, layered usability check:
  `{import: [modules], init?: "torch.cuda", weightsPresent: [repoIds]}`.
- `selfTest`, tiny command run post-install to confirm it actually works.
- `sizes`, per-variant `{cu128, rocm, directml, mps, cpu}` *estimates* for
  pre-detection UI copy (the real shown number is the resolver's diff vs the
  current venv).
- `manifestVersion`, bumped when `groups`/`weights` change; drives "update
  available".

Accelerator is a **separate singleton, not a capability**:

- detected by a Rust probe (GPU vendor + driver version); user override
  persisted.
- maps to the uv torch index / group set; declares `system` prereqs (e.g.
  NVIDIA driver ≥ 570) with a min-version + help URL, checked, not installable.

Resolver, per install request: closure of `requires` → union of `groups` →
diff vs installed → incremental wheel set (uv) + missing `weights`. Manifest is
data; resolver + probe logic live in Rust.

## Control protocol message shapes

One protocol shared by the stdio (local) and WS (remote) adapters; the core
neither knows nor cares which transport carries the frames. Newline-delimited
JSON over stdio (no binary on the wire, see data flow). Every message:
`{v, type, id}` (`v` protocol version; `id` correlates a request with its reply
stream).

Client → backend:

- `request`, `{type:"request", id, op, args}`. `op` ∈ `"transcribe"` |
  `"separate"` | `"alignLyrics"`. `args` carries **source references, not
  bytes**: `{audio: SourceRef, params}` where `SourceRef` = `{kind:"path",
  path}` (local) | `{kind:"upload", uploadId}` (remote).
- `cancel`, `{type:"cancel", id}` (cooperative; long jobs poll it).

Backend → client (a stream per `id`, terminated by exactly one `result` or
`error`):

- `progress`, `{type:"progress", id, stage, frac, message?}` (`stage` e.g.
  `"separating"` | `"onsets"`; `frac` 0..1).
- `log`, `{type:"log", id?, level, message}` (structured diagnostics, distinct
  from the raw stderr text stream).
- `result`, `{type:"result", id, artifacts}`; each artifact is a **ResultRef**:
  `{role:"midi"|"stem"|"audio", ref}` with `ref` = `{kind:"path", path}`
  (local → frontend `convertFileSrc`/`readFile`) | `{kind:"url", url}` (remote)
  | `{kind:"inline", bytesB64}` (tiny only).
- `error`, `{type:"error", id, code, message, recoverable}`.

Rust forwards `request`/`cancel` down, re-emits `progress`/`result`/`error` up
to the webview via Tauri `Channel` keyed by `id`, and resolves
`SourceRef`/`ResultRef` kinds (path ↔ asset URL locally). The identical schema
runs over WS for remote, where the refs resolve to upload ids / signed URLs.
This is what lets **one frontend code path** serve both deployment modes.

The frontend validates **both directions with Zod** (no unchecked `as` casts on
parsed JSON): `encodeClientMessage` parses before sending, `decodeServerMessage`
/ `safeDecodeServerMessage` parse on receipt. Implemented in
`src/net/control_protocol.ts` (+ `control_protocol.test.ts`); the inferred
types are the single source for both wire validation and call-site typing.

## Mobile, later

Committed: **Tauri mobile**, the same web frontend across desktop + iOS /
Android, one codebase. React Native + Expo is rejected (would not reuse the
DOM/Canvas/Web Audio frontend; full rewrite). Mobile is **remote-backend-only**
regardless (never bundle PyTorch on a phone).

## Android (implemented)

Android v1 ships the Tauri shell + the existing web frontend, transcribing over
the existing `HttpBackendClient` (the same transport the web build uses), with
**no sidecar and no capability install** on device. Offline `.jot`
load/edit/playback + MIDI import need no server; transcription / separation /
lyrics work when a server is reachable. iOS is deferred.

- **Backend selection.** `backendClient()` picks: web + mobile -> HTTP, desktop
  -> sidecar unless the user chose Hosted (Settings -> Advanced). The platform
  is a compile-time `__IS_MOBILE__` (Vite `define` from Tauri's
  `TAURI_ENV_PLATFORM`); `isDesktopShell()` = `isTauri() && !isMobile` gates the
  sidecar/capability UI off on mobile.
- **Transcriber URL.** Device-global `AppSettings` store (persisted to
  localStorage, separate from the per-song `SettingsStore`): `backendMode` +
  `transcriberUrl` (default `https://drumjot.kumo.dev`, only the default is
  compiled in; `VITE_TRANSCRIBER_URL` overrides it for dev/docker/e2e, empty =
  origin-relative `/api`). The base every request composes against is
  `<url>/api`. Surfaced in the new Settings -> Advanced tab.
- **Rust.** The sidecar, capability installer, and portable-path / env-redirect
  code are `#[cfg(desktop)]`-gated; the Android `cdylib` builds with just the
  webview + the fs/dialog plugins. The webview origin (`http://tauri.localhost`)
  is added to the transcriber's `cors_origins`; the backend is HTTPS-only so no
  Android cleartext config is needed (debug builds allow cleartext for a LAN box).
- **Build.** `bun run android:{init,dev,build}`. Targets `aarch64` + `armv7`
  only, producing a universal APK. The desktop `bundle.resources` (Python sidecar
  + uv) would otherwise fail the Android build's resource-existence check (the
  host's platform-config + base config leak into the Android merge, and a
  deep-merge override can't clear them); the build/dev scripts pass `--config
  '{"bundle":{"resources":[]}}'`, whose replace-merge does. `gen/android` is
  regenerated + gitignored.
- **Still to validate on a real device:** mobile-webview Web Audio latency, 120
  fps canvas perf, touch interactions (the headless box can build/assemble the
  APK but can't run it).

## Decisions resolved & remaining

Resolved:

- **Desktop default = local transcription.** Remote becomes an option later,
  surfaced in the new settings dialog another agent is building first; the
  per-capability remote-fulfillment path gets wired behind that.
- **WebKit is a committed target** (we want it for the web build anyway), so its
  fallbacks are required regardless of Tauri. See the audit above: no blockers,
  one concrete fix (backdrop-filter prefix) + a WebKitGTK version floor.
- **Mobile = Tauri** (React Native dropped).
- **Capability manifest schema** and **control protocol message shapes** drafted
  above; refine against implementation.

Remaining:

- **Minimum WebKitGTK version** for the Linux target, pin it (gates
  `@property`, OffscreenCanvas, AudioWorklet quality).
- **Prefetch vs lazy weights**, whether a capability install pulls its HF
  weights up front (predictable size, longer install) or on first real use
  (faster install, a second wait later). Likely per-capability.

## References

- Tauri sidecar / asset protocol / drag-drop:
  https://v2.tauri.app/develop/sidecar/ ,
  https://v2.tauri.app/security/asset-protocol/
- uv license: https://docs.astral.sh/uv/reference/policies/license/
- PyTorch wheels bundle CUDA libs (not the driver):
  https://github.com/pytorch/pytorch/issues/100974
- Apple MPS (GPU, not ANE):
  https://developer.apple.com/metal/pytorch/
- ROCm PyTorch (Linux-only upstream):
  https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/pytorch-compatibility.html
