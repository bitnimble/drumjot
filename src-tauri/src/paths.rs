//! Where the app keeps writable state.
//!
//! Portable builds (a `portable` marker file next to the exe) keep EVERYTHING -
//! the capability venv, uv + torch + HuggingFace caches, the downloaded Python,
//! the sidecar's outputs + scratch, the webview's data - under `<exe_dir>/data`,
//! so deleting that folder removes all of it. Installed builds use the OS user
//! app-LOCAL-data dir: never next to the exe (which may be a non-writable Program
//! Files), and local rather than roaming since the venv + model downloads are
//! multi-GB.

use std::path::{Path, PathBuf};

use tauri::path::BaseDirectory;
use tauri::{AppHandle, Manager};

/// `<exe_dir>/data` iff a `portable` marker file sits next to the exe (the
/// portable zip ships it; installers don't).
pub fn portable_data_root() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    dir.join("portable").exists().then(|| dir.join("data"))
}

/// Root for all writable state: the portable data dir, else the OS user
/// app-local-data dir.
pub fn data_root(app: &AppHandle) -> Result<PathBuf, String> {
    if let Some(root) = portable_data_root() {
        return Ok(root);
    }
    app.path().app_local_data_dir().map_err(|e| e.to_string())
}

fn set_dir(key: &str, path: PathBuf) {
    let _ = std::fs::create_dir_all(&path);
    set_var(key, path.as_os_str());
}

fn set_var(key: &str, val: impl AsRef<std::ffi::OsStr>) {
    // SAFETY: called once at startup, before any threads/webview that read env.
    unsafe { std::env::set_var(key, val) };
}

/// Point the pipeline's read-only model checkpoints at the bundled resources, and
/// pick the onset backend by what's bundled. Only fires in a packaged build (the
/// bundled transcriber resource is present); dev keeps the Settings/.env defaults.
/// beat_transformer.pt must be bundled for transcribe; onsets default to ADTOF
/// (weights ship in the pip package) unless a learned-onset run dir was bundled.
pub fn init_checkpoint_env(app: &AppHandle) {
    let resource = |rel: &str| {
        app.path()
            .resolve(rel, BaseDirectory::Resource)
            .ok()
            .filter(|p| p.exists())
    };
    if resource("python/transcriber").is_none() {
        return; // dev / unbundled: leave Settings defaults in place
    }
    if let Some(beat) = resource("python/transcriber/checkpoints/beat_transformer.pt") {
        set_var("BEAT_TRANSFORMER_CHECKPOINT", beat.as_os_str());
    }
    match resource("python/learned_onsets") {
        Some(dir) => set_var("LEARNED_ONSETS_CHECKPOINT", dir.as_os_str()),
        None => set_var("USE_LEARNED_ONSETS", "false"),
    }
}

/// Point every dependency's cache/download/state dir under `root` via process env
/// (inherited by the spawned uv + sidecar). `full` (portable only) also redirects
/// TEMP - so the frontend's staged inputs + the Python scratch + the fs-plugin
/// `$TEMP` all land under `root` - and the WebView2 user-data folder; an installed
/// build leaves TEMP + the webview folder at their user-writable OS defaults.
pub fn redirect_env(root: &Path, full: bool) {
    let cache = root.join("cache");
    // transcriber Settings + the sidecar outputs. Their defaults (/models,
    // /cache, /outputs) are Docker paths, invalid for a packaged desktop app, so
    // these MUST be set in both modes.
    set_dir("MODELS_DIR", root.join("models"));
    set_dir("CACHE_DIR", cache.join("transcriber"));
    set_dir("DRUMJOT_OUTPUTS_DIR", root.join("outputs"));
    // torch / HuggingFace model downloads (MERT, etc.) + uv's package cache and
    // its managed-Python install.
    set_dir("HF_HOME", cache.join("huggingface"));
    set_dir("TORCH_HOME", cache.join("torch"));
    set_dir("XDG_CACHE_HOME", cache.clone());
    set_dir("UV_CACHE_DIR", cache.join("uv"));
    set_dir("UV_PYTHON_INSTALL_DIR", cache.join("uv-python"));
    if full {
        let tmp = root.join("tmp");
        set_dir("TMPDIR", tmp.clone());
        set_dir("TEMP", tmp.clone());
        set_dir("TMP", tmp);
        #[cfg(windows)]
        set_dir("WEBVIEW2_USER_DATA_FOLDER", root.join("webview"));
    }
}
