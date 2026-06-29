//! Hardware detection + capability install state.
//!
//! The accelerator probe picks the torch wheel variant the frontend offers;
//! the capability-state file is the persisted record the (point-of-use)
//! installer writes on success and the frontend `CapabilityStore` reads. Actual
//! dependency installation (uv sync of the multi-GB stack) is intentionally not
//! performed here, see the spec's capability-mechanism section.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use serde::Serialize;
use tauri::ipc::Channel;
use tauri::path::BaseDirectory;
use tauri::{AppHandle, Manager};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

/// Lowest NVIDIA driver major version the cu128 wheels run on (see the
/// transcriber pyproject note: cu128 needs driver 570+).
const CUDA_MIN_DRIVER_MAJOR: u32 = 570;

#[derive(Serialize)]
pub struct AcceleratorInfo {
    /// `cuda` | `mps` | `cpu` (rocm/directml detection is future work).
    pub kind: String,
    pub gpu_name: Option<String>,
    pub driver_version: Option<String>,
    /// NVIDIA driver new enough for the cu128 build.
    pub meets_cuda_min: bool,
}

impl AcceleratorInfo {
    fn plain(kind: &str) -> Self {
        Self {
            kind: kind.to_string(),
            gpu_name: None,
            driver_version: None,
            meets_cuda_min: false,
        }
    }
}

#[tauri::command]
pub async fn detect_accelerator() -> AcceleratorInfo {
    if cfg!(target_os = "macos") {
        return AcceleratorInfo::plain("mps");
    }
    if let Some(info) = detect_nvidia().await {
        return info;
    }
    AcceleratorInfo::plain("cpu")
}

async fn detect_nvidia() -> Option<AcceleratorInfo> {
    // Bound the probe: a wedged driver (post-CUDA-crash, during a GPU reset) can
    // make nvidia-smi hang, which would otherwise park the command forever.
    let out = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        Command::new("nvidia-smi")
            .args(["--query-gpu=name,driver_version", "--format=csv,noheader"])
            .output(),
    )
    .await
    .ok()? // timed out -> treat as no NVIDIA
    .ok()?; // spawn/io error -> ditto
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().next()?.trim();
    let mut parts = line.split(',').map(|s| s.trim().to_string());
    let gpu_name = parts.next().filter(|s| !s.is_empty());
    let driver_version = parts.next().filter(|s| !s.is_empty());
    let meets_cuda_min = driver_version
        .as_deref()
        .and_then(driver_major)
        .map(|m| m >= CUDA_MIN_DRIVER_MAJOR)
        .unwrap_or(false);
    Some(AcceleratorInfo {
        kind: "cuda".to_string(),
        gpu_name,
        driver_version,
        meets_cuda_min,
    })
}

fn driver_major(v: &str) -> Option<u32> {
    v.split('.').next()?.parse().ok()
}

fn state_path(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir.join("capabilities.json"))
}

fn read_states(app: &AppHandle) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    let path = state_path(app)?;
    if !path.exists() {
        return Ok(serde_json::Map::new());
    }
    let text = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    serde_json::from_str(&text).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn capability_states(app: AppHandle) -> Result<serde_json::Value, String> {
    Ok(serde_json::Value::Object(read_states(&app)?))
}

#[tauri::command]
pub fn set_capability_installed(app: AppHandle, id: String, installed: bool) -> Result<(), String> {
    let mut states = read_states(&app)?;
    states.insert(id, serde_json::json!({ "installed": installed }));
    let path = state_path(&app)?;
    let text = serde_json::to_string_pretty(&states).map_err(|e| e.to_string())?;
    // Write to a temp sibling then rename, so a crash mid-write can't leave a
    // truncated/corrupt capabilities.json.
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, text).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, &path).map_err(|e| e.to_string())
}

/// Path to the python interpreter inside a venv (platform-specific layout).
pub fn venv_python(venv: &Path) -> PathBuf {
    if cfg!(windows) {
        venv.join("Scripts").join("python.exe")
    } else {
        venv.join("bin").join("python")
    }
}

/// The app-managed capability venv (separate from the dev `transcriber/.venv`).
pub fn app_venv(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(app.path().app_data_dir().map_err(|e| e.to_string())?.join("venv"))
}

/// The `uv` to run: explicit override, else the binary bundled at
/// `$RESOURCE/bin/uv` (packaged builds), else `uv` on PATH (dev). The bundled
/// copy is marked executable on unix (resources don't preserve the mode).
fn resolve_uv(app: &AppHandle) -> String {
    if let Ok(p) = std::env::var("DRUMJOT_UV") {
        return p;
    }
    let rel = if cfg!(windows) { "bin/uv.exe" } else { "bin/uv" };
    if let Ok(bundled) = app.path().resolve(rel, BaseDirectory::Resource) {
        if bundled.exists() {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(&bundled, std::fs::Permissions::from_mode(0o755));
            }
            return bundled.to_string_lossy().into_owned();
        }
    }
    "uv".to_string()
}

/// Directory holding the transcriber pyproject + uv.lock the capability groups
/// are defined in: explicit override, else the bundled `$RESOURCE/python/
/// transcriber` (packaged), else the in-repo `../transcriber` (dev).
fn resolve_transcriber_dir(app: &AppHandle) -> PathBuf {
    if let Ok(d) = std::env::var("DRUMJOT_TRANSCRIBER_DIR") {
        return PathBuf::from(d);
    }
    if let Ok(bundled) = app.path().resolve("python/transcriber", BaseDirectory::Resource) {
        if bundled.exists() {
            return bundled;
        }
    }
    PathBuf::from("../transcriber")
}

/// Progress for a capability install, streamed to the webview.
#[derive(Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InstallEvent {
    Line { line: String },
    Done,
    Error { message: String },
}

/// Install (or re-resolve) the app capability venv to exactly `groups` via
/// `uv sync --no-default-groups --group <g>…`. `groups` is the union of every
/// capability that should be present afterwards (uv sync replaces the env), so
/// the frontend passes the full desired set, not just the new capability's.
/// uv's progress lines (stderr) stream through `on_event`.
#[tauri::command]
pub async fn install_capability(
    app: AppHandle,
    id: String,
    groups: Vec<String>,
    on_event: Channel<InstallEvent>,
) -> Result<(), String> {
    let venv = app_venv(&app)?;
    let dir = resolve_transcriber_dir(&app);
    log::info!("[install:{id}] uv sync groups={groups:?} -> {}", venv.display());

    let uv = resolve_uv(&app);
    let mut cmd = Command::new(&uv);
    // Pin 3.11 to match the bundled cp311 wheels (see prepare-desktop-resources).
    cmd.arg("sync").arg("--no-default-groups").arg("--python").arg("3.11");
    for group in &groups {
        cmd.arg("--group").arg(group);
    }
    let mut child = cmd
        .current_dir(&dir)
        .env("UV_PROJECT_ENVIRONMENT", &venv)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("failed to spawn uv ({uv}): {e}"))?;

    let stdout = child.stdout.take().ok_or("uv has no stdout")?;
    let stderr = child.stderr.take().ok_or("uv has no stderr")?;
    // uv reports progress on stderr; forward both streams as lines.
    let forward = |reader: tokio::process::ChildStdout| {
        let sink = on_event.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(reader).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let _ = sink.send(InstallEvent::Line { line });
            }
        })
    };
    let out_task = forward(stdout);
    let err_sink = on_event.clone();
    let err_task = tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let _ = err_sink.send(InstallEvent::Line { line });
        }
    });

    let status = child.wait().await.map_err(|e| e.to_string())?;
    let _ = out_task.await;
    let _ = err_task.await;

    if status.success() {
        let _ = on_event.send(InstallEvent::Done);
        Ok(())
    } else {
        let message = format!("uv sync failed ({status})");
        let _ = on_event.send(InstallEvent::Error { message: message.clone() });
        Err(message)
    }
}
