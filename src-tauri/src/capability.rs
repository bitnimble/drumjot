//! Hardware detection + capability install state.
//!
//! The accelerator probe picks the torch wheel variant the frontend offers;
//! the capability-state file is the persisted record the (point-of-use)
//! installer writes on success and the frontend `CapabilityStore` reads. Actual
//! dependency installation (uv sync of the multi-GB stack) is intentionally not
//! performed here, see the spec's capability-mechanism section.

use std::path::PathBuf;

use serde::Serialize;
use tauri::{AppHandle, Manager};
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
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=name,driver_version", "--format=csv,noheader"])
        .output()
        .await
        .ok()?;
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
    std::fs::write(&path, text).map_err(|e| e.to_string())
}
