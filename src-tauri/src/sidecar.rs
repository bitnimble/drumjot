//! Rust broker between the webview and the Python transcriber sidecar.
//!
//! The webview never talks to the sidecar directly: it invokes `run_job`,
//! passing a Tauri `Channel`, and the broker spawns the Python process, writes
//! the request frame to its stdin, and re-emits each control-protocol frame it
//! reads from stdout back up the channel. No bound TCP port, no socket, the
//! sidecar is fully isolated behind this process. See
//! `docs/superpowers/specs/2026-06-29-desktop-app-design.md`.

use std::collections::HashMap;
use std::process::Stdio;

use serde_json::Value;
use tauri::ipc::Channel;
use tauri::{AppHandle, Manager, State};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::{oneshot, Mutex};

use crate::capability::venv_python;

/// In-flight jobs keyed by request `id`; the value cancels the read loop.
#[derive(Default)]
pub struct SidecarState {
    jobs: Mutex<HashMap<String, oneshot::Sender<()>>>,
}

/// Resolve the Python interpreter that runs the sidecar: the capability-managed
/// app venv when present, else the dev `transcriber/.venv` so the broker is
/// exercisable end-to-end. Override with `DRUMJOT_SIDECAR_PYTHON`.
pub fn resolve_python(app: &AppHandle) -> String {
    if let Ok(p) = std::env::var("DRUMJOT_SIDECAR_PYTHON") {
        return p;
    }
    if let Ok(dir) = app.path().app_data_dir() {
        let py = venv_python(&dir.join("venv"));
        if py.exists() {
            return py.to_string_lossy().into_owned();
        }
    }
    "../transcriber/.venv/bin/python".to_string()
}

/// Run one backend job. `request` is a validated client control-protocol frame
/// (the frontend builds it via `encodeClientMessage`); each backend frame is
/// forwarded verbatim through `on_event`. Resolves when the sidecar emits a
/// terminal `result`/`error`, the stream closes, or the job is cancelled.
#[tauri::command]
pub async fn run_job(
    app: AppHandle,
    state: State<'_, SidecarState>,
    request: Value,
    on_event: Channel<Value>,
) -> Result<(), String> {
    let id = request
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or("request missing id")?
        .to_string();

    let python = resolve_python(&app);
    let mut command = Command::new(&python);
    command
        .args(["-u", "-m", "app.sidecar"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    // Artifacts land in the asset-protocol-scoped appdata dir (see lib.rs setup
    // + the fs:allow-read-file capability) so the webview can read/convert them.
    if let Ok(dir) = app.path().app_data_dir() {
        let outputs = dir.join("outputs");
        let _ = std::fs::create_dir_all(&outputs);
        command.env("DRUMJOT_OUTPUTS_DIR", &outputs);
    }
    let mut child = command
        .spawn()
        .map_err(|e| format!("failed to spawn sidecar ({python}): {e}"))?;

    let mut stdin = child.stdin.take().ok_or("sidecar has no stdin")?;
    let stdout = child.stdout.take().ok_or("sidecar has no stdout")?;
    let stderr = child.stderr.take().ok_or("sidecar has no stderr")?;

    let mut line = serde_json::to_string(&request).map_err(|e| e.to_string())?;
    line.push('\n');
    stdin
        .write_all(line.as_bytes())
        .await
        .map_err(|e| format!("failed to write request: {e}"))?;
    stdin.flush().await.map_err(|e| e.to_string())?;

    let (cancel_tx, mut cancel_rx) = oneshot::channel::<()>();
    state.jobs.lock().await.insert(id.clone(), cancel_tx);

    // stderr is diagnostics only (the protocol owns stdout); drain it to the log.
    tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(l)) = lines.next_line().await {
            log::info!("[sidecar] {l}");
        }
    });

    let mut reader = BufReader::new(stdout).lines();
    let outcome = loop {
        tokio::select! {
            next = reader.next_line() => match next {
                Ok(Some(raw)) => {
                    let raw = raw.trim();
                    if raw.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<Value>(raw) {
                        Ok(frame) => {
                            let terminal = matches!(
                                frame.get("type").and_then(|t| t.as_str()),
                                Some("result") | Some("error")
                            );
                            on_event.send(frame).map_err(|e| e.to_string())?;
                            if terminal {
                                break Ok(());
                            }
                        }
                        Err(e) => log::warn!("[sidecar] dropping malformed frame: {e}: {raw}"),
                    }
                }
                Ok(None) => break Ok(()),
                Err(e) => break Err(format!("sidecar read error: {e}")),
            },
            _ = &mut cancel_rx => {
                let cancel = format!("{{\"v\":1,\"type\":\"cancel\",\"id\":\"{id}\"}}\n");
                let _ = stdin.write_all(cancel.as_bytes()).await;
                let _ = stdin.flush().await;
                let _ = child.start_kill();
                break Ok(());
            }
        }
    };

    state.jobs.lock().await.remove(&id);
    let _ = child.wait().await;
    outcome
}

/// Cooperatively cancel the job with the matching `id`.
#[tauri::command]
pub async fn cancel_job(state: State<'_, SidecarState>, id: String) -> Result<(), String> {
    if let Some(tx) = state.jobs.lock().await.remove(&id) {
        let _ = tx.send(());
    }
    Ok(())
}
