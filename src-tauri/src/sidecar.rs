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
use tauri::{AppHandle, Runtime, State};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::{oneshot, Mutex};

use crate::capability::{app_venv, venv_python};

/// In-flight jobs keyed by request `id`; the value cancels the read loop.
#[derive(Default)]
pub struct SidecarState {
    jobs: Mutex<HashMap<String, oneshot::Sender<()>>>,
}

/// Resolve the Python interpreter that runs the sidecar: the capability-managed
/// app venv when present, else the dev `transcriber/.venv` so the broker is
/// exercisable end-to-end. Override with `DRUMJOT_SIDECAR_PYTHON`.
pub fn resolve_python<R: Runtime>(app: &AppHandle<R>) -> String {
    if let Ok(p) = std::env::var("DRUMJOT_SIDECAR_PYTHON") {
        return p;
    }
    if let Ok(venv) = app_venv(app) {
        let py = venv_python(&venv);
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
pub async fn run_job<R: Runtime>(
    app: AppHandle<R>,
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
    // The sidecar writes artifacts to DRUMJOT_OUTPUTS_DIR, which the broker sets
    // in the process env at startup (paths::redirect_env) + scopes for the webview
    // (lib.rs setup); the child inherits it. No per-job override needed.
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
    let stderr_task = tokio::spawn(async move {
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
                            // Break (don't `?`-return) on a send failure so the
                            // jobs-map removal + child reap below still run.
                            if let Err(e) = on_event.send(frame) {
                                break Err(e.to_string());
                            }
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
                let cancel = serde_json::json!({"v": 1, "type": "cancel", "id": id}).to_string();
                let _ = stdin.write_all(format!("{cancel}\n").as_bytes()).await;
                let _ = stdin.flush().await;
                let _ = child.start_kill();
                break Ok(());
            }
        }
    };

    state.jobs.lock().await.remove(&id);
    // Close the sidecar's stdin so its blocking readline loop sees EOF and the
    // process exits. Without this, on the normal terminal-frame path the sidecar
    // is still waiting for more input and child.wait() below deadlocks. (The
    // cancel path already start_kill()ed it; the stdout-EOF path already exited.)
    drop(stdin);
    let _ = child.wait().await;
    // Drain remaining stderr before returning (the pipe closed when the child
    // exited, so this completes promptly) rather than detaching the task.
    let _ = stderr_task.await;
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

#[cfg(test)]
mod tests {
    use super::*;
    use tauri::ipc::InvokeResponseBody;
    use tauri::test::{mock_builder, mock_context, noop_assets};
    use tauri::Manager;

    // Serialises the env-mutating tests in this file (only DRUMJOT_SIDECAR_PYTHON).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn mock_app() -> tauri::App<tauri::test::MockRuntime> {
        let app = mock_builder()
            .build(mock_context(noop_assets()))
            .expect("failed to build mock app");
        app.manage(SidecarState::default());
        app
    }

    /// A `Channel<Value>` that records every frame the broker forwards.
    fn recording_channel() -> (Channel<Value>, std::sync::Arc<std::sync::Mutex<Vec<Value>>>) {
        let frames = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = frames.clone();
        let channel = Channel::new(move |body: InvokeResponseBody| {
            if let InvokeResponseBody::Json(s) = body {
                if let Ok(v) = serde_json::from_str::<Value>(&s) {
                    sink.lock().unwrap().push(v);
                }
            }
            Ok(())
        });
        (channel, frames)
    }

    /// Sets an env var and restores its prior value on drop, so a panicking
    /// assertion can't leak it to other tests in the process.
    struct EnvVarGuard {
        key: &'static str,
        prev: Option<std::ffi::OsString>,
    }
    impl EnvVarGuard {
        fn set(key: &'static str, value: &std::ffi::OsStr) -> Self {
            let prev = std::env::var_os(key);
            std::env::set_var(key, value);
            Self { key, prev }
        }
    }
    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match self.prev.take() {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }

    #[test]
    fn resolve_python_prefers_the_env_override() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _env = EnvVarGuard::set("DRUMJOT_SIDECAR_PYTHON", std::ffi::OsStr::new("/custom/python3"));
        let app = mock_app();
        assert_eq!(resolve_python(app.handle()), "/custom/python3");
    }

    #[test]
    fn cancel_unknown_job_is_a_no_op() {
        let app = mock_app();
        let state = app.state::<SidecarState>();
        let r = tauri::async_runtime::block_on(cancel_job(state, "no-such-job".into()));
        assert!(r.is_ok());
    }

    // Drives the broker against a fake sidecar that speaks the control protocol:
    // it must forward each frame up the channel and resolve on the terminal one.
    #[cfg(unix)]
    #[test]
    fn run_job_forwards_frames_until_the_terminal_result() {
        use std::os::unix::fs::PermissionsExt;
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // Ignores the `-u -m app.sidecar` args, consumes the request line, then
        // emits one progress + one terminal result frame and exits.
        let script = std::env::temp_dir().join(format!("drumjot-fake-sidecar-{}.sh", std::process::id()));
        std::fs::write(
            &script,
            r#"#!/bin/sh
read _req
printf '{"v":1,"type":"progress","id":"job1","stage":"separating","frac":0.5}\n'
printf '{"v":1,"type":"result","id":"job1","artifacts":[]}\n'
"#,
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let _env = EnvVarGuard::set("DRUMJOT_SIDECAR_PYTHON", script.as_os_str());

        let app = mock_app();
        let handle = app.handle().clone();
        let state = app.state::<SidecarState>();
        let (channel, frames) = recording_channel();
        let request = serde_json::json!({
            "v": 1, "type": "request", "id": "job1", "op": "transcribe", "args": {}
        });

        let result = tauri::async_runtime::block_on(run_job(handle, state, request, channel));
        assert!(result.is_ok(), "run_job errored: {result:?}");

        let frames = frames.lock().unwrap();
        assert_eq!(frames.len(), 2, "expected progress + result, got {frames:?}");
        assert_eq!(frames[0]["type"], "progress");
        assert_eq!(frames[1]["type"], "result");
        // The job must be reaped from the in-flight map once it terminates.
        assert!(tauri::async_runtime::block_on(app.state::<SidecarState>().jobs.lock()).is_empty());

        let _ = std::fs::remove_file(&script);
    }
}
