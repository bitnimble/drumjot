mod capability;
mod paths;
mod sidecar;

use tauri::Manager;
use tauri_plugin_fs::FsExt;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Portable mode: redirect every cache/temp/state env var under <exe>/data
    // BEFORE the builder + webview start, so $TEMP and WEBVIEW2_USER_DATA_FOLDER
    // are already pointed there. No-op for an installed build.
    if let Some(root) = paths::portable_data_root() {
        paths::redirect_env(&root, true);
    }
    tauri::Builder::default()
        .plugin(
            tauri_plugin_log::Builder::default()
                .level(log::LevelFilter::Info)
                .build(),
        )
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .manage(sidecar::SidecarState::default())
        .setup(|app| {
            let root = paths::data_root(app.handle())?;
            // Installed build: redirect caches/downloads/state under the OS
            // app-local-data dir (TEMP + the webview folder stay at their
            // user-writable OS defaults). Portable already redirected in run().
            if paths::portable_data_root().is_none() {
                paths::redirect_env(&root, false);
            }
            // Point the pipeline at the bundled model checkpoints + pick the
            // onset backend by what's bundled (packaged build only).
            paths::init_checkpoint_env(app.handle());
            // Let the webview read the sidecar's artifacts (MIDI via plugin-fs,
            // stems via the asset protocol / convertFileSrc) out of the outputs
            // dir, wherever data_root put it. Scope is locked to that dir.
            let outputs = root.join("outputs");
            let _ = std::fs::create_dir_all(&outputs);
            let _ = app.fs_scope().allow_directory(&outputs, true);
            let _ = app.asset_protocol_scope().allow_directory(&outputs, true);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            sidecar::run_job,
            sidecar::cancel_job,
            capability::detect_accelerator,
            capability::capability_states,
            capability::set_capability_installed,
            capability::install_capability,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
