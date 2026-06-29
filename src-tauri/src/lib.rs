mod capability;
mod sidecar;

use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
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
            // Let the webview load artifacts the sidecar writes under appdata via
            // the asset protocol (convertFileSrc). Scope is locked to that dir,
            // not the whole filesystem.
            if let Ok(dir) = app.path().app_data_dir() {
                let outputs = dir.join("outputs");
                let _ = std::fs::create_dir_all(&outputs);
                let _ = app.asset_protocol_scope().allow_directory(&outputs, true);
            }
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
