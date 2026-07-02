// The Python sidecar, the capability installer, and the writable-data-root
// plumbing are all desktop-only: Android can't spawn a Python process or run
// `uv`/`nvidia-smi`, and transcription there goes over the HTTP backend (see
// the frontend `backendClient()`). Gate them out so the Android `cdylib`
// builds with just the webview + the file/dialog plugins.
#[cfg(desktop)]
mod capability;
#[cfg(desktop)]
mod paths;
#[cfg(desktop)]
mod sidecar;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Portable mode: redirect every cache/temp/state env var under <exe>/data
    // BEFORE the builder + webview start, so $TEMP and WEBVIEW2_USER_DATA_FOLDER
    // are already pointed there. No-op for an installed build. Desktop-only.
    #[cfg(desktop)]
    if let Some(root) = paths::portable_data_root() {
        paths::redirect_env(&root, true);
    }

    let builder = tauri::Builder::default()
        .plugin(
            tauri_plugin_log::Builder::default()
                .level(log::LevelFilter::Info)
                .build(),
        )
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init());

    // WebdriverIO e2e: the in-app Execute API + the embedded WebDriver server.
    // Behind the `wdio` cargo feature, so a shipped build never carries them.
    #[cfg(feature = "wdio")]
    let builder = builder
        .plugin(tauri_plugin_wdio::init())
        .plugin(tauri_plugin_wdio_webdriver::init());
    // The wdio capability is granted inside the desktop `.setup()` below (Tauri
    // keeps a single setup slot, so a second `.setup()` here would silently
    // clobber the desktop one in the e2e build where both cfgs are active).

    // Desktop wires the sidecar broker + capability commands and points the
    // pipeline at the bundled data root; mobile ships none of that.
    #[cfg(desktop)]
    let builder = builder
        .manage(sidecar::SidecarState::default())
        .setup(|app| {
            use tauri::Manager;
            use tauri_plugin_fs::FsExt;
            // wdio e2e: grant the plugin permissions at runtime (a static
            // capability file would break release builds where the plugins are
            // absent). Lives here, not in the wdio block, so it doesn't compete
            // for the single setup slot.
            #[cfg(feature = "wdio")]
            {
                let _ = app.handle().add_capability(include_str!("wdio_capability.json"));
            }
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
            capability::available_disk_space,
        ]);

    builder
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
