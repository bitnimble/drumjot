// Thin wrapper around the Tauri CLI that redirects build artifacts per
// `DRUMJOT_BUILD_DIR` (see build_env.ts). Every `bun run tauri …` / android:* /
// desktop build routes through here, so a single env var relocates the heavy
// Rust/Tauri outputs off the local disk. Args after the script name pass
// straight through (`… tauri-build.ts android build --apk` → `tauri android
// build --apk`).
import { spawnSync } from 'node:child_process';
import { buildOutputEnv, relocateAndroidArtifacts } from './build_env';

const args = process.argv.slice(2);
const extra = buildOutputEnv();
const env = { ...process.env, ...extra };
if (extra.CARGO_TARGET_DIR) {
  console.error(`[tauri-build] CARGO_TARGET_DIR=${extra.CARGO_TARGET_DIR}`);
}
const r = spawnSync('tauri', args, { stdio: 'inherit', env });
// `android build` packages the APK/AAB outside CARGO_TARGET_DIR; move it too.
if (r.status === 0 && args[0] === 'android' && args[1] === 'build') {
  relocateAndroidArtifacts();
}
process.exit(r.status ?? 1);
