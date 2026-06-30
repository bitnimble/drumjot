import { copyFileSync, existsSync, mkdirSync, readdirSync, statSync, unlinkSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';

/**
 * Build-output relocation. `DRUMJOT_BUILD_DIR` (set in `.env`, e.g. to a roomy
 * `/codebox-workspace` path) moves the heavy Rust/Tauri build artifacts off the
 * repo and the local disk. It maps to `CARGO_TARGET_DIR`, which covers the
 * whole `target/` tree (the bulk, ~17 GB across the desktop + Android NDK
 * targets), desktop bundles (`$CARGO_TARGET_DIR/release/bundle`), every Android
 * `.so`, and the wdio build. Unset means the in-repo default
 * (`src-tauri/target`). The gradle-packaged Android APK is handled separately
 * by {@link relocateAndroidArtifacts} (CARGO_TARGET_DIR can't reach it).
 */
export function buildOutputEnv(): Record<string, string> {
  const dir = process.env.DRUMJOT_BUILD_DIR?.trim();
  if (!dir) return {};
  return { CARGO_TARGET_DIR: resolve(dir, 'cargo-target') };
}

const ANDROID_OUTPUTS = 'src-tauri/gen/android/app/build/outputs';

/**
 * Move the gradle-packaged Android artifacts into `$DRUMJOT_BUILD_DIR/{apk,aab}`.
 * `CARGO_TARGET_DIR` can't reach them, they land in the gitignored
 * `gen/android/app/build/outputs`, so this finishes the redirect and gives one
 * stable install location off the repo. No-op when `DRUMJOT_BUILD_DIR` is unset.
 * Copy + unlink rather than rename, since the build dir is usually a different
 * mount than `gen/android`.
 */
export function relocateAndroidArtifacts(): void {
  const dir = process.env.DRUMJOT_BUILD_DIR?.trim();
  if (!dir) return;
  const root = resolve(dir);
  for (const [sub, ext] of [['apk', '.apk'], ['bundle', '.aab']] as const) {
    const dest = join(root, ext.slice(1));
    for (const file of filesByExt(join(ANDROID_OUTPUTS, sub), ext)) {
      mkdirSync(dest, { recursive: true });
      const target = join(dest, basename(file));
      copyFileSync(file, target);
      unlinkSync(file);
      console.error(`[tauri-build] moved ${basename(file)} -> ${target}`);
    }
  }
}

function filesByExt(root: string, ext: string): string[] {
  if (!existsSync(root)) return [];
  return readdirSync(root).flatMap((entry) => {
    const p = join(root, entry);
    if (statSync(p).isDirectory()) return filesByExt(p, ext);
    return p.endsWith(ext) ? [p] : [];
  });
}
