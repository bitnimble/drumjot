// Builds the desktop binary for the WebdriverIO e2e run (see wdio.conf.ts):
//   1. the frontend with VITE_WDIO=1 so the @wdio/tauri-plugin frontend hook is
//      bundled (it's dead-code-eliminated from every normal build);
//   2. the `app` binary with `--features wdio` (the in-app Execute API + the
//      embedded WebDriver server) and `withGlobalTauri` enabled, which the wdio
//      plugin requires. Both are test-only and never touch a shipped build.
import { spawnSync } from 'node:child_process';

function run(cmd: string, args: string[], env: Record<string, string>): void {
  const r = spawnSync(cmd, args, { stdio: 'inherit', env: { ...process.env, ...env } });
  if (r.status !== 0) process.exit(r.status ?? 1);
}

run('bun', ['run', 'build'], { VITE_WDIO: '1' });
// `tauri/custom-protocol` makes the binary serve the embedded `frontendDist`
// (the production path) instead of the dev URL `bun run dev` would host -- it's
// the feature the `tauri` CLI always passes for a real build, and the switch
// that flips Tauri out of dev mode. `wdio` adds the e2e plugins.
run(
  'cargo',
  ['build', '--manifest-path', 'src-tauri/Cargo.toml', '--features', 'wdio,tauri/custom-protocol'],
  // `withGlobalTauri` exposes `window.__TAURI__`, which @wdio/tauri-plugin reads
  // to wire its execute/mock API. Test-only (passed via env to this e2e build,
  // never in tauri.conf.json), so a shipped build doesn't expose the global.
  { TAURI_CONFIG: JSON.stringify({ app: { withGlobalTauri: true } }) },
);
