// Stages the Python backend + the `uv` binary into src-tauri/resources/ so a
// packaged build can self-install capabilities (the Rust `install_capability`
// runs `uv sync` against the bundled transcriber pyproject; the sidecar runs
// from the app venv uv creates). Wired into tauri.conf's beforeBuildCommand;
// a no-op for `tauri dev`, which uses the in-repo ../transcriber directly.
//
// Layout produced (mirrored into the app's $RESOURCE/ at bundle time):
//   resources/python/transcriber/{pyproject.toml,uv.lock,app/}
//   resources/python/dsp/{pyproject.toml,drumjot_dsp/}   (../dsp for uv.sources)
//   resources/bin/uv[.exe]                                (host uv, if found)
import { cp, mkdir, rm, copyFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, join, delimiter } from 'node:path';
import { fileURLToPath } from 'node:url';

const repo = join(dirname(fileURLToPath(import.meta.url)), '..');
const out = join(repo, 'src-tauri', 'resources');
const pyOut = join(out, 'python');
const binOut = join(out, 'bin');

const skipJunk = (src) =>
  !src.includes('__pycache__') && !src.endsWith('.pyc') && !src.includes('.egg-info');

function findOnPath(name) {
  const exts = process.platform === 'win32' ? ['.exe', '.cmd', ''] : [''];
  for (const dir of (process.env.PATH || '').split(delimiter)) {
    for (const ext of exts) {
      const candidate = join(dir, name + ext);
      if (existsSync(candidate)) return candidate;
    }
  }
  return null;
}

await rm(pyOut, { recursive: true, force: true });
await mkdir(join(pyOut, 'transcriber'), { recursive: true });
await mkdir(join(pyOut, 'dsp'), { recursive: true });
await mkdir(binOut, { recursive: true });

// transcriber: dep spec + the app package (source for the sidecar + pipeline).
await copyFile(join(repo, 'transcriber/pyproject.toml'), join(pyOut, 'transcriber/pyproject.toml'));
await copyFile(join(repo, 'transcriber/uv.lock'), join(pyOut, 'transcriber/uv.lock'));
await cp(join(repo, 'transcriber/app'), join(pyOut, 'transcriber/app'), {
  recursive: true,
  filter: skipJunk,
});

// dsp: the `drumjot-dsp` path source (transcriber pyproject references ../dsp).
await copyFile(join(repo, 'dsp/pyproject.toml'), join(pyOut, 'dsp/pyproject.toml'));
await cp(join(repo, 'dsp/drumjot_dsp'), join(pyOut, 'dsp/drumjot_dsp'), {
  recursive: true,
  filter: skipJunk,
});

// uv: copy the build host's binary so a clean target machine needs nothing
// preinstalled (uv itself fetches a managed Python when it syncs). Resolved at
// runtime from $RESOURCE/bin, else PATH.
const uv = findOnPath('uv');
if (uv) {
  const dest = join(binOut, process.platform === 'win32' ? 'uv.exe' : 'uv');
  await copyFile(uv, dest);
  console.log(`[desktop-resources] bundled uv from ${uv}`);
} else {
  console.warn('[desktop-resources] uv not found on PATH; not bundled (runtime falls back to PATH uv)');
}

console.log(`[desktop-resources] staged Python backend -> ${pyOut}`);
