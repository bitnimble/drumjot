// Stages the Python backend + the `uv` binary into src-tauri/resources/ so a
// packaged build can self-install capabilities on a clean machine -- with NO
// system git and NO C compiler needed at runtime. Wired into tauri.conf's
// beforeBuildCommand; a no-op for `tauri dev`, which uses the in-repo
// ../transcriber directly.
//
// The transcriber pins a few deps to git (madmom is Cython, so it also builds
// from source). To avoid requiring git + a toolchain on the user's machine, we
// prebuild those as wheels here (build host has git + a compiler), bundle the
// wheelhouse, rewrite the bundled pyproject to install them from there
// (find-links + pinned versions), pin Python to match the wheel ABI, and
// re-lock. Runtime `uv sync` then installs everything from wheels.
//
// Layout produced (mirrored into the app's $RESOURCE/ at bundle time):
//   resources/python/transcriber/{pyproject.toml,uv.lock,app/,wheels/}
//   resources/python/dsp/{pyproject.toml,drumjot_dsp/}   (../dsp for uv.sources)
//   resources/bin/uv[.exe]                                (host uv, if found)
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { copyFile, cp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { delimiter, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// App venv Python. Pins the prebuilt-wheel ABI (cp311) and the runtime venv so
// they match; keep in sync with the Rust installer's `uv sync --python`.
const PY = '3.11';

const repo = join(dirname(fileURLToPath(import.meta.url)), '..');
const out = join(repo, 'src-tauri', 'resources');
const pyOut = join(out, 'python');
const binOut = join(out, 'bin');
const wheelCache = join(out, 'wheel-cache'); // persists across builds; not bundled

const skipJunk = (src: string): boolean =>
  !src.includes('__pycache__') && !src.endsWith('.pyc') && !src.includes('.egg-info');

function findOnPath(name: string): string | null {
  const exts = process.platform === 'win32' ? ['.exe', '.cmd', ''] : [''];
  for (const dir of (process.env.PATH ?? '').split(delimiter)) {
    for (const ext of exts) {
      const candidate = join(dir, name + ext);
      if (existsSync(candidate)) return candidate;
    }
  }
  return null;
}

function run(cmd: string, args: string[]): void {
  const r = spawnSync(cmd, args, { stdio: 'inherit' });
  if (r.status !== 0) {
    throw new Error(`command failed (${r.status ?? r.signal}): ${cmd} ${args.join(' ')}`);
  }
}

const depName = (spec: string): string => spec.split('@')[0].trim();
const venvPython = (venv: string): string =>
  join(venv, process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');

// --- stage the Python source (sidecar + pipeline + dsp path source) ---------
await rm(pyOut, { recursive: true, force: true });
await mkdir(join(pyOut, 'transcriber'), { recursive: true });
await mkdir(join(pyOut, 'dsp'), { recursive: true });
await mkdir(binOut, { recursive: true });

await copyFile(join(repo, 'transcriber/pyproject.toml'), join(pyOut, 'transcriber/pyproject.toml'));
await copyFile(join(repo, 'transcriber/uv.lock'), join(pyOut, 'transcriber/uv.lock'));
await cp(join(repo, 'transcriber/app'), join(pyOut, 'transcriber/app'), {
  recursive: true,
  filter: skipJunk,
});
await copyFile(join(repo, 'dsp/pyproject.toml'), join(pyOut, 'dsp/pyproject.toml'));
await cp(join(repo, 'dsp/drumjot_dsp'), join(pyOut, 'dsp/drumjot_dsp'), {
  recursive: true,
  filter: skipJunk,
});

// --- bundle uv (uv fetches its own managed Python when it syncs) ------------
const uv = findOnPath('uv');
if (!uv) {
  // The shipped app can't function without uv: every capability install is a
  // `uv sync`, and the git deps are vendored to wheels here (which needs uv).
  // A `console.warn` here once let a uv-less bundle ship silently broken, so
  // fail the build instead.
  throw new Error(
    '[desktop-resources] uv not found on PATH; cannot stage the desktop bundle. ' +
      'Install uv (https://docs.astral.sh/uv/) before building.',
  );
}
await copyFile(uv, join(binOut, process.platform === 'win32' ? 'uv.exe' : 'uv'));
console.log(`[desktop-resources] bundled uv from ${uv}`);

// --- vendor the git/source deps as prebuilt wheels --------------------------
const pyprojectPath = join(pyOut, 'transcriber', 'pyproject.toml');
let pyproject = await readFile(pyprojectPath, 'utf8');
const gitSpecs = [...pyproject.matchAll(/"([^"]+ @ git\+[^"]+)"/g)].map((m) => m[1]);

if (gitSpecs.length === 0) {
  console.log('[desktop-resources] no git deps to vendor');
} else {
  await mkdir(wheelCache, { recursive: true });
  const cached = (await readdir(wheelCache)).filter((f) => f.endsWith('.whl'));
  const wheelFor = (spec: string, pool: string[]): string | undefined => {
    const norm = depName(spec).replace(/-/g, '_').toLowerCase();
    return pool.find((w) => w.toLowerCase().startsWith(`${norm}-`));
  };

  if (!gitSpecs.every((s) => wheelFor(s, cached))) {
    // Build the wheels in a throwaway py-pinned venv (needs git + a compiler,
    // which the build host has). Cached so madmom's Cython compile is one-time.
    const buildenv = join(out, 'wheel-buildenv');
    await rm(buildenv, { recursive: true, force: true });
    run(uv, ['venv', '--python', PY, buildenv]);
    run(uv, ['pip', 'install', '--python', buildenv, 'pip', 'wheel']);
    run(venvPython(buildenv), ['-m', 'pip', 'wheel', '--no-deps', '--wheel-dir', wheelCache, ...gitSpecs]);
    await rm(buildenv, { recursive: true, force: true });
    console.log(`[desktop-resources] built ${gitSpecs.length} wheels`);
  } else {
    console.log('[desktop-resources] reusing cached wheels');
  }

  const wheelhouse = join(pyOut, 'transcriber', 'wheels');
  await mkdir(wheelhouse, { recursive: true });
  const wheels = (await readdir(wheelCache)).filter((f) => f.endsWith('.whl'));
  for (const w of wheels) {
    await copyFile(join(wheelCache, w), join(wheelhouse, w));
  }

  // Rewrite the bundled pyproject: pin Python to the wheel ABI, swap each git
  // spec for the pinned wheel version, and add the wheelhouse as a find-links.
  // Each rewrite asserts its marker was present: a silent no-op (e.g. pyproject
  // reformatted) would otherwise ship a bundle that still pins >=3.11 or lacks
  // find-links and falls back to git/compiler at install time.
  const mustReplace = (text: string, find: string, repl: string, what: string): string => {
    if (!text.includes(find)) {
      throw new Error(`bundled pyproject rewrite failed: ${what} marker not found (reformatted?)`);
    }
    return text.replace(find, repl);
  };
  pyproject = mustReplace(
    pyproject,
    'requires-python = ">=3.11"',
    `requires-python = "==${PY}.*"`,
    'requires-python',
  );
  for (const spec of gitSpecs) {
    const wheel = wheelFor(spec, wheels);
    if (!wheel) {
      throw new Error(`no wheel built for ${depName(spec)}`);
    }
    pyproject = mustReplace(
      pyproject,
      `"${spec}"`,
      `"${depName(spec)}==${wheel.split('-')[1]}"`,
      depName(spec),
    );
  }
  pyproject = mustReplace(
    pyproject,
    '[tool.uv]\n',
    '[tool.uv]\nfind-links = ["wheels"]\n',
    'find-links',
  );
  await writeFile(pyprojectPath, pyproject);

  // Re-lock so the runtime install pulls the 3 deps from the wheelhouse (no git).
  run(uv, ['lock', '--directory', join(pyOut, 'transcriber')]);
  console.log('[desktop-resources] vendored git deps as wheels + re-locked');
}

console.log(`[desktop-resources] staged Python backend -> ${pyOut}`);
