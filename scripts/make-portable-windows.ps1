#!/usr/bin/env pwsh
# Assemble a portable (no-installer) Windows build of the Drumjot desktop app and
# zip it for distribution / CI artifacts.
#
# Portable build = the app exe + the bundled resources (Python backend + uv) laid
# out exactly as the installed app expects: Tauri's BaseDirectory::Resource
# resolves `python/` and `bin/` next to the exe (see resolve_uv /
# resolve_transcriber_dir in src-tauri/src/capability.rs), so dropping them
# beside the exe lets it run from any folder without installing. (macOS .app and
# the Linux AppImage are already portable; this is Windows-only.)
#
# Relies on the system WebView2 runtime (Evergreen), same as the installer; a
# machine without it would need the Fixed Version runtime bundled (not done here).
#
# Run after `tauri build` (it produces target/release/<exe> and stages
# src-tauri/resources via the beforeBuildCommand). From the repo root:
#   pwsh scripts/make-portable-windows.ps1
[CmdletBinding()]
param(
  # Display name for the produced exe + zip.
  [string]$AppName = 'Drumjot',
  # Cargo-produced binary name in target/release (package name in Cargo.toml).
  [string]$SourceExe = 'app.exe'
)
$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$releaseDir = Join-Path $repo 'src-tauri/target/release'
$resourcesDir = Join-Path $repo 'src-tauri/resources'
$portableDir = Join-Path $releaseDir 'portable'
$zipPath = Join-Path $releaseDir "$AppName-portable-windows.zip"

$exe = Join-Path $releaseDir $SourceExe
if (-not (Test-Path $exe)) {
  throw "App exe not found at $exe. Run ``tauri build`` first."
}
foreach ($sub in 'python', 'bin') {
  if (-not (Test-Path (Join-Path $resourcesDir $sub))) {
    throw "Resource '$sub' missing under $resourcesDir. Run ``bun run desktop:resources`` (the build's beforeBuildCommand) first."
  }
}

# Fresh output dir.
if (Test-Path $portableDir) { Remove-Item -Recurse -Force $portableDir }
New-Item -ItemType Directory -Force -Path $portableDir | Out-Null

# Exe (rename app.exe -> the product name for a friendlier portable drop;
# resource resolution is by directory, not exe name, so this is safe).
Copy-Item $exe (Join-Path $portableDir "$AppName.exe")

# Any loose runtime DLLs next to the exe (e.g. WebView2Loader.dll when it isn't
# statically linked). A no-op when there are none.
Get-ChildItem $releaseDir -Filter *.dll -ErrorAction SilentlyContinue |
  Copy-Item -Destination $portableDir

# Bundled resources at the layout the app resolves (exe_dir/python, exe_dir/bin).
Copy-Item (Join-Path $resourcesDir 'python') (Join-Path $portableDir 'python') -Recurse
Copy-Item (Join-Path $resourcesDir 'bin') (Join-Path $portableDir 'bin') -Recurse

# Marker that flips the app into portable mode (see src-tauri/src/paths.rs): it
# keeps ALL writable state - venv, uv/torch/HF caches, downloaded Python, temp,
# outputs, webview data - under .\data next to the exe, so deleting this folder
# removes everything. Installers omit it (they use the OS user app-data dir).
New-Item -ItemType File -Force -Path (Join-Path $portableDir 'portable') | Out-Null

# Zip it (overwrite any previous).
if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
Compress-Archive -Path (Join-Path $portableDir '*') -DestinationPath $zipPath

Write-Host "Portable build : $portableDir"
Write-Host "Portable zip   : $zipPath"
