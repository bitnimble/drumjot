/**
 * True end-to-end test that ONNX is loaded + executed *through the desktop app*.
 *
 * Drives the real WebKitGTK webview (via @wdio/tauri-service) to invoke the
 * `beats` sidecar op on a small click track, exercising the full stack:
 *   webview `invoke('run_job')` -> Rust broker -> Python sidecar -> ONNX Beat
 *   This! model -> beats streamed back over the Tauri Channel.
 * It asserts the result reports `engine === 'onnx'` (the ONNX model actually
 * ran, not a torch/librosa fallback) and a sane 120 BPM grid.
 *
 * Needs `beat_this.fp16.onnx` provisioned into the app's data dir (the sidecar
 * reads models from `<app-local-data>/models`; a MODELS_DIR env is ignored by the
 * app). Skipped (whole suite) when it's absent, mirroring
 * transcriber/tests/test_onnx_model_e2e.py. Beat This! runs on onnxruntime by
 * default (DRUMJOT_BEAT_ONNX). Globals (`browser`, `expect`) come from
 * @wdio/globals; `window.__TAURI__` from the wdio build's withGlobalTauri.
 */
import { spawnSync } from 'node:child_process'
import { existsSync, mkdtempSync } from 'node:fs'
import { homedir, tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const repoRoot = fileURLToPath(new URL('../', import.meta.url))
const venvPython = join(repoRoot, 'transcriber/.venv/bin/python3')
const makeClick = join(repoRoot, 'e2e-tauri/fixtures/make_click.py')
// The app reads models from its own data dir (paths::redirect_env forces
// settings.models_dir = <app-local-data>/models), NOT a MODELS_DIR env. Provision
// there: `MODELS_DIR=<this> python -m app.pipeline.provision transcription`.
const appDataModels = join(
  process.env.XDG_DATA_HOME ?? join(homedir(), '.local/share'),
  'dev.drumjot.studio',
  'models',
)
const modelFile = resolve(appDataModels, 'beat_this.fp16.onnx')

let audioPath: string

describe('Beat detection through the app (ONNX Beat This!)', function () {
  before(function () {
    if (!existsSync(modelFile)) {
      // No provisioned model -> the app would fall back / error; skip like the
      // python gate does. Point MODELS_DIR at a dir with beat_this.fp16.onnx.
      this.skip()
    }
    audioPath = join(mkdtempSync(join(tmpdir(), 'drumjot-e2e-')), 'click.wav')
    const gen = spawnSync(venvPython, [makeClick, audioPath, '120', '12'], { encoding: 'utf8' })
    if (gen.status !== 0) {
      throw new Error(`click-track fixture generation failed: ${gen.stderr || gen.error}`)
    }
  })

  it('runs the ONNX beat model on a click track and returns a 120 BPM grid', async () => {
    // The sidecar spawns Python, imports the pipeline, loads the 40 MB onnx, and
    // runs inference; the first run is the slow one.
    await browser.setTimeout({ script: 120_000 })

    const outcome = await browser.execute(
      async (path: string, jobId: string) => {
        const core = (window as unknown as { __TAURI__: { core: {
          invoke: (cmd: string, args: unknown) => Promise<unknown>
          Channel: new () => { onmessage: (frame: unknown) => void }
        } } }).__TAURI__.core

        const frames: Array<Record<string, unknown>> = []
        const channel = new core.Channel()
        const done = new Promise<void>((res) => {
          channel.onmessage = (frame) => {
            frames.push(frame as Record<string, unknown>)
            const t = (frame as { type?: string }).type
            if (t === 'result' || t === 'error') res()
          }
        })

        let invokeError: string | undefined
        try {
          await core.invoke('run_job', {
            request: {
              v: 1,
              type: 'request',
              id: jobId,
              op: 'beats',
              args: { audio: { kind: 'path', path }, params: {} },
            },
            onEvent: channel,
          })
          await done
        } catch (err) {
          invokeError = String(err)
        }
        return { frames, invokeError }
      },
      audioPath,
      `e2e-${Date.now()}`,
    )

    // browser.execute serializes a returned `undefined` field to `null`, so an
    // absent error is null here; only a real failure is a non-null string.
    if (outcome.invokeError != null) {
      throw new Error(`run_job invoke failed: ${outcome.invokeError}`)
    }

    const result = outcome.frames.at(-1) as
      | { type: string; data?: { engine: string; count: number; beats: number[]; downbeats: number[] } }
      | undefined
    expect(result?.type).toBe('result')

    const data = result!.data!
    // The ONNX Beat This! model actually ran (not the torch / librosa fallback).
    expect(data.engine).toBe('onnx')
    expect(data.count).toBeGreaterThan(10)

    const beats = data.beats
    const diffs = beats.slice(1).map((b, i) => b - beats[i])
    const median = diffs.slice().sort((a, b) => a - b)[Math.floor(diffs.length / 2)]
    expect(median).toBeGreaterThan(0.4) // ~0.5s inter-beat interval at 120 BPM
    expect(median).toBeLessThan(0.6)
    expect(data.downbeats.length).toBeGreaterThanOrEqual(1)

    // The pipeline reported a beat-detection stage (progress streamed too).
    const stages = outcome.frames.filter((f) => f.type === 'progress').map((f) => f.stage)
    expect(stages).toContain('beats')
  })
})
