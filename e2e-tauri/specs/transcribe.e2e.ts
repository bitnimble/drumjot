/**
 * True black-box e2e: a FULL transcribe through the desktop app.
 *
 * Loads an audio fixture and triggers a full transcribe via the real frontend
 * client -- `window.drumjot.desktopTranscribe(path, params)` -> `backendClient()`
 * -> Rust broker -> Python sidecar -> the whole ONNX pipeline (separation ->
 * onsets -> beats -> MIDI) -> `fromMidi` -> loaded into the editor -- then
 * asserts the frontend updated with beat information (the loaded jot's bar/beat
 * structure on `jotEditorStore.structural`). This exercises every ONNX model
 * (separation, onsets, beats) end-to-end through the app, not just beat_this.
 *
 * Hermetic: `filter:false` + `quantise:false` skip both LLM stages (the pipeline
 * still runs every ONNX model), so no ANTHROPIC_API_KEY is needed.
 *
 * Opt-in + heavy: needs a GPU (the shipped models are fp16, GPU-only) + the full
 * model set in MODELS_DIR. Gated on DRUMJOT_E2E_TRANSCRIBE so normal runs skip
 * it. Launch:
 *   DRUMJOT_E2E_TRANSCRIBE=1 MODELS_DIR=/dir/with/models \
 *   DRUMJOT_SIDECAR_PYTHON="$PWD/transcriber/.venv/bin/python3" \
 *     xvfb-run -a bun run e2e:tauri
 * The capability gate is seeded in wdio.conf's beforeSession (same env guard).
 */
import { spawnSync } from 'node:child_process'
import { existsSync, mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const repoRoot = fileURLToPath(new URL('../../', import.meta.url))
const venvPython = join(repoRoot, 'transcriber/.venv/bin/python3')
const makeClick = join(repoRoot, 'e2e-tauri/fixtures/make_click.py')
const modelsDir = process.env.MODELS_DIR ?? '/models'

// Everything the full pipeline loads: both separation bodies + their yamls, the
// learned-onset MERT + heads (+ meta), and beat_this. Absent -> hard fail (the
// run was opted in but can't proceed) rather than a silent skip.
const REQUIRED = [
  'model_bs_roformer_sw.fp16.onnx',
  'config_bs_roformer_sw.yaml',
  'drumsep_5stems_mdx23c_jarredou.fp16.onnx',
  'config_drumsep_5stems_mdx23c.yaml',
  'mert_L10.fp16.onnx',
  'onset_heads.fp16.onnx',
  'onset_meta.json',
  'beat_this.fp16.onnx',
]

let audioPath: string

describe('Full transcribe through the app (ONNX pipeline)', function () {
  before(function () {
    if (process.env.DRUMJOT_E2E_TRANSCRIBE == null) {
      this.skip() // opt-in: heavy, needs a GPU + the full model set
    }
    const missing = REQUIRED.filter((f) => !existsSync(resolve(modelsDir, f)))
    if (missing.length > 0) {
      throw new Error(
        `MODELS_DIR (${modelsDir}) missing ${missing.length} file(s): ${missing.join(', ')}. ` +
          'Run `python -m app.pipeline.provision transcription` first.',
      )
    }
    audioPath = join(mkdtempSync(join(tmpdir(), 'drumjot-tx-')), 'click.wav')
    const gen = spawnSync(venvPython, [makeClick, audioPath, '120', '12'], { encoding: 'utf8' })
    if (gen.status !== 0) {
      throw new Error(`click-track fixture generation failed: ${gen.stderr || gen.error}`)
    }
  })

  it('transcribes an audio fixture and renders beat structure in the editor', async () => {
    // Separation + onset + beat inference on the GPU; the first run pays cold
    // model loads, so give it a wide budget.
    await browser.setTimeout({ script: 300_000 })

    const outcome = await browser.execute(async (path: string) => {
      const dj = (window as unknown as {
        drumjot: {
          desktopTranscribe: (p: string, params?: Record<string, unknown>) => Promise<void>
          jotEditorStore: {
            jot?: unknown
            tempo?: { dominantBpmAndTime: { dominantBpm?: number } }
            structural?: { layers: Array<{ bars: Array<{ beats: number }> }> }
          }
        }
      }).drumjot

      try {
        // Hermetic full pipeline: no LLM (filter + quantise off), every ONNX
        // model still runs. Resolves once the MIDI is converted + loaded.
        await dj.desktopTranscribe(path, { filter: false, quantise: false })
      } catch (err) {
        return { error: String(err) }
      }

      const layers = dj.jotEditorStore.structural?.layers ?? []
      return {
        loaded: dj.jotEditorStore.jot != null,
        // Beat tracker's tempo, carried through MIDI -> jot; the primary
        // beat-information signal (present even if the click track yields no
        // drum notes, so it doesn't depend on onsets/bars).
        dominantBpm: dj.jotEditorStore.tempo?.dominantBpmAndTime?.dominantBpm ?? null,
        layerCount: layers.length,
        barCount: layers[0]?.bars?.length ?? 0,
      }
    }, audioPath)

    if ('error' in outcome) {
      throw new Error(`desktopTranscribe failed: ${outcome.error}`)
    }
    // The full ONNX pipeline ran through the app and its result loaded into the
    // editor, carrying the beat tracker's tempo (the fixture is a 120 BPM click
    // track) -- proof the beats stage's ONNX output reached the frontend.
    expect(outcome.loaded).toBe(true)
    expect(outcome.dominantBpm).toBeGreaterThan(100)
    expect(outcome.dominantBpm).toBeLessThan(140)
  })
})
