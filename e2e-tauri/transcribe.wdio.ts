/**
 * End-to-end TRANSCRIPTION e2e: a full transcribe through the desktop app,
 * verifying the transcribed *notes* (not just the beat grid).
 *
 * Loads a known drum-loop fixture and triggers a full transcribe via the real
 * frontend client -- `window.drumjot.desktopTranscribe(path, params)` ->
 * `backendClient()` -> Rust broker -> Python sidecar -> the whole ONNX pipeline
 * (separation -> onsets -> beats -> MIDI) -> `fromMidi` -> loaded into the editor
 * -- then asserts the frontend rendered a transcription that matches the input:
 * the beat tracker's tempo AND actual drum notes across multiple lanes (the
 * fixture is a rock beat: kick 4-on-the-floor, snare backbeat, hi-hat 8ths).
 * Exercises every ONNX model (separation, onsets, beats) end-to-end via the real
 * WebKitGTK webview (@wdio/tauri-service), not just beat_this.
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
 */
import { spawnSync } from 'node:child_process'
import { existsSync, mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const repoRoot = fileURLToPath(new URL('../', import.meta.url))
const venvPython = join(repoRoot, 'transcriber/.venv/bin/python3')
const makeDrums = join(repoRoot, 'e2e-tauri/fixtures/make_drums.py')
const modelsDir = process.env.MODELS_DIR ?? '/models'
const enabled = process.env.DRUMJOT_E2E_TRANSCRIBE != null
const BPM = 120
const BARS = 8

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

// Mark the transcription capability installed over real IPC (persists to the
// data dir's capabilities.json). desktopTranscribe's point-of-use gate reads
// this at boot; the real sidecar still runs on DRUMJOT_SIDECAR_PYTHON, so no
// actual install happens -- this only flips the UI gate.
function setTranscriptionInstalled(installed: boolean): Promise<unknown> {
  return browser.execute(
    (v: boolean) =>
      (
        window as unknown as {
          __TAURI__: { core: { invoke: (c: string, a: unknown) => Promise<unknown> } }
        }
      ).__TAURI__.core.invoke('set_capability_installed', { id: 'transcription', installed: v }),
    installed,
  )
}

let audioPath: string

describe('End-to-end transcription through the app (full ONNX pipeline)', function () {
  before(async function () {
    if (!enabled) {
      this.skip() // opt-in: heavy, needs a GPU + the full model set
    }
    const missing = REQUIRED.filter((f) => !existsSync(resolve(modelsDir, f)))
    if (missing.length > 0) {
      throw new Error(
        `MODELS_DIR (${modelsDir}) missing ${missing.length} file(s): ${missing.join(', ')}. ` +
          'Run `python -m app.pipeline.provision transcription` first.',
      )
    }
    audioPath = join(mkdtempSync(join(tmpdir(), 'drumjot-tx-')), 'drums.wav')
    const gen = spawnSync(venvPython, [makeDrums, audioPath, String(BPM), String(BARS)], {
      encoding: 'utf8',
    })
    if (gen.status !== 0) {
      throw new Error(`drum-loop fixture generation failed: ${gen.stderr || gen.error}`)
    }

    // Install the capability, then reload so the boot capability-refresh reads it
    // as ready and desktopTranscribe's gate passes with no install prompt.
    await setTranscriptionInstalled(true)
    await browser.refresh()
    await browser.waitUntil(
      () => browser.execute(() => typeof (window as { drumjot?: unknown }).drumjot !== 'undefined'),
      { timeout: 30_000, timeoutMsg: 'app did not re-boot after reload' },
    )
  })

  after(async () => {
    // Reset the flag we flipped so the data dir isn't left dirty for later runs.
    if (enabled) await setTranscriptionInstalled(false).catch(() => {})
  })

  it('transcribes a drum loop into the correct notes + tempo', async () => {
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
            structural?: {
              layers: Array<{ bars: Array<{ tracks: Record<string, { notes: unknown[] }> }> }>
            }
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

      // Aggregate the transcribed notes per drum lane across every layer/bar.
      const notesByLane: Record<string, number> = {}
      let totalNotes = 0
      for (const layer of dj.jotEditorStore.structural?.layers ?? []) {
        for (const bar of layer.bars) {
          for (const [pitch, track] of Object.entries(bar.tracks ?? {})) {
            const count = track.notes?.length ?? 0
            notesByLane[pitch] = (notesByLane[pitch] ?? 0) + count
            totalNotes += count
          }
        }
      }
      return {
        loaded: dj.jotEditorStore.jot != null,
        dominantBpm: dj.jotEditorStore.tempo?.dominantBpmAndTime?.dominantBpm ?? null,
        notesByLane,
        totalNotes,
        laneCount: Object.values(notesByLane).filter((c) => c > 0).length,
      }
    }, audioPath)

    if ('error' in outcome) {
      throw new Error(`desktopTranscribe failed: ${outcome.error}`)
    }
    // Surface the transcription so the first GPU run's actual output is visible
    // (the correctness thresholds below are intentionally lenient until then).
    console.log('[transcribe e2e] result:', JSON.stringify(outcome))

    // The full ONNX pipeline ran and its result loaded into the editor.
    expect(outcome.loaded).toBe(true)
    // Beat correctness: the tracker recovered the fixture's tempo (120 BPM).
    expect(outcome.dominantBpm).toBeGreaterThan(100)
    expect(outcome.dominantBpm).toBeLessThan(140)
    // Transcription correctness: real drum notes were produced (not an empty or
    // beats-only result), spread across multiple lanes -- the fixture is a
    // multi-instrument rock beat (kick 4-on-the-floor + snare backbeat + hats),
    // so a faithful transcription lands notes in several lanes.
    //
    // NOTE: thresholds are provisional (separation + onset detection isn't exact
    // and this hasn't been run on a GPU yet). On the first real run, tighten
    // against the logged `notesByLane` -- e.g. assert kick 'k' ~= BARS*4 and
    // snare 's' ~= BARS*2 within tolerance, and check per-beat positions.
    expect(outcome.totalNotes).toBeGreaterThanOrEqual(BARS) // >= ~1 hit/bar minimum
    expect(outcome.laneCount).toBeGreaterThanOrEqual(2)
  })
})
