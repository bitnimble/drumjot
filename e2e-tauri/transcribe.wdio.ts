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
 * transcription model set provisioned into the APP's data dir. The desktop app
 * forces `settings.models_dir` to `<app-local-data>/models` (paths::redirect_env),
 * so the sidecar reads models from there -- a MODELS_DIR passed on the CLI is
 * ignored. Provision it there once, exactly as a real capability install does:
 *   MODELS_DIR="$HOME/.local/share/dev.drumjot.studio/models" \
 *     transcriber/.venv/bin/python -m app.pipeline.provision transcription
 * Then run (gated on DRUMJOT_E2E_TRANSCRIBE so normal runs skip it):
 *   DRUMJOT_E2E_TRANSCRIBE=1 xvfb-run -a bun run e2e:tauri
 */
import { spawnSync } from 'node:child_process'
import { existsSync, mkdtempSync } from 'node:fs'
import { homedir, tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const repoRoot = fileURLToPath(new URL('../', import.meta.url))
const venvPython = join(repoRoot, 'transcriber/.venv/bin/python3')
const makeDrums = join(repoRoot, 'e2e-tauri/fixtures/make_drums.py')
// Where the app actually reads models from (its Linux app-local-data dir), NOT a
// MODELS_DIR env -- the app overrides that. Matches Tauri's app_local_data_dir.
const appDataModels = join(
  process.env.XDG_DATA_HOME ?? join(homedir(), '.local/share'),
  'dev.drumjot.studio',
  'models',
)
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
// data dir's capabilities.json, the same state a real install writes).
// desktopTranscribe's gate confirms against this on disk (CapabilityPresenter),
// so it passes without an install prompt; the sidecar runs the dev transcriber
// venv, so no actual uv install happens.
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
  // The full ONNX pipeline (separation -> onsets -> beats) + cold GPU model
  // loads runs well past wdio.conf's 60s default; give the suite room.
  this.timeout(300_000)

  before(async function () {
    if (!enabled) {
      this.skip() // opt-in: heavy, needs a GPU + the full model set
    }
    const missing = REQUIRED.filter((f) => !existsSync(resolve(appDataModels, f)))
    if (missing.length > 0) {
      throw new Error(
        `App models dir (${appDataModels}) missing ${missing.length} file(s): ${missing.join(', ')}. ` +
          `Provision them there: MODELS_DIR="${appDataModels}" ` +
          'python -m app.pipeline.provision transcription',
      )
    }
    audioPath = join(mkdtempSync(join(tmpdir(), 'drumjot-tx-')), 'drums.wav')
    const gen = spawnSync(venvPython, [makeDrums, audioPath, String(BPM), String(BARS)], {
      encoding: 'utf8',
    })
    if (gen.status !== 0) {
      throw new Error(`drum-loop fixture generation failed: ${gen.stderr || gen.error}`)
    }

    // Mark the transcription capability installed (real IPC -> capabilities.json,
    // the same persisted state a real install writes). No reload needed:
    // desktopTranscribe's gate confirms against this on-disk state before
    // prompting (CapabilityPresenter.requestCapability), so it passes even though
    // the in-memory store loaded the pre-install state at boot.
    await setTranscriptionInstalled(true)
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
    // Surface the transcription for debugging a threshold miss.
    console.log('[transcribe e2e] result:', JSON.stringify(outcome))

    // The full ONNX pipeline ran and its result loaded into the editor.
    expect(outcome.loaded).toBe(true)
    // Beat correctness: the tracker recovered the fixture's tempo (120 BPM).
    expect(outcome.dominantBpm).toBeGreaterThan(115)
    expect(outcome.dominantBpm).toBeLessThan(125)

    // Transcription correctness: the fixture is a known rock beat -- kick
    // 4-on-the-floor (BARS*4 = 32 hits), snare backbeat (BARS*2 = 16), hi-hat
    // 8ths (BARS*8 = 64). Assert each lane lands near its expected count (wide
    // enough for fp16-GPU non-determinism + the odd missed/extra onset -- a real
    // run gets ~31/16/63) AND that the pattern's density ordering holds. An
    // empty, single-lane, or scrambled transcription fails all of these.
    const k = outcome.notesByLane.k ?? 0
    const s = outcome.notesByLane.s ?? 0
    const h = outcome.notesByLane.h ?? 0
    expect(k).toBeGreaterThanOrEqual(BARS * 4 - 8) // kick ~ every beat (32)
    expect(k).toBeLessThanOrEqual(BARS * 4 + 4)
    expect(s).toBeGreaterThanOrEqual(BARS * 2 - 5) // snare ~ backbeats (16)
    expect(s).toBeLessThanOrEqual(BARS * 2 + 5)
    expect(h).toBeGreaterThanOrEqual(BARS * 8 - 16) // hat ~ every 8th (64)
    expect(h).toBeLessThanOrEqual(BARS * 8 + 8)
    // Density ordering that defines the groove: hats (8ths) > kicks (quarters) > snares (backbeat).
    expect(h).toBeGreaterThan(k)
    expect(k).toBeGreaterThan(s)
  })
})
