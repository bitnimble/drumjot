import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { Page } from '@playwright/test';

/**
 * Stub the transcriber backend so e2e never depends on the Python
 * service (or a GPU) being up. Intercepts the two routes the frontend
 * can hit through the Vite `/api` proxy:
 *
 *   - `POST /api/transcribe` -> a canned, parseable TranscribeResponse.
 *   - `GET  /api/outputs/**` -> the WAV tone fixture, so the stem-URL
 *     path (drum_stem_url / no_drums_url) is exercisable without the
 *     real /outputs StaticFiles mount.
 *
 * The canned `jot_dsl` is intentionally minimal-but-real DSL so the
 * client-side parser produces a renderable jot.
 */

const TONE_WAV = readFileSync(
  fileURLToPath(new URL('../fixtures/tone.wav', import.meta.url)),
);

export const CANNED_JOT_DSL = '{{ bpm: 120, time: "4/4", title: "Mock Transcription" }}\n| h h h h h h h h | k . s . k . s . |\n';

export type MockTranscriberOptions = {
  /** Override the DSL the fake backend returns. */
  jotDsl?: string;
  /** Populate stem URLs in the response (defaults to true). */
  withStemUrls?: boolean;
};

export async function mockTranscriber(
  page: Page,
  opts: MockTranscriberOptions = {},
): Promise<void> {
  const jotDsl = opts.jotDsl ?? CANNED_JOT_DSL;
  const withStemUrls = opts.withStemUrls ?? true;

  await page.route('**/api/transcribe', async (route) => {
    const body = {
      jot_dsl: jotDsl,
      metadata: {
        initial_tempo: 120,
        initial_time_signature: [4, 4],
        duration_seconds: 4,
        stems_used: ['k', 's', 'h'],
        bars: [],
        has_tempo_changes: false,
        has_time_sig_changes: false,
      },
      refinement: null,
      best_of_k: null,
      candidates: {},
      debug_dir: null,
      drum_stem_url: withStemUrls ? '/outputs/e2e-mock/drum_stem.flac' : null,
      no_drums_url: withStemUrls ? '/outputs/e2e-mock/no_drums.flac' : null,
    };
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(body),
    });
  });

  await page.route('**/api/outputs/**', async (route) => {
    // The fixture is a WAV; the response Content-Type is what
    // decodeAudioData keys off, and the bytes are valid PCM, so the
    // .flac path name is cosmetic here (mock only).
    await route.fulfill({
      status: 200,
      contentType: 'audio/wav',
      body: TONE_WAV,
    });
  });
}
