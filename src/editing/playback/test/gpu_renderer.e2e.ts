import { expect, test } from '@playwright/test';

/**
 * Diagnostic for the rendering backend the perf suite's frame-budget
 * assertions implicitly depend on. Default headless Chromium uses SwiftShader
 * (CPU raster), which makes the 120fps medians contention-sensitive. Set
 * `E2E_GPU=1` to launch with the hardware-accelerated ANGLE/EGL flags (see
 * `playwright.config.ts`); this spec then reports the active GL renderer and
 * fails if it's still software, so you can tell whether the GPU actually
 * engaged in this container.
 *
 * Skipped unless `E2E_GPU` is set, so the normal suite never depends on a GPU.
 * Run it with: `E2E_GPU=1 bun run e2e src/editing/playback/test/gpu_renderer.e2e.ts`
 */
test.skip(!process.env.E2E_GPU, 'GPU rendering check only runs with E2E_GPU=1');

/** SwiftShader / llvmpipe / generic software-rasteriser markers. */
const SOFTWARE = /swiftshader|llvmpipe|software|microsoft basic/i;

test('Chromium renders on the GPU (not SwiftShader)', async ({ page }) => {
  await page.goto('/');

  // The unmasked WebGL renderer is the authoritative backend signal:
  // "Google SwiftShader" = software; the card's name (often wrapped in
  // "ANGLE (NVIDIA ...)") = hardware.
  const gl = await page.evaluate(() => {
    const canvas = document.createElement('canvas');
    const ctx = (canvas.getContext('webgl2') ??
      canvas.getContext('webgl')) as WebGLRenderingContext | null;
    if (!ctx) return null;
    const dbg = ctx.getExtension('WEBGL_debug_renderer_info');
    return {
      renderer: String(ctx.getParameter(dbg ? dbg.UNMASKED_RENDERER_WEBGL : ctx.RENDERER)),
      vendor: String(ctx.getParameter(dbg ? dbg.UNMASKED_VENDOR_WEBGL : ctx.VENDOR)),
    };
  });
  console.log('[GPU] WebGL renderer:', JSON.stringify(gl));

  // Supplementary: chrome://gpu's GL_RENDERER line + a couple of feature
  // statuses, logged for context. Wrapped because chrome:// navigation can be
  // restricted depending on the launch mode.
  try {
    await page.goto('chrome://gpu');
    const summary = await page.evaluate(() => {
      const text = document.body.innerText;
      const grab = (label: string) =>
        text.split('\n').find((l) => l.includes(label))?.trim() ?? `${label}: (not found)`;
      return [grab('GL_RENDERER'), grab('Canvas:'), grab('WebGL:'), grab('Vulkan:')].join(' | ');
    });
    console.log('[GPU] chrome://gpu:', summary);
  } catch (e) {
    console.log('[GPU] chrome://gpu unavailable:', (e as Error).message);
  }

  expect(gl, 'no WebGL context, GPU process likely failed to start').not.toBeNull();
  expect(gl!.renderer, `still software-rendered (${gl!.renderer})`).not.toMatch(SOFTWARE);
});
