/**
 * Tauri desktop smoke test (WebDriver via tauri-driver).
 *
 * Launches the built release binary and asserts the webview actually loaded the
 * Drumjot frontend *inside the Tauri runtime*: the window title, the #app React
 * root mounting real content, and the Tauri IPC bridge being present (i.e. this
 * is the desktop shell wiring the frontend to the Rust broker + Python sidecar,
 * not a plain browser tab). It exercises the shell only -- the ONNX inference
 * path is covered headlessly by transcriber/tests/test_onnx_model_e2e.py.
 */
describe('Drumjot desktop shell', () => {
  it('opens a window rendering the frontend', async () => {
    await expect(browser).toHaveTitle('Drumjot')

    const app = await $('#app')
    await app.waitForExist({ timeout: 30_000 })
    await browser.waitUntil(async () => (await app.$$('*')).length > 0, {
      timeout: 30_000,
      timeoutMsg: 'frontend never rendered into #app',
    })
  })

  it('runs inside the Tauri runtime (IPC bridge present)', async () => {
    const hasTauri = await browser.execute(
      // __TAURI_INTERNALS__ exists only in the Tauri webview, never a browser.
      () => typeof (window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__ !== 'undefined',
    )
    expect(hasTauri).toBe(true)
  })
})
