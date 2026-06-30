// Desktop smoke + IPC test against the REAL Tauri shell: the service launches
// the built binary and drives the actual WebKitGTK webview (the engine the
// Chromium Playwright suite can't cover). Layered so a failure pinpoints where:
// DOM render -> plain executeScript -> the Tauri global -> real IPC -> mocking.
// Globals (`browser`, `$`, `$$`, `expect`) come from @wdio/globals.
describe('Drumjot desktop shell (real WebKitGTK webview)', () => {
  it('serves the bundle from the Tauri app', async () => {
    expect(await browser.getTitle()).toBe('Drumjot');
  });

  it('boots the React editor in the real webview', async () => {
    // index.tsx mounts into #app; a child appearing proves the JS bundle ran in
    // the actual Tauri webview (loro WASM init, store graph, toolbar/score).
    await $('#app > *').waitForExist({ timeout: 20_000, timeoutMsg: '#app never rendered' });
    expect((await $$('#app *')).length).toBeGreaterThan(20);
  });

  it('runs JS in the webview via plain WebDriver executeScript', async () => {
    // Vanilla browser.execute (NOT browser.tauri.execute) -- the seam that lets
    // shared web-test logic probe window.* state on the real engine.
    const booted = await browser.execute(
      () => typeof (window as { jotPlayer?: unknown }).jotPlayer !== 'undefined',
    );
    expect(booted).toBe(true);
  });

  it('exposes the global Tauri API (withGlobalTauri)', async () => {
    const hasInvoke = await browser.execute(
      () =>
        typeof (window as { __TAURI__?: { core?: { invoke?: unknown } } }).__TAURI__?.core
          ?.invoke === 'function',
    );
    expect(hasInvoke).toBe(true);
  });

  it('round-trips a real Tauri command over IPC', async () => {
    // webview -> Rust -> back, the seam Playwright stubs. capability_states reads
    // the persisted capability map (an object) with no side effects.
    const states = await browser.tauri.execute(({ core }) => core.invoke('capability_states'));
    expect(states).not.toBeNull();
    expect(typeof states).toBe('object');
  });

  it('mocks a Tauri command through the wdio plugin', async () => {
    const mock = await browser.tauri.mock('capability_states');
    await mock.mockReturnValue({ transcription: { installed: true } });
    const states = await browser.tauri.execute(({ core }) => core.invoke('capability_states'));
    expect(states).toEqual({ transcription: { installed: true } });
  });
});
