// Desktop capability/hardware commands over the REAL Tauri IPC transport against
// the actual binary. This is the end-to-end wire, real `@tauri-apps/api/core`
// invoke -> real serde serialization -> real command dispatch -> real on-disk
// state -> back, that no other layer reaches: the Rust `tauri::test` runs on the
// mock runtime (never serializes), the `@tauri-apps/api/mocks` unit tests stub
// the transport, and Playwright can't see these desktop-only commands at all
// (`desktopCapabilities()` is null in the web build).
describe('Desktop capability + hardware commands (real IPC)', () => {
  it('round-trips set_capability_installed -> capability_states', async () => {
    // A throwaway id so the test never flips a real capability's installed
    // state; the command persists any string key to capabilities.json.
    const read = (): Promise<Record<string, { installed: boolean }>> =>
      browser.tauri.execute(({ core }) => core.invoke('capability_states'));
    const setTrue = (): Promise<void> =>
      browser.tauri.execute(({ core }) =>
        core.invoke('set_capability_installed', { id: '__e2e_roundtrip__', installed: true }),
      );
    const setFalse = (): Promise<void> =>
      browser.tauri.execute(({ core }) =>
        core.invoke('set_capability_installed', { id: '__e2e_roundtrip__', installed: false }),
      );
    try {
      await setTrue();
      expect(await read()).toMatchObject({ __e2e_roundtrip__: { installed: true } });
      await setFalse();
      expect(await read()).toMatchObject({ __e2e_roundtrip__: { installed: false } });
    } finally {
      // Unconditional reset so a failed assertion can't leave the data dir dirty.
      await setFalse().catch(() => {});
    }
  });

  it('detect_accelerator serializes the camelCase shape hardware_info.tsx reads', async () => {
    const info = await browser.tauri.execute(({ core }) => core.invoke('detect_accelerator'));
    expect(['cuda', 'mps', 'cpu']).toContain(info.kind);
    // hardware_info.tsx consumes `meetsCudaMin` / `gpuName` / `driverVersion`;
    // the Rust struct must serialize to those exact keys or the panel silently
    // reads undefined (Device "Unknown", spurious "driver too old" on CUDA).
    expect(typeof info.meetsCudaMin).toBe('boolean');
    expect(info).not.toHaveProperty('meets_cuda_min');
  });

  it('cancel_job for an unknown id is a no-op over IPC (does not reject)', async () => {
    const ok = await browser.tauri.execute(async ({ core }) => {
      await core.invoke('cancel_job', { id: 'no-such-job' });
      return true;
    });
    expect(ok).toBe(true);
  });
});
