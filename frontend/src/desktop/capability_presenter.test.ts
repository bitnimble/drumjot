import { describe, expect, it } from 'bun:test';
import {
  ACCELERATOR_TIER_BYTES,
  capabilityById,
} from './capability_manifest';
import { CapabilityPresenter } from './capability_presenter';
import { CapabilityStore } from './capability_store';
import {
  type AcceleratorInfo,
  type CapabilityStateEntry,
  type DesktopBridge,
} from './desktop_bridge';

class MockBridge implements DesktopBridge {
  accelerator: AcceleratorInfo = { kind: 'cuda', meetsCudaMin: true };
  states: Record<string, CapabilityStateEntry> = {};
  installCalls: string[] = [];
  /** Group sets passed to each installCapability (uv sync) call. */
  groupCalls: string[][] = [];

  async detectAccelerator(): Promise<AcceleratorInfo> {
    return this.accelerator;
  }
  async capabilityStates(): Promise<Record<string, CapabilityStateEntry>> {
    return this.states;
  }
  async setCapabilityInstalled(id: string, installed: boolean): Promise<void> {
    this.installCalls.push(id);
    this.states[id] = { installed };
  }
  async installCapability(
    _id: string,
    groups: string[],
    onProgress: (line: string) => void,
  ): Promise<void> {
    this.groupCalls.push(groups);
    onProgress('Resolved 1 package');
  }
  async runJob(): Promise<void> {}
  async cancelJob(): Promise<void> {}
}

function make(): { presenter: CapabilityPresenter; store: CapabilityStore; bridge: MockBridge } {
  const store = new CapabilityStore();
  const bridge = new MockBridge();
  return { presenter: new CapabilityPresenter({ store, bridge }), store, bridge };
}

const flush = (): Promise<void> => new Promise((r) => setTimeout(r, 0));

describe('CapabilityPresenter', () => {
  it('refresh loads the accelerator + per-capability status', async () => {
    const { presenter, store, bridge } = make();
    bridge.states = { transcription: { installed: true } };
    await presenter.refresh();
    expect(store.accelerator?.kind).toBe('cuda');
    expect(store.statusOf('transcription')).toBe('ready');
    expect(store.statusOf('lyrics')).toBe('not-installed');
  });

  it('counts the accelerator tier once, then dedups it', async () => {
    const { presenter, store } = make();
    store.accelerator = { kind: 'cuda', meetsCudaMin: true };

    const first = presenter.incrementalBytes(['transcription']);
    expect(first).toBe(
      capabilityById('transcription').ownApproxBytes + ACCELERATOR_TIER_BYTES.cuda,
    );

    await presenter.install('transcription');
    // Accelerator already present → lyrics shows only its own weights.
    expect(presenter.incrementalBytes(['lyrics'])).toBe(capabilityById('lyrics').ownApproxBytes);
  });

  it('pulls prerequisites into the incremental size', () => {
    const { presenter, store } = make();
    store.accelerator = { kind: 'cuda', meetsCudaMin: true };
    // japanese requires lyrics (which needs the accelerator tier).
    expect(presenter.incrementalBytes(['lyrics.japanese'])).toBe(
      capabilityById('lyrics.japanese').ownApproxBytes +
        capabilityById('lyrics').ownApproxBytes +
        ACCELERATOR_TIER_BYTES.cuda,
    );
  });

  it('install transitions to ready and persists', async () => {
    const { presenter, store, bridge } = make();
    await presenter.install('transcription');
    expect(store.statusOf('transcription')).toBe('ready');
    expect(bridge.installCalls).toContain('transcription');
  });

  it('install walks prerequisites in order', async () => {
    const { presenter, store, bridge } = make();
    await presenter.install('lyrics.japanese');
    expect(store.statusOf('lyrics')).toBe('ready');
    expect(store.statusOf('lyrics.japanese')).toBe('ready');
    expect(bridge.installCalls.indexOf('lyrics')).toBeLessThan(
      bridge.installCalls.indexOf('lyrics.japanese'),
    );
  });

  it('uv-installs with the capability group set', async () => {
    const { presenter, bridge } = make();
    await presenter.install('transcription');
    expect(bridge.groupCalls.at(-1)).toEqual(['transcription']);
  });

  it('re-syncs the union of prior + new groups', async () => {
    const { presenter, bridge } = make();
    await presenter.install('transcription');
    await presenter.install('lyrics');
    // Installing lyrics must keep transcription's group (uv sync replaces env).
    expect(new Set(bridge.groupCalls.at(-1))).toEqual(new Set(['transcription', 'lyrics']));
  });

  it('skips uv for a credentials-only capability', async () => {
    const { presenter, store, bridge } = make();
    await presenter.install('ai-assist');
    expect(store.statusOf('ai-assist')).toBe('ready');
    expect(bridge.groupCalls).toHaveLength(0);
  });

  it('ensure runs a ready capability immediately', () => {
    const { presenter, store } = make();
    store.statuses.set('transcription', 'ready');
    let ran = false;
    const immediate = presenter.ensure('transcription', () => {
      ran = true;
    });
    expect(immediate).toBe(true);
    expect(ran).toBe(true);
  });

  it('ensure queues a not-installed capability and replays the intent after install', async () => {
    const { presenter, store } = make();
    let ran = false;
    const immediate = presenter.ensure('transcription', () => {
      ran = true;
    });
    expect(immediate).toBe(false);
    expect(ran).toBe(false);
    await flush();
    expect(store.statusOf('transcription')).toBe('ready');
    expect(ran).toBe(true);
  });

  it('requestCapability resolves true immediately when ready (no prompt)', async () => {
    const { presenter, store } = make();
    store.statuses.set('transcription', 'ready');
    await expect(presenter.requestCapability('transcription')).resolves.toBe(true);
    expect(store.pendingGate).toBeUndefined();
  });

  it('requestCapability opens the prompt for a not-installed capability', async () => {
    const { presenter, store } = make();
    const pending = presenter.requestCapability('transcription');
    expect(store.pendingGate).toBe('transcription');
    // confirm → install → resolves true, prompt closes
    await presenter.confirmGate();
    await expect(pending).resolves.toBe(true);
    expect(store.statusOf('transcription')).toBe('ready');
    expect(store.pendingGate).toBeUndefined();
  });

  it('cancelGate resolves the pending request as false and closes the prompt', async () => {
    const { presenter, store } = make();
    const pending = presenter.requestCapability('transcription');
    expect(store.pendingGate).toBe('transcription');
    presenter.cancelGate();
    await expect(pending).resolves.toBe(false);
    expect(store.pendingGate).toBeUndefined();
    expect(store.statusOf('transcription')).toBe('not-installed');
  });

  it('requestCapability skips the prompt for a credentials-only capability', async () => {
    const { presenter, store } = make();
    await expect(presenter.requestCapability('ai-assist')).resolves.toBe(true);
    expect(store.pendingGate).toBeUndefined();
  });

  it('a second requestCapability supersedes the first (which resolves false)', async () => {
    const { presenter, store } = make();
    const first = presenter.requestCapability('transcription');
    const second = presenter.requestCapability('transcription');
    await expect(first).resolves.toBe(false);
    expect(store.pendingGate).toBe('transcription');
    await presenter.confirmGate();
    await expect(second).resolves.toBe(true);
  });
});
