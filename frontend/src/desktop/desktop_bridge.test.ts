import { afterEach, beforeEach, describe, expect, test } from 'bun:test';
import { clearMocks, mockIPC } from '@tauri-apps/api/mocks';
import { TauriBridge } from './desktop_bridge';
import { type ClientMessage, type ServerMessage } from 'src/net/control_protocol';

// `@tauri-apps/api/mocks` drives the real `@tauri-apps/api/core` invoke/Channel
// machinery through `window.__TAURI_INTERNALS__`, so the bridge exercises its
// actual code path (not a hand-rolled stub). bun has no `window`; point it at
// globalThis (which already carries `crypto`, used to mint callback ids).
const g = globalThis as unknown as { window?: unknown };

beforeEach(() => {
  g.window = globalThis;
});

afterEach(() => {
  clearMocks();
  delete g.window;
});

/** A Channel arrives at the mock handler as the live JS instance; drive its
 *  onmessage to simulate the broker streaming frames up. */
function channelOf(args: unknown): { onmessage: (frame: unknown) => void } {
  return (args as { onEvent: { onmessage: (frame: unknown) => void } }).onEvent;
}

describe('TauriBridge', () => {
  test('detectAccelerator returns the backend payload', async () => {
    mockIPC((cmd) => {
      if (cmd === 'detect_accelerator') {
        return { kind: 'cuda', gpuName: 'RTX 3080', driverVersion: '570.1', meetsCudaMin: true };
      }
    });
    const info = await new TauriBridge().detectAccelerator();
    expect(info).toEqual({
      kind: 'cuda',
      gpuName: 'RTX 3080',
      driverVersion: '570.1',
      meetsCudaMin: true,
    });
  });

  test('capabilityStates returns the persisted map', async () => {
    mockIPC((cmd) => {
      if (cmd === 'capability_states') return { transcription: { installed: true } };
    });
    expect(await new TauriBridge().capabilityStates()).toEqual({
      transcription: { installed: true },
    });
  });

  test('setCapabilityInstalled invokes with id + installed', async () => {
    let captured: unknown;
    mockIPC((cmd, args) => {
      if (cmd === 'set_capability_installed') captured = args;
    });
    await new TauriBridge().setCapabilityInstalled('transcription', true);
    expect(captured).toEqual({ id: 'transcription', installed: true });
  });

  test('runJob decodes each streamed frame and forwards it to onEvent', async () => {
    mockIPC((cmd, args) => {
      if (cmd !== 'run_job') return;
      const ch = channelOf(args);
      ch.onmessage({ v: 1, type: 'progress', id: 'job1', stage: 'separating', frac: 0.5 });
      ch.onmessage({ v: 1, type: 'result', id: 'job1', artifacts: [] });
    });

    const request: ClientMessage = {
      v: 1,
      type: 'request',
      id: 'job1',
      op: 'transcribe',
      args: { audio: { kind: 'path', path: '/a.wav' }, params: {} },
    };
    const got: ServerMessage[] = [];
    await new TauriBridge().runJob(request, (m) => got.push(m));

    expect(got.map((m) => m.type)).toEqual(['progress', 'result']);
    expect(got[0]).toMatchObject({ type: 'progress', frac: 0.5 });
  });

  test('runJob drops a schema-invalid frame instead of forwarding it', async () => {
    mockIPC((cmd, args) => {
      if (cmd !== 'run_job') return;
      const ch = channelOf(args);
      ch.onmessage({ v: 1, type: 'bogus' }); // not a ServerMessage
      ch.onmessage({ v: 1, type: 'result', id: 'job1', artifacts: [] });
    });
    const request: ClientMessage = {
      v: 1,
      type: 'request',
      id: 'job1',
      op: 'transcribe',
      args: { audio: { kind: 'path', path: '/a.wav' }, params: {} },
    };
    const got: ServerMessage[] = [];
    await new TauriBridge().runJob(request, (m) => got.push(m));
    expect(got.map((m) => m.type)).toEqual(['result']);
  });

  test('installCapability forwards uv line frames to onProgress', async () => {
    mockIPC((cmd, args) => {
      if (cmd !== 'install_capability') return;
      const ch = channelOf(args);
      ch.onmessage({ type: 'line', line: 'Resolving dependencies' });
      ch.onmessage({ type: 'line', line: 'Downloading wheels' });
      ch.onmessage({ type: 'done' });
    });
    const lines: string[] = [];
    await new TauriBridge().installCapability('transcription', ['transcription'], (l) =>
      lines.push(l),
    );
    expect(lines).toEqual(['Resolving dependencies', 'Downloading wheels']);
  });

  test('installCapability rejects when the invoke errors', async () => {
    mockIPC((cmd) => {
      if (cmd === 'install_capability') throw new Error('uv sync failed');
    });
    await expect(
      new TauriBridge().installCapability('transcription', ['transcription'], () => {}),
    ).rejects.toThrow('uv sync failed');
  });
});
