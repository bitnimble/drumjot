import { describe, expect, it } from 'bun:test';
import {
  PROTOCOL_VERSION,
  type Artifact,
  type ClientMessage,
  type ServerMessage,
} from './control_protocol';
import { type DesktopBridge } from 'src/desktop/desktop_bridge';
import { SidecarBackendClient } from './sidecar_backend';

class MockBridge implements DesktopBridge {
  frames: ServerMessage[] = [];
  lastRequest: ClientMessage | undefined;
  cancelledId: string | undefined;

  async detectAccelerator() {
    return { kind: 'cpu' as const, meetsCudaMin: false };
  }
  async capabilityStates() {
    return {};
  }
  async setCapabilityInstalled() {}
  async installCapability() {}
  async cancelJob(id: string) {
    this.cancelledId = id;
  }
  async runJob(request: ClientMessage, onEvent: (msg: ServerMessage) => void) {
    this.lastRequest = request;
    for (const frame of this.frames) onEvent(frame);
  }
}

const midiArtifact: Artifact = { role: 'midi', ref: { kind: 'path', path: '/out/pred.mid' } };

function frame(msg: Partial<ServerMessage> & { type: ServerMessage['type'] }): ServerMessage {
  return { v: PROTOCOL_VERSION, id: 'job', ...msg } as ServerMessage;
}

describe('SidecarBackendClient', () => {
  it('sends the op + path + params and returns artifacts and data', async () => {
    const bridge = new MockBridge();
    bridge.frames = [frame({ type: 'result', artifacts: [midiArtifact], data: { lines: [] } })];
    const client = new SidecarBackendClient(bridge);

    const result = await client.run('transcribe', { kind: 'path', path: '/in.wav' }, { quantise: true });

    expect(bridge.lastRequest?.type).toBe('request');
    const req = bridge.lastRequest as Extract<ClientMessage, { type: 'request' }>;
    expect(req.op).toBe('transcribe');
    expect(req.args.audio).toEqual({ kind: 'path', path: '/in.wav' });
    expect(req.args.params).toEqual({ quantise: true });
    expect(result.artifacts).toEqual([midiArtifact]);
    expect(result.data).toEqual({ lines: [] });
  });

  it('forwards progress frames to onProgress', async () => {
    const bridge = new MockBridge();
    bridge.frames = [
      frame({ type: 'progress', stage: 'stems_all', frac: 0.2, message: 'go' }),
      frame({ type: 'result', artifacts: [] }),
    ];
    const client = new SidecarBackendClient(bridge);
    const seen: Array<{ stage: string; frac: number }> = [];

    await client.run('separate', { kind: 'path', path: '/in.wav' }, {}, {
      onProgress: (p) => seen.push({ stage: p.stage, frac: p.frac }),
    });

    expect(seen).toEqual([{ stage: 'stems_all', frac: 0.2 }]);
  });

  it('throws when the job emits an error frame', async () => {
    const bridge = new MockBridge();
    bridge.frames = [frame({ type: 'error', code: 'EBAD', message: 'boom', recoverable: false })];
    const client = new SidecarBackendClient(bridge);

    await expect(client.run('alignLyrics', { kind: 'path', path: '/in.wav' }, {})).rejects.toThrow(
      /EBAD: boom/,
    );
  });

  it('resolves inline + url refs to bytes; inline/url media URLs', async () => {
    const client = new SidecarBackendClient(new MockBridge());
    const inline = btoa('hi');
    expect(await client.resolveBytes({ kind: 'inline', bytesB64: inline })).toEqual(
      new Uint8Array([104, 105]),
    );
    expect(client.resolveMediaUrl({ kind: 'url', url: 'https://x/y.flac' })).toBe('https://x/y.flac');
    expect(client.resolveMediaUrl({ kind: 'inline', bytesB64: inline })).toContain('base64,');
  });
});
