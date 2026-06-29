import { describe, expect, it } from 'bun:test';
import {
  PROTOCOL_VERSION,
  type ClientMessage,
  type ResultRef,
  type ServerMessage,
} from 'src/net/control_protocol';
import { CapabilityPresenter } from './capability_presenter';
import { CapabilityStore } from './capability_store';
import {
  type AcceleratorInfo,
  type CapabilityStateEntry,
  type DesktopBridge,
} from './desktop_bridge';
import { DesktopTranscriber } from './desktop_transcribe';

class MockBridge implements DesktopBridge {
  states: Record<string, CapabilityStateEntry> = {};
  lastRequest: ClientMessage | undefined;
  emit: (request: ClientMessage, onEvent: (m: ServerMessage) => void) => void = (req, onEvent) => {
    onEvent({ v: PROTOCOL_VERSION, type: 'progress', id: req.id, stage: 'onsets', frac: 0.5 });
    onEvent({
      v: PROTOCOL_VERSION,
      type: 'result',
      id: req.id,
      artifacts: [
        { role: 'midi', ref: { kind: 'path', path: '/out/pred.mid' } },
        { role: 'stem', ref: { kind: 'path', path: '/out/stem_k.flac' } },
      ],
    });
  };

  async detectAccelerator(): Promise<AcceleratorInfo> {
    return { kind: 'cpu', meetsCudaMin: false };
  }
  async capabilityStates(): Promise<Record<string, CapabilityStateEntry>> {
    return this.states;
  }
  async setCapabilityInstalled(id: string, installed: boolean): Promise<void> {
    this.states[id] = { installed };
  }
  async installCapability(): Promise<void> {}
  async runJob(request: ClientMessage, onEvent: (m: ServerMessage) => void): Promise<void> {
    this.lastRequest = request;
    this.emit(request, onEvent);
  }
  async cancelJob(): Promise<void> {}
}

function build(): {
  store: CapabilityStore;
  bridge: MockBridge;
  transcriber: DesktopTranscriber;
} {
  const store = new CapabilityStore();
  const bridge = new MockBridge();
  const capabilities = new CapabilityPresenter({ store, bridge });
  const toBytes = async (): Promise<Uint8Array> => new Uint8Array([1, 2, 3]);
  const toUrl = (ref: ResultRef): string => (ref.kind === 'path' ? `asset://${ref.path}` : 'x');
  return { store, bridge, transcriber: new DesktopTranscriber(bridge, capabilities, toBytes, toUrl) };
}

describe('DesktopTranscriber', () => {
  it('auto-installs the capability, runs the job, returns MIDI + stem URLs', async () => {
    const { store, bridge, transcriber } = build();
    const stages: string[] = [];
    const result = await transcriber.transcribe('/music/song.wav', {
      params: { quantise: false },
      onProgress: (stage) => stages.push(stage),
    });
    expect(store.isReady('transcription')).toBe(true);
    const req = bridge.lastRequest;
    expect(req?.type).toBe('request');
    if (req?.type === 'request') {
      expect(req.op).toBe('transcribe');
    }
    expect(stages).toContain('onsets');
    expect(result.midi).toEqual(new Uint8Array([1, 2, 3]));
    expect(result.audioUrls).toEqual(['asset:///out/stem_k.flac']);
  });

  it('throws on a result with no MIDI', async () => {
    const { store, bridge, transcriber } = build();
    store.statuses.set('transcription', 'ready');
    bridge.emit = (req, onEvent) =>
      onEvent({ v: PROTOCOL_VERSION, type: 'result', id: req.id, artifacts: [] });
    await expect(transcriber.transcribe('/music/song.wav')).rejects.toThrow(/no MIDI/);
  });

  it('throws on an error frame', async () => {
    const { store, bridge, transcriber } = build();
    store.statuses.set('transcription', 'ready');
    bridge.emit = (req, onEvent) =>
      onEvent({
        v: PROTOCOL_VERSION,
        type: 'error',
        id: req.id,
        code: 'boom',
        message: 'kaboom',
        recoverable: false,
      });
    await expect(transcriber.transcribe('/music/song.wav')).rejects.toThrow(/boom/);
  });
});
