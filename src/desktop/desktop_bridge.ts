import { Channel, invoke } from '@tauri-apps/api/core';
import {
  type ClientMessage,
  type ServerMessage,
  safeDecodeServerValue,
} from 'src/net/control_protocol';

export type AcceleratorKind = 'cuda' | 'rocm' | 'directml' | 'mps' | 'cpu';

export type AcceleratorInfo = {
  kind: AcceleratorKind;
  gpuName?: string;
  driverVersion?: string;
  /** NVIDIA driver new enough for the cu128 build. */
  meetsCudaMin: boolean;
};

export type CapabilityStateEntry = { installed: boolean };

/**
 * The Rust commands the desktop frontend drives. An interface (not a direct
 * `invoke` call site) so `CapabilityPresenter` is unit-testable against a mock
 * and the web build never touches a Tauri API.
 */
export interface DesktopBridge {
  detectAccelerator(): Promise<AcceleratorInfo>;
  capabilityStates(): Promise<Record<string, CapabilityStateEntry>>;
  setCapabilityInstalled(id: string, installed: boolean): Promise<void>;
  runJob(request: ClientMessage, onEvent: (msg: ServerMessage) => void): Promise<void>;
  cancelJob(id: string): Promise<void>;
}

/** Real bridge backed by Tauri `invoke` + `Channel`. Construct only when
 *  {@link isTauri} is true. */
export class TauriBridge implements DesktopBridge {
  detectAccelerator(): Promise<AcceleratorInfo> {
    return invoke<AcceleratorInfo>('detect_accelerator');
  }

  capabilityStates(): Promise<Record<string, CapabilityStateEntry>> {
    return invoke<Record<string, CapabilityStateEntry>>('capability_states');
  }

  async setCapabilityInstalled(id: string, installed: boolean): Promise<void> {
    await invoke('set_capability_installed', { id, installed });
  }

  async runJob(request: ClientMessage, onEvent: (msg: ServerMessage) => void): Promise<void> {
    const channel = new Channel<unknown>();
    channel.onmessage = (frame) => {
      const decoded = safeDecodeServerValue(frame);
      if (decoded.ok) {
        onEvent(decoded.message);
      }
    };
    await invoke('run_job', { request, onEvent: channel });
  }

  async cancelJob(id: string): Promise<void> {
    await invoke('cancel_job', { id });
  }
}
