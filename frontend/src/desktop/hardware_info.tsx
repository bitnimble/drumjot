import { observer } from 'mobx-react-lite';
import React from 'react';
import { type AcceleratorInfo, type AcceleratorKind } from './desktop_bridge';
import { CAPABILITIES } from './capability_manifest';
import { desktopCapabilities } from './desktop_services';
import styles from './hardware_info.module.css';

const KIND_LABELS: Record<AcceleratorKind, string> = {
  cuda: 'CUDA (NVIDIA)',
  rocm: 'ROCm (AMD)',
  directml: 'DirectML',
  mps: 'Metal (Apple)',
  cpu: 'CPU (no GPU acceleration)',
};

type Tone = 'good' | 'warn' | 'muted';
type Status = { label: string; tone: Tone; detail?: string };

/**
 * Read-only hardware / acceleration readout for Settings → Hardware. Shows the
 * detected accelerator (type / device / driver) and whether GPU acceleration is
 * actually active, and if not, why (CPU-only, stale driver, or the GPU runtime
 * not downloaded yet). Desktop-only.
 */
export const HardwareInfo = observer(function HardwareInfo() {
  const deps = desktopCapabilities();
  if (deps == null) return null;
  const { store } = deps;
  const accel = store.accelerator;
  if (accel == null) {
    return <p className={styles.note}>Detecting hardware…</p>;
  }

  // "Accelerated" needs both supported hardware AND the GPU runtime installed;
  // separation carries the shared torch tier (accelerator: 'required').
  const runtimeInstalled = CAPABILITIES.some(
    (c) => c.accelerator === 'required' && store.isReady(c.id),
  );
  const status = acceleration(accel, runtimeInstalled);

  return (
    <div className={styles.grid}>
      <Row label="Acceleration" value={status.label} tone={status.tone} />
      <Row label="Type" value={KIND_LABELS[accel.kind] ?? accel.kind} />
      <Row label="Device" value={accel.gpuName ?? ', '} />
      {accel.driverVersion != null && <Row label="Driver" value={accel.driverVersion} />}
      {status.detail != null && <p className={styles.note}>{status.detail}</p>}
    </div>
  );
});

function acceleration(accel: AcceleratorInfo, runtimeInstalled: boolean): Status {
  if (accel.kind === 'cpu') {
    return {
      label: 'Unavailable',
      tone: 'muted',
      detail: 'No supported GPU was detected; compute runs on the CPU.',
    };
  }
  if (accel.kind === 'cuda' && !accel.meetsCudaMin) {
    const driver = accel.driverVersion != null ? ` (${accel.driverVersion})` : '';
    return {
      label: 'Unavailable',
      tone: 'warn',
      detail: `Your NVIDIA driver${driver} is older than the CUDA 12.8 runtime needs (570+). Update your driver to enable GPU acceleration.`,
    };
  }
  if (!runtimeInstalled) {
    return {
      label: 'Available, not installed',
      tone: 'muted',
      detail:
        'Your GPU is supported. Install a capability (Stem separation, transcription, or lyrics) to download the GPU runtime and turn on acceleration.',
    };
  }
  return { label: 'Active', tone: 'good' };
}

function Row({ label, value, tone }: { label: string; value: string; tone?: Tone }): React.ReactNode {
  return (
    <div className={styles.row}>
      <span className={styles.label}>{label}</span>
      <span className={tone != null ? styles[tone] : styles.value}>{value}</span>
    </div>
  );
}
