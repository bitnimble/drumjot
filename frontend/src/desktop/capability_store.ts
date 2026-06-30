import { makeAutoObservable, observable } from 'mobx';
import { type CapabilityId } from './capability_manifest';
import { type AcceleratorInfo } from './desktop_bridge';

export type CapabilityStatus =
  | 'not-installed'
  | 'installing'
  | 'ready'
  | 'update-available'
  | 'error';

/**
 * Desktop capability state: the detected accelerator + per-capability status.
 * Data only (observables + simple read accessors); install orchestration,
 * probes, and the point-of-use gate live on {@link CapabilityPresenter}.
 */
export class CapabilityStore {
  accelerator: AcceleratorInfo | undefined = undefined;
  statuses: Map<CapabilityId, CapabilityStatus> = new Map();
  /** Capability awaiting the user's point-of-use install decision (drives the
   *  install prompt modal); undefined when no prompt is open. */
  pendingGate: CapabilityId | undefined = undefined;
  /** Latest uv output line while a capability is `installing` (uv gives no
   *  clean 0..1, so we surface its progress text instead of a bar). */
  installLog: Map<CapabilityId, string> = new Map();
  /** Last error message per capability, when its status is `error`. */
  errors: Map<CapabilityId, string> = new Map();

  constructor() {
    makeAutoObservable(this, {
      statuses: observable.shallow,
      installLog: observable.shallow,
      errors: observable.shallow,
    });
  }

  statusOf(id: CapabilityId): CapabilityStatus {
    return this.statuses.get(id) ?? 'not-installed';
  }

  isReady(id: CapabilityId): boolean {
    return this.statusOf(id) === 'ready';
  }
}
