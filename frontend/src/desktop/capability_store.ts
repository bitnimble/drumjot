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
  /** 0..1 install progress while a capability is `installing`. */
  installProgress: Map<CapabilityId, number> = new Map();
  /** Last error message per capability, when its status is `error`. */
  errors: Map<CapabilityId, string> = new Map();

  constructor() {
    makeAutoObservable(this, {
      statuses: observable.shallow,
      installProgress: observable.shallow,
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
