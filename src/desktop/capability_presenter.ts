import { makeAutoObservable, runInAction } from 'mobx';
import {
  ACCELERATOR_TIER_BYTES,
  CAPABILITIES,
  type CapabilityId,
  capabilityById,
} from './capability_manifest';
import { type CapabilityStore } from './capability_store';
import { type DesktopBridge } from './desktop_bridge';

export type CapabilityPresenterDeps = {
  store: CapabilityStore;
  bridge: DesktopBridge;
};

/**
 * Owns all capability mutation: hardware/state refresh, the point-of-use gate,
 * and install orchestration. The actual uv sync of the multi-GB stack is
 * performed by the (parked) installer behind `bridge.setCapabilityInstalled`;
 * this drives status + persists state + replays any queued point-of-use intent.
 */
export class CapabilityPresenter {
  readonly store: CapabilityStore;
  private readonly bridge: DesktopBridge;
  /** Action queued by a gate, replayed once its capability becomes ready. */
  private readonly pendingIntents: Map<CapabilityId, () => void> = new Map();

  constructor(deps: CapabilityPresenterDeps) {
    this.store = deps.store;
    this.bridge = deps.bridge;
    makeAutoObservable(this, { store: false });
  }

  /** Load detected accelerator + persisted install state into the store. */
  async refresh(): Promise<void> {
    const [accelerator, states] = await Promise.all([
      this.bridge.detectAccelerator(),
      this.bridge.capabilityStates(),
    ]);
    runInAction(() => {
      this.store.accelerator = accelerator;
      for (const cap of CAPABILITIES) {
        const installed = states[cap.id]?.installed ?? false;
        this.store.statuses.set(cap.id, installed ? 'ready' : 'not-installed');
      }
    });
  }

  /** Incremental download for installing `ids` (+ prereqs), deduping the shared
   *  accelerator tier and skipping already-installed capabilities. The big
   *  number shows on the first accelerator-needing capability; later ones show
   *  only their own weights. */
  incrementalBytes(ids: CapabilityId[]): number {
    const want = this.closure(ids);
    let bytes = 0;
    let needsAccelerator = false;
    for (const id of want) {
      if (this.store.isReady(id)) {
        continue;
      }
      const cap = capabilityById(id);
      bytes += cap.ownApproxBytes;
      if (cap.accelerator === 'required') {
        needsAccelerator = true;
      }
    }
    if (needsAccelerator && !this.acceleratorInstalled()) {
      const variant = this.store.accelerator?.kind ?? 'cpu';
      bytes += ACCELERATOR_TIER_BYTES[variant];
    }
    return bytes;
  }

  /** Gate a point-of-use action: run it immediately if the capability is ready,
   *  otherwise queue it and start installing. Returns true when it ran now. */
  ensure(id: CapabilityId, action: () => void): boolean {
    if (this.store.isReady(id)) {
      action();
      return true;
    }
    this.pendingIntents.set(id, action);
    void this.install(id);
    return false;
  }

  /** Install a capability and its prereqs in dependency order. */
  async install(id: CapabilityId): Promise<void> {
    const order = [...this.closure([id])];
    try {
      for (const dep of order) {
        if (this.store.isReady(dep)) {
          continue;
        }
        runInAction(() => {
          this.store.statuses.set(dep, 'installing');
          this.store.installProgress.set(dep, 0);
          this.store.errors.delete(dep);
        });
        await this.bridge.setCapabilityInstalled(dep, true);
        runInAction(() => {
          this.store.statuses.set(dep, 'ready');
          this.store.installProgress.delete(dep);
        });
      }
      const intent = this.pendingIntents.get(id);
      if (intent != null) {
        this.pendingIntents.delete(id);
        intent();
      }
    } catch (err) {
      runInAction(() => {
        this.store.statuses.set(id, 'error');
        this.store.errors.set(id, err instanceof Error ? err.message : String(err));
      });
    }
  }

  /** Transitive prereq closure of `ids`, in install order: a capability's
   *  prerequisites come before it (post-order over the acyclic `requires`). */
  private closure(ids: CapabilityId[]): Set<CapabilityId> {
    const out = new Set<CapabilityId>();
    const visit = (id: CapabilityId): void => {
      if (out.has(id)) {
        return;
      }
      for (const req of capabilityById(id).requires) {
        visit(req);
      }
      out.add(id);
    };
    ids.forEach(visit);
    return out;
  }

  private acceleratorInstalled(): boolean {
    return CAPABILITIES.some((c) => c.accelerator === 'required' && this.store.isReady(c.id));
  }
}
