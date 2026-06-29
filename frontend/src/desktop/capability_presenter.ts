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
 * and install orchestration. Installing drives the Rust `install_capability`
 * command (a `uv sync` of the app venv to the union of desired groups), streams
 * its progress into the store, persists state, and replays any queued
 * point-of-use intent.
 */
export class CapabilityPresenter {
  readonly store: CapabilityStore;
  private readonly bridge: DesktopBridge;
  /** Action queued by a gate, replayed once its capability becomes ready. */
  private readonly pendingIntents: Map<CapabilityId, () => void> = new Map();
  /** Resolver for the in-flight point-of-use prompt (see requestCapability). */
  private gateResolve: ((ok: boolean) => void) | undefined;

  constructor(deps: CapabilityPresenterDeps) {
    this.store = deps.store;
    this.bridge = deps.bridge;
    makeAutoObservable<CapabilityPresenter, 'gateResolve'>(this, {
      store: false,
      gateResolve: false,
    });
  }

  /** Point-of-use gate: resolves true if the capability is already installed,
   *  otherwise opens the install prompt (sets `store.pendingGate`) and resolves
   *  once the user confirms+install succeeds (true) or cancels (false). */
  requestCapability(id: CapabilityId): Promise<boolean> {
    if (this.store.isReady(id)) {
      return Promise.resolve(true);
    }
    if (capabilityById(id).kind !== 'deps') {
      // credentials/system capabilities aren't installed via uv; treat as ready.
      return Promise.resolve(true);
    }
    // A prompt is already open (concurrent gate): abandon the prior waiter so it
    // resolves false rather than hanging when this one supersedes it.
    this.gateResolve?.(false);
    return new Promise<boolean>((resolve) => {
      this.gateResolve = resolve;
      runInAction(() => {
        this.store.pendingGate = id;
      });
    });
  }

  /** Confirm the open prompt: install the pending capability (streaming
   *  progress), then resolve the gate. Leaves the prompt open showing the
   *  error if the install fails, so the user can retry or cancel. */
  async confirmGate(): Promise<void> {
    const id = this.store.pendingGate;
    if (id == null) {
      return;
    }
    await this.install(id);
    if (this.store.isReady(id)) {
      runInAction(() => {
        this.store.pendingGate = undefined;
      });
      this.gateResolve?.(true);
      this.gateResolve = undefined;
    }
  }

  /** Dismiss the open prompt without installing; the gated action is abandoned. */
  cancelGate(): void {
    runInAction(() => {
      this.store.pendingGate = undefined;
    });
    this.gateResolve?.(false);
    this.gateResolve = undefined;
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

  /** Install a capability and its prereqs: one `uv sync` of the app venv to the
   *  union of all desired groups, then persist + mark the closure ready and
   *  replay any queued point-of-use intent. */
  async install(id: CapabilityId): Promise<void> {
    const order = [...this.closure([id])];
    const fresh = order.filter((dep) => !this.store.isReady(dep));
    if (fresh.length === 0) {
      this.replayIntent(id);
      return;
    }
    runInAction(() => {
      for (const dep of fresh) {
        this.store.statuses.set(dep, 'installing');
        this.store.errors.delete(dep);
      }
    });
    try {
      // `deps` capabilities need the uv sync; `credentials` (the LLM key) and
      // `system` ones pull no packages.
      if (order.some((dep) => capabilityById(dep).kind === 'deps')) {
        await this.bridge.installCapability(id, this.installGroups(order), (line) => {
          runInAction(() => {
            for (const dep of fresh) {
              this.store.installLog.set(dep, line);
            }
          });
        });
      }
      await Promise.all(fresh.map((dep) => this.bridge.setCapabilityInstalled(dep, true)));
      runInAction(() => {
        for (const dep of fresh) {
          this.store.statuses.set(dep, 'ready');
          this.store.installLog.delete(dep);
        }
      });
      this.replayIntent(id);
    } catch (err) {
      runInAction(() => {
        for (const dep of fresh) {
          this.store.statuses.set(dep, 'error');
          this.store.installLog.delete(dep);
        }
        this.store.errors.set(id, err instanceof Error ? err.message : String(err));
      });
    }
  }

  private replayIntent(id: CapabilityId): void {
    const intent = this.pendingIntents.get(id);
    if (intent != null) {
      this.pendingIntents.delete(id);
      intent();
    }
  }

  /** The uv group set the venv should hold after installing `closureIds`: the
   *  union of those capabilities' groups plus every already-ready capability's,
   *  since `uv sync` replaces the env and must keep prior capabilities present. */
  private installGroups(closureIds: CapabilityId[]): string[] {
    const ids = new Set<CapabilityId>(closureIds);
    for (const cap of CAPABILITIES) {
      if (this.store.isReady(cap.id)) {
        ids.add(cap.id);
      }
    }
    const groups = new Set<string>();
    for (const id of ids) {
      for (const group of capabilityById(id).groups) {
        groups.add(group);
      }
    }
    return [...groups];
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
