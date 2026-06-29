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
 * its progress into the store, and persists state.
 */
export class CapabilityPresenter {
  readonly store: CapabilityStore;
  private readonly bridge: DesktopBridge;
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
    const resolve = this.gateResolve;
    if (id == null || resolve == null) {
      return;
    }
    // Detach this waiter before the await: a concurrent requestCapability that
    // supersedes the prompt mid-install must not clobber it or resolve the wrong
    // promise.
    this.gateResolve = undefined;
    await this.install(id);
    if (this.store.isReady(id)) {
      runInAction(() => {
        if (this.store.pendingGate === id) this.store.pendingGate = undefined;
      });
      resolve(true);
    } else if (this.store.pendingGate === id) {
      // Install failed and nothing superseded us: re-arm so the prompt's Retry
      // resolves this same waiter.
      this.gateResolve = resolve;
    } else {
      // Superseded during a failed install: abandon this waiter.
      resolve(false);
    }
  }

  /** Dismiss the open prompt without installing; the gated action is abandoned. */
  cancelGate(): void {
    const resolve = this.gateResolve;
    this.gateResolve = undefined;
    runInAction(() => {
      this.store.pendingGate = undefined;
    });
    resolve?.(false);
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

  /** Install a capability and its prereqs. Thin wrapper over {@link installAll}. */
  async install(id: CapabilityId): Promise<void> {
    return this.installAll([id]);
  }

  /** Install one or more capabilities and their prereqs: one `uv sync` of the app
   *  venv to the union of all desired groups, then persist + mark the whole
   *  closure ready. Lets the first-run dialog install a multi-capability
   *  selection (e.g. separation + lyrics) in a single sync. */
  async installAll(ids: CapabilityId[]): Promise<void> {
    const order = [...this.closure(ids)];
    const fresh = order.filter((dep) => !this.store.isReady(dep));
    if (fresh.length === 0) {
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
        await this.bridge.installCapability(fresh[0], this.installGroups(order), (line) => {
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
    } catch (err) {
      runInAction(() => {
        const message = err instanceof Error ? err.message : String(err);
        for (const dep of fresh) {
          this.store.statuses.set(dep, 'error');
          this.store.installLog.delete(dep);
          this.store.errors.set(dep, message);
        }
      });
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
