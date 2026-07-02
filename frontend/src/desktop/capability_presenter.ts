import { makeAutoObservable, runInAction } from 'mobx';
import {
  ACCELERATOR_TIER_BYTES,
  CAPABILITIES,
  type CapabilityId,
  capabilityById,
} from './capability_manifest';
import { type CapabilityStore } from './capability_store';
import { type DesktopBridge } from './desktop_bridge';

/** Peak install footprint over the raw download size: the venv unpacks wheels
 *  and models decompress, so require this much headroom before warning. */
const DISK_INSTALL_SAFETY_FACTOR = 1.5;

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
  /** Tail of the install queue; installs chain off this so only one runs at a
   *  time (see installAll). */
  private installChain: Promise<void> = Promise.resolve();

  constructor(deps: CapabilityPresenterDeps) {
    this.store = deps.store;
    this.bridge = deps.bridge;
    makeAutoObservable<CapabilityPresenter, 'gateResolve' | 'installChain'>(this, {
      store: false,
      gateResolve: false,
      installChain: false,
    });
  }

  /** Point-of-use gate: resolves true if the capability is already installed,
   *  otherwise opens the install prompt (sets `store.pendingGate`) and resolves
   *  once the user confirms+install succeeds (true) or cancels (false). */
  async requestCapability(id: CapabilityId): Promise<boolean> {
    if (this.store.isReady(id)) {
      return true;
    }
    if (capabilityById(id).kind !== 'deps') {
      // credentials/system capabilities aren't installed via uv; treat as ready.
      return true;
    }
    // The in-memory status can be a stale `not-installed`: the boot `refresh()`
    // may not have landed yet (it's async, and `detectAccelerator` can be slow),
    // or the on-disk state changed since load. Confirm against the persisted
    // source of truth before showing an install prompt for something the user
    // already has -- otherwise a genuinely-installed capability spuriously
    // prompts to reinstall if the gate is hit early enough.
    await this.syncInstalledFromDisk();
    if (this.store.isReady(id)) {
      return true;
    }
    // About to prompt: refresh free disk space so the gate can warn up front if
    // the download wouldn't fit.
    await this.refreshDiskSpace();
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

  /** Read the persisted capability states and mark any that are installed as
   *  ready. Upgrade-only: never downgrades, so it can't clobber an in-flight
   *  `installing`; it only corrects a stale/not-yet-loaded `not-installed` for a
   *  capability that's actually present on disk. */
  private async syncInstalledFromDisk(): Promise<void> {
    const states = await this.bridge.capabilityStates();
    runInAction(() => {
      for (const cap of CAPABILITIES) {
        if (states[cap.id]?.installed && !this.store.isReady(cap.id)) {
          this.store.statuses.set(cap.id, 'ready');
        }
      }
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
    } else if (this.store.pendingGate === id && this.gateResolve == null) {
      // Install failed and nothing superseded us: re-arm so the prompt's Retry
      // resolves this same waiter.
      this.gateResolve = resolve;
    } else {
      // Superseded, either by a different capability's prompt, or by a concurrent
      // requestCapability for this same id that already installed its own resolver
      // (which we must not overwrite, or that caller hangs forever): abandon this
      // waiter.
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

  /** Load detected accelerator + persisted install state into the store.
   *  Merge, don't clobber: this fires once at boot and its `detectAccelerator`
   *  probe can be slow, so an install may complete while it's still awaiting. A
   *  blind write of the boot snapshot would revert that cap's `installing`/`ready`
   *  to `not-installed` and re-prompt the user, so it upgrades to `ready` from
   *  disk but never downgrades a cap the store already considers in-flight/done.
   *  A transport failure is logged rather than left as an unhandled rejection
   *  that would wedge the "Detecting hardware…" readout. */
  async refresh(): Promise<void> {
    void this.refreshDiskSpace();
    try {
      const [accelerator, states] = await Promise.all([
        this.bridge.detectAccelerator(),
        this.bridge.capabilityStates(),
      ]);
      runInAction(() => {
        this.store.accelerator = accelerator;
        for (const cap of CAPABILITIES) {
          const installed = states[cap.id]?.installed ?? false;
          if (installed) {
            this.store.statuses.set(cap.id, 'ready');
          } else if (this.store.statusOf(cap.id) === 'not-installed') {
            // Only (re)assert not-installed for a cap that isn't already mid-
            // install, ready, or errored from an operation fresher than our snapshot.
            this.store.statuses.set(cap.id, 'not-installed');
          }
        }
      });
    } catch (err) {
      console.error('[capability] refresh failed', err);
    }
  }

  /** Query + cache free disk space for the pre-install space check. Best-effort:
   *  a failure leaves `availableBytes` unchanged, so the UI won't block on an
   *  unknown value. */
  async refreshDiskSpace(): Promise<void> {
    try {
      const bytes = await this.bridge.availableDiskSpace();
      runInAction(() => {
        this.store.availableBytes = bytes;
      });
    } catch (err) {
      console.error('[capability] disk-space query failed', err);
    }
  }

  /** Whether the data-root volume has room to install `ids` (the incremental
   *  download times {@link DISK_INSTALL_SAFETY_FACTOR}). `undefined` when free
   *  space isn't known yet, callers must NOT block on an unknown. */
  hasEnoughSpaceFor(ids: CapabilityId[]): boolean | undefined {
    const free = this.store.availableBytes;
    if (free == null) return undefined;
    return free >= this.incrementalBytes(ids) * DISK_INSTALL_SAFETY_FACTOR;
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
   *  selection (e.g. separation + lyrics) in a single sync.
   *
   *  Serialised through {@link installChain}: two concurrent `uv sync`s on the
   *  same venv would race/corrupt it, and a later install's group set depends on
   *  what earlier ones added, so overlapping installs queue rather than run
   *  together. `fresh` is recomputed when each run actually starts, so a queued
   *  install skips anything an earlier one already brought in. */
  installAll(ids: CapabilityId[]): Promise<void> {
    const next = this.installChain.then(() => this.runInstall(ids));
    // runInstall catches its own errors, so the chain never rejects; the catch
    // is belt-and-suspenders so one failure can't wedge the queue.
    this.installChain = next.catch(() => {});
    return next;
  }

  private async runInstall(ids: CapabilityId[]): Promise<void> {
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
      // `system` ones pull no packages. Guard on `fresh` (not `order`): an
      // already-installed deps-cap in the closure must not trigger a needless
      // re-sync when every fresh capability is credentials/system.
      if (fresh.some((dep) => capabilityById(dep).kind === 'deps')) {
        // `id` is just the install's log label on the Rust side; name the whole
        // batch rather than an arbitrary first element.
        await this.bridge.installCapability(fresh.join('+'), this.installGroups(order), (line) => {
          runInAction(() => {
            for (const dep of fresh) {
              this.store.installLog.set(dep, line);
            }
          });
        });
      }
      // Persist sequentially, NOT Promise.all: each call is a read-modify-write
      // of the one capabilities.json on the Rust side, so concurrent writes race
      // (lost update + shared temp-file collision).
      for (const dep of fresh) {
        await this.bridge.setCapabilityInstalled(dep, true);
      }
      runInAction(() => {
        for (const dep of fresh) {
          this.store.statuses.set(dep, 'ready');
          this.store.installLog.delete(dep);
        }
      });
    } catch (err) {
      runInAction(() => {
        const base = err instanceof Error ? err.message : String(err);
        for (const dep of fresh) {
          // The invoke rejection is the Rust side's generic "uv sync failed
          // (status)"; the actionable detail (resolver conflict, no disk space,
          // network) is the last uv line we streamed into installLog. Fold it
          // into the surfaced error before clearing the log.
          const tail = this.store.installLog.get(dep);
          this.store.statuses.set(dep, 'error');
          this.store.installLog.delete(dep);
          this.store.errors.set(dep, tail != null && tail !== '' ? `${base}\n${tail}` : base);
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
