/**
 * Render-counter test hook. Zero cost in normal operation: a single
 * property check that bails when `window.__perf` is unset. An e2e test
 * (or the console) opts in by setting `window.__perf = {}`, after which
 * every instrumented component bumps its per-name counter on each render.
 * Reading the object back shows exactly which components re-rendered over
 * a measured interaction.
 *
 * Used by `e2e/zoom-rerender.spec.ts` to guard the invariant that zooming
 * does NOT re-render `JotView` (and therefore not its score subtree). The
 * `--px-per-beat` CSS-variable design plus the stable-prop isolation in
 * `createJotView` keep zoom off the React render path; this hook lets the
 * test fail loudly if a future change reintroduces the cascade. Keep call
 * sites minimal, add one only when there's a test asserting on it.
 */
export function perfProbe(name: string): void {
  const w = globalThis as unknown as { __perf?: Record<string, number> };
  const counts = w.__perf;
  if (!counts) return;
  counts[name] = (counts[name] ?? 0) + 1;
}
