/**
 * Lane-side (drum/instrument-row) mute/solo/volume filter shared by the
 * transport, the drum scheduler, and the mixer store.
 *
 * Split out of `player.ts` so the scheduler can import it without a cycle
 * back through the transport; `player.ts` re-exports these to keep the
 * historical import path stable.
 */

export type PlayerFilter = {
  /** Track keys (`layerId/lane`, see `trackKey`) the user has muted. */
  mutedTracks: ReadonlySet<string>;
  /**
   * When solo is active, ONLY these tracks are audible (others behave
   * as if muted). Soloed-AND-muted = muted; explicit mute always wins so
   * the user can keep solo on while temporarily silencing a soloed row.
   * Keyed by track key (`layerId/lane`).
   */
  soloedTracks: ReadonlySet<string>;
  /**
   * True when a solo is engaged *anywhere*, on an instrument row OR an
   * audio track. Solo is a single global mode shared across both
   * domains: as soon as the user solos any row, every non-soloed row
   * (drums *and* music) drops out. Computed by the store, which is the
   * only place that sees both the lane and audio-track solo sets.
   */
  soloActive: boolean;
  /**
   * True when this section's master mute is engaged. Silences every row
   * in the section regardless of per-row mute/solo state; mirrors the
   * bus-gain pin to 0, so the scheduler skips events that would not have
   * sounded anyway and the UI can dim the rows uniformly.
   */
  sectionMasterMuted: boolean;
  /**
   * True when this section's master solo is engaged. Acts as if every
   * row in the section were soloed (only for the purpose of the solo
   * exclusion rule); without this, soloing Drums master would set
   * `soloActive` but leave `soloedTracks` empty, silencing every drum
   * row.
   */
  sectionMasterSoloed: boolean;
  /** Per-track volume multiplier in [0, 1], keyed by track key; missing = full (1). */
  volumes: ReadonlyMap<string, number>;
};

export const PASSTHROUGH_FILTER: PlayerFilter = {
  mutedTracks: new Set(),
  soloedTracks: new Set(),
  soloActive: false,
  sectionMasterMuted: false,
  sectionMasterSoloed: false,
  volumes: new Map(),
};

/** @param track the event's track key (`layerId/lane`, see `trackKey`). */
export function isAudibleUnder(track: string, filter: PlayerFilter): boolean {
  if (filter.sectionMasterMuted) return false;
  if (filter.mutedTracks.has(track)) return false;
  if (
    filter.soloActive &&
    !filter.sectionMasterSoloed &&
    !filter.soloedTracks.has(track)
  ) {
    return false;
  }
  if ((filter.volumes.get(track) ?? 1) <= 0) return false;
  return true;
}
