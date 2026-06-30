import type { Modifier, Sticking } from 'src/schema/schema';
import { DEFAULT_VELOCITY } from 'src/dynamics/dynamics';

/**
 * Static field metadata for the Note properties editor: the modifier list +
 * labels, the Roll-conflict set, the volume scale, and the sticking options.
 * Pure data shared by the store, presenter, and view.
 */

/** Roll lives in its own checkbox above the modifier list. The other
 *  multi-stroke ornaments it's incompatible with are disabled while Roll is
 *  on. (Accent/ghost aren't modifiers, they're loudness, set via Volume.) */
export const ROLL_DISABLED_MODIFIERS: ReadonlySet<Modifier> = new Set<Modifier>([
  'fl',
  'dr',
  'rf',
  'z',
]);

/** The modifier checkboxes, in display order, with human labels (the enum
 *  order). Loudness, including accent/ghost feel; is the Volume control. */
export const MODIFIER_FIELDS: ReadonlyArray<{ mod: Modifier; label: string }> = [
  { mod: 'c', label: 'Closed' },
  { mod: 'h', label: 'Half-open' },
  { mod: 'o', label: 'Open' },
  { mod: 'f', label: 'Foot' },
  { mod: 's', label: 'Splash' },
  { mod: 'r', label: 'Rimshot' },
  { mod: 'x', label: 'Cross-stick' },
  { mod: 'z', label: 'Buzz' },
  { mod: 'k', label: 'Choke' },
  { mod: 'm', label: 'Mute' },
  { mod: 'l', label: 'Let-ring' },
  { mod: 'fl', label: 'Flam' },
  { mod: 'dr', label: 'Drag' },
  { mod: 'rf', label: 'Ruff' },
];

/** Sticking radio options. `none` clears the (optional) sticking. */
export const STICKING_FIELDS: ReadonlyArray<{ value: Sticking | 'none'; label: string }> = [
  { value: 'none', label: 'None' },
  { value: 'r', label: 'R' },
  { value: 'l', label: 'L' },
  { value: 'rf', label: 'RF' },
  { value: 'lf', label: 'LF' },
];

// ---------- Volume (0-10 UI scale <-> 0-127 velocity) ----------

/** Top of the UI loudness scale. */
export const VOLUME_UI_MAX = 10;
/** velocity = round(ui * VELOCITY_PER_STEP); 10 steps span the full 0-127. */
const VELOCITY_PER_STEP = 127 / VOLUME_UI_MAX;

/** Dynamic markers shown beside the volume value at their UI step (matches
 *  {@link VOLUME_TO_VELOCITY} in dynamics.ts). */
export const VOLUME_UI_LABELS: Readonly<Record<number, string>> = {
  1: 'pp',
  3: 'p',
  5: 'mp',
  6: 'mf',
  8: 'f',
  10: 'ff',
};

export function clampVolumeUi(ui: number): number {
  return Math.min(Math.max(Math.round(ui), 0), VOLUME_UI_MAX);
}

/** Stored velocity (0-127) -> UI step (0-10). */
export function velocityToUi(velocity: number | undefined): number {
  return clampVolumeUi(Math.round((velocity ?? DEFAULT_VELOCITY) / VELOCITY_PER_STEP));
}

/** UI step (0-10) -> stored velocity (0-127). */
export function uiToVelocity(ui: number): number {
  return Math.min(Math.max(Math.round(clampVolumeUi(ui) * VELOCITY_PER_STEP), 0), 127);
}

/** Micro-timing nudge granularity (ms). */
export const MICRO_TIMING_STEP_MS = 1;

/** Beat nudge granularity in quarter-note beats (a 16th at 4/4). */
export const BEAT_STEP = 0.25;
