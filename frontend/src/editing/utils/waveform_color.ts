/**
 * Hex literal used by Canvas paint sites for the waveform's stroke colour,
 * shared by the score's onset-timing visualization, the minimap waveform,
 * and the mixer's audio-track rows. Mirrors `--color-pattern-2` in
 * `src/design_tokens.css` (sky blue); Canvas 2D needs a literal RGB string
 * and reading it via `getComputedStyle(...).getPropertyValue('--color-pattern-2')`
 * per paint forces a style flush, so we duplicate the value here and accept
 * the carve-out from "no naked color literals" (AGENTS.md §5.8). If the CSS
 * token's hex changes, update this in lockstep.
 */
export const WAVEFORM_PAINT_COLOR = '#5ba8e8';
