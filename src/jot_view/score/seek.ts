import type React from 'react';

/**
 * Shared click-to-seek handler for the score bars row and the audio-track
 * waveforms. Bails on clicks that originated on a note, pattern label,
 * or anything else tagged `data-noseek` so those keep their own
 * behaviour. `e.currentTarget` is the bars-row element whose left edge
 * is x=0 in `bar.x` space, so `clientX - rect.left` is the bars-row-
 * local pixel regardless of horizontal scroll.
 */
export function seekFromClick(
  e: React.MouseEvent<HTMLDivElement>,
  onSeek: (x: number) => void
): void {
  if (e.button !== 0) return;
  if ((e.target as HTMLElement).closest('[data-noseek]')) return;
  const rect = e.currentTarget.getBoundingClientRect();
  onSeek(e.clientX - rect.left);
}
