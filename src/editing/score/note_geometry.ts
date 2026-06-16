import { Box, Point } from 'src/utils/geom';
import type { StructLayer, StructNote } from 'src/editing/structure/structure_store';

/**
 * Pixel ↔ note geometry helpers for the marquee and drag-move interactions.
 * These read the live DOM (note rects, the scroll wrapper) and so are only
 * ever called from pointer-event handlers, never a render, effect, or
 * per-frame path (see AGENTS.md §5.9). `pxPerBeat` is globally uniform (a
 * single zoom × density factor, not per-bar), so a horizontal pixel delta maps
 * linearly to a beat delta.
 */

/** The scroll-content wrapper that hosts the notes (and the marquee / frame
 *  overlays). Notes' positions are interpreted in its local coordinate space,
 *  which is also the space the marquee box is built in. */
function scrollContent(): HTMLElement | null {
  return document.querySelector<HTMLElement>('[data-jot-scroll-content]');
}

/** All note ids whose rendered glyph centre falls inside `box` (scroll-content
 *  local coords). Resolves ids to the supplied current `StructNote`s. */
export function notesInBox(box: Box, byId: ReadonlyMap<string, StructNote>): StructNote[] {
  const wrap = scrollContent();
  if (!wrap) return [];
  const wr = wrap.getBoundingClientRect();
  const out: StructNote[] = [];
  for (const el of wrap.querySelectorAll<HTMLElement>('[data-note-id]')) {
    const r = el.getBoundingClientRect();
    const centre = new Point(r.left + r.width / 2 - wr.left, r.top + r.height / 2 - wr.top);
    if (!box.encloses(centre)) continue;
    const note = byId.get(el.dataset.noteId ?? '');
    if (note) out.push(note);
  }
  return out;
}

/** Bounding box (scroll-content local coords, with `pad`) enclosing the DOM
 *  glyphs of the given note ids, or null if none are mounted. */
export function boundingBoxOfNotes(ids: ReadonlySet<string>, pad = 4): Box | null {
  const wrap = scrollContent();
  if (!wrap || ids.size === 0) return null;
  const wr = wrap.getBoundingClientRect();
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let found = false;
  for (const el of wrap.querySelectorAll<HTMLElement>('[data-note-id]')) {
    if (!ids.has(el.dataset.noteId ?? '')) continue;
    const r = el.getBoundingClientRect();
    const x = r.left - wr.left;
    const y = r.top - wr.top;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x + r.width);
    maxY = Math.max(maxY, y + r.height);
    found = true;
  }
  if (!found) return null;
  return new Box(minX - pad, minY - pad, maxX - minX + 2 * pad, maxY - minY + 2 * pad);
}

/** A flat map of every current note keyed by id, for resolving DOM-attribute
 *  ids back to live `StructNote`s. */
export function notesById(layers: readonly StructLayer[]): Map<string, StructNote> {
  const map = new Map<string, StructNote>();
  for (const layer of layers) {
    for (const bar of layer.bars) {
      for (const lane of Object.keys(bar.tracks)) {
        for (const note of bar.tracks[lane].notes) map.set(note.id, note);
      }
    }
  }
  return map;
}

/**
 * Remap one lane to another by the same row offset the anchor moved, over the
 * rendered `laneOrder`. The anchor lands on `toLane`; every other selected
 * note shifts by the same number of rows, preserving the group's vertical
 * arrangement. Identity when source and target match or aren't in the order.
 */
export function buildLaneMap(
  laneOrder: readonly string[],
  fromLane: string,
  toLane: string
): (lane: string) => string {
  if (fromLane === toLane) return (l) => l;
  const fromIdx = laneOrder.indexOf(fromLane);
  const toIdx = laneOrder.indexOf(toLane);
  if (fromIdx < 0 || toIdx < 0) return (l) => l;
  const rowDelta = toIdx - fromIdx;
  return (lane) => {
    const i = laneOrder.indexOf(lane);
    if (i < 0) return lane;
    const j = Math.min(Math.max(i + rowDelta, 0), laneOrder.length - 1);
    return laneOrder[j];
  };
}
