/**
 * Drumjot brand mark. The SVG itself lives at `public/favicon.svg`
 * (single source of truth, also referenced by the favicon `<link>`
 * in `index.html`); this component just renders it via `<img>`.
 *
 * The mark depicts three lanes (kick / hi-hat / snare, bottom→top)
 * with notes in the order 1, 2, 1, 3, coloured to mirror the in-app
 * track palette (`palette` in src/jot.ts).
 */
export const Logo = ({ size = 56, title }: { size?: number; title?: string }) => (
  <img
    src="/favicon.svg"
    alt={title ?? ''}
    title={title}
    width={size}
    height={size}
    draggable={false}
  />
);
