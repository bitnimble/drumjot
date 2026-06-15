import React from 'react';

/**
 * Story-only layout helpers (NOT a `*.stories.tsx`, so the stories glob
 * skips it). A component's variants/states all live in a single story,
 * each wrapped in a labelled {@link Variant} (an HTML `<fieldset>` +
 * `<legend>`) and stacked by {@link Gallery}, rather than scattered across
 * one story per state.
 */

/** One labelled section in a gallery — a `<fieldset>` with the variant
 *  name as its `<legend>`. */
export function Variant({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <fieldset style={{ margin: 0, borderRadius: 6, padding: '10px 14px' }}>
      <legend
        style={{
          padding: '0 4px',
          fontSize: 11,
          textTransform: 'uppercase',
          letterSpacing: '0.04em',
          opacity: 0.6,
        }}
      >
        {label}
      </legend>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center' }}>
        {children}
      </div>
    </fieldset>
  );
}

/** Vertical stack of {@link Variant} sections — the body of a gallery story. */
export function Gallery({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14, alignItems: 'flex-start' }}>
      {children}
    </div>
  );
}
