import React from 'react';
import { RenderedJotContext } from '../contexts';

/**
 * Read the live `pxPerBeat` off the active jot via the MobX scope
 * already used by surrounding `observer`s. Used by chunk draws to
 * sample the current zoom at rasterisation time without forcing the
 * chunk's render path to take a `jot` prop (the bars/structure don't
 * change with zoom; passing the whole jot just to read this one
 * field would dirty the chunks on every wheel tick).
 */
export function useLiveJotPxPerBeat(): number {
  // `RenderedJotContext` is provided at the JotView root; null only
  // outside the View (tests). In that case fall back to 1, which is
  // safe (chunks just don't render).
  const jot = React.useContext(RenderedJotContext);
  return jot?.pxPerBeat ?? 1;
}
