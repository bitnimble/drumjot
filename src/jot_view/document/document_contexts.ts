import React from 'react';
import { RenderedJot } from 'src/jot';
import { BarTiming } from 'src/jot_view/playback';

/**
 * The active {@link RenderedJot} for the current view (the document's
 * `currentJot`). Provided once at the JotView level so deep consumers
 * (today: NoteProvenanceDetails' timing-drift visualization, which reads
 * `effectiveDrumOffsetBeats` to account for the user-applied Beat-offset
 * slider as a separate stage in the detected → final chain) don't have to
 * thread the jot down through MixerView → InstrumentRow → BarView →
 * NoteView.
 *
 * `null` outside the View; consumers should fall back to a sensible
 * "no offset / nothing to show" default in that case.
 */
export const RenderedJotContext = React.createContext<RenderedJot | null>(null);

/**
 * Per-bar audio-time timings (start + duration, in seconds) for the
 * current jot, keyed by {@link StructuralBar.index}. Computed once at
 * the JotView level so deep consumers (today: NoteProvenanceDetails'
 * "Final position" row) can read a bar's absolute audio time without
 * depending on the playback timeline — the player's timeline is
 * `EMPTY_TIMELINE` until the first Play, but the math only needs the
 * jot's structure + tempos, so building it eagerly here makes the
 * lookup work even on an idle score.
 *
 * Keyed by `bar.index` rather than by `StructuralBar` reference because
 * the rendering chain shallow-clones bars (InstrumentRow rewrites
 * `tracks` for its lane) — the original reference doesn't survive the
 * walk down to NoteView. `bar.index` is preserved across those clones
 * and across `drumOffsetBeats` reflows, so it's the stable key.
 *
 * `null` outside the View or when the jot has no voices/bars.
 */
export const BarTimingsContext = React.createContext<
  ReadonlyMap<number, BarTiming> | null
>(null);
