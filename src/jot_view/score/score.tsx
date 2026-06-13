import { observer } from 'mobx-react-lite';
import React from 'react';
import { RenderedJot } from 'src/jot';
import sharedStyles from '../../jot_view.module.css';

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

/**
 * Decides whether a label popover anchored to `anchorRef` should flip
 * above its anchor instead of below. Default placement is below; flips
 * when below-placement would extend past the score scroll viewport's
 * bottom edge; which sits flush with the playback bar's top, so
 * "extends past" means "hidden behind the playback bar / debug panel".
 *
 * Measured synchronously on open in `useLayoutEffect` so the flip class
 * is applied before paint (no one-frame flash of a wrongly-placed
 * label). Not re-measured on scroll/zoom: the popover is transient and
 * users dismiss + re-open if the score moves under them.
 *
 * Falls back to below-placement when neither side fits; better to
 * partially clip the bottom of the label than to cover the notehead.
 */
/**
 * Read the artist string from wherever a loader plausibly stashed it.
 * Today only the RLRR (Paradiddle map) loader surfaces an artist, on
 * `globalMetadata.rlrr.recordingMetadata.artist`; a top-level
 * `globalMetadata.artist` is accepted too so hand-authored DSL or a
 * future loader can populate it directly. Anything non-string or empty
 * returns `undefined`, which makes the call site fall back to the
 * title alone.
 */
export function extractArtist(jot: RenderedJot): string | undefined {
  const meta = jot.globalMetadata as Record<string, unknown>;
  const direct = meta.artist;
  if (typeof direct === 'string' && direct.trim() !== '') return direct.trim();
  const rlrr = meta.rlrr as { recordingMetadata?: { artist?: unknown } } | undefined;
  const rlrrArtist = rlrr?.recordingMetadata?.artist;
  if (typeof rlrrArtist === 'string' && rlrrArtist.trim() !== '') return rlrrArtist.trim();
  return undefined;
}

/**
 * Display string for the score's `<h2>`. Appends ` - <artist>` when the
 * artist is known (RLRR-loaded charts today), so the header reads
 * "Song Name - Artist Name". When no artist is known the title stands
 * alone. Empty when the jot has neither title nor artist; the caller
 * shows the "Untitled jot" placeholder in that case.
 */
export function formatDisplayTitle(jot: RenderedJot): string {
  const title = jot.title.trim();
  const artist = extractArtist(jot);
  if (title && artist) return `${title} - ${artist}`;
  if (title) return title;
  if (artist) return artist;
  return '';
}

export function formatSubtitle(jot: RenderedJot): string {
  const parts: string[] = [];
  const { bpm: globalBpm, time: globalTime, vol } = jot.globalMetadata;
  const { dominantBpm, dominantTime } = jot.dominantBpmAndTime;

  if (dominantBpm !== undefined) parts.push(`${dominantBpm} bpm`);
  else if (typeof globalBpm === 'number') parts.push(`${globalBpm} bpm`);
  else if (globalBpm) parts.push(`${globalBpm.start ?? '?'}-${globalBpm.end} bpm`);

  const time = dominantTime ?? globalTime;
  if (time) parts.push(`${time.count}/${time.unit}`);
  if (typeof vol === 'string') parts.push(vol);
  else if (vol) parts.push(`${vol.start ?? '?'} -> ${vol.end}`);
  return parts.join('  -  ');
}

export const Legend = observer(({ jot }: { jot: RenderedJot }) => {
  // Aggregate unique pitches across all voices, in first-seen order.
  // Cached on the jot itself (`legendPitches`) so the walk is shared
  // across observers and only recomputes when the structural cache
  // changes, not on every zoom tick.
  const entries = jot.legendPitches;
  if (entries.length === 0) return null;
  return (
    <div className={sharedStyles.legend}>
      {entries.map(([pitch, info]) => (
        <span key={pitch} className={sharedStyles.legendChip}>
          <span className={sharedStyles.legendSwatch} style={{ background: info.color }} />
          <strong>{pitch}</strong>
          {info.name ? <span>{info.name}</span> : null}
        </span>
      ))}
    </div>
  );
});
