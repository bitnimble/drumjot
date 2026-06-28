import { observer } from 'mobx-react-lite';
import type { MutableJot } from 'src/schema/schema';
import type { PaletteStore } from 'src/editing/palette/palette_store';
import sharedStyles from '../jot_editor.module.css';

/** Parse the opaque `globalMetadataJson` residual (artist / vol / provenance /
 *  custom keys) back into a plain object for the header's display reads. The
 *  structurally-modelled metadata (title, bpm, time) is read from its own
 *  reactive field instead. Returns `{}` when absent or malformed. */
function residualMetadata(jot: MutableJot): Record<string, unknown> {
  const raw = jot.globalMetadataJson;
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? (parsed as Record<string, unknown>) : {};
  } catch {
    return {};
  }
}

/**
 * Read the artist string from wherever a loader plausibly stashed it.
 * Today only the RLRR (Paradiddle map) loader surfaces an artist, on
 * `rlrr.recordingMetadata.artist`; a top-level `artist` key is accepted too
 * so hand-authored DSL or a future loader can populate it directly. Anything
 * non-string or empty returns `undefined`, which makes the call site fall
 * back to the title alone.
 */
export function extractArtist(jot: MutableJot): string | undefined {
  const meta = residualMetadata(jot);
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
export function formatDisplayTitle(jot: MutableJot): string {
  const title = jot.title.trim();
  const artist = extractArtist(jot);
  if (title && artist) return `${title} - ${artist}`;
  if (title) return title;
  if (artist) return artist;
  return '';
}

export function formatSubtitle(jot: MutableJot): string {
  const parts: string[] = [];
  const vol = residualMetadata(jot).vol;
  const { dominantBpm, dominantTime } = jot.dominantBpmAndTime;

  if (dominantBpm !== undefined) parts.push(`${dominantBpm} bpm`);
  else parts.push(`${jot.bpm} bpm`);

  if (dominantTime) parts.push(`${dominantTime.count}/${dominantTime.unit}`);
  if (typeof vol === 'string') parts.push(vol);
  else if (vol && typeof vol === 'object') {
    const v = vol as { start?: unknown; end?: unknown };
    parts.push(`${v.start ?? '?'} -> ${v.end}`);
  }
  return parts.join('  -  ');
}

/**
 * Per-jot colour/name legend shown above the score. Aggregates unique
 * lanes across all layers (in first-seen order, cached on the jot as
 * `legendLanes`).
 */
export const Legend = observer(({ palette }: { palette: PaletteStore }) => {
  const entries = palette.legend;
  if (entries.length === 0) return null;
  return (
    <div className={sharedStyles.legend}>
      {entries.map(([lane, info]) => (
        <span
          key={lane}
          className={sharedStyles.legendChip}
          data-testid="legend-chip"
          data-lane={lane}
        >
          <span
            className={sharedStyles.legendSwatch}
            style={{ background: info.color }}
            data-testid="legend-swatch"
          />
          <strong>{lane}</strong>
          {info.name ? <span data-testid="legend-name">{info.name}</span> : null}
        </span>
      ))}
    </div>
  );
});
