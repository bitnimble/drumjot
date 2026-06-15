import { observer } from 'mobx-react-lite';
import { Jot } from 'src/schema/dsl/dsl';
import type { TempoPresenter } from 'src/editing/playback/tempo_presenter';
import type { PaletteStore } from 'src/editing/palette/palette_store';
import sharedStyles from '../jot_editor.module.css';

/**
 * Read the artist string from wherever a loader plausibly stashed it.
 * Today only the RLRR (Paradiddle map) loader surfaces an artist, on
 * `globalMetadata.rlrr.recordingMetadata.artist`; a top-level
 * `globalMetadata.artist` is accepted too so hand-authored DSL or a
 * future loader can populate it directly. Anything non-string or empty
 * returns `undefined`, which makes the call site fall back to the
 * title alone.
 */
export function extractArtist(source: Jot): string | undefined {
  const meta = source.globalMetadata as Record<string, unknown>;
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
export function formatDisplayTitle(source: Jot): string {
  const title = source.title.trim();
  const artist = extractArtist(source);
  if (title && artist) return `${title} - ${artist}`;
  if (title) return title;
  if (artist) return artist;
  return '';
}

export function formatSubtitle(source: Jot, tempo: TempoPresenter): string {
  const parts: string[] = [];
  const { bpm: globalBpm, time: globalTime, vol } = source.globalMetadata;
  const { dominantBpm, dominantTime } = tempo.dominantBpmAndTime;

  if (dominantBpm !== undefined) parts.push(`${dominantBpm} bpm`);
  else if (typeof globalBpm === 'number') parts.push(`${globalBpm} bpm`);
  else if (globalBpm) parts.push(`${globalBpm.start ?? '?'}-${globalBpm.end} bpm`);

  const time = dominantTime ?? globalTime;
  if (time) parts.push(`${time.count}/${time.unit}`);
  if (typeof vol === 'string') parts.push(vol);
  else if (vol) parts.push(`${vol.start ?? '?'} -> ${vol.end}`);
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
