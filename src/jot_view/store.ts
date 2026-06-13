import { comparer, makeAutoObservable, reaction, runInAction } from 'mobx';
import { computedFn } from 'mobx-utils';
import {
  DebugBundleManifest,
  loadDebugZip,
  NO_DRUMS_KEY,
  NoteProvenanceEntry,
  NoteProvenanceFile,
} from 'src/debug_zip';
import { Instrument } from 'src/dsl';
import { ExampleJot } from 'src/fakes';
import { DrumInstrumentKind, defaultKindForPitch } from 'src/instruments';
import { RenderedJot, px } from 'src/jot';
import {
  AlignLyricsRequest,
  LyricLine,
  LyricsSource,
  LyricsTrackId,
  alignLyricsForced,
  lyricsStore,
  nameLooksLikeVocals,
  parseEnhancedLrc,
  stripLyricNoise,
} from 'src/lyrics';
import { fromMidi } from 'src/midi';
import { parse, ParseError } from 'src/parser';
import {
  AudioTrackFilter,
  AudioTrackId,
  AudioTrackRole,
  isAudibleUnder,
  isAudioTrackAudibleUnder,
  jotPlayer,
  PlayerFilter,
  buildTimeline,
  xToTime,
} from 'src/playback';
import { pickDominantBpmAndTime } from 'src/playback/timeline';
import { loadParadbZip } from 'src/rlrr';
import {
  BeatInput,
  DrumSeparator,
  LlmModel,
  stemUrl,
  titleFromFilename,
  transcriber,
  TranscribeProgress,
  TranscribeStage,
  TranscriptionSummary,
} from 'src/transcriber';
import type { NoteProvenanceContextValue } from './contexts';
import { transcribeSuccessToastMessage } from './toasts_messages';
import { toastStore } from './toasts';
import { SettingsStore } from './stores/settings_store';
import { DocumentStore } from './stores/document_store';

/** Long-running lyric-alignment indicator. `queued` is the wait state
 *  while the request sits behind another in-flight GPU job (a transcribe
 *  or another align); `aligning` is once it owns the GPU and forced
 *  alignment is actually running. Success and failure surface as toasts
 *  (see `./toasts.ts`). */
export type LyricsAlignStatus =
  | { phase: 'idle' }
  | { phase: 'queued'; detail: string }
  | { phase: 'aligning'; detail: string };

/** Long-running stem-split indicator for an audio track. `mix` covers
 *  stage 1 (`stems_all`, isolating drums + drumless backing from a full
 *  mix); `pieces` covers stage 2 (`stems_per`, splitting an isolated
 *  drum stem into per-instrument pieces). Only the in-flight phase is
 *  modelled here; success and failure surface as toasts. The store
 *  exposes {@link beginAudioTrackSplit} / {@link endAudioTrackSplit} so
 *  the future server wiring just brackets its work with those calls
 *  and the per-row spinner picks it up automatically. */
export type AudioTrackSplitStatus = { phase: 'splitting'; kind: 'mix' | 'pieces' };

/** Long-running transcribe indicator. Only the in-flight `uploading`
 *  phase is modelled here; success and failure surface as toasts. */
export type TranscribeStatus =
  | { phase: 'idle' }
  | {
      phase: 'uploading';
      filename: string;
      /** Current pipeline stage (`stems_all`, `beats`, `transcribe`, …)
       *  reported by the server's NDJSON progress stream. `undefined`
       *  until the first stage event arrives; the initial "uploading"
       *  read covers everything before the first stage starts. */
      stage?: TranscribeStage;
      /** Optional in-stage detail, e.g. "filtering 3/5 instruments
       *  (latest: snare)". Cleared whenever the stage advances. */
      substage?: string;
    };

export type TranscribeOptions = {
  debug: boolean;
  beatInput: BeatInput;
  /** Stage-2 separator: `mdx23c` (default) or the opt-in `larsnet`. */
  drumSeparator: DrumSeparator;
  /** Model for the three Opus-by-default classification stages. */
  llmModel: LlmModel;
  /** Run the optional `quantise` pipeline stage. False = no snap; every
   *  onset keeps its raw detected time, the MIDI emitter writes it as
   *  a near-grid tick + sub-slot offset, and the UI / playback honour
   *  the offset so nothing re-snaps on load. */
  quantise: boolean;
  /** Run the LLM residual pass inside the quantise stage. Disables that
   *  pass when false; geometric + envelope + grid still run. No-op
   *  when `quantise` is false. */
  quantiseUseLlm: boolean;
};

// `GridLineSettings` + the grid-line / uniform-waveform display state
// moved to `SettingsStore`. Re-exported so existing type imports from
// `./store` keep working during the carve-up.
export type { GridLineSettings } from './stores/settings_store';

/**
 * Pixels-per-bar at zoom = 1. Same numeric value as `ViewConfig.barWidth`'s
 * own default so existing layouts are unchanged for users who never touch
 * the slider.
 */
export const BASE_BAR_WIDTH = 448;
export const MIN_ZOOM = 0.1;
export const MAX_ZOOM = 4.0;

/** Snap a CSS-pixel value to the nearest 1/dpr boundary. The
 *  `.scrollViewport`'s `transform: translate3d(--scroll-x × -1px, ...)`
 *  is composited at device-pixel resolution; if scroll values are
 *  sub-device-pixel (e.g. 100.3 CSS px on a 2x display = 200.6 device
 *  px), the compositor bilinearly interpolates the bitmap each frame
 *  and the interpolation distribution shifts as scroll advances,
 *  producing a visible ~1px back-and-forth wobble during auto-follow.
 *  Snapping keeps every scroll value on the device grid: at dpr=2
 *  that's 0.5 CSS-px steps, fine enough that 120-fps auto-follow at
 *  120 BPM still advances smoothly (~3-4 device pixels per frame at
 *  pxPerBeat 112, more at higher zoom).
 *  Also used to lock `--playhead-x` to the same grid as `scrollX` in
 *  `PlayheadPosVar` so the centred playhead doesn't drift sub-pixel
 *  against the bars below it. */
export function snapToDevicePx(x: number): number {
  if (typeof window === 'undefined') return x;
  const dpr = window.devicePixelRatio || 1;
  return Math.round(x * dpr) / dpr;
}
// Row volume faders are pure attenuation (0 = silent, 1 = unscaled).
// The kit's overall loudness is handled by the drum master gain.
export const VOLUME_STEP = 0.05;

// Sticky gutter column width (px). Default matches the legacy
// hardcoded 132px so existing layouts are unchanged; the user can drag
// the gutter's right edge to widen it when long track names are clipped
// with `…` and `fit-content` would be too jumpy.
export const DEFAULT_GUTTER_WIDTH = 132;
// Floor at the width needed to fit the row gutter's minimum content:
// padding + drag handle + a short volume slider + the M/S button pair.
export const MIN_GUTTER_WIDTH = 128;
export const MAX_GUTTER_WIDTH = 480;

export function clampVolume(v: number): number {
  if (!Number.isFinite(v)) return 1;
  return Math.max(0, Math.min(1, v));
}

// `TrackKey` + `trackKeyEq` were moved to `src/tracks.ts` alongside the
// `MixerContext` interface they participate in. Re-exported here so
// existing callers (and the JSDoc references below to
// `JotViewStore.trackOrder`) keep working without an import churn.
import {
  buildDebugBundleTrackOrder,
  trackKeyEq,
  type TrackKey,
  InstrumentTrack,
  INSTRUMENT_FALLBACK_COLOR,
  type MixerContext,
} from 'src/tracks';
export type { TrackKey };
export { trackKeyEq };

/**
 * Default top-to-bottom mixer ordering for drum-instrument kinds when
 * the user hasn't manually reordered rows: top-of-kit cymbals first,
 * then drums from high to low, with kick last. `custom` falls to the
 * very bottom. Drives `collectJotPitches`, so both `syncTrackOrder` and
 * the debug-bundle / ParaDB track-order layouts adopt it.
 */
const DEFAULT_MIXER_KIND_ORDER: readonly DrumInstrumentKind[] = [
  'crash',
  'ride',
  'hihat',
  'tom',
  'snare',
  'kick',
  'custom',
];

/**
 * Best-effort `DrumInstrumentKind` from an instrument's display name.
 * Used to recover a sensible mixer position for rows whose loader
 * stamped `kind: 'custom'` despite a recognisable name; e.g. ParaDB
 * fallback-allocated pitches (a `BP_Snare2_C` not in our class table
 * still arrives with `name = 'BP_Snare2_C'`). Substring-based; the
 * patterns mirror the names produced by the RLRR / MIDI / transcriber
 * loaders.
 */
function inferKindFromInstrumentName(name: string | undefined): DrumInstrumentKind | undefined {
  if (!name) return undefined;
  const n = name.toLowerCase();
  if (/\bkick\b|\bbass\s*drum\b/.test(n)) return 'kick';
  if (/\bsnare\b/.test(n)) return 'snare';
  if (/hi.?hat/.test(n)) return 'hihat';
  if (/\bride\b/.test(n)) return 'ride';
  if (/\bcrash\b|\bchina\b|\bsplash\b/.test(n)) return 'crash';
  if (/\bfloor\s*tom\b|\btom\b/.test(n)) return 'tom';
  return undefined;
}

/**
 * Floor toms render below regular toms within the tom group. Detected
 * from the instrument name; pitch letter `f` is the GM importer's
 * convention for the floor tom (see `src/instruments.ts`) so it counts
 * even when the instrument has no display name.
 */
function isFloorTom(instrument: Instrument | undefined, pitch: string): boolean {
  if (instrument?.name && /floor/i.test(instrument.name)) return true;
  return pitch === 'f';
}

/**
 * Sort tuple for the default mixer order: [kind rank, intra-kind rank,
 * pitch]. Kind comes from the parsed `Instrument` when available;
 * `kind: 'custom'` falls back to a name heuristic, then to the pitch
 * letter's default kind (so a raw DSL jot without `instrumentMapping`
 * still lands on the canonical layout). Intra-kind rank only matters
 * for toms today (regular before floor).
 */
function defaultMixerSortKey(
  pitch: string,
  instrument: Instrument | undefined
): [number, number, string] {
  let kind: DrumInstrumentKind = instrument?.kind ?? 'custom';
  if (kind === 'custom') {
    const fromName = inferKindFromInstrumentName(instrument?.name);
    if (fromName) kind = fromName;
  }
  if (kind === 'custom') {
    const fromLetter = defaultKindForPitch(pitch);
    if (fromLetter !== 'custom') kind = fromLetter;
  }
  const kindRank = DEFAULT_MIXER_KIND_ORDER.indexOf(kind);
  const subRank = kind === 'tom' && isFloorTom(instrument, pitch) ? 1 : 0;
  return [kindRank === -1 ? DEFAULT_MIXER_KIND_ORDER.length : kindRank, subRank, pitch];
}

/**
 * Map a transcriber-side provenance pitch tag onto the jot's pitch
 * letter. The transcriber's hi-hat split (`transcriber/app/pipeline/
 * hihat_split.py`) routes open-hi-hat onsets through synthetic pitch
 * `H` so the filter LLM can see closed (`h`) and open (`H`) hits as
 * separate lanes; from_midi.ts then folds those back into the standard
 * `h:o` notation. Provenance lookups (debug-details popover, "show
 * filtered" ghost overlays) have to canonicalise the same way so the
 * jot's `note.pitch = 'h'` finds entries the provenance stored under
 * `'H'`. Adding new synthetic-pitch routes (e.g. a future ride-bell
 * split) means adding a case here.
 */
function canonicalProvenancePitch(transcriberPitch: string): string {
  if (transcriberPitch === 'H') return 'h';
  return transcriberPitch;
}

/**
 * Pitches that appear anywhere in the rendered jot, sorted into the
 * default mixer ordering (see {@link DEFAULT_MIXER_KIND_ORDER}). A
 * pitch that shows up in two voices is listed once at its first
 * appearance; ordering reads each pitch's resolved `Instrument` (from
 * the first bar that has a track for it) so ParaDB / debug-bundle
 * loads, whose instrument names carry the kind even when the pitch
 * letter has been fallback-allocated, still land on the canonical
 * layout.
 *
 * Reads the zoom-invariant structural cache (not `jot.resolved`) so the
 * mixer-order reaction that wraps this doesn't re-evaluate on every
 * wheel tick; pitch identity is a function of the source DSL, not the
 * pixel layout.
 */
export function collectJotPitches(jot: RenderedJot | undefined): string[] {
  if (!jot) return [];
  const out: string[] = [];
  const instrumentByPitch = new Map<string, Instrument>();
  for (const voice of jot.structure.voices) {
    for (const p of voice.pitches) {
      if (!out.includes(p)) out.push(p);
    }
    for (const bar of voice.bars) {
      for (const [pitch, track] of Object.entries(bar.tracks)) {
        if (!instrumentByPitch.has(pitch)) {
          instrumentByPitch.set(pitch, track.instrument);
        }
      }
    }
  }
  out.sort((a, b) => {
    const ka = defaultMixerSortKey(a, instrumentByPitch.get(a));
    const kb = defaultMixerSortKey(b, instrumentByPitch.get(b));
    if (ka[0] !== kb[0]) return ka[0] - kb[0];
    if (ka[1] !== kb[1]) return ka[1] - kb[1];
    return ka[2].localeCompare(kb[2]);
  });
  return out;
}

export class JotViewStore {
  transcribeStatus: TranscribeStatus = { phase: 'idle' };
  /** UI-controlled options for the next transcribe call. `debug=true`
   *  so the run is resumable. */
  transcribeOptions: TranscribeOptions = {
    debug: true,
    beatInput: 'full_mix',
    drumSeparator: 'mdx23c',
    llmModel: 'claude-haiku-4-5-20251001',
    quantise: true,
    quantiseUseLlm: false,
  };
  /** Server-side picker of recent /transcribe runs that can be resumed.
   *  Populated by {@link refreshRecentTranscriptions}; an empty array
   *  before the first fetch (the picker shows "Loading…" in that state).
   *  Refreshed lazily whenever the toolbar opens the Transcribe dropdown
   *  so the operator sees their just-completed run without needing to
   *  reload the page. */
  recentTranscriptions: TranscriptionSummary[] = [];
  /** True once {@link refreshRecentTranscriptions} has completed at least
   *  one fetch (success or empty). The Load → Recent submenu uses this to
   *  decide whether to issue the initial fetch on first open or use the
   *  cache; the Transcribe dropdown refreshes eagerly on each open so it
   *  doesn't read this flag. */
  recentTranscriptionsLoaded: boolean = false;
  /** True while an in-flight {@link refreshRecentTranscriptions} is
   *  resolving. Drives the spinner inside the Load → Recent submenu. */
  recentTranscriptionsLoading: boolean = false;
  /** Folder name of the currently-selected recent transcription, or
   *  `undefined` when nothing is selected. Drives the stage picker (we
   *  read `resumable_stages` off the matching summary). */
  selectedResumeFolder: string | undefined = undefined;
  /** Stage the user has picked to resume from. `undefined` until they
   *  pick one; reset whenever {@link selectedResumeFolder} changes so
   *  stale picks from one folder can't leak into another folder's
   *  request. */
  selectedResumeStage: TranscribeStage | undefined = undefined;
  /** Which flow the Transcribe dropdown is showing: a fresh upload
   *  (`new`) or resume-from-debug-folder (`resume`). Defaults to `new`
   *  since that's the only flow available before any runs exist; the
   *  toolbar coerces the rendered mode back to `new` whenever the
   *  recent-runs list is empty so a stale `resume` selection can't
   *  surface an empty form. */
  transcribeMode: 'new' | 'resume' = 'new';
  /** Horizontal zoom multiplier; 1.0 = `BASE_BAR_WIDTH` pixels per bar. */
  zoom: number = 1;
  /** DSL pitches the user has muted via the row-gutter M button. */
  mutedPitches: Set<string> = new Set();
  /**
   * DSL pitches the user has soloed. When non-empty, ONLY these rows
   * are audible; this and `mutedPitches` are pushed to the player via
   * an autorun so toggles take effect live during playback.
   */
  soloedPitches: Set<string> = new Set();
  /** Audio-track ids the user has muted via the gutter M button. */
  mutedAudioTracks: Set<AudioTrackId> = new Set();
  /** Soloed audio-track ids; same semantics as `soloedPitches`. */
  soloedAudioTracks: Set<AudioTrackId> = new Set();
  /**
   * Section-master mute / solo. These act on the whole bus, not by
   * editing the per-row M/S sets; see {@link isAudioSectionAudible} and
   * {@link isDrumSectionAudible} for the audibility formula and
   * `JotPlayer.setAudioMasterAudible` / `.setDrumMasterAudible` for the
   * audio-graph side. Master-solo is folded into {@link soloActive} so
   * it participates in the same cross-domain "if anything is soloed,
   * non-soloed rows fall silent" rule the per-row solos already follow.
   */
  audioMasterMuted: boolean = false;
  drumMasterMuted: boolean = false;
  audioMasterSoloed: boolean = false;
  drumMasterSoloed: boolean = false;
  /**
   * Per-row volume faders, 0..1 (1 = full). Sparse: a row absent from
   * the map plays at full volume. Pitch volumes scale note velocity in
   * the scheduler; audio-track volumes scale the track's GainNode.
   */
  pitchVolumes: Map<string, number> = new Map();
  audioTrackVolumes: Map<AudioTrackId, number> = new Map();
  /**
   * Per-pitch {@link InstrumentTrack} view-models keyed by DSL pitch
   * letter. Each holds the user's per-instrument note-colour override
   * (sparse; `_color === undefined` falls back to the jot's palette
   * default). Survives jot reloads so a customisation made on one song
   * persists onto matching pitches in the next song; a kept-in-sync
   * reaction drops entries for pitches no longer present in any
   * loaded jot (see the constructor reaction).
   *
   * Audio-track colour overrides live on the {@link AudioTrack}
   * instance itself; they're not stored here.
   */
  instrumentTracks: Map<string, InstrumentTrack> = new Map();
  /**
   * User-customizable order of mixer rows. Each entry is either a
   * loaded audio track id or a DSL pitch letter; the mixer renders rows
   * top-to-bottom in this exact order, with audio and drum-instrument
   * rows freely interleavable.
   *
   * Kept in sync with the live set of audio tracks (added via
   * {@link loadAudioTrack}, removed via {@link clearAudioTrack}) and
   * the pitches in the current jot through a reaction in the
   * constructor — entries that no longer correspond to anything are
   * dropped, new audio tracks append after the last audio entry (or to
   * the top if none), and new pitches append at the end so manual
   * reorderings survive reloads.
   */
  trackOrder: TrackKey[] = [];
  /**
   * Last loaded transcriber debug bundle (`.zip`), if any. Carries the
   * captured logs + per-stage timings produced server-side during a
   * transcribe run, so the UI's DebugPanel can show what happened end-
   * to-end without requiring a `docker compose logs` round trip.
   * Replaced when a new bundle is loaded; otherwise survives jot/audio
   * changes.
   */
  lastDebugBundle: DebugBundleManifest | undefined = undefined;
  /**
   * Per-note debug provenance from the loaded debug bundle, if the
   * bundle came from a filter-mode transcribe run. Keyed by DSL pitch
   * letter → list of every detected onset (kept and rejected). The
   * NoteView selection label looks up its provenance by matching
   * `note.metadata.midi.tick` against entries' `tick`; the
   * FilteredOnsetView renders the `kept=false` entries as ghost
   * overlays gated by {@link showFilteredOnsets}. `undefined` until a
   * filter-mode bundle is loaded; cleared when a new (non-bundle) song
   * replaces the current one.
   */
  noteProvenance: NoteProvenanceFile | undefined = undefined;
  /**
   * Toolbar checkbox: show rejected onsets as dashed ghost overlays.
   * Only meaningful when {@link noteProvenance} is loaded; the checkbox
   * is hidden when there's nothing to show. Default off so a freshly
   * loaded bundle reads as just "the score" until the operator opts in.
   */
  showFilteredOnsets: boolean = false;
  /**
   * When true, the score auto-scrolls horizontally during playback to
   * keep the playhead pinned to the viewport's centre
   * (`PlayheadAutoScroller` in `jot_view.tsx`). Toggle off via the
   * button above the playhead label to scroll freely while playing
   *; useful for previewing an upcoming section without pausing.
   * Session-only; resets to true on reload.
   */
  followPlayhead: boolean = true;
  /**
   * When true, transitioning to the playing state re-enables
   * {@link followPlayhead} if the user disabled it *during* the previous
   * playback session (pan, minimap drag, or the follow-button toggle
   * while playing). An off-state set while idle/paused is treated as
   * deliberate and survives the play press. Session-only, defaults on.
   */
  autoFollowOnPlay: boolean = true;
  /**
   * Internal: was the current `followPlayhead === false` set during
   * playback (transient, eligible for auto-re-enable on next play) or
   * during idle/paused (deliberate, must survive). Always false while
   * `followPlayhead` is true. See {@link setFollowPlayhead}.
   */
  private followDisabledIsTransient: boolean = false;
  /** Whether the DebugPanel is expanded, small UI state, kept here so
   * the toolbar toggle and the panel itself stay in sync. */
  debugPanelOpen: boolean = false;
  /** Lyrics search modal visibility. */
  lyricsSearchOpen: boolean = false;
  /** Lyrics plain-text load modal visibility. */
  lyricsTextOpen: boolean = false;
  /** Height of the DebugPanel (px) when expanded; adjusted by dragging
   * the resize handle along its top edge. */
  debugPanelHeight: number = 280;
  /** Width (px) of the sticky mixer/score gutter column; user-resizable
   * by dragging the gutter's right edge. Propagated to every gutter
   * element through the `--gutter-width` CSS variable set on the JotView
   * container. */
  gutterWidth: number = DEFAULT_GUTTER_WIDTH;
  /**
   * Virtual horizontal scroll offset (px) for the score viewport.
   *
   * The score doesn't use native overflow scrolling: `.jotContainer` is
   * `overflow: hidden`, and an inner `.scrollViewport` wrapper translates
   * by `(-scrollX, -scrollY)` via CSS `transform`. Driving scroll
   * through this observable instead of `el.scrollLeft` gives subpixel
   * precision (browsers integer-snap `scrollLeft`, which made the
   * playhead wobble ~1px against the bars during auto-follow), and
   * makes scroll position reactive so observers (the minimap viewport
   * box) just `observer()` it and re-render with no `scroll` event hookup.
   *
   * Clamped to `[0, contentWidth - viewportWidth]` inside `setScrollX`.
   * The bounds come from `_viewportWidth` / `_contentWidth` which
   * JotView's ResizeObservers feed via `setViewportSize` /
   * `setContentSize`.
   */
  scrollX: number = 0;
  /** Virtual vertical scroll offset (px). See `scrollX`. */
  scrollY: number = 0;
  /**
   * Cached viewport (jotContainer clientWidth/Height) and content
   * (scrollViewport offsetWidth/Height) dimensions, fed by a
   * ResizeObserver in JotView via `setViewportSize` / `setContentSize`.
   * Used internally to clamp `scrollX` / `scrollY` to `[0, content -
   * viewport]`, and also read by per-frame observers that derive what
   * to paint from the visible viewport (e.g. the waveform-chunk
   * visibility slice in `mixer.tsx`; see AGENTS.md §5.9 on the no-DOM-
   * layout-reads rule).
   *
   * MobX-observable (the constructor's `makeAutoObservable` only marks
   * `transcribeController` / `lyricsAlignControllers` as non-observable;
   * everything else defaults to observable). The underscore prefix is
   * historical and signals "consumers should generally go through the
   * dedicated setters" rather than "non-reactive"; an observer that
   * reads any of these will re-run when its specific field changes,
   * and only that observer (MobX tracks per-field).
   */
  _viewportWidth: number = 0;
  _viewportHeight: number = 0;
  _contentWidth: number = 0;
  _contentHeight: number = 0;
  /**
   * Controller for the in-flight `/transcribe` request, if any. The
   * "Stop" toolbar button calls `.abort()` here; the request's
   * AbortSignal is passed into `transcriber.transcribe` which forwards
   * it to `fetch` so the request is genuinely cancelled at the
   * network layer rather than just discarding the response.
   */
  transcribeController: AbortController | undefined;

  /**
   * Wrap an async file-load with the modal overlay's bookkeeping (the
   * loading counter / label live on {@link DocumentStore}). Errors
   * propagate; the finally block guarantees the counter decrements even if
   * the inner promise rejects, so a failed load never leaves the overlay
   * stuck on screen.
   */
  private async withLoading<T>(label: string, fn: () => Promise<T>): Promise<T> {
    runInAction(() => {
      if (this.document.loadingCount === 0) this.document.loadingLabel = label;
      this.document.loadingCount += 1;
    });
    try {
      return await fn();
    } finally {
      runInAction(() => {
        this.document.loadingCount -= 1;
        if (this.document.loadingCount === 0) this.document.loadingLabel = undefined;
      });
    }
  }

  /** Data-only stores carved out of this (transitional) store. Held as
   * references so the orchestration still living here can read/write them;
   * excluded from observability (they're already-observable stores). */
  readonly document: DocumentStore;
  readonly settings: SettingsStore;

  constructor(document: DocumentStore, settings: SettingsStore) {
    this.document = document;
    this.settings = settings;
    makeAutoObservable(this, {
      transcribeController: false,
      lyricsAlignControllers: false,
      document: false,
      settings: false,
    });
    // Wire ourselves in as the player's mixer context so freshly-
    // constructed AudioTracks can resolve grouped-instrument colour
    // inheritance. Done before any reactions fire so loadAudioTrack
    // calls made during the same tick see a populated context.
    jotPlayer.attachMixerContext(this as MixerContext);

    // Prune instrument-track view-models for pitches no longer present
    // in the active jot. The override is store-owned and survives jot
    // reloads, so a pitch that comes back in a later jot picks up its
    // previous override; we only forget pitches that disappeared from
    // the current jot to keep the map from growing unboundedly across
    // a long session. `fireImmediately` runs the prune once on boot
    // (a no-op when the map is empty).
    reaction(
      () => new Set(this.jotPitches),
      (pitches) => {
        for (const p of Array.from(this.instrumentTracks.keys())) {
          if (!pitches.has(p)) this.instrumentTracks.delete(p);
        }
      },
      { fireImmediately: true, equals: comparer.structural }
    );

    // Push mute / solo state to the player whenever it changes. While
    // playback is in flight, the player cancels and reschedules events
    // so the toggle takes effect immediately (including bringing
    // previously-muted rows back mid-song). When idle, the filter is
    // just stored for the next play().
    //
    // This MUST be a `reaction`, not an `autorun`: `setFilter` both
    // reads (`scheduleEvents` → `isAudibleUnder(..., this.currentFilter)`)
    // and writes (`this.currentFilter = filter`) an observable on the
    // MobX-observable player while playing. An `autorun` tracks reads
    // made during the effect, so it would depend on the very observable
    // it writes — a non-converging reaction that MobX bails on, after
    // which the UI (observer components reading store state directly)
    // keeps updating but the filter stops reaching the player (e.g.
    // un-solo visually clears yet audio stays soloed). `reaction` only
    // tracks the data selector; the effect runs untracked, so the
    // player's internal reads/writes stay out of the dependency graph.
    // The data fn returns the `pitchFilter` computed; that getter
    // snapshots its Sets/Maps so the structural comparer can actually
    // detect mute/solo/volume changes. Sharing the live Set/Map
    // references would defeat the comparer (prev and next cached
    // snapshots would point to the same mutated instance, so a deep
    // walk sees no diff) and the reaction would silently stop firing
    // after the initial seed; see the `pitchFilter` getter's doc
    // comment.
    reaction(
      () => this.pitchFilter,
      (filter) => jotPlayer.setFilter(filter),
      { fireImmediately: true, equals: comparer.structural }
    );
    // Same shape for audio tracks; observed mutations push immediately
    // so toggling M/S on a track is sample-accurate during playback
    // (per-track GainNode flip, no source recreation). Same
    // read-and-write-the-same-observable hazard as above
    // (`setAudioTrackFilter` reads/writes `currentAudioTrackFilter`), so this is a
    // `reaction` for the same reason.
    reaction(
      () => this.audioTrackFilter,
      (filter) => jotPlayer.setAudioTrackFilter(filter),
      { fireImmediately: true, equals: comparer.structural }
    );
    // Push the section-audibility booleans to the player so master mute
    // and master solo can flip the bus gain to 0 without touching the
    // per-row M/S sets. fireImmediately to seed the initial unmuted state.
    reaction(
      () => this.isAudioSectionAudible,
      (audible) => jotPlayer.setAudioMasterAudible(audible),
      { fireImmediately: true }
    );
    reaction(
      () => this.isDrumSectionAudible,
      (audible) => jotPlayer.setDrumMasterAudible(audible),
      { fireImmediately: true }
    );
    // Seed the player's live drum↔audio offset from each loaded jot's
    // transcribed lead-in (`globalMetadata.drumsT0Sec`). Tracking
    // `currentJot` (an observable reference) re-fires whenever a new jot
    // is loaded, resetting the offset to that recording's value; manual
    // nudges via the Offset control persist until the next load. We read
    // `globalMetadata` (the raw source) rather than `resolved` so seeding
    // doesn't force a layout pass.
    reaction(
      () => {
        const raw = this.document.currentJot?.globalMetadata.drumsT0Sec;
        return typeof raw === 'number' && raw > 0 ? raw : 0;
      },
      (offsetSec) => jotPlayer.setDrumsT0Sec(offsetSec),
      { fireImmediately: true }
    );

    // Keep `trackOrder` synced with the live audio-track set and the
    // current jot's pitches. The reaction fires whenever either changes;
    // dropped rows are removed and newly-discovered rows are slotted at
    // a sensible default position (new audio tracks → end of the audio
    // block, new pitches → end of the list) so the user's drag-and-drop
    // ordering of surviving rows is preserved. `fireImmediately` seeds
    // the initial ordering on construction.
    reaction(
      () => ({
        audioIds: Array.from(jotPlayer.audioTracks.keys()),
        pitches: this.jotPitches,
        lyricsIds: lyricsStore.trackIds.slice(),
      }),
      ({ audioIds, pitches, lyricsIds }) => this.syncTrackOrder(audioIds, pitches, lyricsIds),
      { fireImmediately: true }
    );
  }

  /**
   * Drop entries from {@link trackOrder} that no longer correspond to a
   * live audio track, jot pitch, or lyrics track; then append the
   * missing ones at a sensible default position so the row appears
   * immediately:
   *   - new audio track  → after the last existing audio entry (or top
   *     of the list if no audio entries exist yet)
   *   - new pitch        → end of the list
   *   - new lyrics row   → just after the last existing lyrics row,
   *     keeping the lyrics group contiguous. The very first lyrics row
   *     (when none exist yet) goes to the top of the list. User can drag
   *     it elsewhere; its position survives subsequent reactions because
   *     the filter step preserves surviving entries.
   *
   * Existing entries keep their relative order so a user drag survives
   * an audio-track add/remove or a jot reload that didn't change the
   * pitch set.
   */
  private syncTrackOrder(
    audioIds: AudioTrackId[],
    pitches: readonly string[],
    lyricsIds: readonly LyricsTrackId[]
  ): void {
    const wanted: TrackKey[] = [
      ...lyricsIds.map((id) => ({ kind: 'lyrics' as const, id })),
      ...audioIds.map((id) => ({ kind: 'audio' as const, id })),
      ...pitches.map((pitch) => ({ kind: 'instrument' as const, pitch })),
    ];
    const next: TrackKey[] = this.trackOrder.filter((k) => wanted.some((w) => trackKeyEq(w, k)));
    for (const w of wanted) {
      if (next.some((k) => trackKeyEq(k, w))) continue;
      if (w.kind === 'lyrics') {
        // Slot a new lyrics row just after the last existing lyrics
        // entry so lyrics rows stay contiguous by default. When no
        // lyrics rows exist yet, default to the very top of the mixer
        // (above any audio / instrument rows); the filter step above
        // preserves the position the user drags it to on subsequent
        // runs.
        let insertAt: number | undefined;
        for (let i = next.length - 1; i >= 0; i--) {
          if (next[i].kind === 'lyrics') {
            insertAt = i + 1;
            break;
          }
        }
        if (insertAt === undefined) {
          next.unshift(w);
        } else {
          next.splice(insertAt, 0, w);
        }
      } else if (w.kind === 'audio') {
        // Slot a new audio track in just after the last existing audio
        // entry so audio rows stay contiguous by default.
        let insertAt = 0;
        for (let i = next.length - 1; i >= 0; i--) {
          if (next[i].kind === 'audio') {
            insertAt = i + 1;
            break;
          }
        }
        next.splice(insertAt, 0, w);
      } else {
        next.push(w);
      }
    }
    // The reaction fires whenever its data fn returns a new object, even
    // when the underlying sets are unchanged (zoom-driven layout passes
    // mint a fresh `pitches` array, audio-track mute toggles re-emit
    // `audioIds` of the same content, etc.). Skip the observable
    // assignment when the order is structurally identical — identity
    // *and* groupId — so downstream consumers (the mixer renderer)
    // don't pay an unnecessary re-render but a real grouping change
    // still propagates.
    if (
      next.length === this.trackOrder.length &&
      next.every(
        (k, i) => trackKeyEq(k, this.trackOrder[i]) && k.groupId === this.trackOrder[i].groupId
      )
    ) {
      return;
    }
    this.trackOrder = next;
  }

  /**
   * Reorder the mixer by moving the row at `fromIdx` to position
   * `toIdx`. Both indices refer to positions in the *current*
   * `trackOrder` (so `toIdx` is interpreted before the removal); a
   * no-op move (`fromIdx === toIdx`) is silently dropped.
   *
   * Drives the gutter drag-and-drop: the dragged row's start index is
   * the source; the drop target's "insert before me" index is the
   * destination. Used by the keyboard reorder shortcuts too (Alt+
   * Up/Down on a focused row).
   *
   * Group semantics on drop: the moved row's `groupId` is replaced with
   * the post-move adjacent rows' shared group, if any:
   *   - dropped inside an existing group (above + below in same group)
   *     → join that group
   *   - dropped at a group boundary, between solos, or at top/bottom
   *     → becomes solo (`groupId = undefined`)
   * This way a row can be dropped into a group by aiming for the middle
   * of it, and out of a group by aiming for a boundary or the end —
   * without any explicit "leave group" / "join group" UI.
   */
  moveTrack(fromIdx: number, toIdx: number): void {
    if (fromIdx < 0 || fromIdx >= this.trackOrder.length) return;
    const clamped = Math.max(0, Math.min(this.trackOrder.length, toIdx));
    if (clamped === fromIdx || clamped === fromIdx + 1) return;
    const next = this.trackOrder.slice();
    const [moved] = next.splice(fromIdx, 1);
    // After the removal a `toIdx` that was past the source shifts down
    // by one; before the source it's unaffected.
    const adjusted = clamped > fromIdx ? clamped - 1 : clamped;
    const above = adjusted > 0 ? next[adjusted - 1] : undefined;
    const below = adjusted < next.length ? next[adjusted] : undefined;
    const newGroupId =
      above && below && above.groupId !== undefined && above.groupId === below.groupId
        ? above.groupId
        : undefined;
    // Spread keeps the discriminant intact while overwriting groupId; a
    // direct assignment would widen the type and lose narrowing.
    let repositioned: TrackKey;
    if (moved.kind === 'audio') {
      repositioned = { kind: 'audio', id: moved.id, groupId: newGroupId };
    } else if (moved.kind === 'instrument') {
      repositioned = { kind: 'instrument', pitch: moved.pitch, groupId: newGroupId };
    } else {
      repositioned = { kind: 'lyrics', id: moved.id, groupId: newGroupId };
    }
    next.splice(adjusted, 0, repositioned);
    this.trackOrder = next;
  }

  /**
   * Solo is one global mode across both the pitch and audio-track domains: any
   * soloed row (drum *or* music) puts every non-soloed row; in either
   * domain; into the "solo-excluded" state. Without this, soloing a
   * drum to practise it would leave the backing music playing. The two
   * section-master solos count too, so soloing the Drums master silences
   * the Audio section even when no individual rows are soloed.
   */
  get soloActive(): boolean {
    return (
      this.soloedPitches.size > 0 ||
      this.soloedAudioTracks.size > 0 ||
      this.audioMasterSoloed ||
      this.drumMasterSoloed
    );
  }

  /**
   * Whether the audio section's bus is currently audible. Master mute
   * always wins; under an active solo the section is audible only if it
   * is master-soloed OR has at least one soloed row. Pushed to
   * {@link JotPlayer.setAudioMasterAudible} via a constructor reaction.
   */
  get isAudioSectionAudible(): boolean {
    if (this.audioMasterMuted) return false;
    if (!this.soloActive) return true;
    return this.audioMasterSoloed || this.soloedAudioTracks.size > 0;
  }

  /** Mirror of {@link isAudioSectionAudible} for the drum section. */
  get isDrumSectionAudible(): boolean {
    if (this.drumMasterMuted) return false;
    if (!this.soloActive) return true;
    return this.drumMasterSoloed || this.soloedPitches.size > 0;
  }

  /**
   * Live {@link PlayerFilter} view onto the per-pitch mute/solo/volume
   * state. Sets and Maps are *snapshotted* on each read (small entries;
   * sparse mute/solo sets, one entry per pitch with a fader nudge), so
   * the downstream `reaction(..., comparer.structural)` that pushes this
   * to the player can actually detect changes. Sharing the store's live
   * Set/Map references here would defeat the comparer: the prev and next
   * cached values both point to the same mutated instance, so a deep-
   * equal walk sees no diff and the reaction never refires; mute/solo
   * toggles update the UI but the player never learns about them.
   */
  get pitchFilter(): PlayerFilter {
    return {
      mutedPitches: new Set(this.mutedPitches),
      soloedPitches: new Set(this.soloedPitches),
      soloActive: this.soloActive,
      sectionMasterMuted: this.drumMasterMuted,
      sectionMasterSoloed: this.drumMasterSoloed,
      volumes: new Map(this.pitchVolumes),
    };
  }

  /** Mirror of {@link pitchFilter} for the audio-track domain. */
  get audioTrackFilter(): AudioTrackFilter {
    return {
      mutedAudioTracks: new Set(this.mutedAudioTracks),
      soloedAudioTracks: new Set(this.soloedAudioTracks),
      soloActive: this.soloActive,
      sectionMasterMuted: this.audioMasterMuted,
      sectionMasterSoloed: this.audioMasterSoloed,
      volumes: new Map(this.audioTrackVolumes),
    };
  }

  /**
   * Pitches that appear anywhere in the rendered jot, in the default
   * mixer ordering. Thin wrapper over {@link collectJotPitches} so the
   * `syncTrackOrder` reaction and any future consumer tracks a single
   * MobX-memoised computed rather than re-walking the jot structure on
   * every read.
   */
  get jotPitches(): readonly string[] {
    return collectJotPitches(this.document.currentJot);
  }

  /**
   * Drum-pitch lane order derived from {@link trackOrder}, dropping audio
   * + lyrics rows and keeping only pitches in the user's mixer order.
   * Pattern brackets in {@link MixerView} use this to know whether a
   * given row is the topmost / bottommost participant of a pattern span
   * so the bracket reads as one continuous outline across rows.
   */
  get pitchOrder(): readonly string[] {
    return this.trackOrder.flatMap((k) => (k.kind === 'instrument' ? [k.pitch] : []));
  }

  /**
   * Index of the topmost instrument row in {@link trackOrder}. The mixer
   * hosts score-wide chrome (tuplet brackets, lead-in label) on that row
   * because it belongs to the score as a whole, not to any one
   * instrument. `-1` when no instrument row exists yet.
   */
  get firstInstrumentIdx(): number {
    return this.trackOrder.findIndex((k) => k.kind === 'instrument');
  }

  /**
   * Bundle the per-note debug provenance into the shape
   * {@link NoteProvenanceContext} consumers expect, or `null` when no
   * filter-mode bundle is loaded. Memoised through the MobX computed
   * graph so the `audioFilenameByPitch` Map (rebuilt from the manifest's
   * plain-object `mapping`) is only re-constructed when the underlying
   * provenance / bundle / toggle changes.
   */
  get provenanceContextValue(): NoteProvenanceContextValue | null {
    const provenance = this.noteProvenance;
    if (!provenance) return null;
    return {
      byTick: this.noteProvenanceByTick,
      rejectedByPitch: this.filteredOnsetsByPitch,
      leadBars: provenance.lead_bars ?? 0,
      showFiltered: this.showFilteredOnsets,
      beatAlignmentOffsetSec: provenance.beat_alignment_offset_sec ?? null,
      beatAlignCoarseOffsetSec: provenance.beat_align_coarse_offset_sec ?? null,
      beatAlignFineOffsetSec: provenance.beat_align_fine_offset_sec ?? null,
      // Bundle manifest mapping is `Record<string, string>`; rebuild it
      // as a Map for ergonomic .get() lookups inside the per-onset
      // timing visualization. Empty when the current bundle didn't ship
      // a manifest (hand-authored jots, legacy bundles).
      audioFilenameByPitch: new Map(Object.entries(this.lastDebugBundle?.mapping ?? {})),
    };
  }

  toggleAudioMasterMute() {
    this.audioMasterMuted = !this.audioMasterMuted;
  }

  toggleDrumMasterMute() {
    this.drumMasterMuted = !this.drumMasterMuted;
  }

  /** Enabling solo clears the matching master-mute so the section can
   * actually be heard; mirrors `toggleSolo` for per-row state. */
  toggleAudioMasterSolo() {
    if (this.audioMasterSoloed) {
      this.audioMasterSoloed = false;
    } else {
      this.audioMasterSoloed = true;
      this.audioMasterMuted = false;
    }
  }

  toggleDrumMasterSolo() {
    if (this.drumMasterSoloed) {
      this.drumMasterSoloed = false;
    } else {
      this.drumMasterSoloed = true;
      this.drumMasterMuted = false;
    }
  }

  toggleMute(pitch: string) {
    if (this.mutedPitches.has(pitch)) this.mutedPitches.delete(pitch);
    else this.mutedPitches.add(pitch);
  }

  toggleSolo(pitch: string) {
    if (this.soloedPitches.has(pitch)) {
      this.soloedPitches.delete(pitch);
    } else {
      this.soloedPitches.add(pitch);
      this.mutedPitches.delete(pitch);
    }
  }

  /**
   * Whether a given drum pitch is currently audible under the live
   * mute / solo / volume state. `computedFn` memoises per-argument, so
   * the per-row gutter observer for pitch `k` only re-renders when `k`'s
   * audibility actually flips; toggling mute on pitch `s` doesn't pull
   * in `k`'s observer the way a method call would.
   */
  isPitchAudible = computedFn((pitch: string): boolean => {
    return isAudibleUnder(pitch, this.pitchFilter);
  });

  pitchVolume(pitch: string): number {
    return this.pitchVolumes.get(pitch) ?? 1;
  }

  setPitchVolume(pitch: string, v: number) {
    this.pitchVolumes.set(pitch, clampVolume(v));
  }

  toggleAudioTrackMute(id: AudioTrackId) {
    if (this.mutedAudioTracks.has(id)) this.mutedAudioTracks.delete(id);
    else this.mutedAudioTracks.add(id);
  }

  toggleAudioTrackSolo(id: AudioTrackId) {
    if (this.soloedAudioTracks.has(id)) {
      this.soloedAudioTracks.delete(id);
    } else {
      this.soloedAudioTracks.add(id);
      this.mutedAudioTracks.delete(id);
    }
  }

  /** Mirror of {@link isPitchAudible} for the audio-track domain. */
  isAudioTrackAudible = computedFn((id: AudioTrackId): boolean => {
    return isAudioTrackAudibleUnder(id, this.audioTrackFilter);
  });

  audioTrackVolume(id: AudioTrackId): number {
    return this.audioTrackVolumes.get(id) ?? 1;
  }

  setAudioTrackVolume(id: AudioTrackId, v: number) {
    this.audioTrackVolumes.set(id, clampVolume(v));
  }

  /**
   * Lazily-constructed {@link InstrumentTrack} for a DSL pitch. The
   * track's fallback closure reads the active jot's palette default for
   * the pitch, so a jot reload that re-shuffles palette slots updates
   * unfilled tracks automatically. Caches the instance in
   * {@link instrumentTracks} so the picker reads/writes the same MobX
   * observable from every callsite; the constructor reaction prunes
   * dead entries when pitches leave every loaded jot.
   *
   * Also serves the {@link MixerContext} the player's AudioTrack
   * colour resolution calls back into.
   */
  getInstrumentTrack(pitch: string): InstrumentTrack {
    let track = this.instrumentTracks.get(pitch);
    if (track) return track;
    track = new InstrumentTrack(
      pitch,
      () => this.document.currentJot?.defaultPaletteColorFor(pitch) ?? INSTRUMENT_FALLBACK_COLOR
    );
    this.instrumentTracks.set(pitch, track);
    return track;
  }

  /**
   * Load an audio file as a new audio track and update the status pill
   * on failure. Decoding goes through the shared `AudioContext`, so the
   * call has to occur inside a user gesture (the file-picker click
   * satisfies that). Every call appends an independent track — load N
   * files to get N tracks. Returns the new track's id, or `undefined`
   * if the load failed (so callers can e.g. default it to muted).
   */
  async loadAudioTrack(
    file: File,
    pitch?: string,
    role?: AudioTrackRole
  ): Promise<AudioTrackId | undefined> {
    return this.withLoading(`Loading ${file.name}…`, async () => {
      try {
        return await jotPlayer.loadAudioTrack(file, pitch, role);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        toastStore.showError(`Audio track load failed: ${message}`);
        return undefined;
      }
    });
  }

  clearAudioTrack(id: AudioTrackId): void {
    jotPlayer.clearAudioTrack(id);
    // Drop the removed track's mute/solo/volume so it doesn't linger
    // (ids are never reused, so the entries would be dead weight); and
    // critically so clearing the only soloed audio track doesn't leave
    // a phantom solo silencing everything else. The colour override
    // lives on the AudioTrack instance itself and is freed alongside it
    // when the player drops the track.
    this.mutedAudioTracks.delete(id);
    this.soloedAudioTracks.delete(id);
    this.audioTrackVolumes.delete(id);
  }

  /**
   * Per-track stem-split status; read by the audio-track row to render a
   * loading spinner alongside the label while a split is in flight.
   * Sparse map: a row absent from this map renders without a spinner.
   *
   * Today the actual server-side wiring (POST the track's PCM to the
   * `stems_all` / `stems_per` stage) is still deferred; the in-progress
   * state is plumbed through here so future wiring just brackets its
   * work with {@link beginAudioTrackSplit} / {@link endAudioTrackSplit}
   * and the existing per-row UI picks it up unchanged.
   */
  audioTrackSplitStatuses: Map<AudioTrackId, AudioTrackSplitStatus> = new Map();

  /** Mark an audio track as currently being split. Drives the per-row
   *  spinner; safe to call multiple times (latest call wins). */
  beginAudioTrackSplit(id: AudioTrackId, kind: 'mix' | 'pieces'): void {
    this.audioTrackSplitStatuses.set(id, { phase: 'splitting', kind });
  }

  /** Clear the splitting status for an audio track once the work has
   *  finished (success, failure, or cancellation). */
  endAudioTrackSplit(id: AudioTrackId): void {
    this.audioTrackSplitStatuses.delete(id);
  }

  /**
   * Stub: invoked from the audio-track overflow menu's "Split into
   * drums + backing" item. The actual transcriber-side wiring (POST the
   * track's PCM to a single-stage `stems_all` endpoint, then auto-load
   * the resulting drum stem + drumless backing as fresh audio tracks)
   * is deferred to a follow-up change; surface a status pill so the
   * click visibly does something in the meantime.
   */
  splitAudioTrackFromMix(id: AudioTrackId): void {
    const track = jotPlayer.audioTracks.get(id);
    const name = track ? track.filename : id;
    toastStore.showError(`Split into drums + backing on "${name}" isn't wired up yet.`);
  }

  /**
   * Stub: invoked from the audio-track overflow menu's "Split into
   * kick / snare / hi-hat / cymbals" item. See
   * {@link splitAudioTrackFromMix} for the same TODO.
   */
  splitAudioTrackDrumPieces(id: AudioTrackId): void {
    const track = jotPlayer.audioTracks.get(id);
    const name = track ? track.filename : id;
    toastStore.showError(`Split into drum pieces on "${name}" isn't wired up yet.`);
  }

  /** Drop every loaded audio track. Used when a new source (e.g. a
   * ParaDB pack) replaces the current song, otherwise the previous
   * song's tracks linger and play over the new one. */
  clearAllAudioTracks(): void {
    for (const id of Array.from(jotPlayer.audioTracks.keys())) {
      this.clearAudioTrack(id);
    }
  }

  /** Reset the per-pitch mixer (mute/solo/volume). These are keyed by
   * DSL pitch letter, not by song, so without this a mute/solo/fader
   * set on one song silently bleeds onto the next song's matching rows
   * when a new source replaces the current one. */
  resetPitchMixer(): void {
    this.mutedPitches.clear();
    this.soloedPitches.clear();
    this.pitchVolumes.clear();
  }

  setJot(jot: RenderedJot | undefined) {
    this.document.currentJot = jot;
    // External setJot calls invalidate the example pointer + any
    // previously-loaded debug provenance (provenance is per-bundle and
    // doesn't survive a wholesale jot replacement).
    this.document.currentExampleId = undefined;
    this.clearNoteProvenance();
    // Lyrics are tied to a specific recording; a new jot means they no
    // longer apply. See `src/lyrics/store.ts` for the lifecycle rationale.
    this.clearLyrics();
    // Replace the song wholesale: stop any in-flight playback so the
    // playhead, scheduled drum events, and idle cue from the previous
    // jot don't leak onto the new one.
    jotPlayer.stop();
  }

  /** Drop the debug bundle's per-note provenance + reset the toolbar
   * visibility toggle. Called from every loader that replaces the
   * current song outside the bundle path so stale debug info from a
   * previous bundle doesn't leak onto the new score. */
  private clearNoteProvenance() {
    this.noteProvenance = undefined;
    this.showFilteredOnsets = false;
  }

  /** Replace the toolbar's `Show filtered` checkbox state. */
  setShowFilteredOnsets(show: boolean) {
    this.showFilteredOnsets = show;
  }

  setLyricsSearchOpen(open: boolean) {
    this.lyricsSearchOpen = open;
  }

  setLyricsTextOpen(open: boolean) {
    this.lyricsTextOpen = open;
  }

  /** Identifies which filtered-onset popover is pinned open. The key is
   * `${pitch}:${detected_time_sec}` (rejected onsets have `tick === null`,
   * so we can't use it); `undefined` means none pinned. Hover-only popovers
   * don't go through here. */
  pinnedFilteredOnsetKey: string | undefined = undefined;

  setPinnedFilteredOnsetKey(key: string | undefined) {
    this.pinnedFilteredOnsetKey = key;
  }

  toggleFollowPlayhead() {
    this.setFollowPlayhead(!this.followPlayhead);
  }

  /**
   * Set {@link followPlayhead} and tag whether the off-state is
   * transient (set while playing) or deliberate (set while idle/paused).
   * Idempotent: redundant calls don't reshuffle the transient tag so e.g.
   * a pan during playback can't promote an already-deliberate off-state
   * into a transient one.
   */
  setFollowPlayhead(on: boolean) {
    if (on === this.followPlayhead) return;
    this.followPlayhead = on;
    this.followDisabledIsTransient = on ? false : jotPlayer.state === 'playing';
  }

  setAutoFollowOnPlay(on: boolean) {
    this.autoFollowOnPlay = on;
  }

  /**
   * Pre-indexed view onto `noteProvenance` for the per-note selection
   * label lookup. Keyed by `${pitch}:${tick}` so `NoteView` can attach
   * provenance to its note in O(1) instead of scanning the per-pitch
   * list on every render. Recomputed when `noteProvenance` changes.
   *
   * Pitch keys are canonicalised through {@link canonicalProvenancePitch}
   * so the rendered jot's pitch letter (what `NoteView` builds the
   * lookup key from) matches the provenance regardless of any synthetic
   * routing pitches the transcriber pipeline used (today: `H` for open
   * hi-hat, which `from_midi.ts` collapses back into `h:o`).
   */
  get noteProvenanceByTick(): Map<string, NoteProvenanceEntry> {
    const out = new Map<string, NoteProvenanceEntry>();
    const provenance = this.noteProvenance;
    if (!provenance) return out;
    for (const [pitch, entries] of Object.entries(provenance.per_pitch)) {
      const jotPitch = canonicalProvenancePitch(pitch);
      for (const entry of entries) {
        if (entry.tick === null || !entry.kept) continue;
        out.set(`${jotPitch}:${entry.tick}`, entry);
      }
    }
    return out;
  }

  /**
   * Per-pitch list of rejected onsets the {@link FilteredOnsetView}
   * renders. Built once from `noteProvenance` and cached via MobX so
   * the per-instrument row doesn't re-filter on every render. Out-of-range
   * entries (those that fell outside the beat-tracked region) are
   * dropped — they have no displayable bar to anchor against.
   */
  get filteredOnsetsByPitch(): Map<string, NoteProvenanceEntry[]> {
    const out = new Map<string, NoteProvenanceEntry[]>();
    const provenance = this.noteProvenance;
    if (!provenance) return out;
    for (const [pitch, entries] of Object.entries(provenance.per_pitch)) {
      const rejected = entries.filter((e) => !e.kept && !e.out_of_range);
      if (rejected.length === 0) continue;
      // Canonicalise so the consuming instrument row (which keys by the
      // jot's `note.pitch`) finds entries even when the transcriber
      // routed them through a synthetic pitch like `H` for open hat; // merge into the existing bucket rather than overwriting so the
      // closed (`h`) and open (`H` → `h`) rejected lists land together.
      const jotPitch = canonicalProvenancePitch(pitch);
      const existing = out.get(jotPitch);
      if (existing) {
        existing.push(...rejected);
      } else {
        out.set(jotPitch, rejected);
      }
    }
    return out;
  }

  setExamples(examples: readonly ExampleJot[]) {
    this.document.examples = examples;
  }

  loadExample(id: string) {
    const example = this.document.examples.find((e) => e.id === id);
    if (!example) return;
    this.document.currentJot = new RenderedJot(example.jot, this.document.viewConfig);
    this.document.currentExampleId = id;
    this.clearNoteProvenance();
    this.clearLyrics();
    jotPlayer.stop();
  }

  setDebug(enabled: boolean) {
    this.transcribeOptions.debug = enabled;
  }

  setBeatInput(input: BeatInput) {
    this.transcribeOptions.beatInput = input;
  }

  setDrumSeparator(separator: DrumSeparator) {
    this.transcribeOptions.drumSeparator = separator;
  }

  setLlmModel(model: LlmModel) {
    this.transcribeOptions.llmModel = model;
  }

  setQuantise(enabled: boolean) {
    this.transcribeOptions.quantise = enabled;
  }

  setQuantiseUseLlm(enabled: boolean) {
    this.transcribeOptions.quantiseUseLlm = enabled;
  }

  setSelectedResumeFolder(folder: string | undefined) {
    this.selectedResumeFolder = folder;
    // Clearing the folder (or picking a different one) invalidates any
    // stage selection — different folders have different `resumable_stages`,
    // so a stale pick could land on a stage missing its prerequisites.
    this.selectedResumeStage = undefined;
  }

  setSelectedResumeStage(stage: TranscribeStage | undefined) {
    this.selectedResumeStage = stage;
  }

  setTranscribeMode(mode: 'new' | 'resume') {
    this.transcribeMode = mode;
  }

  setZoom(z: number) {
    const clamped = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z));
    this.zoom = clamped;
    this.document.viewConfig.barWidth = px(BASE_BAR_WIDTH * clamped);
  }

  /**
   * Upload an audio file to the transcriber service, parse the returned
   * Drumjot DSL, and load the resulting Jot. Updates `transcribeStatus`
   * so the toolbar can show progress / errors.
   *
   * A single in-flight transcription is tracked via `transcribeController`.
   * Calling `cancelTranscribe()` aborts the underlying `fetch` request and
   * surfaces a cancelled state on the toolbar; starting a new
   * transcription while one is in flight will abort the previous one
   * first (defensive - the UI disables the button during upload, but the
   * console-level `loadDsl` API doesn't).
   */
  async transcribeAudio(file: File): Promise<void> {
    if (this.transcribeController) {
      this.transcribeController.abort();
    }
    const controller = new AbortController();
    this.transcribeController = controller;
    runInAction(() => {
      this.transcribeStatus = { phase: 'uploading', filename: file.name };
    });
    try {
      const response = await transcriber.transcribe(file, {
        debug: this.transcribeOptions.debug,
        beatInput: this.transcribeOptions.beatInput,
        drumSeparator: this.transcribeOptions.drumSeparator,
        llmModel: this.transcribeOptions.llmModel,
        quantise: this.transcribeOptions.quantise,
        quantiseUseLlm: this.transcribeOptions.quantiseUseLlm,
        signal: controller.signal,
        onProgress: (event) => this.applyProgress(file.name, event),
      });
      await this.applyTranscribeResponse(response, file.name, controller.signal);
    } catch (err) {
      this.handleTranscribeError(err, controller, 'Transcribe');
    } finally {
      if (this.transcribeController === controller) {
        this.transcribeController = undefined;
      }
      // The folder list has a new entry (the just-finished run); refresh
      // best-effort so the picker is up to date without the operator
      // having to reopen the dropdown.
      void this.refreshRecentTranscriptions();
    }
  }

  /**
   * Re-run the transcribe pipeline from a chosen stage against a
   * previously-cached debug folder. Same status / auto-load semantics as
   * {@link transcribeAudio}: progress pill while in flight, the response
   * either parses straight (DSL mode) or auto-loads the rebuilt debug
   * bundle (filter mode), and the resume controller shares
   * `transcribeController` so the Stop button cancels both flows.
   */
  async resumeTranscribe(folder: string, stage: TranscribeStage): Promise<void> {
    if (this.transcribeController) {
      this.transcribeController.abort();
    }
    const controller = new AbortController();
    this.transcribeController = controller;
    const label = `${folder} from ${stage}`;
    runInAction(() => {
      this.transcribeStatus = { phase: 'uploading', filename: label };
    });
    try {
      const response = await transcriber.resume({
        resumeFolder: folder,
        resumeStage: stage,
        beatInput: this.transcribeOptions.beatInput,
        drumSeparator: this.transcribeOptions.drumSeparator,
        llmModel: this.transcribeOptions.llmModel,
        quantise: this.transcribeOptions.quantise,
        quantiseUseLlm: this.transcribeOptions.quantiseUseLlm,
        signal: controller.signal,
        onProgress: (event) => this.applyProgress(label, event),
      });
      // The resumed run reuses the original folder, so the original
      // upload filename is the most informative pill label — fall back
      // to the resume folder name when the server doesn't know it.
      const fallbackName =
        this.recentTranscriptions.find((t) => t.folder === folder)?.original_filename ?? folder;
      await this.applyTranscribeResponse(response, fallbackName, controller.signal);
    } catch (err) {
      this.handleTranscribeError(err, controller, 'Resume');
    } finally {
      if (this.transcribeController === controller) {
        this.transcribeController = undefined;
      }
      void this.refreshRecentTranscriptions();
    }
  }

  /**
   * Shared post-transcribe handling. The backend produces a MIDI
   * prediction; we auto-load the bundled debug.zip so the score (via
   * `from_midi.ts`), audio tracks, and note provenance hydrate in one
   * go without the user having to download and re-load the zip by hand.
   */
  private async applyTranscribeResponse(
    response: Awaited<ReturnType<typeof transcriber.transcribe>>,
    fallbackName: string,
    signal: AbortSignal
  ): Promise<void> {
    const bundleUrl = stemUrl(response.debug_zip_url ?? null);
    if (!bundleUrl) {
      runInAction(() => {
        this.transcribeStatus = { phase: 'idle' };
      });
      toastStore.showError('Transcriber returned no debug bundle.');
      return;
    }
    const ok = await this.autoLoadDebugBundle(bundleUrl, fallbackName, signal);
    if (!ok) {
      // The auto-loader already surfaced the specific failure as an
      // error toast; clear the busy pill back to idle and bail.
      runInAction(() => {
        this.transcribeStatus = { phase: 'idle' };
      });
      return;
    }
    runInAction(() => {
      this.transcribeStatus = { phase: 'idle' };
    });
    toastStore.showSuccess(
      transcribeSuccessToastMessage({
        filename: fallbackName,
        tempo: response.metadata.initial_tempo,
        hasTempoChanges: response.metadata.has_tempo_changes,
        hasTimeSigChanges: response.metadata.has_time_sig_changes,
        barCount: response.metadata.bars.length,
        debugDir: response.debug_dir ?? null,
        debugZipUrl: bundleUrl,
      }),
      {
        title: response.debug_dir
          ? `Debug artifacts saved to ${response.debug_dir} (under ./debug/ on the host with the default docker-compose mount).`
          : undefined,
      }
    );
  }

  /**
   * Fetch the debug zip from `url`, parse it, and load every artifact
   * via {@link applyDebugBundle}. The predicted-MIDI score, audio
   * tracks, note provenance, and stage timings / logs all come along
   * in one round trip.
   *
   * Returns `true` on success, `false` if either the fetch or the
   * parse failed (in which case the caller surfaces an error pill).
   */
  private async autoLoadDebugBundle(
    url: string,
    fallbackName: string,
    signal: AbortSignal
  ): Promise<boolean> {
    let bundle: Awaited<ReturnType<typeof loadDebugZip>>;
    try {
      const res = await fetch(url, { signal });
      if (!res.ok) {
        throw new Error(`fetch ${url} failed (${res.status})`);
      }
      const blob = await res.blob();
      const file = new File([blob], `${fallbackName}.debug.zip`, {
        type: 'application/zip',
      });
      bundle = await loadDebugZip(file);
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') throw err;
      // eslint-disable-next-line no-console
      console.warn('Auto-load debug bundle failed:', err);
      return false;
    }
    try {
      const ok = await this.applyDebugBundle(bundle, fallbackName);
      return ok;
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') throw err;
      // eslint-disable-next-line no-console
      console.warn('Auto-load debug bundle apply failed:', err);
      return false;
    }
  }

  /** Shared transcribe / resume failure handler. Routes aborts to idle
   *  (user cancelled), everything else to the error pill. */
  /**
   * Fold one streamed `TranscribeProgress` event into the live
   * `transcribeStatus` pill so the user sees the pipeline advancing
   * through each stage. `stage` events with `phase='start'` set the
   * current stage and clear any substage label from the previous one;
   * `substage` events overwrite the in-stage detail without changing
   * the stage itself. `phase='end'` is ignored for UI purposes — the
   * pill rolls straight from one stage's `start` to the next stage's
   * `start`, which reads more clearly than briefly showing "(done)".
   */
  private applyProgress(filename: string, event: TranscribeProgress): void {
    runInAction(() => {
      const status = this.transcribeStatus;
      // If the request was aborted or already terminal (success/error)
      // before this late event fires, ignore — late progress shouldn't
      // resurrect the spinner over an idle/success/error pill.
      if (status.phase !== 'uploading') return;
      if (event.kind === 'stage' && event.phase === 'start') {
        this.transcribeStatus = {
          phase: 'uploading',
          filename,
          stage: event.stage,
        };
      } else if (event.kind === 'substage') {
        this.transcribeStatus = {
          phase: 'uploading',
          filename,
          stage: event.stage,
          substage: event.detail,
        };
      }
    });
  }

  private handleTranscribeError(err: unknown, controller: AbortController, verb: string): void {
    // AbortError surfaces as DOMException with name='AbortError' (and
    // wraps as TypeError in some runtimes when the fetch was already
    // aborted at start). Treat the user-initiated cancellation
    // distinctly from real errors so we don't show a scary red pill.
    const isAbort =
      controller.signal.aborted || (err instanceof DOMException && err.name === 'AbortError');
    if (isAbort) {
      runInAction(() => {
        this.transcribeStatus = { phase: 'idle' };
      });
      return;
    }
    const message =
      err instanceof ParseError
        ? `${verb} returned invalid DSL: ${err.message}`
        : err instanceof Error
          ? err.message
          : String(err);
    runInAction(() => {
      this.transcribeStatus = { phase: 'idle' };
    });
    toastStore.showError(`${verb} failed: ${message}`);
  }

  /**
   * Refresh the recent-transcriptions picker from the server. Failures
   * are logged but never surfaced — the picker just stays as-is, which
   * is the right behaviour when the backend is briefly unavailable.
   * Safe to call from a fire-and-forget context.
   */
  async refreshRecentTranscriptions(): Promise<void> {
    runInAction(() => {
      this.recentTranscriptionsLoading = true;
    });
    try {
      const list = await transcriber.listTranscriptions();
      runInAction(() => {
        this.recentTranscriptions = list;
        this.recentTranscriptionsLoaded = true;
        // Drop the selection if its target folder vanished server-side
        // (e.g. operator pruned the debug dir between dropdown opens).
        if (
          this.selectedResumeFolder !== undefined &&
          !list.some((s) => s.folder === this.selectedResumeFolder)
        ) {
          this.selectedResumeFolder = undefined;
          this.selectedResumeStage = undefined;
        }
      });
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn('Could not refresh recent transcriptions:', err);
    } finally {
      runInAction(() => {
        this.recentTranscriptionsLoading = false;
      });
    }
  }

  /**
   * Load a previously produced transcription's debug bundle straight from
   * the server's `/outputs/<folder>/debug.zip` without re-running any
   * pipeline stage. The bundle carries the kept-onset MIDI score, the
   * per-stem audio, and the run's logs / stage timings, so this is the
   * cheap way to reopen a finished run.
   *
   * Errors land on the shared status pill, mirroring the explicit
   * "Load debug bundle" file picker. Wrapped in `withLoading` so the
   * modal overlay reads as one continuous load even though the inner
   * `applyDebugBundle` may itself trigger nested audio-track loads.
   */
  async loadRecentTranscription(folder: string): Promise<void> {
    const url = stemUrl(`/outputs/${encodeURIComponent(folder)}/debug.zip`);
    if (!url) return;
    const summary = this.recentTranscriptions.find((s) => s.folder === folder);
    const fallbackName = summary?.original_filename ?? folder;
    return this.withLoading(`Loading ${fallbackName}…`, async () => {
      try {
        const res = await fetch(url);
        if (!res.ok) {
          throw new Error(`fetch ${url} failed (${res.status})`);
        }
        const blob = await res.blob();
        const file = new File([blob], `${fallbackName}.debug.zip`, {
          type: 'application/zip',
        });
        const bundle = await loadDebugZip(file);
        const ok = await this.applyDebugBundle(bundle, fallbackName);
        if (!ok) {
          toastStore.showError(`Could not parse score from ${fallbackName}.`);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        toastStore.showError(`Could not load ${fallbackName}: ${message}`);
      }
    });
  }

  /**
   * Abort the in-flight transcription, if any. No-op when nothing is
   * running. The next `transcribeAudio` call resumes normally.
   */
  cancelTranscribe() {
    if (!this.transcribeController) return;
    this.transcribeController.abort();
    this.transcribeController = undefined;
  }

  /**
   * Read a Drumjot DSL file from the user's machine and load it as the
   * current jot. Parse failures surface as error toasts.
   */
  async loadJotFile(file: File): Promise<void> {
    return this.withLoading(`Loading ${file.name}…`, async () => {
      let text: string;
      try {
        text = await file.text();
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        toastStore.showError(`Could not read ${file.name}: ${message}`);
        return;
      }
      try {
        const jot = parse(text);
        if (!jot.title) {
          const derivedTitle = titleFromFilename(file.name);
          if (derivedTitle) jot.title = derivedTitle;
        }
        runInAction(() => {
          this.document.currentJot = new RenderedJot(jot, this.document.viewConfig);
          this.document.currentExampleId = undefined;
          // A bare jot file has no provenance; drop whatever the
          // previous bundle put there so the selection label doesn't
          // surface stale debug data on the new song's notes.
          this.clearNoteProvenance();
          this.clearLyrics();
          jotPlayer.stop();
        });
      } catch (err) {
        const message =
          err instanceof ParseError
            ? `Could not parse ${file.name}: ${err.message}`
            : err instanceof Error
              ? err.message
              : String(err);
        toastStore.showError(message);
      }
    });
  }

  /**
   * Read a Standard MIDI File from the user's machine, convert it to a
   * Jot via {@link fromMidi}, and load it as the current jot. Like
   * {@link loadJotFile}, conversion runs entirely client-side and
   * failures surface through the shared `transcribeStatus` pill.
   */
  async loadMidiFile(file: File): Promise<void> {
    return this.withLoading(`Loading ${file.name}…`, async () => {
      let bytes: ArrayBuffer;
      try {
        bytes = await file.arrayBuffer();
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        toastStore.showError(`Could not read ${file.name}: ${message}`);
        return;
      }
      try {
        const jot = fromMidi(bytes);
        if (!jot.title) {
          const derivedTitle = titleFromFilename(file.name);
          if (derivedTitle) jot.title = derivedTitle;
        }
        runInAction(() => {
          this.document.currentJot = new RenderedJot(jot, this.document.viewConfig);
          this.document.currentExampleId = undefined;
          // Same reasoning as in loadJotFile: a bare MIDI load shouldn't
          // surface stale provenance from a previous debug bundle.
          this.clearNoteProvenance();
          this.clearLyrics();
          jotPlayer.stop();
        });
      } catch (err) {
        const message =
          err instanceof Error ? `Could not convert ${file.name}: ${err.message}` : String(err);
        toastStore.showError(message);
      }
    });
  }

  /**
   * Load a ParaDB / Paradiddle map pack (`.zip`): convert its `.rlrr`
   * chart to a Jot and auto-load its audio tracks so the pack is
   * immediately play-along ready. Audio decoding shares the
   * `AudioContext`, so this must run inside the file-picker's user
   * gesture (the same constraint as {@link loadAudioTrack}). Errors surface
   * through the shared status pill, matching {@link loadJotFile}.
   */
  async loadParadbMap(file: File): Promise<void> {
    return this.withLoading(`Loading ${file.name}…`, async () => {
      let map: Awaited<ReturnType<typeof loadParadbZip>>;
      try {
        map = await loadParadbZip(file);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        toastStore.showError(`Could not load ${file.name}: ${message}`);
        return;
      }

      const jot = map.jot;
      if (!jot.title) {
        const derivedTitle = titleFromFilename(file.name);
        if (derivedTitle) jot.title = derivedTitle;
      }
      runInAction(() => {
        // Replace the song wholesale: drop any audio tracks from a
        // previously loaded map/transcription so they don't play over
        // the new pack's tracks, and reset the per-pitch mixer so an
        // old song's mute/solo/faders don't bleed onto the new rows.
        this.clearAllAudioTracks();
        this.resetPitchMixer();
        this.document.currentJot = new RenderedJot(jot, this.document.viewConfig);
        this.document.currentExampleId = undefined;
        this.clearNoteProvenance();
        this.clearLyrics();
        jotPlayer.stop();
      });

      // Audio tracks are best-effort: a chart with the score loaded is
      // still useful even if one is absent or fails to decode.
      // loadAudioTrack already reports its own failures on the status pill.
      // Drum tracks load too but start muted; you're playing the drums,
      // so the backing music should be the only thing you hear by default.
      //
      // Lyrics alignment is deliberately NOT auto-fired here: vocals
      // separation (BS-Roformer) eats a chunk of GPU time, and most
      // ParaDB loads don't need lyrics. The user kicks it off explicitly
      // via the Lyrics menu (or the LRCLIB search modal) when they want
      // synced lyrics.
      //
      // Decode in parallel; `decodeAudioData` runs on browser-side
      // codec threads so concurrent calls overlap, cutting the song +
      // drums decode wall time roughly in half. Mirrors the debug-
      // bundle loader's approach.
      const resolved = await Promise.all(
        map.audioTracks.map(async (track) => {
          const id = await this.loadAudioTrack(track.file, undefined, track.role);
          return { id, defaultMuted: track.defaultMuted };
        })
      );
      runInAction(() => {
        for (const { id, defaultMuted } of resolved) {
          if (id && defaultMuted) this.mutedAudioTracks.add(id);
        }
      });
    });
  }

  /**
   * Score a ParaDB `.zip` map against its own audio via the transcriber's
   * `POST /score`, surfacing the result as a toast (full result to the
   * console). A development test harness for the corpus-filtering scorer
   * (`transcriber/app/scoring`); unlike {@link loadParadbMap} it does NOT
   * touch the current score, it only reports a quality number.
   */
  async scoreParadbMap(file: File): Promise<void> {
    return this.withLoading(`Scoring ${file.name}…`, async () => {
      try {
        const result = await transcriber.scoreParadb(file);
        const offsetMs = (result.offset_sec * 1000).toFixed(0);
        toastStore.showSuccess(
          `${file.name}: ${result.score_corrected}/100 corrected ` +
            `(raw ${result.score}) · offset ${offsetMs} ms · ` +
            `tempo ${result.tempo_ratio.toFixed(3)}× · ${result.audio_reference}`,
          { title: 'See the browser console for the full per-lane breakdown.' }
        );
        // eslint-disable-next-line no-console
        console.log('Alignment score', file.name, result);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        toastStore.showError(`Could not score ${file.name}: ${message}`);
      }
    });
  }

  /**
   * Load a transcriber debug `.zip` bundle: parse the embedded
   * `final.jot`, load every audio track in the manifest's `mapping`, and
   * stash the manifest (stage timings + log stream) on
   * {@link lastDebugBundle} so the {@link DebugPanel} can show it.
   *
   * Behaves like {@link loadParadbMap}: replaces the current song
   * wholesale (drops previously loaded audio tracks, resets the pitch
   * mixer), runs entirely client-side, and surfaces errors on the
   * shared status pill.
   *
   * The `no_drums` entry (drumless backing audio) is auto-defaulted to
   * unmuted; the per-pitch stems are defaulted to muted, mirroring the
   * "drum tracks are reference-only, you're playing them" convention
   * from the ParaDB loader — the drums you hear should be the smplr-
   * scheduled ones from the score, not a re-decoded stem layered on top.
   */
  async loadDebugBundleFile(file: File): Promise<void> {
    return this.withLoading(`Loading ${file.name}…`, async () => {
      let bundle: Awaited<ReturnType<typeof loadDebugZip>>;
      try {
        bundle = await loadDebugZip(file);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        toastStore.showError(`Could not load ${file.name}: ${message}`);
        return;
      }
      const ok = await this.applyDebugBundle(bundle, file.name);
      if (!ok) {
        toastStore.showError(`Could not parse score from ${file.name}.`);
      }
    });
  }

  /**
   * Apply an already-parsed {@link DebugBundle} to the store: replace the
   * current song with the bundle's score (DSL → MIDI fallback), load each
   * audio track, pair stems with their instrument rows, and mount the
   * manifest on the DebugPanel.
   *
   * Returns `true` if a score was loaded, `false` if neither `final.jot`
   * nor `prediction.mid` could be turned into a jot (the audio tracks
   * still load either way so the operator can at least listen).
   *
   * Status-pill management is left to the caller — `loadDebugBundleFile`
   * sets it to idle/error on completion, while `transcribeAudio` keeps
   * its success pill visible after the auto-load.
   */
  private async applyDebugBundle(
    bundle: Awaited<ReturnType<typeof loadDebugZip>>,
    fallbackName: string
  ): Promise<boolean> {
    runInAction(() => {
      this.clearAllAudioTracks();
      this.resetPitchMixer();
      this.clearLyrics();
      this.lastDebugBundle = bundle.manifest;
      // Replace (or clear) the per-note debug provenance whenever a
      // new bundle loads. Older bundles may not carry one (e.g. a
      // hand-built or legacy zip); the absent-case clears the previous
      // bundle's provenance so it doesn't leak onto the new score.
      this.noteProvenance = bundle.noteProvenance ?? undefined;
      // Reset the visibility toggle so a freshly loaded bundle reads
      // as just "the score"; operator opts into the ghost overlays.
      this.showFilteredOnsets = false;
      // Bundles come from the transcribe pipeline, which routinely
      // emits triplet subdivisions; the 48ths grid is the LCM of 16ths
      // + triplets so it visualises both. Override the store-wide 16ths
      // default for this load specifically.
      this.settings.gridLines = {
        mainBeat: true,
        subBeat16: false,
        subBeatQuarterTriplet: false,
        subBeatTriplet: false,
        subBeat48: true,
      };
    });

    // The bundle's score is the `prediction.mid` produced by the
    // transcribe stage; `src/midi/from_midi.ts` converts it to a Jot.
    let scoreLoaded = false;
    if (bundle.predictionMidi) {
      try {
        const jot = fromMidi(bundle.predictionMidi);
        if (!jot.title) {
          const derivedTitle = titleFromFilename(fallbackName);
          if (derivedTitle) jot.title = derivedTitle;
        }
        // The beats stage's `align_beats_to_*` shift is already baked
        // into `prediction.mid`'s tick grid (see `compute_bar_tick_grid`
        // in `transcriber/app/pipeline/onsets_midi.py`), so the loaded
        // MIDI is at the aligned positions and the Beat control starts
        // at 0. The applied alignment is still visible per-note in the
        // selection popup as the "Beat alignment" row sourced from
        // `noteProvenance.beat_alignment_offset_sec`.
        runInAction(() => {
          const rendered = new RenderedJot(jot, this.document.viewConfig);
          this.document.currentJot = rendered;
          this.document.currentExampleId = undefined;
          jotPlayer.stop();
        });
        scoreLoaded = true;
      } catch (err) {
        const message =
          err instanceof Error ? `Could not convert prediction.mid: ${err.message}` : String(err);
        toastStore.showError(message);
      }
    }

    // Decode every audio track in parallel, `decodeAudioData` runs on
    // browser-side codec threads, so concurrent calls overlap well and
    // turn what used to be a one-by-one wait into a single combined
    // wait. `Promise.all` preserves input order so the resolved array
    // still matches `bundle.audioTracks` (which is already in manifest
    // order; `no_drums` first, then pitch letters), keeping the
    // post-load pair-with-instrument-row logic stable. The bundle
    // loader dedupes by filename, so each `track` here represents one
    // unique file; we bind every key in `track.keys` to the resulting
    // `AudioTrackId` so a shared stem (e.g. `stem_c.mp3` serving both
    // crash and ride after the cymbal split) is loaded once and looked
    // up under either key.
    const resolved = await Promise.all(
      bundle.audioTracks.map(async (track) => {
        // The audio-row's `pitch` (used by the mixer for waveform
        // tinting) takes the first non-`no_drums` key; for a stem
        // shared across pitches, this picks the first-mentioned pitch
        // in the manifest, which is good enough since the tint is
        // cosmetic and both siblings live in the same colour family.
        const primaryKey = track.keys.find((k) => k !== NO_DRUMS_KEY);
        // Role classification: any track whose only key is `no_drums`
        // is the Demucs drumless mix; everything else came from the
        // per-pitch split (a key shared between multiple pitches still
        // counts as a single drum piece for menu purposes).
        const role: AudioTrackRole = primaryKey === undefined ? 'no-drums' : 'drum-piece';
        const id = await this.loadAudioTrack(track.file, primaryKey, role);
        return { keys: track.keys, id };
      })
    );
    const loadedByKey = new Map<string, AudioTrackId>();
    const toMute: AudioTrackId[] = [];
    for (const { keys, id } of resolved) {
      if (!id) continue;
      let muteThis = false;
      for (const key of keys) {
        loadedByKey.set(key, id);
        // Mute the per-pitch stems by default so the (audible) drums
        // come from the smplr score scheduler; the drumless backing
        // stays unmuted. Multiple keys → still one mute, since they
        // share the same `id`.
        if (key !== NO_DRUMS_KEY) muteThis = true;
      }
      if (muteThis) toMute.push(id);
    }

    // Batch the mute updates and the reorder into a single observable
    // mutation so the mixer renders once at the end instead of once
    // per loaded track.
    runInAction(() => {
      for (const id of toMute) this.mutedAudioTracks.add(id);
      this.applyDebugBundleTrackOrder(loadedByKey);
    });

    return scoreLoaded;
  }

  /**
   * Re-order the mixer after a debug bundle is loaded so each per-pitch
   * audio stem sits immediately above its instrument row, with any
   * unmatched audio (e.g. the `no_drums` backing) at the top. The
   * ordering itself is the pure {@link buildDebugBundleTrackOrder} (see
   * there for the full layout + the shared-stem dedupe); this just feeds
   * it the current jot's pitches.
   *
   * The `syncTrackOrder` reaction won't reshuffle the result, it only
   * ever drops stale entries and appends new ones, both of which are
   * no-ops right after a fresh bundle load.
   */
  private applyDebugBundleTrackOrder(loadedByKey: ReadonlyMap<string, AudioTrackId>): void {
    const pitches = collectJotPitches(this.document.currentJot);
    this.trackOrder = buildDebugBundleTrackOrder(pitches, loadedByKey);
  }

  /** Toggle the {@link DebugPanel}'s open state without forgetting the bundle. */
  toggleDebugPanel(): void {
    this.debugPanelOpen = !this.debugPanelOpen;
  }

  /** Resize the {@link DebugPanel}. Clamped so it can't shrink past the
   * header or grow past the viewport (with headroom for the toolbar). */
  setDebugPanelHeight(px: number): void {
    const max = Math.max(120, this._viewportHeight - 160);
    this.debugPanelHeight = Math.min(max, Math.max(80, px));
  }

  /** Resize the sticky gutter column. Clamped to a sensible range so a
   * runaway drag can't collapse the controls or push the bars row off
   * screen. */
  setGutterWidth(px: number): void {
    if (!Number.isFinite(px)) return;
    this.gutterWidth = Math.min(MAX_GUTTER_WIDTH, Math.max(MIN_GUTTER_WIDTH, px));
  }

  /**
   * Cache the score viewport's pixel dimensions. Fed by a ResizeObserver
   * on `.jotContainer` in JotView. Re-clamps `scrollX` / `scrollY` so a
   * resize that shrinks the viewport (or grows it past the content's
   * extent) doesn't leave the scroll parked off the new end.
   */
  setViewportSize(width: number, height: number): void {
    this._viewportWidth = width;
    this._viewportHeight = height;
    this.scrollX = this.clampScrollX(this.scrollX);
    this.scrollY = this.clampScrollY(this.scrollY);
  }

  /**
   * Cache the scroll-content's pixel dimensions (the inner
   * `.scrollViewport` wrapper's offset width / height, fed by a
   * ResizeObserver in JotView). Re-clamps as above: zooming out shrinks
   * the content and the user might land past the new max.
   */
  setContentSize(width: number, height: number): void {
    this._contentWidth = width;
    this._contentHeight = height;
    this.scrollX = this.clampScrollX(this.scrollX);
    this.scrollY = this.clampScrollY(this.scrollY);
  }

  setScrollX(x: number): void {
    this.scrollX = this.clampScrollX(snapToDevicePx(x));
  }

  setScrollY(y: number): void {
    this.scrollY = this.clampScrollY(snapToDevicePx(y));
  }

  setScrollBy(dx: number, dy: number): void {
    this.scrollX = this.clampScrollX(snapToDevicePx(this.scrollX + dx));
    this.scrollY = this.clampScrollY(snapToDevicePx(this.scrollY + dy));
  }

  /**
   * Reset the horizontal scroll to the score's start. Used on Stop
   * transitions so a fresh Play shows the lead-in. Deliberately does
   * NOT touch `scrollY`: the user might have scrolled down to see
   * lower tracks; their vertical view shouldn't snap back just because
   * they pressed Stop, only the horizontal playhead-tracking axis.
   */
  resetScrollX(): void {
    this.scrollX = 0;
  }

  /**
   * Clamp a tentative target to `[0, contentSize - viewportSize]`. Public
   * so callers (zoom anchor) can sequence "compute target → setScrollX"
   * deterministically; setScrollX itself goes through this.
   */
  clampScrollX(x: number): number {
    const max = Math.max(0, this._contentWidth - this._viewportWidth);
    if (!(x > 0)) return 0;
    if (x > max) return max;
    return x;
  }

  clampScrollY(y: number): number {
    const max = Math.max(0, this._contentHeight - this._viewportHeight);
    if (!(y > 0)) return 0;
    if (y > max) return max;
    return y;
  }

  /**
   * Quarter-note-beat window currently on screen (plus a one-viewport
   * buffer on each side so a fast scroll doesn't outrun the rendered
   * region before the next frame). Drives horizontal score
   * virtualisation: each row renders only the bars / ticks / words whose
   * beat span intersects this range, so a 5-minute song keeps a bounded
   * DOM + CSSOM regardless of length.
   *
   * Derived purely from `JotViewStore` observables (`scrollX`,
   * `_viewportWidth`) and the jot's zoom-driven `pxPerBeat`, so it
   * re-derives on scroll / zoom / resize and nothing else, no DOM
   * layout reads (AGENTS.md §5.9). The bars row is positioned in
   * score-px where beat 0 sits at x=0 and the virtual scroll window is
   * `[scrollX, scrollX + viewportWidth]` (the same convention the
   * waveform-chunk visibility check uses); the sticky gutter's width is
   * comfortably absorbed by the buffer, so it's omitted here.
   *
   * `null` means "windowing disabled, render everything": returned
   * before the first ResizeObserver tick (viewport size unknown) and
   * whenever `pxPerBeat` is degenerate, so initial paint / non-laid-out
   * test environments still render the full score.
   */
  get visibleBeatRange(): { startBeat: number; endBeat: number } | null {
    const ppb = this.document.currentJot?.pxPerBeat ?? 0;
    const vw = this._viewportWidth;
    if (ppb <= 0 || vw <= 0) return null;
    const buffer = vw;
    return {
      startBeat: (this.scrollX - buffer) / ppb,
      endBeat: (this.scrollX + vw + buffer) / ppb,
    };
  }

  /**
   * Read a synced-lyrics file (LRC, or a text file in LRC format) from
   * disk and push it into the session lyrics store. Empty / unparseable
   * inputs surface a failure message on the shared status pill instead
   * of silently doing nothing.
   */
  async loadLyricsFile(file: File): Promise<void> {
    return this.withLoading(`Loading ${file.name}…`, async () => {
      let text: string;
      try {
        text = await file.text();
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        toastStore.showError(`Could not read ${file.name}: ${message}`);
        return;
      }
      // Enhanced-LRC aware: word-tagged files load as word-aligned
      // tracks (with per-word durations), plain line-level LRC parses
      // exactly as before. A leading `[offset:±ms]` restores the saved
      // offset nudge.
      const { lines, offsetSec } = parseEnhancedLrc(text);
      if (lines.length === 0) {
        toastStore.showError(`No synced lyrics found in ${file.name}.`);
        return;
      }
      runInAction(() => {
        const id = lyricsStore.add(lines, {
          source: 'file',
          sourceLabel: `File · ${file.name}`,
        });
        if (offsetSec !== 0) lyricsStore.setOffsetSec(id, offsetSec);
      });
    });
  }

  /**
   * Apply a synced-lyrics result the LRCLIB modal picked. The modal
   * parses the candidate's LRC and hands us the lines + the picked
   * match's identifying fields. Source label always reads `LRCLIB · …`;
   * word-level upgrades replace the lines in-place but keep the source.
   *
   * When `opts.wordLevel` is true, the LRCLIB lines load immediately
   * (so the row shows up right away with line-level timing) and a
   * background CTC forced-alignment job runs against an auto-picked
   * audio track. Success replaces the lines with word-timed
   * versions; failure leaves the line-level lines in place and surfaces
   * the error on the status pill.
   */
  applyLrclibResult(
    lines: readonly LyricLine[],
    match: { trackName: string; artistName: string },
    opts: { wordLevel: boolean } = { wordLevel: false }
  ): void {
    const trackId = lyricsStore.add(lines, {
      source: 'lrclib',
      sourceLabel: `LRCLIB · ${match.trackName} - ${match.artistName}`,
    });
    toastStore.showSuccess(`Loaded ${match.trackName} by ${match.artistName} from LRCLIB`, {
      testId: 'lyrics-search-loaded',
    });
    if (opts.wordLevel) {
      void this.runWordLevelAlignmentForLrclib(trackId, lines, match);
    }
  }

  /**
   * Auto-pick an audio track and run CTC forced-alignment against it
   * using the LRCLIB lines as authoritative text. The picked track
   * + kind drive whether the backend's vocals separator runs first
   * (`mix` = run separation; `vocals` = skip it).
   *
   * No-op (with a status pill error) when no audio tracks are loaded;
   * the modal disables the word-level checkbox in that case so this is
   * a programming-error safety net rather than a user-reachable path.
   */
  private async runWordLevelAlignmentForLrclib(
    targetTrackId: LyricsTrackId,
    lines: readonly LyricLine[],
    match: { trackName: string; artistName: string }
  ): Promise<void> {
    const pick = this.pickAudioTrackForAlignment();
    if (!pick) {
      toastStore.showError('Word-level alignment needs an audio track; load one first.');
      return;
    }
    const track = jotPlayer.audioTracks.get(pick.id);
    if (!track) return;
    const file = new File([track.sourceBlob], track.filename, { type: track.sourceBlob.type });
    const label = `${match.trackName} - ${match.artistName}`;
    await this.alignLyricsForced(
      targetTrackId,
      {
        kind: pick.kind,
        file,
        realign: {
          lines: lines.map((l) => ({ startSec: l.startSec, text: l.text })),
        },
      },
      label,
      {
        source: 'lrclib',
        sourceLabel: `LRCLIB · ${match.trackName} - ${match.artistName}`,
      }
    );
  }

  /**
   * Pick the loaded audio track most likely to carry vocals + the
   * separator mode to feed it to the CTC aligner with. Heuristic priority:
   *
   *   1. Any track whose filename looks like vocals → `vocals` (skip
   *      separation).
   *   2. First non-drums track (role ≠ `drums` / `drum-piece`) → `mix`
   *      (separator extracts vocals first).
   *   3. Fallback: first track regardless → `mix` (even a drums-only
   *      track is worth trying once over erroring out; the separator
   *      may still find faint vocal bleed; if not the user gets a
   *      "no speech found" message and can load a better track).
   *
   * Returns undefined only when no audio tracks are loaded.
   */
  private pickAudioTrackForAlignment(): { id: AudioTrackId; kind: 'mix' | 'vocals' } | undefined {
    const tracks = Array.from(jotPlayer.audioTracks.values());
    if (tracks.length === 0) return undefined;
    for (const t of tracks) {
      if (nameLooksLikeVocals(t.filename)) {
        return { id: t.id, kind: 'vocals' };
      }
    }
    for (const t of tracks) {
      if (t.role !== 'drums' && t.role !== 'drum-piece') {
        return { id: t.id, kind: 'mix' };
      }
    }
    return { id: tracks[0].id, kind: 'mix' };
  }

  /**
   * Push pasted / typed plain-text lyrics into the session lyrics store.
   *
   * Plain text has no timestamps, so we synthesise them by spreading
   * the lines evenly across the song's known duration (longest loaded
   * audio track > rendered jot's timeline > 60 s fallback). The spread
   * serves two ends: lines are immediately visible across the row
   * (otherwise they'd all stack at beat 0 and collapse to an invisible
   * point), and `opts.wordLevel`'s re-time path gets non-degenerate
   * starting estimates for wav2vec2 (whose search window for each line
   * is `[startSec, nextLine.startSec]` - all-zero starts collapse every
   * segment to the same audio window).
   *
   * Strips section markers like `[Chorus]` / `[Verse 1]` (any line whose
   * trimmed content is wrapped in a single pair of brackets) because
   * pastes from Genius and similar lyric sites carry them and they
   * aren't sung. Also strips parenthetical asides and music glyphs via
   * {@link stripLyricNoise}, so echo lines like `(I'm screaming…)` and
   * interlude markers like `♪ ♪ ♪` drop out. Returns the number of
   * lines actually loaded so the caller can surface a "nothing usable
   * in this paste" error.
   *
   * When `opts.wordLevel` is true and an audio track is loaded, fires
   * the same background CTC forced-alignment used by the LRCLIB
   * word-level path: the spread lines land immediately, then word-
   * timed versions replace them on success.
   */
  applyPlainTextLyrics(text: string, opts: { wordLevel?: boolean } = {}): number {
    const cleaned: string[] = [];
    for (const raw of text.split(/\r?\n/)) {
      const trimmed = raw.trim();
      if (trimmed.length === 0) continue;
      if (/^\[[^\]]*\]$/.test(trimmed)) continue;
      const stripped = stripLyricNoise(trimmed);
      if (stripped.length === 0) continue;
      cleaned.push(stripped);
    }
    if (cleaned.length === 0) return 0;
    const spreadSec = this.computeLyricsSpreadSec();
    // Linear `i / N` spread (not `i / (N-1)`) leaves the final 1/N of
    // the song as buffer past the last line, which is closer to how
    // real lyrics sit relative to a recording's tail (intro & outro
    // are often instrumental). First line lands at 0.
    const lines: LyricLine[] = cleaned.map((t, i) => ({
      startSec: (spreadSec * i) / cleaned.length,
      text: t,
    }));
    const trackId = lyricsStore.add(lines, {
      source: 'plaintext',
      sourceLabel: 'Plain text',
    });
    if (opts.wordLevel) {
      void this.runWordLevelAlignmentForPlainText(trackId, lines);
    }
    return lines.length;
  }

  /** Best-effort duration in seconds across which to spread untimed
   *  lyric lines. Prefers loaded audio (matches the realign domain),
   *  then the rendered jot's timeline, then a small default. */
  private computeLyricsSpreadSec(): number {
    let longestAudio = 0;
    for (const t of jotPlayer.audioTracks.values()) {
      if (t.durationSec > longestAudio) longestAudio = t.durationSec;
    }
    if (longestAudio > 0) return longestAudio;
    if (this.document.currentJot) {
      const tl = buildTimeline(this.document.currentJot);
      if (tl.totalDurationSec > 0) return tl.totalDurationSec;
    }
    return 60;
  }

  /** Mirror of {@link runWordLevelAlignmentForLrclib} for the plain-
   *  text source. Picks an audio track and runs CTC forced alignment
   *  using the spread lines as authoritative text; on success the
   *  lines are replaced with word-timed versions while the source
   *  label stays "Plain text". */
  private async runWordLevelAlignmentForPlainText(
    targetTrackId: LyricsTrackId,
    lines: readonly LyricLine[]
  ): Promise<void> {
    const pick = this.pickAudioTrackForAlignment();
    if (!pick) {
      toastStore.showError('Word-level alignment needs an audio track; load one first.');
      return;
    }
    const track = jotPlayer.audioTracks.get(pick.id);
    if (!track) return;
    const file = new File([track.sourceBlob], track.filename, { type: track.sourceBlob.type });
    await this.alignLyricsForced(
      targetTrackId,
      {
        kind: pick.kind,
        file,
        realign: {
          lines: lines.map((l) => ({ startSec: l.startSec, text: l.text })),
        },
      },
      'Plain text',
      { source: 'plaintext', sourceLabel: 'Plain text' }
    );
  }

  /**
   * Drop every lyrics row and abort every in-flight align. Called by
   * wholesale-song-reload paths (`loadJotFile`, `loadParadbMap`,
   * `applyDebugBundle`) so stale lyrics + still-running aligns can't
   * leak onto the new song.
   */
  clearLyrics(): void {
    lyricsStore.clear();
    this.cancelAllLyricsAlign();
  }

  /**
   * Remove a single lyrics row, aborting that row's in-flight align if
   * any. Routed through here (rather than `lyricsStore.remove(id)`
   * directly) so the lyrics store stays unaware of the per-track align
   * state held on `JotViewStore`.
   */
  removeLyricsTrack(id: LyricsTrackId): void {
    const ctrl = this.lyricsAlignControllers.get(id);
    if (ctrl) {
      ctrl.abort();
      this.lyricsAlignControllers.delete(id);
    }
    runInAction(() => {
      this.lyricsAlignStatuses.delete(id);
    });
    lyricsStore.remove(id);
  }

  /**
   * Per-track Whisper alignment state. Each row aligning at the same
   * time has its own AbortController and status entry; absence of an
   * entry means that row is idle. Per-track concurrency lets users
   * align a duet's two vocal lines without one cancelling the other,
   * and lets the per-row spinner show *which* row is currently working
   * (the toolbar busy pill, in contrast, just shows a generic "any
   * aligning" boolean).
   *
   * The controller map is non-observable; statuses are observable so
   * `lyricsAnyAligning` and the per-row spinner re-render on change.
   */
  lyricsAlignControllers: Map<LyricsTrackId, AbortController> = new Map();
  lyricsAlignStatuses: Map<LyricsTrackId, LyricsAlignStatus> = new Map();

  /** Aggregate lyrics-alignment state across all rows, for the toolbar
   *  busy pill (which doesn't display *which* row; the per-row spinner
   *  does). `aligning` wins over `queued` so that once any row owns the
   *  GPU the pill reads as actively working; `queued` shows only while
   *  every in-flight row is still waiting its turn. The backend
   *  serialises GPU work, so at most one row is `aligning` at a time. */
  get lyricsAlignBusyPhase(): 'idle' | 'queued' | 'aligning' {
    let anyQueued = false;
    for (const s of this.lyricsAlignStatuses.values()) {
      if (s.phase === 'aligning') return 'aligning';
      if (s.phase === 'queued') anyQueued = true;
    }
    return anyQueued ? 'queued' : 'idle';
  }

  /**
   * Run CTC forced-alignment against the given input source and
   * upgrade `targetTrackId`'s lines on success. The caller supplies the
   * {@link LyricsSource} and source label to re-apply, so the row's
   * gutter label doesn't get rewritten to a hardcoded LRCLIB string
   * when the plain-text flow runs through here.
   *
   * Per-target concurrency: a second align on the SAME track aborts the
   * first (the newer pick wins). Aligns on DIFFERENT tracks run
   * concurrently from this layer's perspective; the backend serialises
   * them GPU-wise.
   */
  private async alignLyricsForced(
    targetTrackId: LyricsTrackId,
    req: AlignLyricsRequest,
    label: string,
    opts: { source: LyricsSource; sourceLabel: string }
  ): Promise<void> {
    const existing = this.lyricsAlignControllers.get(targetTrackId);
    if (existing) {
      existing.abort();
      this.lyricsAlignControllers.delete(targetTrackId);
    }
    const controller = new AbortController();
    this.lyricsAlignControllers.set(targetTrackId, controller);
    runInAction(() => {
      this.lyricsAlignStatuses.set(targetTrackId, { phase: 'aligning', detail: label });
    });
    let lines: LyricLine[];
    try {
      lines = await alignLyricsForced(req, {
        signal: controller.signal,
        onProgress: (event) => {
          // The stream emits `queued` while waiting behind another GPU
          // job, then `running` once alignment starts. Flip the per-row
          // status so the spinner/pill read "Queued…" vs "Aligning…".
          // Guard against a newer align (or a clear) that raced in while
          // we were waiting: only this controller may touch the status.
          if (this.lyricsAlignControllers.get(targetTrackId) !== controller) {
            return;
          }
          runInAction(() => {
            this.lyricsAlignStatuses.set(targetTrackId, {
              phase: event.kind === 'queued' ? 'queued' : 'aligning',
              detail: label,
            });
          });
        },
      });
    } catch (err) {
      if (controller.signal.aborted) {
        // A newer align on the same track (or a wholesale jot replace)
        // cancelled us; don't overwrite their state. The newer caller
        // already set either its own aligning status or cleared back to
        // idle for this track.
        return;
      }
      const message = err instanceof Error ? err.message : String(err);
      runInAction(() => {
        this.lyricsAlignStatuses.delete(targetTrackId);
      });
      toastStore.showError(`Lyrics align failed: ${message}`);
      return;
    } finally {
      if (this.lyricsAlignControllers.get(targetTrackId) === controller) {
        this.lyricsAlignControllers.delete(targetTrackId);
      }
    }
    if (lines.length === 0) {
      runInAction(() => {
        this.lyricsAlignStatuses.delete(targetTrackId);
      });
      toastStore.showError(`No lyrics were aligned (the aligner found no speech in ${label}).`);
      return;
    }
    runInAction(() => {
      lyricsStore.replace(targetTrackId, lines, {
        source: opts.source,
        sourceLabel: opts.sourceLabel,
      });
      this.lyricsAlignStatuses.delete(targetTrackId);
    });
  }

  /**
   * Abort every in-flight Whisper alignment and clear the statuses.
   * Called by wholesale-song-reload paths so slow aligns from the
   * previous song can't land lines onto the new one.
   */
  private cancelAllLyricsAlign(): void {
    for (const ctrl of this.lyricsAlignControllers.values()) {
      ctrl.abort();
    }
    this.lyricsAlignControllers.clear();
    runInAction(() => {
      this.lyricsAlignStatuses.clear();
    });
  }

  async playCurrent(): Promise<void> {
    const jot = this.document.currentJot;
    if (!jot) return;
    // Pass the laid-out RenderedJot (not its source) so the player's
    // timeline reads live bar widths — the playhead then tracks correctly
    // across zoom changes.
    await jotPlayer.play(jot);
  }

  stopPlayback(): void {
    jotPlayer.stop();
  }

  /** Current beat-grid offset (quarter-note beats) on the loaded jot. */
  get drumOffsetBeats(): number {
    return this.document.currentJot?.drumOffsetBeats ?? 0;
  }

  /**
   * Slide every drum note across the bar grid by `beats` quarter-note
   * beats to realign a consistently mis-detected groove (see
   * {@link RenderedJot.drumOffsetBeats}). Reflows the score reactively and
   * reschedules in-flight playback so the change is heard immediately.
   */
  setDrumOffset(beats: number): void {
    const jot = this.document.currentJot;
    if (!jot) return;
    // Slider semantics: the user is re-labeling note positions on the
    // notational grid (e.g. "this hit is on 1/48, not 3/48"), not
    // re-timing the drums against the audio recording. So when the
    // shift moves every note by Δ beats in jot time, compensate the
    // audio offset by the same magnitude in the opposite direction so
    // the audio-track waveform tracks the noteheads instead of sliding
    // out from under them. Uses the dominant bpm (the tempo the song
    // spends the most audio time at, excluding lead-in bars) rather
    // than globalMetadata.bpm, because transcribed bundles store a
    // back-solved lead-in tempo as the first setTempo event and that
    // value can be very different from the song's actual rate. Per-bar
    // tempo variation still leaves a few-ms-per-note residual; same
    // caveat as the Drum-offset row in the debug panel.
    const deltaBeats = beats - jot.drumOffsetBeats;
    if (Math.abs(deltaBeats) > 1e-12) {
      const { dominantBpm } = pickDominantBpmAndTime(jot);
      const bpm = dominantBpm ?? 120;
      const deltaSec = (deltaBeats * 60) / bpm;
      jotPlayer.setDrumsT0Sec(jotPlayer.drumsT0Sec - deltaSec);
    }
    jot.setDrumOffset(beats);
    jotPlayer.refreshDrumSchedule(jot);
  }

  /**
   * Click-to-seek. `x` is a pixel offset within the bars row — the same
   * coordinate space `bar.x` / the playhead use (origin at the left
   * edge of the bars region, after the gutter). While playing this
   * scrubs live; while idle it parks the playhead and the next Play
   * starts from there. Uses the live timeline when one exists so a
   * mid-playback scrub reads the exact bars being played.
   */
  seekToX(x: number): void {
    const jot = this.document.currentJot;
    if (!jot) return;
    const timeline = jotPlayer.timeline.bars.length > 0 ? jotPlayer.timeline : buildTimeline(jot);
    jotPlayer.seek(jot, xToTime(timeline, x));
  }

  /**
   * Single transport action shared by the spacebar shortcut and the
   * toolbar's play/pause button:
   *   idle    -> play the current jot from the start
   *   playing -> pause (freezes the clock, playhead stays put)
   *   paused  -> resume from the same spot
   * `loading` is intentionally a no-op so a double-press during the
   * one-time sample fetch can't stack two `play()` calls.
   */
  async togglePlayPause(): Promise<void> {
    switch (jotPlayer.state) {
      case 'idle':
        this.maybeReenableFollowOnPlay();
        await this.playCurrent();
        break;
      case 'playing':
        await jotPlayer.pause();
        break;
      case 'paused':
        this.maybeReenableFollowOnPlay();
        await jotPlayer.resume();
        break;
    }
  }

  /**
   * Restore {@link followPlayhead} on the idle/paused → playing
   * transition when the off-state was set during the previous playback
   * session (pan, minimap drag, or follow-button toggle while playing).
   * No-op when {@link autoFollowOnPlay} is off, when follow is already
   * on, or when the user deliberately disabled it while idle/paused.
   */
  private maybeReenableFollowOnPlay() {
    if (!this.autoFollowOnPlay) return;
    if (this.followPlayhead) return;
    if (!this.followDisabledIsTransient) return;
    this.setFollowPlayhead(true);
  }
}
