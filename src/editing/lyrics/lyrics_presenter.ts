import { makeAutoObservable, runInAction } from 'mobx';
import { AlignLyricsRequest, alignLyricsForced, nameLooksLikeVocals } from 'src/lyrics/forced_align';
import { LyricLine, stripLyricNoise } from 'src/lyrics/lrc';
import { LyricsSource, LyricsTrackId, lyricsStore } from 'src/lyrics/store';
import { AudioTrackId } from 'src/editing/playback/audio_tracks';
import { jotPlayer } from 'src/editing/playback/player';
import { toastStore } from '../../ui/toasts/toasts';
import { isBackendUnreachable } from 'src/net/backend_fetch';
import { JotEditorStore } from '../jot_editor_store';
import { LyricsAlignStore } from './lyrics_align_store';
import type { Resettable } from '../session_reset';

/**
 * Orchestration over {@link LyricsAlignStore}: the lyrics-load flows
 * (LRCLIB picks, pasted plain text), per-track CTC forced-alignment, and
 * the modal-visibility flags. The sole owner of the per-track align
 * AbortControllers. Reads {@link JotEditorStore} only to size the
 * plain-text spread against the current jot's timeline.
 */
export class LyricsPresenter implements Resettable {
  readonly lyricsAlign: LyricsAlignStore;
  readonly jotEditorStore: JotEditorStore;

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

  constructor(lyricsAlign: LyricsAlignStore, jotEditorStore: JotEditorStore) {
    this.lyricsAlign = lyricsAlign;
    this.jotEditorStore = jotEditorStore;
    makeAutoObservable(this, {
      lyricsAlign: false,
      jotEditorStore: false,
      lyricsAlignControllers: false,
    });
  }

  // --- modal visibility ---

  setLyricsSearchOpen(open: boolean) {
    this.lyricsAlign.lyricsSearchOpen = open;
  }

  setLyricsTextOpen(open: boolean) {
    this.lyricsAlign.lyricsTextOpen = open;
  }

  // --- LRCLIB ---

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

  // --- plain text ---

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
    if (this.jotEditorStore.tempo) {
      const tl = this.jotEditorStore.tempo.timeline;
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

  // --- lifecycle ---

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

  /** Session reset: same as {@link clearLyrics} (drop every lyrics row +
   *  abort in-flight aligns). Lyrics are tied to a specific recording, so a
   *  new song always starts with none. */
  reset(): void {
    this.clearLyrics();
  }

  /**
   * Remove a single lyrics row, aborting that row's in-flight align if
   * any. Routed through here (rather than `lyricsStore.remove(id)`
   * directly) so the lyrics store stays unaware of the per-track align
   * state held here.
   */
  removeLyricsTrack(id: LyricsTrackId): void {
    const ctrl = this.lyricsAlignControllers.get(id);
    if (ctrl) {
      ctrl.abort();
      this.lyricsAlignControllers.delete(id);
    }
    runInAction(() => {
      this.lyricsAlign.lyricsAlignStatuses.delete(id);
    });
    lyricsStore.remove(id);
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
      this.lyricsAlign.lyricsAlignStatuses.set(targetTrackId, { phase: 'aligning', detail: label });
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
            this.lyricsAlign.lyricsAlignStatuses.set(targetTrackId, {
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
      runInAction(() => {
        this.lyricsAlign.lyricsAlignStatuses.delete(targetTrackId);
      });
      // backendFetch already surfaced the generic "Server is down" toast.
      if (isBackendUnreachable(err)) return;
      const message = err instanceof Error ? err.message : String(err);
      toastStore.showError(`Lyrics align failed: ${message}`);
      return;
    } finally {
      if (this.lyricsAlignControllers.get(targetTrackId) === controller) {
        this.lyricsAlignControllers.delete(targetTrackId);
      }
    }
    if (lines.length === 0) {
      runInAction(() => {
        this.lyricsAlign.lyricsAlignStatuses.delete(targetTrackId);
      });
      toastStore.showError(`No lyrics were aligned (the aligner found no speech in ${label}).`);
      return;
    }
    runInAction(() => {
      lyricsStore.replace(targetTrackId, lines, {
        source: opts.source,
        sourceLabel: opts.sourceLabel,
      });
      this.lyricsAlign.lyricsAlignStatuses.delete(targetTrackId);
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
      this.lyricsAlign.lyricsAlignStatuses.clear();
    });
  }
}
