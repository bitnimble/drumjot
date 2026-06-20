import { makeAutoObservable, runInAction } from 'mobx';
import { loadDebugZip, NO_DRUMS_KEY } from 'src/editing/provenance/debug_zip';
import { ExampleJot } from 'src/fakes/fakes';
import { Jot } from 'src/schema/dsl/dsl';
import { parseEnhancedLrc } from 'src/lyrics/enhanced_lrc';
import { lyricsStore } from 'src/lyrics/store';
import { fromMidi } from 'src/midi/from_midi';
import { ParseError } from 'src/schema/dsl/parser/errors';
import { parse } from 'src/schema/dsl/parser/parser';
import { AudioTrackId, AudioTrackRole } from 'src/editing/playback/audio_tracks';
import { jotPlayer } from 'src/editing/playback/player';
import { loadParadbZip } from 'src/schema/rlrr/paradb';
import { titleFromFilename, transcriber } from 'src/editing/transcribe/transcriber';
import { toastStore } from '../ui/toasts/toasts';
import { backendFetch, isBackendUnreachable } from 'src/net/backend_fetch';
import { JotEditorStore } from './jot_editor_store';
import { SettingsPresenter } from '../settings/settings_presenter';
import { MixerPresenter } from './mixer/mixer_presenter';
import { ProvenancePresenter } from './provenance/provenance_presenter';
import { LyricsPresenter } from './lyrics/lyrics_presenter';
import type { SessionReset } from './session_reset';
import {
  AUDIO_ENTRY_PREFIX,
  decodeMutableJotFile,
  encodeMutableJotFile,
  isMutableJotFile,
  JOT_FILE_VERSION,
  type AudioEntryInput,
  type DecodedMutableJotFile,
  type JotEditorMetadata,
  type MutableJotFile,
  type PersistedAudioTrack,
} from './persistence/jot_file';

/**
 * Song-load orchestration over {@link JotEditorStore}: the file loaders
 * (.jot / MIDI / ParaDB / debug bundle / lyrics / audio track), the
 * example picker, and the shared debug-bundle apply path. The sole writer
 * of the loaded song's peer fields / `currentExampleId` and the
 * loading-overlay counter.
 *
 * A wholesale song replace touches several other domains (drop stale
 * audio tracks + lane mixer, clear lyrics, reset per-note provenance,
 * pick the transcribe grid). Rather than write those stores directly,
 * this delegates to the owning sibling presenters so each store keeps a
 * single writer.
 */
export class JotEditorPresenter {
  readonly jotEditorStore: JotEditorStore;
  readonly settingsPresenter: SettingsPresenter;
  readonly mixerPresenter: MixerPresenter;
  readonly provenancePresenter: ProvenancePresenter;
  readonly lyricsPresenter: LyricsPresenter;

  /**
   * Registry of every persistent peer's session reset, fired exactly once
   * at the start of each wholesale load. Injected after construction
   * ({@link attachSessionReset}) because some resettable peers (selection,
   * editing) are built after this presenter at the composition root.
   * `undefined` only in the window before that wiring (e.g. a presenter
   * constructed standalone in a unit test that never loads a song).
   */
  private sessionReset: SessionReset | undefined = undefined;

  constructor(
    jotEditorStore: JotEditorStore,
    settingsPresenter: SettingsPresenter,
    mixerPresenter: MixerPresenter,
    provenancePresenter: ProvenancePresenter,
    lyricsPresenter: LyricsPresenter
  ) {
    this.jotEditorStore = jotEditorStore;
    this.settingsPresenter = settingsPresenter;
    this.mixerPresenter = mixerPresenter;
    this.provenancePresenter = provenancePresenter;
    this.lyricsPresenter = lyricsPresenter;
    makeAutoObservable<this, 'sessionReset'>(this, {
      jotEditorStore: false,
      settingsPresenter: false,
      mixerPresenter: false,
      provenancePresenter: false,
      lyricsPresenter: false,
      sessionReset: false,
    });
  }

  /** Wire in the composition root's {@link SessionReset} registry. Called
   *  once by `createJotEditor` after every resettable peer exists. */
  attachSessionReset(sessionReset: SessionReset): void {
    this.sessionReset = sessionReset;
  }

  /**
   * Reset every persistent peer to its fresh-session state, the shared
   * prelude to every wholesale song load (`.jot` / MIDI / ParaDB / debug
   * bundle / example / Save-file open). Replaces the per-loader hand-clears
   * that used to drift out of sync (the plain `.jot`/MIDI loaders once
   * forgot to drop the previous song's audio tracks + lane mixer). Run
   * inside the caller's `runInAction`, before the new song is installed.
   */
  private resetSession(): void {
    this.sessionReset?.reset();
  }

  /**
   * Wrap an async file-load with the modal overlay's bookkeeping (the
   * loading counter / label live on {@link JotEditorStore}). Errors
   * propagate; the finally block guarantees the counter decrements even if
   * the inner promise rejects, so a failed load never leaves the overlay
   * stuck on screen. Public so the transcribe flows (which also show the
   * overlay for their auto-load step) can reuse it.
   */
  async withLoading<T>(label: string, fn: () => Promise<T>): Promise<T> {
    runInAction(() => {
      if (this.jotEditorStore.loadingCount === 0) this.jotEditorStore.loadingLabel = label;
      this.jotEditorStore.loadingCount += 1;
    });
    try {
      return await fn();
    } finally {
      runInAction(() => {
        this.jotEditorStore.loadingCount -= 1;
        if (this.jotEditorStore.loadingCount === 0) this.jotEditorStore.loadingLabel = undefined;
      });
    }
  }

  /**
   * Load an audio file as a new audio track and update the status pill
   * on failure. Decoding goes through the shared `AudioContext`, so the
   * call has to occur inside a user gesture (the file-picker click
   * satisfies that). Every call appends an independent track, load N
   * files to get N tracks. Returns the new track's id, or `undefined`
   * if the load failed (so callers can e.g. default it to muted).
   */
  async loadAudioTrack(
    file: File,
    lane?: string,
    role?: AudioTrackRole
  ): Promise<AudioTrackId | undefined> {
    return this.withLoading(`Loading ${file.name}…`, async () => {
      try {
        return await jotPlayer.loadAudioTrack(file, lane, role);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        toastStore.showError(`Audio track load failed: ${message}`);
        return undefined;
      }
    });
  }

  /**
   * Install `source` as the loaded song (or clear it with `undefined`).
   * Delegates the reactive-document + peer construction to
   * {@link JotEditorStore.loadSource} (the single writer of those fields);
   * callers wrap this in their own `runInAction` alongside the per-load
   * bookkeeping (example pointer, provenance/lyrics clears, `jotPlayer.stop()`).
   */
  private installJot(source: Jot | undefined): void {
    this.jotEditorStore.loadSource(source);
  }

  setJot(source: Jot | undefined) {
    runInAction(() => {
      // Wholesale replace: return every persistent peer to its fresh state
      // (drops stale audio tracks / mixer / lyrics / debug provenance,
      // stops playback) before installing the new song.
      this.resetSession();
      this.installJot(source);
      this.jotEditorStore.currentExampleId = undefined;
    });
  }

  setExamples(examples: readonly ExampleJot[]) {
    this.jotEditorStore.examples = examples;
  }

  loadExample(id: string) {
    const example = this.jotEditorStore.examples.find((e) => e.id === id);
    if (!example) return;
    runInAction(() => {
      this.resetSession();
      this.installJot(example.jot);
      this.jotEditorStore.currentExampleId = id;
    });
  }

  /**
   * Read a `.jot` file from the user's machine and load it. Format-aware:
   * the same menu entry opens both flavours of `.jot` (see
   * `persistence/jot_file.ts`). A {@link isMutableJotFile mutable binary}
   * file is decoded and installed with its saved editor metadata; otherwise
   * the bytes are DSL text and go through the parser. Read / parse / decode
   * failures surface as error toasts.
   */
  async loadJotFile(file: File): Promise<void> {
    return this.withLoading(`Loading ${file.name}…`, async () => {
      let bytes: Uint8Array;
      try {
        bytes = new Uint8Array(await file.arrayBuffer());
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        toastStore.showError(`Could not read ${file.name}: ${message}`);
        return;
      }
      if (isMutableJotFile(bytes)) {
        await this.installMutableFileBytes(bytes, file.name);
        return;
      }
      // DSL text path: the original `.jot`. Decode the same bytes as UTF-8.
      const text = new TextDecoder().decode(bytes);
      try {
        const jot = parse(text);
        if (!jot.title) {
          const derivedTitle = titleFromFilename(file.name);
          if (derivedTitle) jot.title = derivedTitle;
        }
        runInAction(() => {
          this.resetSession();
          this.installJot(jot);
          this.jotEditorStore.currentExampleId = undefined;
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
   * Load a mutable-format `.jot` directly (the binary flavour). Public so a
   * caller that already knows the file is a save bundle (or e2e) can skip the
   * sniff; the format-agnostic entry point is {@link loadJotFile}.
   */
  async loadMutableFile(file: File): Promise<void> {
    return this.withLoading(`Loading ${file.name}…`, async () => {
      const bytes = new Uint8Array(await file.arrayBuffer());
      await this.installMutableFileBytes(bytes, file.name);
    });
  }

  /**
   * Decode mutable-format bytes and install them as the current song:
   * reset every peer, seed the document straight from the saved snapshot
   * (preserving the edits a DSL round-trip would drop), re-apply the file's
   * editor metadata (drum mixer / display settings / palette), then re-decode
   * the embedded audio tracks. Decode failures (corrupt zip, wrong format tag,
   * a newer file version) surface as an error toast and leave the current song
   * untouched.
   */
  private async installMutableFileBytes(bytes: Uint8Array, filename: string): Promise<void> {
    let decoded: DecodedMutableJotFile;
    try {
      decoded = await decodeMutableJotFile(bytes);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      toastStore.showError(`Could not open ${filename}: ${message}`);
      return;
    }
    const { file, audio } = decoded;
    runInAction(() => {
      this.resetSession();
      this.jotEditorStore.loadState(file.document, file.source);
      this.jotEditorStore.currentExampleId = undefined;
      this.applyEditorMetadata(file.editor);
    });
    // Audio tracks are re-decoded AFTER the document install (decode is async +
    // needs the AudioContext) and best-effort, mirroring the ParaDB / debug
    // loaders: a missing or undecodable track doesn't fail the whole load.
    await this.restoreAudioTracks(file.editor.audioTracks, audio);
  }

  /** Re-apply a loaded save file's (non-audio) editor metadata over the
   *  freshly-reset, installed song. Each block is optional, an older / partial
   *  file just leaves the reset defaults in place. Audio tracks are handled
   *  separately by {@link restoreAudioTracks} (they need async decode). */
  private applyEditorMetadata(editor: JotEditorMetadata): void {
    if (editor.settings) this.settingsPresenter.applySettings(editor.settings);
    if (editor.palette) this.jotEditorStore.viewConfig.palette = [...editor.palette];
    if (editor.mixer) this.mixerPresenter.applyTrackMixerState(editor.mixer);
  }

  /** Re-decode each embedded audio track from the container and re-apply its
   *  saved per-track mixer state to the freshly-minted track id. Best-effort:
   *  `loadAudioTrack` reports its own failures and returns `undefined`, so one
   *  bad track doesn't abort restoring the rest. */
  private async restoreAudioTracks(
    tracks: PersistedAudioTrack[] | undefined,
    audio: Map<string, Uint8Array>
  ): Promise<void> {
    if (!tracks?.length) return;
    for (const track of tracks) {
      const bytes = audio.get(track.entry);
      if (!bytes) continue;
      const file = new File([new Uint8Array(bytes)], track.filename);
      const id = await this.loadAudioTrack(file, track.lane, track.role);
      if (!id) continue;
      runInAction(() => {
        this.mixerPresenter.applyAudioTrackState(id, {
          muted: track.muted,
          soloed: track.soloed,
          volume: track.volume,
        });
      });
    }
  }

  /**
   * Assemble the current session into a mutable `.jot` save bundle, the JSON
   * envelope plus the embedded audio bytes, or `undefined` when no song is
   * loaded. Captures the edited document snapshot (the lossless superset), the
   * transitional DSL `source`, the editor metadata the DSL can't carry (drum
   * mixer, display settings, palette), and every loaded audio track / stem
   * (source bytes + per-track mixer state). The manifest entry paths are
   * generated in lockstep with the `audio` byte array so they pair up at
   * decode.
   */
  private async buildSaveBundle(): Promise<
    { file: MutableJotFile; audio: AudioEntryInput[] } | undefined
  > {
    const document = this.jotEditorStore.snapshot();
    const source = this.jotEditorStore.source;
    if (!document || !source) return undefined;
    const settings = this.settingsPresenter.settings;
    const mixer = this.mixerPresenter.mixer;

    const audio: AudioEntryInput[] = [];
    const audioTracks: PersistedAudioTrack[] = [];
    let i = 0;
    for (const track of jotPlayer.audioTracks.values()) {
      const entry = audioEntryName(i++, track.filename);
      // Store the source bytes verbatim (original codec); only zip-deflate the
      // uncompressed PCM formats (WAV / AIFF), where it actually shrinks them.
      audio.push({
        entry,
        bytes: new Uint8Array(await track.sourceBlob.arrayBuffer()),
        compress: !isAlreadyCompressedAudio(track.filename),
      });
      audioTracks.push({
        entry,
        filename: track.filename,
        role: track.role,
        lane: track.lane,
        muted: mixer.mutedAudioTracks.has(track.id),
        soloed: mixer.soloedAudioTracks.has(track.id),
        volume: mixer.audioTrackVolume(track.id),
      });
    }

    const file: MutableJotFile = {
      format: 'drumjot-mutable',
      version: JOT_FILE_VERSION,
      savedAt: new Date().toISOString(),
      document,
      source,
      editor: {
        mixer: this.mixerPresenter.trackMixerState(),
        settings: {
          gridLines: { ...settings.gridLines },
          uniformWaveforms: settings.uniformWaveforms,
          mergeLayers: settings.mergeLayers,
        },
        palette: [...this.jotEditorStore.viewConfig.palette],
        audioTracks: audioTracks.length > 0 ? audioTracks : undefined,
      },
    };
    return { file, audio };
  }

  /** Encode the current session and return the mutable-format bytes, or
   *  `undefined` when nothing is loaded. The serialisable form behind
   *  {@link saveMutableFile} (and the programmatic / e2e entry point). */
  async toMutableBytes(): Promise<Uint8Array | undefined> {
    const bundle = await this.buildSaveBundle();
    return bundle ? encodeMutableJotFile(bundle.file, bundle.audio) : undefined;
  }

  /**
   * Save the current session to a mutable `.jot` file via a browser
   * download named after the song title. No-op (with a toast) when nothing
   * is loaded.
   */
  async saveMutableFile(): Promise<void> {
    const bundle = await this.buildSaveBundle();
    if (!bundle) {
      toastStore.showError('Nothing to save yet, load or transcribe a song first.');
      return;
    }
    const bytes = await encodeMutableJotFile(bundle.file, bundle.audio);
    const title = bundle.file.document.title?.trim() || 'untitled';
    downloadBytes(bytes, `${sanitizeFilename(title)}.jot`);
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
          this.resetSession();
          this.installJot(jot);
          this.jotEditorStore.currentExampleId = undefined;
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
        // Replace the song wholesale: drop the previous song's audio
        // tracks, mixer, lyrics, and debug provenance, and stop playback,
        // before installing the new pack (its own audio tracks load below).
        this.resetSession();
        this.installJot(jot);
        this.jotEditorStore.currentExampleId = undefined;
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
      this.mixerPresenter.muteAudioTracks(
        resolved.filter((r) => r.id && r.defaultMuted).map((r) => r.id as AudioTrackId)
      );
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
        // backendFetch already surfaced the generic "Server is down" toast.
        if (isBackendUnreachable(err)) return;
        const message = err instanceof Error ? err.message : String(err);
        toastStore.showError(`Could not score ${file.name}: ${message}`);
      }
    });
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
   * Load a transcriber debug `.zip` bundle: parse the embedded
   * `final.jot`, load every audio track in the manifest's `mapping`, and
   * stash the manifest (stage timings + log stream) on the provenance
   * store so the {@link DebugPanel} can show it.
   *
   * Behaves like {@link loadParadbMap}: replaces the current song
   * wholesale (drops previously loaded audio tracks, resets the lane
   * mixer), runs entirely client-side, and surfaces errors on the
   * shared status pill.
   *
   * The `no_drums` entry (drumless backing audio) is auto-defaulted to
   * unmuted; the per-lane stems are defaulted to muted, mirroring the
   * "drum tracks are reference-only, you're playing them" convention
   * from the ParaDB loader, the drums you hear should be the smplr-
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
   * Fetch a debug `.zip` from a server URL and load it like the explicit
   * file picker (score + audio + provenance), wrapped in the loading
   * overlay. Errors land on the shared status pill. Used by the recent-
   * transcriptions picker (TranscribePresenter) to reopen a finished run
   * straight from `/outputs/<folder>/debug.zip` without re-running any
   * pipeline stage.
   */
  async loadDebugBundleFromUrl(url: string, fallbackName: string): Promise<void> {
    return this.withLoading(`Loading ${fallbackName}…`, async () => {
      try {
        const res = await backendFetch(url);
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
        // backendFetch already surfaced the generic "Server is down" toast.
        if (isBackendUnreachable(err)) return;
        const message = err instanceof Error ? err.message : String(err);
        toastStore.showError(`Could not load ${fallbackName}: ${message}`);
      }
    });
  }

  /**
   * Apply an already-parsed debug bundle: replace the current song with
   * the bundle's score (MIDI → jot), load each audio track, pair stems
   * with their instrument rows, and mount the manifest on the DebugPanel.
   *
   * Returns `true` if a score was loaded, `false` if `prediction.mid`
   * could not be turned into a jot (the audio tracks still load either
   * way so the operator can at least listen).
   *
   * Status-pill management is left to the caller, `loadDebugBundleFile`
   * sets it to idle/error on completion, while the transcribe flow keeps
   * its success pill visible after the auto-load. Public so the
   * transcribe presenter can reuse it.
   */
  async applyDebugBundle(
    bundle: Awaited<ReturnType<typeof loadDebugZip>>,
    fallbackName: string
  ): Promise<boolean> {
    runInAction(() => {
      // Wholesale replace prelude (drops the previous song's audio tracks,
      // mixer, lyrics, debug provenance; stops playback; returns the grid
      // overlay to its default). The bundle then mounts its own provenance
      // + 48ths overlay on top, so these two calls must follow the reset.
      this.resetSession();
      // Mount the manifest + per-note provenance (or clear it when the
      // bundle didn't ship one) and reset the ghost-overlay toggle.
      this.provenancePresenter.loadDebugBundle(bundle.manifest, bundle.noteProvenance ?? undefined);
      // Bundles come from the transcribe pipeline, which routinely
      // emits triplet subdivisions; the 48ths grid visualises both
      // 16ths and triplets. Override the store-wide 16ths default for
      // this load specifically.
      this.settingsPresenter.useTranscribeGridLines();
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
          this.installJot(jot);
          this.jotEditorStore.currentExampleId = undefined;
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
    // order; `no_drums` first, then lane letters), keeping the
    // post-load pair-with-instrument-row logic stable. The bundle
    // loader dedupes by filename, so each `track` here represents one
    // unique file; we bind every key in `track.keys` to the resulting
    // `AudioTrackId` so a shared stem (e.g. `stem_c.mp3` serving both
    // crash and ride after the cymbal split) is loaded once and looked
    // up under either key.
    const resolved = await Promise.all(
      bundle.audioTracks.map(async (track) => {
        // The audio-row's `lane` (used by the mixer for waveform
        // tinting) takes the first non-`no_drums` key; for a stem
        // shared across lanes, this picks the first-mentioned lane
        // in the manifest, which is good enough since the tint is
        // cosmetic and both siblings live in the same colour family.
        const primaryKey = track.keys.find((k) => k !== NO_DRUMS_KEY);
        // Role classification: any track whose only key is `no_drums`
        // is the Demucs drumless mix; everything else came from the
        // per-lane split (a key shared between multiple lanes still
        // counts as a single drum piece for menu purposes).
        const role: AudioTrackRole = primaryKey === undefined ? 'no-drums' : 'drum-piece';
        const id = await this.loadAudioTrack(track.file, primaryKey, role);
        return { keys: track.keys, id };
      })
    );
    const toMute: AudioTrackId[] = [];
    for (const { keys, id } of resolved) {
      if (!id) continue;
      // Mute the per-lane stems by default so the (audible) drums come
      // from the smplr score scheduler; the drumless backing stays
      // unmuted. A stem shared across keys still mutes once (one `id`).
      // Each track's row placement + waveform tint now derive from
      // `jot.ordering` (the audio entity's `lane` from `loadAudioTrack`),
      // so there's no separate reorder pass here.
      if (keys.some((key) => key !== NO_DRUMS_KEY)) toMute.push(id);
    }

    // Batch the mute updates into a single observable mutation so the
    // mixer renders once at the end instead of once per loaded track.
    runInAction(() => {
      this.mixerPresenter.muteAudioTracks(toMute);
    });

    return scoreLoaded;
  }

  /**
   * Fetch the debug zip from `url`, parse it, and load every artifact
   * via {@link applyDebugBundle}. The predicted-MIDI score, audio
   * tracks, note provenance, and stage timings / logs all come along
   * in one round trip.
   *
   * Returns `true` on success, `false` if either the fetch or the
   * parse failed (in which case the caller surfaces an error pill).
   * Public so the transcribe presenter's post-run auto-load can reuse it.
   */
  async autoLoadDebugBundle(
    url: string,
    fallbackName: string,
    signal: AbortSignal
  ): Promise<boolean> {
    let bundle: Awaited<ReturnType<typeof loadDebugZip>>;
    try {
      // backendFetch surfaces "Server is down" on transport failure; the
      // catch below still logs + returns false so the caller resets to idle.
      const res = await backendFetch(url, { signal });
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
}

/** Trigger a browser download of `bytes` as `filename` via a transient
 *  object-URL anchor. A side-effect of the Save flow (DOM, not store state),
 *  so it lives on this orchestrating presenter's module rather than a store. */
function downloadBytes(bytes: Uint8Array, filename: string): void {
  const blob = new Blob([new Uint8Array(bytes)], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  // Revoke on the next tick so the click's navigation has surely picked up
  // the URL first (revoking synchronously can cancel the download in some
  // browsers).
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

/** Make a song title safe for a download filename: collapse path separators
 *  and characters browsers/OSes dislike to underscores, trim, and cap the
 *  length so the `.jot` suffix always survives. */
function sanitizeFilename(name: string): string {
  const cleaned = name.replace(/[\\/:*?"<>|]+/g, '_').replace(/\s+/g, ' ').trim();
  return (cleaned || 'untitled').slice(0, 120);
}

/** Zip-entry path for the `index`-th saved audio track, keeping the source
 *  file's extension for human inspectability (the path is opaque to load,
 *  which matches it back via the manifest). */
function audioEntryName(index: number, filename: string): string {
  const ext = fileExt(filename);
  return `${AUDIO_ENTRY_PREFIX}${index}.${ext || 'dat'}`;
}

/** Already-compressed audio container extensions, deflating these in the zip
 *  only burns CPU for ~no size win, so they're stored verbatim. Everything
 *  else (WAV / AIFF / unknown) is treated as compressible PCM. */
const COMPRESSED_AUDIO_EXTS = new Set([
  'mp3', 'flac', 'ogg', 'oga', 'm4a', 'mp4', 'aac', 'opus', 'weba', 'webm', 'wma',
]);

/** Whether a loaded audio file's codec is already compressed (so the
 *  container stores it as-is rather than deflating). Lossless FLAC counts, its
 *  own entropy coding leaves nothing for deflate. */
function isAlreadyCompressedAudio(filename: string): boolean {
  return COMPRESSED_AUDIO_EXTS.has(fileExt(filename));
}

/** Lowercased file extension without the dot, or `''` when there is none. */
function fileExt(filename: string): string {
  const dot = filename.lastIndexOf('.');
  return dot > 0 ? filename.slice(dot + 1).toLowerCase().replace(/[^a-z0-9]/g, '') : '';
}
