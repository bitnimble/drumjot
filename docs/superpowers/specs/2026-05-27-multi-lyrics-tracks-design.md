# Multi-track lyrics, design

**Status:** approved (brainstorming); ready for implementation planning
**Date:** 2026-05-27

## Problem

Today the lyrics row is a strict singleton: one `lyricsStore` with one `lines` array, one `TrackKey { kind: 'lyrics' }` deduped by `trackKeyEq`, one row in the mixer, one offset, one in-flight align controller. Every loader (File, LRCLIB, plain text) replaces the singleton's contents via `lyricsStore.load(...)`.

The user wants multiple lyrics tracks coexisting on the same jot. Driving use cases:

- Side-by-side comparison of different alignments / hand-edits / LRCLIB matches.
- Multiple languages of the same song (Japanese / romaji / English translation).
- Multiple songs / sections in the same jot.
- Iterative drafts kept around as references.
- **Overlapping simultaneous vocal lines** (duets, harmonies); two tracks need to be visible at the same playhead position, in separate rows, without one obscuring the other.

The duet case rules out tabbed / active-of-many designs.

## Approach

Mirror the existing `audioTracks` pattern: id-keyed Map of `LyricsTrack`, one mixer row per id, drag-reorderable, additive-only loaders. Pattern parity with audio tracks means the existing row UI / drag mechanics / `syncTrackOrder` infrastructure carries the change with minimal new surface area.

Alternatives considered and rejected:

- **Single row stacking multiple "layers"**; collapses to the same 56px vertical slot. Worst-case for duets (the most demanding use case) is forced into the smallest space.
- **Keep singleton + parallel `lyricsTracks` map**; leaves a dead singleton behind as a half-completed migration.

## Data model

### `LyricsStore`

```ts
export type LyricsTrackId = string;

export type LyricsTrack = {
  readonly id: LyricsTrackId;
  readonly lines: readonly LyricLine[];
  readonly source: LyricsSource;
  readonly sourceLabel: string;
  readonly offsetSec: number;
};

class LyricsStore {
  // observable; iteration order = insertion order = display order seed for syncTrackOrder
  private tracks: Map<LyricsTrackId, LyricsTrack> = new Map();

  add(lines: readonly LyricLine[], opts: { source: LyricsSource; sourceLabel: string }): LyricsTrackId;
  replace(id: LyricsTrackId, lines: readonly LyricLine[], opts?: { source?: LyricsSource; sourceLabel?: string }): void;
  remove(id: LyricsTrackId): void;
  clear(): void;
  setOffsetSec(id: LyricsTrackId, sec: number): void;
  get(id: LyricsTrackId): LyricsTrack | undefined;
  get trackIds(): readonly LyricsTrackId[];
  get hasAnyLyrics(): boolean;
}
```

Active-line / active-word lookups move out of the store. `LyricsRow` calls `activeLineIndexAt(track.lines, audioTimeSec, track.offsetSec)` from `lrc.ts` directly with its own track's slice. The store no longer wraps that function.

### Id allocation

Monotonic counter, formatted as `lyrics-<n>` (e.g. `lyrics-1`, `lyrics-2`). Unique within the session; not persisted.

### Source label disambiguation

`add()` inspects existing tracks' `sourceLabel`s. On collision, suffixes ` (2)`, ` (3)`, etc. until unique. Logic is contained in `add()`; callers always pass the natural label.

### `replace()` semantics

- Preserves the track's existing `offsetSec` (the user may have nudged).
- Preserves `source` / `sourceLabel` unless explicitly overridden via `opts`.
- Used by the LRCLIB+wordLevel upgrade path to swap line-level lines for word-aligned lines in the same row, exactly as today's `lyricsStore.load(...)` does at the end of `alignLyricsWhisper`.

## Mixer integration

### TrackKey

```ts
export type TrackKey =
  | { kind: 'audio'; id: AudioTrackId; groupId?: string }
  | { kind: 'pitch'; pitch: string; groupId?: string }
  | { kind: 'lyrics'; id: LyricsTrackId; groupId?: string };
```

`trackKeyEq`:

```ts
if (a.kind === 'lyrics') {
  return a.id === (b as { kind: 'lyrics'; id: LyricsTrackId }).id;
}
```

### `syncTrackOrder`

```ts
const wanted: TrackKey[] = [
  ...lyricsStore.trackIds.map((id) => ({ kind: 'lyrics' as const, id })),
  ...audioIds.map((id) => ({ kind: 'audio' as const, id })),
  ...pitches.map((pitch) => ({ kind: 'pitch' as const, pitch })),
];
```

Insertion policy mirrors audio:

- A new lyrics row slots in just after the last existing lyrics row, keeping the lyrics group contiguous.
- The very first lyrics row (when no lyrics rows exist yet) goes to the top of the mixer (matches today's `next.unshift(w)` for the singleton).

Existing entries keep their relative position across reactions; the filter step preserves surviving entries. Drag-reorder works without change.

### Mixer render branch

```ts
if (key.kind === 'lyrics') {
  return <LyricsRow key={reactKey} id={key.id} jot={jot} onSeek={onSeek} {...rowProps} />;
}
```

React keys for lyrics rows derive from `id` so reorders don't remount.

## Load paths

### `loadLyricsFile(file)`

```ts
runInAction(() => {
  lyricsStore.add(parsedLines, {
    source: 'file',
    sourceLabel: `File · ${file.name}`,
  });
});
```

### `applyLrclibResult(lines, match, opts)`

```ts
const id = lyricsStore.add(lines, {
  source: 'lrclib',
  sourceLabel: `LRCLIB · ${match.trackName} - ${match.artistName}`,
});
toastStore.showSuccess(...);
if (opts.wordLevel) {
  void this.runWordLevelAlignmentForLrclib(id, lines, match);
}
```

### `applyPlainTextLyrics(text, opts)`

```ts
const id = lyricsStore.add(lines, { source: 'plaintext', sourceLabel: 'Plain text' });
if (opts.wordLevel) {
  void this.runWordLevelAlignmentForPlainText(id, lines);
}
return lines.length;
```

`computeLyricsSpreadSec()` is unchanged; picks the longest audio track's duration as today.

All loaders are additive only. No "replace existing" affordance; the user removes a row via its gutter Clear button and loads a new one.

## Wholesale reload

The existing wholesale-reload callers (`loadJotFile`, `loadParadbMap`, `applyDebugBundle`) call `this.clearLyrics()`. With multi-track that semantic still holds; "clear" means drop every lyrics row. The internal method body changes to:

```ts
clearLyrics(): void {
  lyricsStore.clear();
  this.cancelAllLyricsAlign();   // see Alignment section
}
```

## Word-level alignment

### Per-track state

```ts
// On JotViewStore. Replaces the singletons lyricsAlignController and lyricsAlignStatus.
lyricsAlignControllers: Map<LyricsTrackId, AbortController> = new Map();
lyricsAlignStatuses: Map<LyricsTrackId, LyricsAlignStatus> = new Map();
```

Absence of an entry = idle for that track.

### `alignLyricsWhisper(targetTrackId, req, label, opts)`

Signature gains `targetTrackId: LyricsTrackId` as its first parameter. Body:

1. If `lyricsAlignControllers.has(targetTrackId)` → abort the existing controller and delete it (new align on the same row wins; don't queue).
2. Allocate a fresh `AbortController`; store under `targetTrackId`.
3. Set `lyricsAlignStatuses.set(targetTrackId, { phase: 'aligning', detail: label })`.
4. Await the request; on success call `lyricsStore.replace(targetTrackId, lines, opts)`; preserves `offsetSec`, source can be overridden.
5. On any exit (success / error / abort): if `lyricsAlignControllers.get(targetTrackId) === controller`, delete the map entry; clear the status (or delete the entry; absence = idle).

The existing `controller.signal.aborted` early-return in the catch block is preserved, now keyed by track so a late-completing aborted job can't overwrite a freshly-started one on the same id.

### `cancelAllLyricsAlign()` (was `cancelLyricsAlign()`)

Iterate all controllers, abort each, clear both maps. Called by wholesale-reload paths only.

### Per-track removal

`LyricsRow`'s Clear button routes through a new `JotViewStore.removeLyricsTrack(id)` instead of calling `lyricsStore.remove(id)` directly:

```ts
removeLyricsTrack(id: LyricsTrackId): void {
  const ctrl = this.lyricsAlignControllers.get(id);
  if (ctrl) {
    ctrl.abort();
    this.lyricsAlignControllers.delete(id);
  }
  this.lyricsAlignStatuses.delete(id);
  lyricsStore.remove(id);
}
```

This keeps `lyricsStore` itself unaware of the align state (no coupling to `JotViewStore`).

## Row + toolbar UI

### `LyricsRow`

- Takes `id: LyricsTrackId` prop.
- Reads `const track = lyricsStore.get(id)`; guards `undefined` (id race on the mixer reaction; one-frame possible).
- `lines = track.lines`, `offsetSec = track.offsetSec`, `sourceLabel = track.sourceLabel`.
- Offset stepper: `lyricsStore.setOffsetSec(id, v)`.
- Clear button: `jotViewStore.removeLyricsTrack(id)`.

### Per-track align indicator

A small spinner appears in the row gutter (alongside the source label, not replacing it) when `jotViewStore.lyricsAlignStatuses.get(id)?.phase === 'aligning'`. Tooltip: `Aligning lyrics to audio…`. Spinner styling matches whatever the existing toolbar busy pill uses; if none exists, a minimal CSS-only rotating SVG.

### Toolbar

- "Clear lyrics" menu item removed entirely. Per-row removal subsumes it.
- `onClearLyrics` and `hasLyrics` props on toolbar removed (the latter was only gating the now-removed menu item).
- The existing busy-pill `lyricsAlignStatus` prop is replaced with `lyricsAnyAligning: boolean` (derived: `jotViewStore.lyricsAlignStatuses.size > 0` after absence-means-idle). The pill doesn't display which track; the per-row spinner covers that.

## Tests

In `src/lyrics/__tests__/store.test.ts` (extend existing):

- `add()` returns a fresh id each call.
- `add()` with duplicate `sourceLabel` produces ` (2)`, ` (3)` suffixes; unique labels pass through unchanged.
- `add()` does not modify other tracks' fields.
- `remove(id)` drops one track; others retain their offsets and lines.
- `clear()` drops all; `hasAnyLyrics === false`.
- `setOffsetSec(id, sec)` only mutates the targeted track; clamps to `[LYRICS_OFFSET_MIN_SEC, LYRICS_OFFSET_MAX_SEC]`.
- `replace(id, lines)` preserves `offsetSec`; preserves `source` / `sourceLabel` when not overridden.
- `replace(id, lines, { sourceLabel })` overrides only the label.

In `src/lyrics/__tests__/whisper_align.test.ts`; shift any direct `lyricsStore.load()` exercises to `lyricsStore.add()` and assert on the returned id.

No new tests added for `JotViewStore`'s align controller paths (existing whisper-align tests cover the wire shape; the controller logic is a thin async wrapper).

## Out of scope

- Per-track rename (will arrive as a generic per-row feature for all track types).
- Persistence across reloads (lyrics remain session-only; matches existing product decision).
- Mute/solo for lyrics rows (lyrics are visual, not audible).
- Toolbar UI for "which track is aligning"; per-row spinner is the only surface.

## Files touched

- `src/lyrics/store.ts`; core rewrite.
- `src/lyrics/index.ts`; re-exports updated to include `LyricsTrack`, `LyricsTrackId`.
- `src/lyrics/__tests__/store.test.ts`; extended.
- `src/lyrics/__tests__/whisper_align.test.ts`; `load → add` shift.
- `src/jot_view/store.ts`; `TrackKey`, `trackKeyEq`, `syncTrackOrder`, the three loader paths (`loadLyricsFile`, `applyLrclibResult`, `applyPlainTextLyrics`), `alignLyricsWhisper` (per-track signature + status maps), `cancelAllLyricsAlign`, `removeLyricsTrack`, toolbar prop derivation.
- `src/jot_view/mixer.tsx`; render branch passes `id`.
- `src/jot_view/lyrics_row.tsx`; accepts `id` prop, reads from `lyricsStore.get(id)`, spinner during align, clear routes through `removeLyricsTrack`.
- `src/jot_view/lyrics_row.module.css`; spinner styling.
- `src/jot_view/toolbar.tsx`; remove "Clear lyrics" menu item; drop `onClearLyrics` / `hasLyrics` props; rename `lyricsAlignStatus` → `lyricsAnyAligning: boolean`.
