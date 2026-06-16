# Score stacking & z-index layers

Reference for how the score/mixer composites: which elements form stacking
contexts, the z-index ladder inside each, and the portal pattern that
keeps selection popovers visible over the minimap / playback chrome.
Read this before adding any overlay, popover, badge, or bracket that
needs to escape its row or bar, the clipping and stacking model is
subtle and has bitten us more than once.

The cardinal rule: **z-index only orders siblings within the *same*
stacking context.** A big number does nothing if the element it's fighting
lives in a different context. Most "my overlay is hidden" bugs are a
context-nesting problem, not a "number too small" problem, and some aren't
z-index at all but *paint containment* clipping (see §4) or *overflow*
clipping at the score viewport (see §5).

## 1. Stacking-context tree

Indentation = DOM nesting. `(SC)` marks an element that establishes a new
stacking context, with the property that causes it. Elements without `(SC)`
do **not** create one, so their positioned descendants compete in the
nearest `(SC)` ancestor.

```
.jotContainer            overflow:hidden, clips the score viewport (no SC by itself)
  .scrollViewport  (SC)  transform + will-change:transform   ← root SC for everything in the score
    .timelineHeader / title / subtitle / legend …
    .mixer                 display:flex column, NOT an SC (z-index:auto)
      .gutterMasterRow, .instrumentRow, .musicTrack, .lyricsRow   ← flex items, NOT SCs while z-index:auto
        ─ gutter   (SC)    scrollStickyHorizontal = transform + will-change ; z-index:6
        ─ .barsRow (SC)    position:relative ; z-index:1
            .bar   (SC*)   position:absolute ; content-visibility:auto ⇒ contain:paint   (*see §4)
              .lane            position:relative (NOT an SC), containing block for the notes
                .note  (SC)    transform ; z-index:20/30 when selected / hovered
                .gridLayer*      beat-grid mask layers
              .patternBracket / .tupletBracket
            .filteredOnset (SC) z-index:4 / 25 ; direct child of .barsRow (NOT inside .bar)
            .playhead (SC)   transform ; z-index:5

document.body              ← PopoverPortal target (see §5)
  .noteLabel / .filteredOnsetLabel  (SC, position:fixed, z-index:1100)
```

Key consequences:

- **All `.barsRow`s share one parent SC** (`.scrollViewport`), because the
  rows themselves are `z-index:auto` (not SCs). Every `.barsRow` sits at
  `z-index:1` there, so **ties break by DOM order**, a lower row paints
  over an upper row. The selection popovers no longer live inside a row
  (§5), so they don't need the legacy row-lift to compose with sibling
  rows.
- **`.bar` clips its descendants** (§4), independent of any z-index. The
  popovers escape this clip by being portaled out (§5); any new overlay
  living *inside* `.bar` is still subject to it.

## 2. z-index ladder, per stacking context

Numbers are only comparable **within the same block** below.

### Inside `.scrollViewport` (row level)
| z | element | notes |
|---|---------|-------|
| 1 | `.barsRow` / `.leadInOverlay` | per-row bars strip; ties break by DOM order |
| 5 | `.selectionFrame` | multi-note bounding box; interactive drag-to-move surface, so it must sit above every `.barsRow` (and its notes) to receive the press, below the gutters. No fill, so it doesn't obscure the notes |
| 6 | sticky gutters (`.instrumentRowGutter`, `.musicTrackGutter`, `.lyricsGutter`, `.gutterMasterGutter`, `.timelineHeaderGutter`) | pinned column |
| 7 | `.gutterResizeHandle` | sits above its gutter |

### Inside `.barsRow` (z-index 1)
| z | element | notes |
|---|---------|-------|
| auto | `.bar` | bars paint in DOM order |
| 1 | `.leadInOverlay` | |
| 2 | `.patternBracket` | |
| 3 | `.tupletBracket` | |
| 4 / 25 | `.filteredOnset` / `.filteredOnsetShowingLabel` | ghost onset, raised when its popover is open |
| 5 | `.playhead` | |
| 6 | `.playheadLabel` | |
| 20 / 30 | `.note.noteShowingLabel` / `.note.noteHovered` | selected/hovered note raised above sibling bars (for the note glyph itself; the label is portaled out, §5) |

### App-shell chrome (outside / above the score, separate domains)
| z | element |
|---|---------|
| 5 | `.verticalScrollbar` |
| 20 | `.toolbar` |
| 30 / 40 | dropdown trigger / dropdown panel (`ui/dropdown`) |
| 40 | `.recentTranscriptions` |
| 100 | `.loadingOverlay`, modal backdrop (`ui/modal`) |
| 1000 | toasts |
| 1050 | color picker popover |
| 1100 | **`.noteLabel` / `.filteredOnsetLabel`** (portaled to `document.body`; §5) |

## 3. Notes about the cursor / selection state

`.note.noteShowingLabel { z-index: 20 }` and `.note.noteHovered { z-index: 30 }`
still apply: they lift the note **glyph** above sibling bars / notes in
the same `.barsRow` so the selected / hovered notehead reads clearly.
The popover that hangs off the note is now portaled (§5), so these
z-indexes are only about the note itself, not its label.

## 4. Paint containment clips overlays inside `.bar`

`.bar` has `content-visibility: auto` for long-song virtualization, which
**forces `contain: paint`**, descendants are clipped to the bar's box
even though every `overflow` in the chain is `visible`.
`overflow-clip-margin` buys a few px so edge noteheads aren't clipped,
but anything that hangs further would be clipped if it lived inside
`.bar`.

The selection popover used to live inside `.bar` and needed an explicit
`.bar:has([data-note-label-open]) { content-visibility: visible }`
opt-out for the bar that owned the popover. The popover is now portaled
out of `.bar` entirely (§5), so this rule is gone. Anything new that
lives inside `.bar` and needs to escape needs either an equivalent
opt-out, the portal pattern in §5, or to stay within
`overflow-clip-margin`.

`FilteredOnset` (the ghost glyph) lives directly in `.barsRow`, not
inside `.bar`, so it isn't subject to this clip. Its popover, like the
real note's, is portaled.

## 5. Portaled selection popovers

`.jotContainer { overflow: hidden }` clips its descendants at the score
viewport's edge, the side effect of disabling native scroll so the
transform-driven scroll can run subpixel. Pre-portal, the selection
popover lived inside `.scrollViewport`, which meant it couldn't extend
past the score's bottom edge to overlap the minimap or playback bar
below; flipping above (`usePopoverFlipAbove`) only worked when the
anchor was near the bottom.

`PopoverPortal` (`src/editing/score.tsx`) renders the popover into
`document.body` via `createPortal`. Position is set inline at render
time from the anchor's `getBoundingClientRect()` and applied as
`position: fixed`; the wrapping `observer(...)` HOC subscribes to
`store.scrollX` / `store.scrollY` / `store.zoom` so the popover
re-measures and re-renders whenever the anchor's screen position
changes under the (transform-driven) scroll. Above-flip is preserved,
but the bottom limit is now the window edge rather than the score
scroller bottom, the popover happily overlaps the minimap / playback
chrome until it would actually run off-window.

z-index `1100` sits above every other app-shell layer (modals at 100,
toasts at 1000, color picker at 1050) so the popover paints on top of
anything outside the score that happens to overlap it. `data-popover`
on the wrapper is the stable test selector (`[data-popover="note-label"]`).

Costs the portal accepts:

- `getBoundingClientRect()` reads at render time and on every scroll-tick
  re-render. AGENTS.md §5.9 forbids per-frame rect reads in *hot* paths;
  this is the explicit popover-anchoring exception called out there
  (one read per popover-open re-render of a single element, not a
  per-frame layout loop on the whole tree).
- The anchor's note still carries `.noteShowingLabel` / `.noteHovered`
  for the in-row glyph z-lift (§3); no row-lift is needed any more
  because the popover doesn't live inside the row.

## 6. Contract for new overlays / popovers

Anything that must visually escape its bar / row / score viewport:

1. **Default to portaling.** Use the `PopoverPortal` pattern (§5):
   render into `document.body`, position from the anchor's
   `getBoundingClientRect()`, observe `store.scrollX/Y/zoom` for
   re-measurement, give the wrapper a `data-popover="…"` test hook.
2. **If you must live inside `.bar`** (e.g. an indicator that scrolls
   visually-tight with the notes), respect `content-visibility: auto`
   on `.bar` (§4), either stay within `overflow-clip-margin`, or set
   a marker attribute on the anchor and add a paint-containment opt-out
   keyed on it.
3. **Don't add `opacity < 1`, `transform`, `filter`, or `will-change`**
   to an ancestor row / lane just to get a visual effect, those
   establish a stacking context that re-traps descendants. (See the
   per-note `opacity` dim in `.laneDim .note`, done on the note, not
   the lane, for exactly this reason.)

## 7. Known limitations

- **Top / left of window.** `PopoverPortal` flips above when the popover
  would overflow the window bottom; if it would *also* overflow the
  window top (a tiny window, an anchor near the top edge), the
  fallback is to place below and accept clipping at the window's own
  edge. A scroll-into-view or off-edge translation would handle this,
  not currently implemented.
- **Fixed-px overflows at low zoom.** `.tupletBracket` (`top:-17px`)
  and `.stickingBadge` (`bottom:-10px`) overflow `.bar` by a fixed
  pixel amount, while `overflow-clip-margin` scales with
  `--note-pad-px` (which shrinks with zoom). At very low zoom the clip
  margin can fall under those offsets and nick the bracket tick /
  badge. Cosmetic; not currently handled.

## 8. Regression coverage

`e2e/popover-visibility.spec.ts` selects a note in a non-bottom row and
asserts (via `elementFromPoint`) that the popover is the painted-on-top
element across its whole height, including the part overlapping the
track beneath it. It fails if the portal z-index or position
computation regresses.
