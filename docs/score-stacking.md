# Score stacking & z-index layers

Reference for how the score/mixer composites: which elements form stacking
contexts, the z-index ladder inside each, and the two cross-cutting rules
that keep selection popovers visible. Read this before adding any overlay,
popover, badge, or bracket that needs to escape its row or bar, the
clipping and stacking model is subtle and has bitten us more than once.

The cardinal rule: **z-index only orders siblings within the *same*
stacking context.** A big number does nothing if the element it's fighting
lives in a different context. Most "my overlay is hidden" bugs are a
context-nesting problem, not a "number too small" problem, and some aren't
z-index at all but *paint containment* clipping (see §4).

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
                .note  (SC)    transform ; z-index:20/30 when its label is open
                  .noteLabel (SC)  transform ; z-index:12   ← the selection popover
                .gridLayer*      beat-grid mask layers
              .patternBracket / .tupletBracket
            .filteredOnset (SC) z-index:4 / 25 ; direct child of .barsRow (NOT inside .bar)
              .filteredOnsetLabel (SC) z-index:13
            .playhead (SC)   transform ; z-index:5
```

Key consequences:

- **All `.barsRow`s share one parent SC** (`.scrollViewport`), because the
  rows themselves are `z-index:auto` (not SCs). Every `.barsRow` sits at
  `z-index:1` there, so **ties break by DOM order**, a lower row paints
  over an upper row. That is why a popover hanging into the row below needs
  the row-lift in §3.
- **`.note` / `.noteLabel` carry their own z-index *inside* `.barsRow`'s
  context** (z-index 1). They cannot escape the row on their own; lifting
  has to happen on the row wrapper.
- **`.bar` clips its descendants** (§4), independent of any z-index.

## 2. z-index ladder, per stacking context

Numbers are only comparable **within the same block** below.

### Inside `.scrollViewport` (row level)
| z | element | notes |
|---|---------|-------|
| 1 | `.barsRow` / `.leadInOverlay` | per-row bars strip; ties break by DOM order |
| 6 | sticky gutters (`.instrumentRowGutter`, `.musicTrackGutter`, `.lyricsGutter`, `.gutterMasterGutter`, `.timelineHeaderGutter`) | pinned column |
| 7 | `.gutterResizeHandle` | sits above its gutter |
| 32 | **lifted row** (`.instrumentRow`/`.musicTrack`:has open popover) | see §3 |

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
| 13 | `.filteredOnsetLabel` | onset detail popover |
| 20 / 30 | `.note.noteShowingLabel` / `.note.noteHovered` | selected/hovered note raised above sibling bars so its label wins intra-row |
| 12 | `.noteLabel` | inside `.note`, so effectively above sibling bars via the note's 20/30 |

### App-shell chrome (outside / above the score, separate domains)
| z | element |
|---|---------|
| 5 | `.verticalScrollbar` |
| 20 | `.toolbar` |
| 30 / 40 | dropdown trigger / dropdown panel (`components/dropdown`) |
| 40 | `.recentTranscriptions` |
| 100 | `.loadingOverlay`, modal backdrop (`components/modal`) |
| 1000 | toasts |
| 1050 | color picker popover |

## 3. The row-lift (`data-note-label-open`)

A selection popover hangs **below** (or, when flipped, above) its note,
into the neighbouring row. Because all `.barsRow`s are equal-z siblings in
`.scrollViewport` (§1), the neighbour would paint over it. So:

- `NoteView` / `FilteredOnsetView` set `data-note-label-open` on their
  anchor while the popover is shown (`src/jot_view/score.tsx`).
- `mixer.module.css` lifts the whole row above its siblings:
  ```css
  .instrumentRow:has([data-note-label-open]),
  .musicTrack:has([data-note-label-open]) { z-index: 32; }
  ```
  A data-attribute (not a CSS-module class) because the note classes live
  in a different module's local scope and `:has()` needs to see them.

The lift is on the **row wrapper**, not the note, on purpose: the note's
own z-index only orders it inside `.barsRow`'s context, so it can't clear
the row boundary by itself.

## 4. Paint containment clips overlays (the real "obscured popover" bug)

`.bar` has `content-visibility: auto` for long-song virtualization, which
**forces `contain: paint`**, descendants are clipped to the bar's box even
though every `overflow` in the chain is `visible`. `overflow-clip-margin`
buys a few px so edge noteheads aren't clipped, but a **popover** hangs
tens to hundreds of px past that, into the next row. Clipped, it reads
exactly like "the track below is covering the popover"; but it is not a
z-index problem; the pixels are simply not painted.

Fix / contract:
```css
.bar:has([data-note-label-open]) { content-visibility: visible; }
```
The one bar with an open popover drops paint containment. It is on-screen
by definition (the user selected/hovered it), so the virtualization it
forgoes for that moment costs nothing.

`FilteredOnsetView` lives **directly in `.barsRow`, not inside `.bar`**, so
it is not subject to this clip, it only needs the §3 row-lift.

## 5. Contract for new overlays / popovers

Anything that must visually escape its bar or row:

1. **Set `data-note-label-open`** (or extend the `:has()` selectors) on the
   anchor while shown, so the row lifts above its siblings (§3).
2. **If it lives inside `.bar`**, make sure the bar drops paint containment
   while shown (§4), the existing `.bar:has([data-note-label-open])` rule
   already covers anything keyed on that attribute.
3. **Give the popover itself a z-index** above sibling content in its
   context (notes use 12 inside the 20/30 note).
4. Don't add `opacity < 1`, `transform`, `filter`, or `will-change` to an
   ancestor row/lane to get a visual effect without realising it creates a
   stacking context that re-traps descendants. (See the per-note `opacity`
   dim in `.laneDim .note`, done on the note, not the lane, for exactly
   this reason.)

## 6. Known limitations

- **Score-edge clipping.** `.jotContainer { overflow: hidden }` (the
  virtualised-scroll viewport) clips popovers that extend past the visible
  score edge. `usePopoverFlipAbove` mitigates the bottom edge for the two
  score popovers; a popover near the very top/right can still be cut. A
  portal would fix it but would detach the popover from the scroll
  transform, so it's intentionally not done.
- **Fixed-px overflows at low zoom.** `.tupletBracket` (`top:-17px`) and
  `.stickingBadge` (`bottom:-10px`) overflow `.bar` by a fixed pixel amount,
  while `overflow-clip-margin` scales with `--note-pad-px` (which shrinks
  with zoom). At very low zoom the clip margin can fall under those offsets
  and nick the bracket tick / badge. Cosmetic; not currently handled.

## 7. Regression coverage

`e2e/popover-visibility.spec.ts` selects a note in a non-bottom row and
asserts (via `elementFromPoint`) that the popover is the painted-on-top
element across its whole height, including the part overlapping the track
below. It fails if either the row-lift (§3) or the paint-containment
opt-out (§4) regresses.
