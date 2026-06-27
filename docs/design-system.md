# Frontend design system

The frontend has a small, deliberate design system. **Use it.** Don't
hardcode colors, radii, shadows, or shared text styles in module CSS; that's the drift the design-token lint (`bun run lint:design`, chained
into `bun run build`) catches.

## Source-of-truth files

- **[src/design_tokens.css](../src/design_tokens.css)**, global `:root`
  custom properties for every color, border-radius, and shadow. Loaded
  once from [src/index.tsx](../src/index.tsx) as a plain (non-module)
  stylesheet so the vars are visible to every module. Grouped
  semantically: `--color-bg-*`, `--color-border-*`, `--color-text-*`,
  `--color-accent-*`, `--color-error-*`, `--color-success-*`,
  `--color-busy-*`, `--color-toggle-active-*`,
  `--radius-{xs,sm,md,lg,xl,circle,pill}`,
  `--shadow-{sm,md,lg,note,playhead-label,accent-glow}`.
- **[src/typography.module.css](../src/typography.module.css)**, one
  CSS-modules class per typography **use case**, not per atomic property:
  `heading`, `body`, `bodySm`, `bodyMd`, `bodyEmphasis`, `label`,
  `labelSm`, `metaLabel`, `metaCaption`, `readout`, `readoutSm`,
  `subtle`, `mono`. Each bundles font-size + weight + letter-spacing +
  transform + line-height + family for its use case in one place.
- **[src/ui/](../src/ui/)**, shared React UI primitives + their module
  CSS, one folder per component (`src/ui/<name>/<name>.tsx` +
  `<name>.module.css`; stories in `src/ui/stories/`):
  - `button.module.css`, `<button>` reset + variants (`reset`,
    `primary`, `secondary`, `danger`, `ghost`, `close`). Every
    page-level button composes one of these.
  - `icon_button.module.css` / `icon_button.tsx`, `iconBox` (18×18
    bordered box) and `iconButton` (iconBox + labelSm); the 18×18
    `IconButton` base + `MuteButton` / `SoloButton` / `ClearButton`.
  - `modal.module.css`, `backdrop`, `panel`, `header`, `title`, `body`,
    `footer`.
  - `form.module.css`, `field` (input chrome), `fieldBlock` (textarea).
  - `spinner.module.css`, one `spinner` class (circle + 0.8s rotation
    keyframes defined once).
  - `dropdown.module.css`, `popoverPanel` chrome that
    `dropdownPanel` / `submenuPanel` / etc. compose; `dropdownItem`
    composes `ghost` so dropdown rows are real buttons.
  - Other widgets: `dropdown.tsx`, `tabs.tsx`, `number_stepper.tsx`,
    `checkbox.tsx`. New shared chrome goes here, not inlined into
    page-level modules.

## How consumers use it

```css
/* module CSS: pull in a use case, then layer your own chrome. */
.toolbarLabel {
    composes: labelSm from '../typography.module.css';
    color: var(--color-text-muted-alt);
}

.transcribeButton {
    composes: bodyEmphasis from '../typography.module.css';
    padding: 5px 12px;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    background: var(--color-accent);
    color: var(--color-bg-base);
}
```

In React inline styles, `style={{ color: 'var(--color-…)' }}` strings
work, so colors still flow through tokens.

**`composes:` constraint**, it only works in a rule whose selector is a
*single bare class*: `.foo` is fine, `.foo dt` / `.foo:hover` /
`.foo > .bar` are NOT (build fails with *"composition is only allowed
when selector is single :local class name"*). When typography is needed
on an element targeted by a compound selector (an `<h4>` in a card, a
`<dt>` in a `<dl>`), inline the props and leave a comment naming the use
case it mirrors. Current exceptions: `.debugDetailsList dt`
(score.module.css).

## The rules (1–3 enforced by stylelint via `.stylelintrc.json`)

1. **No hex literals or `rgb()`/`rgba()`/`hsl()`/`hsla()` calls in any
   `src/**/*.css` outside `src/design_tokens.css`.** Every color is a
   named token; need a new one → add it to `design_tokens.css` first.
   (`color-no-hex` + `function-disallowed-list`.)
2. **No direct typography declarations outside
   `src/typography.module.css`.** `font-size`, `font-weight`,
   `font-style`, `letter-spacing`, `text-transform`,
   `font-variant-numeric`, `font-feature-settings` come through
   `composes:`. `font-family` outside the typography file may only be
   `inherit`. True one-offs carry a
   `/* stylelint-disable-next-line property-disallowed-list -- <reason> */`.
   (`property-disallowed-list` + `declaration-property-value-allowed-list`.)
3. **Every `<button>` rule composes from `button.module.css`.** New
   buttons that just want a primary/secondary look land on the existing
   variant with no extra declarations; new visual variants belong in
   `button.module.css`, not re-declared at the call site.
4. **Shared visual chrome goes through a primitive.** A 3rd mixer row
   reaching for a 4th hand-rolled M/S button pair is a smell; it belongs
   as a variant of `IconButton`.

Stylelint runs **without a preset** (presets fire on CSS-modules
conventions like `composes:` and camelCase class names).

## Carve-outs (stay as raw values, by design)

- `LANE_COLORS` in [src/jot.ts](../src/jot.ts), an 8-color data palette
  consumed by JS to set per-lane note colors, not styling chrome.
- Canvas drawing in `mixer.tsx` (`ctx.fillStyle = '#…'`), the Canvas2D
  API requires literal color strings; `var(--…)` would mean a
  `getComputedStyle` per paint.
- True one-off typography (tuplet number italic, 22px transport glyph,
  single-use empty-state text), listed in the typography file's header.

## Adding a new visual primitive

1. *Color/radius/shadow* → add a semantically-named token to
   `design_tokens.css`.
2. *Typography use case* appearing in ≥2 places → add a class to
   `typography.module.css`, `composes:` it.
3. *React UI widget* used by ≥2 pages → drop it under
   `src/ui/<name>/` (its own folder).
4. Run `bun run lint:design` before committing.
