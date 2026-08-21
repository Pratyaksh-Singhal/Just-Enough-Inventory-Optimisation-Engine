# Design system

The resolved system, as the dashboard actually implements it. Read this before changing
`dashboard/index.template.html`, and update it when a token changes.

**Canonical source:** `data/design/Just Enough v2.dc.html` (Claude Design export), transcribed
onto `dashboard/index.template.html`. Where the two disagree, this document records which one
won and why.

**Retired, do not treat as live:**

| source | status |
|---|---|
| Industry wireframe | retired 2026-08-18 — superseded by the v2 export |
| Stitch `DESIGN.md` | retired 2026-08-18 — its palette never shipped |
| `Inventory Optimization Engine.dc.html` | superseded by `Just Enough v2.dc.html` |
| `Just Enough.dc.html` | superseded by v2 |

`design-artifacts/` is WDS scaffolding and is gitignored, so nothing there survives a clone.
This file lives in `docs/` for that reason.

---

## Colour

Dark-only. The page commits to one look rather than shipping a light theme it cannot test.

### Ground and structure

| token | value | role |
|---|---|---|
| `--ground` | `#0C1211` | page background, and the fill of grid cells that must read as holes |
| `--surface` | `#131A18` | cards, panels, table bodies |
| `--surface-2` | `#182120` | table headers only |
| `--line` | `#222E2A` | every border and rule |
| `--line-soft` | `#1B2523` | row separators inside a bordered panel |

### Ink

| token | value | role |
|---|---|---|
| `--ink` | `#E8EFEC` | headings, primary figures, the winning row |
| `--ink-2` | `#A6B5AF` | body copy |
| `--ink-3` | `#7A8A84` | captions, table labels, hints |
| `--ink-4` | `#5F6F69` | mono metadata, axis labels, the quietest legible text |

### Accent

Three hues, each with one meaning. **Semantic, not decorative** — a colour is never chosen
because a block needs variety.

| token | value | means |
|---|---|---|
| `--jade` | `#2FA97C` | the recommended action, and the cheaper outcome |
| `--jade-hi` | `#69DBAA` | headline figures and links — jade at display weight |
| `--amber` | `#C4862B` | waste, and any cost that went **up** |
| `--blue` | `#4F86D4` | missed sales, and a target the user picked rather than derived |

Tints for callout grounds and chips: `--jade-dim`, `--amber-dim`, `--blue-dim`,
`--callout-jade`, `--callout-amber`, `--callout-blue`.

Two greys are not tokens because they appear once each: `#5D6B65` (the neutral bar on Home)
and `#35443E` (the bracket corner ticks).

**The rule that keeps this honest:** a figure that can go either way takes its colour from its
sign, never from its position. The calculator's headline tile is jade when ordering to the
target is cheaper and amber when it is not — a tile that could only ever be green is
advertising, not arithmetic.

---

## Type

Three faces, each with a job. All four weights are vendored in `dashboard/fonts/` and embedded
as base64 at build time — never linked, because a CDN link is blocked wherever the page is
published outside Fly and would fall back to system faces with no error to notice.

| token | face | carries |
|---|---|---|
| `--cond` | Barlow Condensed 600 | headings, buttons, card titles |
| `--sans` | Barlow 400/600 | body copy, table names |
| `--mono` | JetBrains Mono 400 | **every number**, labels, metadata, axis text |

The mono face is not decoration. Tabular figures are what keep an order-quantity column
readable down a table, and `font-variant-numeric: tabular-nums` is set wherever digits line up.

### Scale

Display: `82 · 76 · 64 · 56 · 46 · 44` — one per page, at most.
Headings: `40 · 32 · 30 · 26 · 24 · 22 · 20 · 19`.
Body: `18 · 17 · 16 · 15 · 14`.
Mono UI: `13 · 12 · 11 · 10.5 · 10 · 9.5`.

Letter-spacing: `-0.05em` to `-0.02em` on display sizes, `0` on body, `0.1em`–`0.16em` on
uppercase mono labels.

---

## Layout

| measure | value |
|---|---|
| page gutter | `1320px` max, `32px` sides (`18px` below 700px) |
| reading column | `1000px` — the Why tab, which is one argument read top to bottom |
| wide column | `1200px` — the Proof tab |
| radius | `--r: 3px`, and `4px` on buttons. Nothing is more rounded than that. |

Grids are drawn as a `1px` gap over a `--line` ground with the cells filled `--ground`, so two
adjacent cells never produce a doubled rule.

---

## Rules that exist because breaking them shipped a bug

Each is pinned by a test in `tests/test_dashboard_contract.py`.

1. **Narrow-screen overrides go after the rules they override.** Equal specificity means source
   order decides and a media query adds none. A `@media (max-width: 700px)` block placed above
   its base rules was inert for days while its numbers were adjusted again and again.
2. **Never set `padding` shorthand on a `.wrap` selector.** `main` and `footer` both carry
   `.wrap`, and a class beats an element selector, so the shorthand silently zeroed their
   vertical padding. Use longhands.
3. **No bare `svg { }` rule.** A CSS width overrides an HTML width attribute, so a rule written
   for charts inflated every inline icon to the width of its button. Charts opt in with
   `.fluid`.
4. **No `stroke` or `fill` as an SVG presentation attribute.** `stroke="var(--jade)"` leaves
   stroke at its initial `none` wherever the property does not compute, and the glyph renders
   invisible with nothing logged. Put it in the stylesheet.
5. **A chart's viewBox is chosen at draw time.** SVG text scales with the viewBox, so a wide
   box on a phone shrinks the words as well as the picture — an 880×300 box rendered 97px tall
   with 3px labels in a phone column.
6. **Every figure is computed, never transcribed.** The design file states `28.6%`, `73%`,
   `41%`, `2.5×`. All were correct when it was drawn; all are now read from `DATA` at render
   time, so the page cannot keep asserting them once they stop being true.

---

## Deliberate departures from the design file

See also the project memory note of the same name.

| the file says | the page does | why |
|---|---|---|
| overage `= spoilage×cost + holding×price` → `44%` / `40.5%` | `cost × (spoilage + holding)` → `43.4%` / `40.9%` | `costs.py` is what produces every other number on the site |
| calculator step 02 shows `30 ÷ (30 + 44) = 40.5%` | the accurate derivation | the mock rounds its components before dividing and contradicts its own tile three rows down |
| footer left-aligned at full width | short, centred | content columns are 1000/1200/full, so a left-aligned footer fights whichever it sits under |
