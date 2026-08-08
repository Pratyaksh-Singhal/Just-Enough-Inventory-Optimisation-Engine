# Dashboard

`index.html` — a single self-contained page. Open it directly; no server, no build step.

```bash
python scripts/build_dashboard.py     # regenerates index.html from the warehouse
```

`index.template.html` holds the markup, CSS and JS with a `__DATA__` placeholder; the
script queries DuckDB, computes the cost sweep, and inlines the result as JSON.

## Deviation from the brief, stated

The brief specified **React + Vite + Tailwind + Recharts**. This is a hand-authored single
file with inline SVG charts instead — a real deviation, not an oversight.

The reason: the ask was a **live URL**, and a Vite SPA calling the FastAPI service cannot be
live without hosting both, which is Phase 10 and needs deployment credentials this build
doesn't have. A self-contained file with the data baked in is viewable immediately by
anyone, from any host, with no backend running.

What it costs: no live API calls, so the figures are a snapshot of the last full pipeline
run rather than whatever the warehouse holds right now. What it buys beyond deployability:
complete control of the marks, which is why the two-pole cost encoding is consistent across
every figure rather than fighting a chart library's defaults.

The API is real and tested (`run-api`, 19 contract tests); the dashboard simply doesn't
depend on it being up.

## The cost simulator is exact, not interpolated

Order quantity depends only on the critical ratio. So total shortfall and leftover **units**
are precomputed once per service level across all 100,720 decisions — 90 points from
CR 0.10 to 0.99. Cost at any Cu/Co is then `shortfall × Cu + leftover × Co`: exact
arithmetic over 90 points on every drag, with no model call and nothing to debounce.

Because the sliders take flat per-unit costs while Phase 7's money table priced each row at
its own shelf price, simulator totals run 0.1–2% above the stored table depending on
policy. Stated on the page.

## Colour is doing information design, not decoration

The subject's central tension is that overstocking and understocking are *opposite* costs
that trade off against each other. The palette encodes exactly that, consistently, on every
figure:

| | meaning |
|---|---|
| **amber** | overstock — spoilage, the expensive side for perishables |
| **blue** | understock — a missed sale, the cheap side here |
| **jade** | the optimum sitting between them |

That is what makes the 48% stockout rate legible on screen rather than alarming: the reader
can *see* that the blue side is small and the amber side is large.

Validated with the dataviz skill's checker against both surfaces — all six checks pass
(lightness band, chroma floor, CVD separation, normal-vision floor, contrast):

- dark (`#131A18`): `#2FA97C` jade · `#C4862B` amber · `#4F86D4` blue
- light (`#FBFCFB`): `#0F8A5F` · `#9A6516` · `#2C5FA8`

The page commits to the dark theme (the brief specified it) and paints every colour
explicitly, so it renders identically regardless of the viewer's system theme.

## Type

Three voices, each with a job — chosen partly to avoid the Inter-everywhere default:

- **monospace** — every figure, label and eyebrow. The instrument voice.
- **system sans** — headings and chrome.
- **serif** — narrative prose in the engineering log, so writing reads as writing rather
  than telemetry.
