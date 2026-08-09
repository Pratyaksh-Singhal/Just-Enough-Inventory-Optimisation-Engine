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

## One tab does need a backend

**Full forecast** is the exception, and it is a deliberate one. Every other tab renders from
data baked into the file at build time; that tab uploads a CSV to the tier 2 service, polls
a job, and draws what comes back. It cannot work without `run-service` running.

So it degrades openly rather than silently: opening the tab probes `/health` once, and when
there is nothing there it says so and points at the Order calculator, which needs no backend
and handles the same file. The address is an editable field, prefilled with
`http://127.0.0.1:8001` — **8001, not 8000**, because tier 1's `run-api` owns 8000 and the
two are meant to run side by side.

The probe fires on first open of that tab, not on page load: a visitor who never opens it
should not have a failing request in their network log.

### The forecast chart's colour, and why it is one hue

History and forecast are the same quantity at two points in time, so they share **jade** and
are separated by a channel that is not colour — solid for what happened, dashed for what is
projected. The band is jade at 10%, and the order line is a neutral dotted annotation.

That was not the first attempt. A muted grey history line against a jade forecast failed the
dataviz validator at **ΔE 14.8 normal-vision, under the floor of 15** — two lines a
full-colour reader would struggle to tell apart. Borrowing amber or blue would have been
worse: they carry fixed meanings (overstock, understock) on every other figure here, and
reusing them for "past" and "future" would break the one thing this palette is for.

The order quantity is drawn as a **daily rate**, not its horizon total. A 28-day total of 437
plotted on an axis whose values run 0–60 would sit far off the top of the chart and mean
nothing; 15.6/day is the same decision in the chart's own units.

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
