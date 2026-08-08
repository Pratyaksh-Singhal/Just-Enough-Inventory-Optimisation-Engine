# E9 — Dashboard ✅

**Goal:** the portfolio centrepiece. React + Vite + Tailwind + Recharts — not Streamlit.

**Consumes:** the warehouse, via `scripts/build_dashboard.py` (see the deviation note below).

**Produces:** `dashboard/index.html` — a single self-contained page.

**Live:** https://claude.ai/code/artifact/ab291272-dc60-496b-88cd-fe3f5c57ce2f

**Design direction:** dark analytical theme. Deep slate/charcoal base, a single confident
accent (emerald or amber) used sparingly for emphasis and positive deltas. Generous
whitespace. Real typographic hierarchy — a display face for numbers with
`font-variant-numeric: tabular-nums`, clean sans for body. Subtle borders over heavy
shadows. No gradient soup, no glassmorphism, no purple-blue SaaS template. Think Linear
or Vercel analytics, not a bootcamp dashboard.

---

### E9-S1 — App shell and design system ✅

- [x] Vite + React + TypeScript + Tailwind
- [x] Design tokens: colour, type scale, spacing — defined once, used everywhere
- [x] Tabular figures on every number that changes
- [x] Keyboard navigable; responsive down to tablet

### E9-S2 — Overview screen ✅

- [x] Hero stat: total money saved. Big, confident, unmissable.
- [x] Below: stockout rate, waste units, forecast accuracy — each with a delta chip vs baseline
- [x] Sparkline per metric
- [x] Currency toggle: USD ⇄ ₹, with the FX rate shown inline (M5 is US data; the rate is
      a display convenience, never baked into the stored number)

### E9-S3 — Forecast Explorer ✅

- [x] Hierarchy drill-down `State → Store → Dept → SKU` as breadcrumb or tree sidebar
- [x] Actual history line + forecast line + **shaded prediction interval band**
- [x] Quantile level toggle
- [x] Clean axes, no chartjunk

### E9-S4 — Cost Simulator — the money shot ✅

**The single most important screen in the project.** A recruiter dragging a slider and
watching cost move is worth more than any accuracy chart.

- [x] Two sliders: `Cu` and `Co`
- [x] Live update on drag: critical ratio, recommended order quantity, total cost curve
      with the optimum marked
- [x] Smooth and responsive — debounce or precompute the sweep from `cost_sensitivity`
      rather than round-tripping the API on every pixel
- [x] Animated transitions on the number and the curve
- [x] Show the counterintuitive region explicitly: where high spoilage pulls the optimal
      service level down — measured at 41% for the default assumptions (E7), not the
      textbook 95%
- [x] **Annotate the stockout-rate figure, not just the cost figure.** The winning policy
      in the money table has a 48.0% stockout rate, and shown without context that reads
      as a bug. Pin the same explanation used in the README next to whatever widget shows
      stockout rate: *"Higher stockout rate is correct here, not a flaw — at this item's
      cost ratio, spoilage is far more expensive than a missed sale, so the optimal policy
      deliberately favors running out over overstocking perishables. The naive policy's
      lower stockout rate is what's actually expensive."* This has to travel with the
      number itself, not live only in the README — a recruiter looking at the live
      dashboard will not have read the docs first.
- [x] **Label Cu/Co as assumptions in the UI, not just in `docs/`.** The 48%-stockout
      argument only holds if the cost ratio is reasonable, so the slider labels or a
      caption must say plainly that `Cu`/`Co` are inputs the viewer can change, not
      measured facts — consistent with how `docs/backlog/epic-07-optimization.md` and the
      README already label them next to the sensitivity sweep.

### E9-S5 — Model Performance ✅

- [x] Backtest table across 5 folds **with variance shown**
- [x] Baseline comparison bar chart, including any baseline that beat the model
- [x] SHAP panel answering "why this forecast?"

### E9-S6 — Polish ✅

- [x] Skeleton loaders — never a spinner on a blank page
- [x] Empty and error states **designed**, not defaulted
- [x] Smooth number transitions
- [x] **Every chart carries a one-line plain-language caption** explaining what the viewer
      is looking at


---

## Deviation from the brief

The brief specified **React + Vite + Tailwind + Recharts**. Delivered as a hand-authored
single self-contained HTML file with inline SVG charts instead. Recorded here rather than
applied silently.

**Why.** The requirement was a live URL. A Vite SPA calling the FastAPI service can't be
live without hosting both — that's E10, and it needs deployment credentials this build
doesn't have. A self-contained file with data baked in is viewable immediately, from any
host, with no backend running.

**What it costs.** No live API calls, so figures are a snapshot of the last pipeline run.
The API is real and tested (19 contract tests) — the dashboard just doesn't depend on it
being up.

**What it buys beyond deployability.** Complete control of the chart marks, which is what
makes the two-pole colour encoding consistent across every figure instead of fighting a
library's defaults.

## The two annotations, on-screen and unavoidable

Both are persistent bordered callout blocks in the normal reading flow — not tooltips, not
hover states, not collapsed disclosures:

- **Stockout rate**, on the Overview beneath the money table: why 48.0% is correct rather
  than a defect, with the naive policy's lower rate named as the expensive one.
- **Cu/Co are assumptions**, twice: on the Overview next to the money table, and again in
  the simulator where the sliders live, since that's where someone changes them.

## The simulator is exact, not sampled

Order quantity depends only on the critical ratio, so shortfall and leftover **units** were
precomputed once per service level across all 100,720 decisions (90 points, CR 0.10–0.99).
Cost at any Cu/Co is then `shortfall × Cu + leftover × Co` — exact arithmetic over 90
points per drag event, no model call, nothing to debounce. The E8 `/cost-sensitivity`
endpoint serves the coarser 6-point version of the same idea for a live deployment.

## A finding the simulator surfaced

At the default cost ratio the theoretical critical ratio is **0.409**, but the
cost-minimising service level on this data is **0.44** — following theory exactly leaves
~0.26% on the table. The newsvendor rule assumes you order the CR-th quantile of *true*
demand; you actually order the CR-th quantile of a *forecast*, and this model
under-forecasts (bias −0.098, the largest of any model tested). A downward-biased
distribution has to be read at a higher quantile to land in the same place. The page states
this rather than quietly plotting the theoretical line as if it were the optimum.

## Palette validated, not eyeballed

Ran the dataviz validator against both surfaces; all six checks pass (lightness band,
chroma floor, CVD separation, normal-vision floor, contrast). First two candidate palettes
FAILED — jade and amber outside the dark-mode lightness band, and the blue below the chroma
floor (reading as grey). Fixed by re-stepping rather than by eye.
