# E9 — Dashboard ⬜

**Goal:** the portfolio centrepiece. React + Vite + Tailwind + Recharts — not Streamlit.

**Consumes:** the E8 API.

**Produces:** a deployable static front-end.

**Design direction:** dark analytical theme. Deep slate/charcoal base, a single confident
accent (emerald or amber) used sparingly for emphasis and positive deltas. Generous
whitespace. Real typographic hierarchy — a display face for numbers with
`font-variant-numeric: tabular-nums`, clean sans for body. Subtle borders over heavy
shadows. No gradient soup, no glassmorphism, no purple-blue SaaS template. Think Linear
or Vercel analytics, not a bootcamp dashboard.

---

### E9-S1 — App shell and design system ⬜

- [ ] Vite + React + TypeScript + Tailwind
- [ ] Design tokens: colour, type scale, spacing — defined once, used everywhere
- [ ] Tabular figures on every number that changes
- [ ] Keyboard navigable; responsive down to tablet

### E9-S2 — Overview screen ⬜

- [ ] Hero stat: total money saved. Big, confident, unmissable.
- [ ] Below: stockout rate, waste units, forecast accuracy — each with a delta chip vs baseline
- [ ] Sparkline per metric
- [ ] Currency toggle: USD ⇄ ₹, with the FX rate shown inline (M5 is US data; the rate is
      a display convenience, never baked into the stored number)

### E9-S3 — Forecast Explorer ⬜

- [ ] Hierarchy drill-down `State → Store → Dept → SKU` as breadcrumb or tree sidebar
- [ ] Actual history line + forecast line + **shaded prediction interval band**
- [ ] Quantile level toggle
- [ ] Clean axes, no chartjunk

### E9-S4 — Cost Simulator — the money shot ⬜

**The single most important screen in the project.** A recruiter dragging a slider and
watching cost move is worth more than any accuracy chart.

- [ ] Two sliders: `Cu` and `Co`
- [ ] Live update on drag: critical ratio, recommended order quantity, total cost curve
      with the optimum marked
- [ ] Smooth and responsive — debounce or precompute the sweep from `cost_sensitivity`
      rather than round-tripping the API on every pixel
- [ ] Animated transitions on the number and the curve
- [ ] Show the counterintuitive region explicitly: where high spoilage pulls the optimal
      service level down — measured at 41% for the default assumptions (E7), not the
      textbook 95%
- [ ] **Annotate the stockout-rate figure, not just the cost figure.** The winning policy
      in the money table has a 48.0% stockout rate, and shown without context that reads
      as a bug. Pin the same explanation used in the README next to whatever widget shows
      stockout rate: *"Higher stockout rate is correct here, not a flaw — at this item's
      cost ratio, spoilage is far more expensive than a missed sale, so the optimal policy
      deliberately favors running out over overstocking perishables. The naive policy's
      lower stockout rate is what's actually expensive."* This has to travel with the
      number itself, not live only in the README — a recruiter looking at the live
      dashboard will not have read the docs first.
- [ ] **Label Cu/Co as assumptions in the UI, not just in `docs/`.** The 48%-stockout
      argument only holds if the cost ratio is reasonable, so the slider labels or a
      caption must say plainly that `Cu`/`Co` are inputs the viewer can change, not
      measured facts — consistent with how `docs/backlog/epic-07-optimization.md` and the
      README already label them next to the sensitivity sweep.

### E9-S5 — Model Performance ⬜

- [ ] Backtest table across 5 folds **with variance shown**
- [ ] Baseline comparison bar chart, including any baseline that beat the model
- [ ] SHAP panel answering "why this forecast?"

### E9-S6 — Polish ⬜

- [ ] Skeleton loaders — never a spinner on a blank page
- [ ] Empty and error states **designed**, not defaulted
- [ ] Smooth number transitions
- [ ] **Every chart carries a one-line plain-language caption** explaining what the viewer
      is looking at
