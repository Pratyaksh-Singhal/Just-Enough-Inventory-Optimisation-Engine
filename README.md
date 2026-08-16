# Inventory Optimization Engine

Hierarchical demand forecasting and newsvendor inventory optimization on M5 retail data.

**Forecasting accuracy is not the deliverable. Money saved is.** This project forecasts
SKU-level demand, reconciles those forecasts across a store hierarchy so they stay
coherent, converts the forecast *distribution* into optimal stocking quantities via
newsvendor theory, and reports the cost delta against baseline stocking practice.

> **Build status:** Phases 1–9 complete — data foundation through a live dashboard.
> Phase 10 (deploy) in progress.
>
> **Live dashboard → https://claude.ai/code/artifact/ab291272-dc60-496b-88cd-fe3f5c57ce2f**

**The strongest evidence of engineering judgment here is the record of what went wrong:**
[**`docs/what-went-wrong.md`**](docs/what-went-wrong.md) — decisions this project got wrong
the first time, none of which crashed, and every one of which returned a confident,
plausible, wrong answer.

---

## Headline result

**28.6% lower inventory cost than current practice** — and, more interestingly, **a flat
95% service level costs 2.5× more than doing nothing sophisticated at all.**

On 100,720 stocking decisions priced against realised demand, the cost-optimal service
level for a thin-margin perishable is **41%**, not 95%. Most of that saving comes from the
*policy*, not the forecast: switching policy at a fixed forecast cuts cost 71%, while
switching forecast at a fixed policy cuts it 18%.

---

## The money table

Assumptions: 30% gross margin, 60% spoilage, 2%/day holding → **critical ratio 0.409**.
100,720 (series × day) decisions across 5 rolling-origin folds, priced against **realised**
demand — never against the forecast that produced the order.

| Policy | Stockout rate | Waste units | Holding cost | Total cost | vs naive |
|---|---:|---:|---:|---:|---:|
| Current (naive: last week's sales) | 31.1% | 78,106 | $3,198 | $169,172 | — |
| Fixed 95% service level | 6.1% | 326,957 | $13,118 | $415,644 | **+145.7%** |
| **Newsvendor + our forecast** | 48.0% | **20,714** | **$801** | **$120,797** | **−28.6%** |

The middle row is the point. It achieves the **best stockout rate in the table** and is by
far the most expensive, because hitting 95% requires 327k units of stock that spoil. The
newsvendor policy accepts a *higher* stockout rate than doing nothing (48% vs 31%) and
still costs 28.6% less, because it cuts waste by 73%. On thin-margin perishables an empty
shelf is cheaper than a full bin.

> **The winning policy has a 48.0% stockout rate. That is correct, not a defect.** At this
> item's cost ratio, spoilage is far more expensive than a missed sale, so the optimal
> policy deliberately favors running out over overstocking a perishable. The naive
> policy's *lower* stockout rate (31.1%) is what's actually expensive — it is bought with
> 78k wasted units instead of 21k. Read the stockout-rate column against the total-cost
> column, not on its own.

This argument only holds if the cost ratio behind it is reasonable, so it is worth
repeating here rather than only in the sensitivity section below: **every cost figure in
this table is a stated assumption, not a measured fact.** M5 ships shelf prices, not
margins, spoilage rates or holding costs — the 30% margin, 60% spoilage and 2%/day holding
that produce `CR = 0.409` (see Phase 7 below) are inputs a reader should be free to
disagree with. The sensitivity sweep exists precisely so the 48%-is-correct conclusion can
be checked rather than taken on trust: newsvendor beats the alternatives at every spoilage
rate from 0.0 to 1.0, so the argument does not depend on having picked the assumptions
that flatter it.

All cost figures are USD.

---

## Approach

Baselines → global GBM → hierarchical reconciliation → newsvendor optimization.
Each stage is measured against the one before it, and a baseline that wins is reported
as a win rather than tuned away.

Work is tracked as ten epics in [`docs/BACKLOG.md`](docs/BACKLOG.md). Phases communicate
through named DuckDB tables rather than through code, so each epic states what it consumes
and produces and can be picked up without reading the one before it. Branching follows
[`docs/BRANCHING.md`](docs/BRANCHING.md): `dev` for integration, `main` for phase-complete
work, one tag per phase. Decisions that were wrong the first time — and what the wrong
version actually produced on real data — are recorded in
[`docs/what-went-wrong.md`](docs/what-went-wrong.md).

| Phase | Epic | Status |
|---|---|---|
| 1. Data foundation (DuckDB warehouse) | [E1](docs/backlog/epic-01-data-foundation.md) | ✅ Complete |
| 2. Features + leakage test | [E2](docs/backlog/epic-02-features.md) | ✅ Complete |
| 3. Baselines (Seasonal Naive, Croston/TSB, ETS/AutoARIMA) | [E3](docs/backlog/epic-03-baselines.md) | ✅ Complete |
| 4. LightGBM global model + MLflow | [E4](docs/backlog/epic-04-global-model.md) | ✅ Complete |
| 5. Rolling-origin backtest harness | [E5](docs/backlog/epic-05-backtest.md) | ✅ Complete |
| 6. MinT hierarchical reconciliation | [E6](docs/backlog/epic-06-reconciliation.md) | ✅ Complete |
| 7. Newsvendor optimization layer | [E7](docs/backlog/epic-07-optimization.md) | ✅ Complete |
| 8. FastAPI service | [E8](docs/backlog/epic-08-api.md) | ✅ Complete |
| 9. Dashboard | [E9](docs/backlog/epic-09-dashboard.md) | ✅ Complete |
| 10. Deploy | [E10](docs/backlog/epic-10-ship.md) | 🔜 Next |

---

## Phase 1 — data foundation

### Scoping: 720 series, chosen on purpose

The full M5 dataset is 30,490 series. Modelling all of it turns every backtest into an
overnight job and buries the optimization work — which is the actual point of this
project — under data engineering. So the scope is deliberately cut:

- **State `CA`, category `FOODS`, all 4 stores.** Fresh food is where the newsvendor
  model earns its keep: spoilage makes overstock cost nearly the full unit cost, which
  is what drives the counterintuitive service-level result in Phase 7.
- **60 items per department, all 3 FOODS departments** → 180 items × 4 stores = **720 series**.

Keeping all three departments matters. Cutting to a single department would have hit the
same series count with less work, but it collapses the `Dept` level of the hierarchy to a
single node — and Phase 6's MinT reconciliation would lose an entire level to reconcile
across. Breadth of hierarchy was worth more than simplicity here.

### Why the sample is stratified, not top-N

The obvious way to pick 60 items per department is by sales volume. That would have been
a mistake: **the densest SKUs are the easy ones.** Selecting on volume quietly removes the
intermittent demand that Phase 3's Croston/TSB models exist to handle, and would leave
this project claiming to solve a problem it had engineered out of the dataset.

Instead, each department's items are split into three equal-count bands by their share of
zero-sales days, and 20 items are drawn from each band:

| Dept | Stratum | Items | Zero-day share (min–max) | Mean |
|---|---|---:|---|---:|
| FOODS_1 | dense | 20 | 3%–44% | 30% |
| FOODS_1 | mid | 20 | 45%–61% | 54% |
| FOODS_1 | sparse | 20 | 64%–99% | 81% |
| FOODS_2 | dense | 20 | 9%–42% | 28% |
| FOODS_2 | mid | 20 | 47%–63% | 55% |
| FOODS_2 | sparse | 20 | 64%–97% | 75% |
| FOODS_3 | dense | 20 | 1%–37% | 21% |
| FOODS_3 | mid | 20 | 39%–59% | 49% |
| FOODS_3 | sparse | 20 | 62%–100% | 76% |

The resulting panel is 61.6% zero-sales days overall, with sparse-stratum series reaching
99–100% zeros. The intermittent-demand problem is fully present in the sample.

**The trade-off, stated plainly:** this panel is *not* representative of CA FOODS by
volume. Stratifying by intermittency over-weights slow movers relative to their share of
revenue. Aggregate rupee/dollar results should be read as "cost behaviour on a
demand-diverse basket", not as an extrapolation to the full category. A volume-weighted
re-run is listed under next steps.

Two details that make the sample defensible:

1. **Intermittency is measured before the evaluation period.** The 90-day window used to
   stratify ends 140 days before the panel does — i.e. before the first backtest fold
   begins. Choosing *which SKUs to model* using statistics computed over the evaluation
   period is selection leakage, even though scoping happens once and offline.
   `test_intermittency_is_measured_before_the_backtest_region` pins this: a synthetic
   series that is silent throughout the reference window and dense throughout the
   backtest region must be graded maximally sparse.
2. **Selection order is a hash, not a PRNG.** Items are ordered by
   `hash(item_id || seed)` rather than `random()`. A seeded PRNG's output can shift
   across DuckDB versions, platforms and thread counts; a hash cannot. The sampled
   universe is byte-identical on every rebuild, on any machine.

Items with no price history in the reference window are excluded before stratification —
an item that was never on shelf is 100% zero-days for reasons that have nothing to do
with demand, and would otherwise pollute the sparse stratum.

### Warehouse

Long format in DuckDB, exactly as specified:

```
fact_sales(date, item_id, store_id, dept_id, cat_id, state_id, units, price, snap, event)
```

1,397,520 rows · 720 series · 2011-01-29 → 2016-05-22 · builds in ~30s.

A second table, `dim_calendar`, holds the **full** M5 calendar including the 28 days that
extend past the end of the sales data. Phase 2 needs those future rows to compute
"days until next event" for observations near the end of the training window. This is the
one place where reading ahead of *t* is legitimate — event calendars are genuinely known
in advance — and it is isolated in its own table so that fact.

Two implementation notes:

- **The ETL is pure DuckDB SQL; there is no pandas in the loading path.**
  `sales_train_evaluation.csv` is a 122 MB wide table of 1,947 columns. Melting it in
  pandas means materialising the whole frame in RAM before the scope filter can shrink it.
  DuckDB pushes the filter down to the CSV scan and streams the `UNPIVOT`, so peak memory
  stays flat.
- **The target table is created from explicit DDL with `NOT NULL` and primary key
  constraints, then inserted into.** The insert doubles as a data-quality assertion: if
  the calendar join fans out, or a series has duplicate `(date, item, store)` rows, the
  load fails loudly instead of producing a panel every downstream phase would misread.

`price IS NULL` (22.7% of rows) means the item was not listed at that store that week —
it is an availability signal, not missing data, and Phase 2 treats it as such.

---

## Phase 2 — features, and the leakage gate

27 features over 1,080,714 rows, built in ~3.5s. Full dictionary in
[`docs/features.md`](docs/features.md).

### The rule, and how it is enforced

**No feature on a row targeting date _t_ may use any observation from _t_ or later.**

Because this project forecasts 28 days ahead, that rule is stronger than "not today". A
forecast for _t_ is made at origin `t0 = t − 28`, so the newest actual anyone could have
seen is the one at `t0`. Every units-derived feature is computed on the series shifted by
the horizon.

The shift is not applied per feature and remembered — that approach fails the first time
someone adds a feature in a hurry. It is folded into the SQL window frame by exactly two
functions, and **every** units-derived feature is emitted by one of them. There is no
other code path in the module that reads `units`, so a new lag or rolling feature cannot
be added without inheriting the shift.

### The test is a perturbation probe, not a checklist

`tests/test_no_leakage.py` does not assert anything about individual feature definitions.
It corrupts the actuals at and after some date `C`, rebuilds the panel, and asserts every
feature targeting `t < C + horizon` is bit-identical. That argument holds for **any**
feature, including ones added later by someone who never read the test. A per-feature
assertion would only ever cover the features somebody remembered to write an assertion for.

Two supporting tests keep it honest:

- **Sensitivity** — past the boundary, every units-derived feature *must* move. Without
  this, a builder that emitted constants would pass the gate by doing nothing.
- **Exact boundary** — corrupt a single cell at date `D`. A rolling feature targeting
  `D + 28` reads exactly up to `D` and must change; the same feature targeting `D + 27`
  must not. This pins the frame to `horizon PRECEDING` rather than merely "somewhere in
  the past", which is where off-by-one errors live.

### The one deliberate exception

Calendar, event, SNAP and price features are **not** shifted. A retailer genuinely knows
next month's calendar and its own future shelf prices at forecast time; refusing to use
them would model a business that does not exist.

But that exception is also exactly how leakage gets in by accident, so it is fenced rather
than assumed: the allowlist is pinned by a test against a hard-coded set. Widening it is a
deliberate, reviewable edit — adding a column forces someone to justify why the retailer
really does know it in advance. The perturbation gate corrupts `units`, so it structurally
cannot catch a leak introduced through a non-`units` column; the pinned allowlist is what
covers that gap.

### Panel decisions

- **Pre-listing rows dropped** — 316,806 rows (22.7%) precede the date an item was first
  priced at a store. Those are not zero demand, they are the absence of a product.
  Training on them teaches the model to forecast shelf gaps. Post-introduction gaps are
  kept, flagged `is_listed = false`.
- **Rectangularity is checked, not assumed.** Row-based window frames only equal day-based
  frames on a dense panel. One missing day would silently make every frame reach further
  back in time than intended, and nothing downstream would notice.
- **Partial windows aggregate what exists** rather than returning NULL until full. An
  early `units_roll_mean_90` is a mean of fewer than 90 days — the column name is a
  maximum, not a guarantee. Deliberate, and pinned by a test.

---

## Phase 3 — baselines, and a real finding

Four baselines, scored by the rolling-origin harness built alongside them (E5-S1..S5 — see
the ordering note in [`docs/BACKLOG.md`](docs/BACKLOG.md)): 5 folds × 28-day horizon ×
720 series.

| Model | MASE | RMSSE | Bias |
|---|---:|---:|---:|
| **AutoETS** | **1.024** ± 0.026 | **0.763** ± 0.023 | −0.042 ± 0.065 |
| TSB | 1.045 ± 0.026 | 0.778 ± 0.024 | +0.017 ± 0.078 |
| Croston | 1.112 ± 0.012 | 0.798 ± 0.009 | +0.111 ± 0.057 |
| Seasonal naive | 1.204 ± 0.034 | 0.989 ± 0.029 | −0.063 ± 0.086 |

All four clear seasonal naive comfortably — the honest bar is doing its job.

### The finding: the intermittent-demand specialists lose on their own turf

**Croston loses to plain seasonal naive on the sparse stratum** (MASE 1.127 vs 1.107),
and **AutoETS — a classical method with no special handling for zero-inflated demand —
beats both Croston and TSB across every stratum, sparse included.** This is the opposite
of the textbook expectation, and it is reported as found rather than tuned away, per this
project's own ground rule.

| Stratum | AutoETS | TSB | Croston | Seasonal naive |
|---|---:|---:|---:|---:|
| dense | 1.042 | 1.068 | 1.076 | 1.284 |
| mid | 1.037 | 1.056 | 1.134 | 1.221 |
| sparse | **0.993** | 1.009 | 1.127 | 1.107 |

Two things explain most of it:

1. **Croston and TSB forecast a flat rate** — "probability of a sale" × "size when it
   sells" — constant across the whole 28-day horizon. This panel has real day-of-week
   seasonality (weekend uplift, SNAP-day effects) that a flat-rate method cannot
   represent at all, and that AutoETS's seasonal component captures directly.
2. **TSB earns its inclusion over Croston, just not enough to beat AutoETS.** TSB was
   added specifically because Croston never decays its estimate through a long zero run,
   and the data confirms that reasoning — TSB beats Croston on every stratum and every
   fold. It just isn't enough on its own to also model the weekly cycle.

**Croston's bias is the more consequential number for Phase 7.** +0.11 mean, worst fold
+0.17, is a *systematic* over-forecast, and for fresh food — where overstock cost runs
close to 100% of unit cost — that bias is spoilage, priced directly into whatever policy
uses it. TSB's bias is close to zero (+0.02); AutoETS's is slightly negative (−0.04, a
mild under-forecast). Bias sign and magnitude will matter as much as MASE when Phase 7
picks which forecast the newsvendor layer trusts.

**Consequence for Phase 4:** the LightGBM global model's bar to clear is AutoETS at
MASE 1.024, not the intermittent-demand baselines. If it wins mainly on the sparse
stratum specifically, that's the more interesting result than an aggregate win.

Full breakdown, including the TSB smoothing parameters (a stated assumption — statsforecast
has no optimised TSB variant) and the excluded-series accounting, in
[`docs/backlog/epic-03-baselines.md`](docs/backlog/epic-03-baselines.md).

---

## Phases 4–5 — global model and backtest harness

One LightGBM model across all 720 series, series identity as a feature. 5 folds ×
28-day horizon.

| Model | MASE | RMSSE | Bias |
|---|---:|---:|---:|
| **lgbm** | **1.0192** ± 0.0182 | **0.7576** ± 0.0212 | −0.0982 ± 0.0348 |
| ets | 1.0242 ± 0.0255 | 0.7626 ± 0.0227 | −0.0422 ± 0.0650 |
| tsb | 1.0446 ± 0.0260 | 0.7776 ± 0.0237 | +0.0166 ± 0.0776 |
| croston | 1.1122 ± 0.0123 | 0.7976 ± 0.0085 | +0.1106 ± 0.0571 |
| seasonal_naive | 1.2042 ± 0.0341 | 0.9890 ± 0.0294 | −0.0633 ± 0.0856 |

**The honest reading: LightGBM matches a well-tuned classical method rather than beating
it.** The 0.005 MASE margin over AutoETS sits well inside a fold-to-fold std of 0.018–0.026,
and AutoETS wins outright on fold 0. What the GBM buys is not a large accuracy jump — it is
one model instead of 720 fits, quantile forecasts the classical models don't provide
directly, and SHAP attributions E7 can use to defend an ordering decision.

**Why not deep learning:** at 720 series of daily data a global GBM matches classical
methods, trains in ~13 minutes on CPU, and is directly inspectable. An LSTM or N-BEATS here
would be a larger, slower, less explainable model fitted to less data than it wants.

**LightGBM is the most under-forecasting model in the panel** (bias −0.098, worse than
seasonal naive). Tweedie's point estimate leans toward the mode on data that is 61.6%
zeros. The consequence for Phase 7 is concrete: use the quantiles, not the point forecast.

### Quantile crossings, and where they matter

LightGBM fits each quantile level independently, so nothing forces them to be ordered.
1.74% of rows came out crossed overall — and **1.29% inside the CR 0.5–0.95 band the
newsvendor rule actually selects from**, where a crossing returns a *lower* stocking
quantity for a *higher* service level. Silently wrong, in the direction that causes
stockouts.

Fixed by rearrangement at **read** time, so stored forecasts keep the raw values and the
crossing rate stays auditable. Crossing rate in the CR band: **1.2887% → 0.0000%**.

Sorting rather than PAVA isotonic regression is deliberate: Chernozhukov et al. (2010)
prove rearrangement weakly reduces pinball loss, whereas PAVA minimises squared deviation
from the raw predictions with no guarantee for the loss these are judged by. Verified —
pinball improves at q0.5/q0.9/q0.99 and is worse by 5e-6 at q0.95, reported rather than
rounded into a clean win.

### Stratum-aware routing — investigated, not adopted

The per-stratum breakdown surfaced a genuine finding: **AutoETS beats LightGBM on the
sparse stratum** (MASE 0.9931 vs 1.0090), and beat Croston and TSB there too — their own
supposed specialty.

| Stratum | lgbm | ets | tsb | croston | seasonal_naive |
|---|---:|---:|---:|---:|---:|
| dense | **1.0155** | 1.0421 | 1.0679 | 1.0757 | 1.2843 |
| mid | **1.0328** | 1.0371 | 1.0564 | 1.1340 | 1.2209 |
| sparse | 1.0090 | **0.9931** | 1.0091 | 1.1270 | 1.1068 |

The mechanism is legible: a global GBM learns one function across all series, and the dense
two-thirds dominate the gradient, so its fitted response is tuned to series with regular
structure. On a series selling fewer than one day in five, the lag and rolling features it
leans on are mostly zeros; AutoETS's per-series fit adapts to that specific history.

So a routing hybrid — AutoETS on sparse, LightGBM on dense/mid — was **built and measured**
(`models/hybrid.py`, retained and tested). It improves headline MASE to 1.0139. **It was
not adopted**, for four reasons:

1. It **loses to plain LightGBM on folds 3 and 4** — the two most recent windows, and the
   closest analogue to production. It wins 3 folds, loses 2, with higher variance
   (std 0.0244 vs 0.0182).
2. It is **worse on RMSSE** (0.7585 vs 0.7576).
3. It is **worse on pinball loss at every quantile level** — the distributional metric
   Phase 7 actually consumes.
4. MASE and RMSSE **disagree** on the sparse band: AutoETS is better on typical error,
   LightGBM better on large errors.

That last disagreement can't be settled by preferring one abstract metric. It turns on
whether occasional large misses cost more than routine small ones — which is precisely what
Phase 7's newsvendor cost function decides. Adopting routing now would bake a structural
choice on a metric guess three phases before the evidence exists.

**Deferred to Phase 7 (E7-S6):** once the cost function exists, rerun the dense/mid/sparse
comparison on actual cost delta instead of error-metric proxies, and adopt routing only if
the money justifies the complexity. Perishables skew sparse, so that is where it would pay
if it pays at all. If it doesn't, that gets recorded too — a rejected optimisation with
numbers attached is a result.

**Carried into Phase 6: plain LightGBM with isotonic-sorted quantiles.**

---

## Phase 6 — MinT reconciliation

Forecast each level of a retail hierarchy independently and the numbers don't add up. The
720 SKU forecasts don't sum to the store forecasts; the store forecasts don't sum to the
state forecast. That isn't a rounding nuisance — a replenishment planner and a regional
manager read the same system and get contradictory answers, and only one can be acted on.

Four-level tree over **737 series** (1 state + 4 stores + 12 store×dept + 720 item×store),
reconciled with MinT via Nixtla `hierarchicalforecast`.

### Coherence: proven, not asserted

| Parent → child | Mean gap before | Max gap before | Max gap after |
|---|---:|---:|---:|
| state → store | 19.77 | 42.76 | 4.55e-13 |
| store → store_dept | 4.57 | 23.01 | 2.84e-13 |
| store_dept → item_store | 9.00 | 37.62 | 8.53e-14 |

Reconciled forecasts sum exactly, to floating-point tolerance. Asserted in tests, not just
printed.

### WRMSSE — now computable

**0.81173 → 0.78411** after reconciliation, a **3.4% improvement** on the M5 official
metric. This is the number Phase 5 deliberately declined to report, because WRMSSE weights
RMSSE across the aggregation hierarchy and that hierarchy didn't exist yet.

### The trade-off, and its price

| Level | RMSSE base | RMSSE MinT | MASE base | MASE MinT |
|---|---:|---:|---:|---:|
| state | 0.7144 | **0.6862** | 0.7696 | **0.7183** |
| store | 0.8026 | **0.7807** | 0.8121 | **0.7931** |
| store_dept | 0.9263 | **0.8635** | 0.9809 | **0.9086** |
| item_store | **0.7576** | 0.7615 | **1.0192** | 1.0348 |

**Every aggregate level improves; the bottom level gets ~1.5% worse on MASE.** That is the
expected MinT trade-off and the price is stated rather than buried — coherence is purchased,
not free.

### Which forecast feeds which consumer

That trade-off makes "use the latest forecast" the wrong default, so the routing is decided
explicitly:

| Consumer | Forecast | Why |
|---|---|---|
| **Newsvendor / order quantities** (Phase 7) | **Base, unreconciled**, `item_store` | Orders are placed at `item_store`. Every unit of stockout and spoilage is realised at that grain, so the cost function must optimise against the most accurate forecast *there* — which is the unreconciled one. Coherence with the store total buys nothing when deciding how many units of one SKU go on one shelf. |
| **Aggregate & planning views** (Phase 9) | **MinT-reconciled** | Here coherence *is* the requirement. A regional plan that disagrees with the sum of its stores is unusable regardless of how accurate either number is on its own. |

Both live in the `forecast` table, discriminated by the `reconciled` column, so neither
consumer has to know how the other's forecast was produced. `USE_RECONCILED = False` is a
named constant in the optimizer rather than an implicit default, and
`test_newsvendor_consumes_unreconciled_forecasts` pins it — so switching it becomes a
deliberate, reviewable edit rather than a silent regression to whatever ran last.

### Two deviations, both recorded

**The Category level is dropped.** The brief specifies `State → Store → Category → Dept →
Item`, but Phase 1's scope fixes `cat_id = FOODS`, making Category identical to Store.
Including it would put duplicate rows in the summing matrix, leaving the covariance
rank-deficient — a numerical problem created purely by encoding a level carrying no
information. On a multi-category scope it restores unchanged.

**`wls_struct` instead of `mint_shrink`.** The shrinkage estimator is the textbook default
and was the original choice, but it estimates the residual covariance from *in-sample*
one-step residuals — requiring fitted values for all 737 series in every fold. Phase 4
persists forecasts, not models. `wls_struct` weights each node by the number of bottom
series aggregated into it, using `S` alone, which is exactly the case Wickramasuriya et al.
propose it for when residuals are unavailable.

---

## Phase 7 — the optimization layer

Full detail in [`docs/backlog/epic-07-optimization.md`](docs/backlog/epic-07-optimization.md).
The money table is at the top of this README; three findings behind it are worth pulling out.

### Optimal service level is 41%, not 95%

`CR = Cu / (Cu + Co)`. Understocking costs the lost *margin* — the retailer keeps the unit
it didn't sell. Overstocking a perishable costs close to the whole *unit cost*. With a 30%
margin and 60% spoilage:

```
Cu = price x 0.30              = 0.30
Co = price x 0.70 x 0.62       = 0.434
CR = 0.30 / 0.734              = 0.409
```

Ordering to a 95th percentile on a thin-margin perishable is not caution; it is a
systematic and expensive error, and the money table prices it at +145.7%.

### Sensitivity: the result holds everywhere, and 95% almost never does

| Spoilage | CR | naive | fixed 95% | newsvendor |
|---:|---:|---:|---:|---:|
| 0.0 | 0.955 | 73,225 | 22,094 | **21,756** |
| 0.2 | 0.661 | 105,207 | 153,277 | **86,942** |
| 0.4 | 0.505 | 137,190 | 284,461 | **108,668** |
| 0.6 | 0.409 | 169,172 | 415,644 | **120,797** |
| 0.8 | 0.343 | 201,155 | 546,827 | **128,709** |
| 1.0 | 0.296 | 233,138 | 678,011 | **133,417** |

Newsvendor wins at **every** spoilage rate, so the finding isn't an artifact of one
assumption. **Fixed-95% beats naive only at spoilage ≈ 0** — for any real perishable it is
worse than having no system at all, and at total spoilage it costs 2.9× as much.

### The policy is worth more than the model

| Combination | Total cost |
|---|---:|
| naive forecast + fixed 95% | 800,704 |
| naive forecast + newsvendor CR | 147,688 |
| our forecast + fixed 95% | 415,644 |
| **our forecast + newsvendor CR** | **120,797** |

Holding the forecast fixed and fixing the *policy* cuts cost **71%**. Holding the policy
fixed and improving the *forecast* cuts it **18%**. Most of the value here comes from
asking what service level is actually optimal — not from forecasting better. A team with a
mediocre forecast and the right cost model beats a team with a good forecast and a 95%
target.

### Stratum-aware routing: rejected on cost

The alternative deferred from Phase 4 was finally settled with money rather than a metric
preference. Rebuilt with the full quantile grid and priced, routing costs **+3.58%** — and
is **17.7% worse on the sparse band it was supposed to help.**

Two reasons, both findings in their own right. **AutoETS wins the point forecast and loses
the distribution**: its Gaussian prediction intervals are badly mis-specified for
intermittent count demand, producing 4.5× the waste on sparse series. The newsvendor
consumes the distribution, so a better MASE was simply the wrong qualification. And **the
premise was wrong about where cost lives** — routing was motivated by "perishables skew
sparse", but sparse carries only 20.2% of total cost while dense carries 49.9%. Routing
optimises the smallest pool.

---

## Phase 8 — the API

Full detail in [`docs/backlog/epic-08-api.md`](docs/backlog/epic-08-api.md).

Seven endpoints (`/health`, `/forecast`, `/optimize`, `/backtest`, `/hierarchy`,
`/cost-comparison`, `/cost-sensitivity`), all reads. `test_no_handler_imports_a_trainer`
checks "never train on request" statically — the app module can't import `lightgbm`,
`statsforecast`, `hierarchicalforecast` or `mlflow`, so a handler that started calling
training code would fail a test before it could ship.

### DuckDB's single-writer model decided the connection design, not the other way round

A writer needs the file to itself — no concurrent readers, no concurrent writers. The API
never holds a long-lived connection; every request opens and closes a fresh
`read_only=True` handle. That leaves a write window between requests, but "usually open" is
not "safe" — especially on Windows, where a rename over an open file handle can fail
outright. So the nightly refresh job (E8-S4) never writes to the live file at all: it copies
the warehouse to a shadow path, runs the full pipeline against the copy, and atomically
swaps it in with `os.replace`, retrying a few times against transient contention. Any step
failing, or every swap attempt failing, discards the shadow and leaves the live file
untouched — the job is idempotent and a failure can never corrupt what's already there.

### Latency (in-process, 200 requests per endpoint)

| Endpoint | p50 | p95 | p99 |
|---|---:|---:|---:|
| `GET /health` | 17.9ms | 20.0ms | 22.4ms |
| `POST /forecast` | 26.3ms | 29.7ms | 31.1ms |
| `POST /optimize` | 26.1ms | 28.0ms | 29.9ms |

The floor is DuckDB connection overhead (~15–20ms, visible in `/health`) — the accepted
cost of never holding a connection that could block the nightly writer.

### Two real bugs, both caught by the fixture suite rather than the happy path

**`/optimize` mixed 28 days of quantile curves into one array per store.** With no date
filter, fetching every horizon for a SKU pulled 28 days × 7 quantile levels × 4 stores into
one group, sorted by quantile value with duplicated keys — output that looked entirely
plausible (sane-looking order quantities) over a demand distribution that was structurally
meaningless. Found by checking a row count (784, expected) against what a single coherent
distribution should have (7), not by the numbers looking wrong. Fixed by pinning the
endpoint to a single period — `horizon = 1`, "tomorrow" relative to the latest completed
fold, the classic newsvendor framing.

**An extreme Cu/Co ratio 500'd instead of 422'ing.** `interpolate_quantile` correctly
refuses to extrapolate past the fitted grid, but the handler didn't catch that error. A
test written specifically to probe an edge the happy-path test never reaches caught it
before deploy.

**`/forecast` was serving raw, potentially crossed quantiles.** E4/E5 established that
~1.3% of fitted quantile rows are crossed inside the newsvendor's CR band, and built
`monotonize()` as the read-time fix every consumer must apply. The first version of this
endpoint queried `forecast` directly and forgot it. A fixture test plants a deliberately
crossed grid so this can't regress silently.

---

## Phase 9b — tier 2: the Full Forecast service

Everything above is **tier 1**: a fixed M5 panel, a nightly pipeline, a read-only API over
precomputed results, and a dashboard with the numbers baked in. It is unchanged by any of
what follows.

**Tier 2** takes a user's own CSV and forecasts it. Separate package
(`inventory_engine.service`), separate database, separate port.

| | tier 1 | tier 2 |
|---|---|---|
| data | M5, curated | whatever the user uploads |
| store | DuckDB, one nightly writer | Postgres, concurrent writers |
| compute | nightly batch | arq worker, per request |
| entry point | `run-api` (:8000) | `run-service` (:8001) |
| UI | four tabs, data inlined | the **Full forecast** tab, live |

### Why Postgres and a queue, rather than more DuckDB

DuckDB gives a writer *exclusive* access to the file — documented in `api/deps.py`, worked
around in `api/precompute.py` with a shadow copy and an atomic swap. That is a sound design
for one nightly writer and many readers of a static warehouse, and a hopeless one here,
where every upload and every job transition is an unscheduled write. There is no quiet
moment to swap in.

Model fitting does not happen in a request handler. `tests/test_service_layering.py`
enforces it by AST scan over every handler module in **both** tiers — a widening of tier 1's
`test_no_handler_imports_a_trainer`, which read one file and grepped for four strings.

### The data gate

Tier 2 refuses data it cannot honestly forecast, and names the shortfall:

- ≥ 90 days of history per SKU, ≥ 20 recorded days, gaps ≤ 14 days
- a failing SKU is excluded by name and reason; if *all* fail, the response says
  `90 days of history needed, 21 found` and points at the quick calculator, which works on
  short history precisely because it validates nothing

Nothing is padded or interpolated to get past it. The transformations that *are* applied —
duplicate `(sku, date)` rows summed, negatives read as returns, unparseable dates dropped —
are each counted and reported back.

### Honest model selection

Per-SKU quantile GBM against a seasonal-naive baseline, rolling origin on the user's own
history, fold count derived from what that history supports (90 days at a 28-day horizon is
**two** folds, and the response says so). Selection is on pinball loss at the critical
ratio, because that is the loss function of the number on the purchase order. **Both
methods' scores are always returned**, and when the baseline wins the baseline is served
and the response says by how much. Differences under 5% are reported as "too close to call"
rather than as a victory for whichever way the noise fell.

## What tier 2 sends to third parties

Users upload their own sales history — product names, daily volumes, prices. That is
commercially sensitive, so this section is specific rather than reassuring.

**PostHog** (product analytics) receives six funnel events: `upload_received`,
`upload_rejected`, `dataset_created`, `forecast_enqueued`, `forecast_completed`,
`forecast_failed`.

Their properties come from an **allowlist** in `service/observability.py` — counts,
durations, enums and opaque UUIDs. Not a convention: `Analytics.capture` drops anything
outside `ALLOWED_PROPERTIES` and logs a warning, so a future call site passing `sku=...`
leaks nothing and fails a test. What is sent:

`rows_read` · `sku_count` · `sku_count_admitted` · `sku_count_rejected` · `byte_size` ·
`horizon` · `critical_ratio` · `elapsed_seconds` · `n_folds` · `method_used` (aggregated to
`all_model` / `all_baseline` / `mixed`) · `rejection_reason` (a fixed vocabulary) ·
`dataset_id` / `job_id` / `request_id` · `component` · `environment`

Never sent: product names, unit prices, order quantities, sales figures, filenames, file
contents, or the rendered refusal message (which names SKUs).

Identity is an anonymous per-browser UUID in `localStorage`, sent as `X-Client-ID`. It
identifies a browser, not a person; nothing links it to a name, an email or a company, and
clearing site data resets it. There are no accounts.

**Sentry** (error tracking) receives exceptions from the API and the worker, tagged by
component. `send_default_pii=False`, request bodies are never captured, and a `before_send`
hook drops the request object, breadcrumbs and stack-frame locals, and redacts our own
`sku` / `filename` / `storage_uri` keys.

Stated plainly: **that is not a guarantee that no user data ever reaches Sentry.** An
exception message can contain anything — a pandas error can quote a value. The hook removes
the channels that would carry data in bulk; it cannot sanitise arbitrary strings. An
operator who cannot accept that residual risk should leave `SENTRY_DSN` unset, which
disables the integration entirely.

**Both are off by default.** No key, no integration — the local stack runs with neither, and
`.env.example` ships both blank.

### How long uploads are kept

Being careful about *where* data goes is undermined by keeping it forever, so there is a
finite window: `UPLOAD_RETENTION_DAYS`, 30 by default. A daily job on the worker (03:15)
destroys anything older, and `DELETE /datasets/{id}` does it on demand.

Deletion is a hard delete, not a flag. It removes the stored CSV, the dataset row, the gate
verdicts, the jobs, and the forecast results — including `forecast_results.series`, which
holds a copy of the user's own daily sales values. A soft delete would leave both the file
and that copy in place, which is not what "delete my data" means to the person asking.

**The bytes go first, then the row.** The two orderings fail differently and only one fails
safely: row-first then a crash leaves a file nothing points at — sensitive data lingering
invisibly. Bytes-first then a crash leaves a row pointing at a file that is gone, which is a
404 and a sweep. The purge also removes unreferenced files, which is how a process killed
between writing bytes and committing a row gets cleaned up.

A queued or running forecast against a deleted dataset is failed with a stated reason rather
than left to die on a missing file.

The purge refuses a retention window of zero or less instead of reading it as "delete
everything on the next run".

Logs stay on the operator's own infrastructure: structured JSON to stdout, one object per
line, every line carrying `request_id`. The id flows API → queue → worker, so one upload is
one `grep`:

```
{"component":"api",    "request_id":"TRACE-…", "message":"dataset created",      "admitted":7}
{"component":"api",    "request_id":"TRACE-…", "message":"forecast enqueued",    "horizon":28}
{"component":"worker", "request_id":"TRACE-…", "message":"forecast job starting"}
{"component":"worker", "request_id":"TRACE-…", "message":"forecast job done",    "seconds":8.2}
```

One documented exception: arq prints two banner lines before `on_startup`, the earliest
hook a worker has, so those two lines per process start are plain text. Everything after
them is JSON.

### Running tier 2

```bash
docker compose up -d db redis          # Postgres 16 + Redis 7
alembic upgrade head                   # tier 2 schema
run-service                            # API on :8001 (tier 1's run-api keeps :8000)
arq inventory_engine.service.worker.WorkerSettings
```

Then open the dashboard and use the **Full forecast** tab. Or `docker compose up` for the
whole stack.

---

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install -e ".[dev]"         # add ".[service]" for tier 2
```

Download the three M5 files from
[Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and extract
`sales_train_evaluation.csv`, `calendar.csv` and `sell_prices.csv` into `data/`.
No renaming needed; the loader will tell you exactly what is missing and where it belongs
if you skip this.

```bash
build-warehouse --overwrite      # ~30s
build-features                   # ~4s
pip install -e ".[models]"       # statsforecast, lightgbm, mlflow
run-baselines                    # fit + score all four baselines on 5 folds
run-gbm                          # train the global GBM + quantiles, score vs baselines
run-backtest                     # canonical scoring report (E5)
run-mint                         # MinT reconciliation + WRMSSE (E6)
run-optimize                     # money table, sensitivity, attribution (E7)
pip install -e ".[api]"          # fastapi, uvicorn, pydantic
run-api                           # start the API on :8000
python scripts/build_dashboard.py  # regenerate dashboard/index.html
pytest                           # 207 tests
ruff check . && ruff format --check .
```

Useful flags:

```bash
build-warehouse --all-items                 # every item in scope, 5,748 series
build-warehouse --items-per-dept 100        # bigger budget
build-warehouse --dept FOODS_3 --seed 7     # single dept, different draw
build-features --horizon 7                  # shorter horizon, features shift less
```

---

## Backtest methodology

Rolling-origin, 5 folds, 28-day horizon, origins stepping backwards from the panel end;
training is expanding, not sliding. Never a random split, never a single holdout. Defined
once in `backtest/folds.py` and imported by every model and the scorer, so "model A beats
model B" cannot silently become "model A was evaluated on easier weeks".

Metrics: **MASE** and **RMSSE** (scale-free, survive the 61.6% zero-demand days that make
MAPE undefined — MAPE is deliberately excluded), **Bias/ME** (signed — over-forecast is
dead stock, under-forecast is a stockout, and Phase 7 acts on the sign), and **pinball
loss** per quantile level, scored on the monotonized quantiles because that is how Phase 7
reads them. **WRMSSE** — the M5 official metric — was deliberately not
reported until Phase 6 built the hierarchy it weights over; it now reads 0.81173 base and
0.78411 reconciled.

Breakdowns by intermittency stratum and by horizon are persisted alongside the headline
numbers. Horizon decay turns out **not** to be monotonic — lgbm MASE by forecast week runs
1.0104, 1.0334, 1.0152, 1.0177, so week 2 is the worst, not week 4.

Reported as mean **and spread** across folds, never mean alone; a model scored on fewer
series than another (naive-scale exclusions) is surfaced, not hidden.

---

## What broke and how I fixed it

_Accumulating as the build progresses. Honest log, including what did not work._

- **The brief contradicted itself on scope.** It specified "state CA, category FOODS, all
  4 stores" *and* "~600–800 series". Those are incompatible: CA + FOODS + 4 stores is
  1,437 items × 4 = **5,748 series**. Resolved by quantifying all three readings and
  choosing the stratified 720-series sample above.
- **The first two loader tests to fail were both real bugs, not test bugs.** The
  transaction rollback fired even when no transaction had been opened, masking the
  original exception behind a `TransactionException`; and scope validation happened deep
  inside the SQL, so an invalid state code surfaced as the misleading "scope matched zero
  series" instead of naming the bad input. Both fixed.
- **The first stratification test asserted on a 2-item draw** and failed on sampling
  noise rather than on behaviour. Rewritten to assert how items are *classified* over the
  population, plus a non-overlap check between bands — a stronger property that does not
  depend on which two items were drawn.
- **The leakage gate's sensitivity check failed on `units_lag_365`, and the code was
  fine.** With the corruption cutoff at day 300 of a 520-day fixture, a 365-day lag on a
  28-day horizon reads 393 days behind its target — it could never reach the corrupted
  region, so it looked "immune" for reasons that had nothing to do with the shift logic.
  The fixture was the bug. The gate now uses a late cutoff (large clean region to verify)
  and the sensitivity check an early one (so the longest lag can respond), with the reason
  written down so nobody "fixes" it back.
- **`statsforecast` crashed with `BrokenProcessPool` on the full fit.** Its default
  `n_jobs=-1` re-imports scipy and numba in every worker process; at 1.3M training rows
  that exhausted the Windows paging file mid-fit. Forced `n_jobs=1` — the numba JIT keeps
  it fast enough (346s for three models × 5 folds × 720 series), and the parallel
  version's failure mode was worse than the speedup was worth.
- **Croston and TSB losing to seasonal naive on part of the panel** was not a bug — see
  the Phase 3 finding above. Reported rather than tuned away, per this project's own rule
  that a baseline beating expectations is information, not failure.
- **LightGBM produced 718 series in fold 0 where the baselines produced 720**, and my first
  hypothesis (dead series that never sells) was wrong. The actual cause is a **data-source
  asymmetry**: the GBM reads `feature_panel`, which drops pre-listing rows, while the
  baselines read raw `fact_sales`, which keeps them as genuine zeros. `FOODS_3_595` at two
  stores was first listed 2016-02-13, after fold 0 closes. Impact is provably zero — those
  are exactly the two series already excluded from every model for having no defined naive
  scale, and re-scoring with them explicitly dropped moves every model's MASE by `0.000000`.
  **Left open deliberately**: it is structurally wrong but numerically inert here, and the
  right fix is to intersect series across models per fold rather than rely on that
  coincidence.
- **1.29% of quantile forecasts were crossed inside the exact band Phase 7 selects from.**
  Independently fitted quantile models aren't constrained to be ordered, so q0.90 could
  land above q0.95 — returning a *lower* stocking quantity for a *higher* service level.
  Caught by measuring the crossing rate rather than assuming monotonicity, and fixed by
  rearrangement at read time (1.2887% → 0.0000%).
- **A blanket `DELETE` in the scorer made results depend on call order.** The point and
  quantile scorers both wrote to `backtest_fold_metrics`, and whichever ran second wiped
  the other's rows — producing a plausible-looking table with half the metrics silently
  missing. Each scorer now deletes only the metrics it owns.
- **The stratum-routing hybrid looked like a win on the headline metric and wasn't.** MASE
  1.0139 vs 1.0192 is a real improvement on average, but it loses on the two most recent
  folds, on RMSSE, and on pinball at every quantile level. Investigated, documented, and
  deliberately not adopted — then finally rejected on cost in Phase 7 (+3.58%).
- **The brief's quantile grid could not express its own conclusion.** It specified
  {0.5, 0.9, 0.95, 0.99}, but fresh-food critical ratios land at 0.25–0.63 — *below* the
  lowest fitted level. Every order would have been silently clamped at the median and the
  "optimal service level isn't 95%" finding would have been invisible. Caught by computing
  realistic critical ratios *before* wiring the optimizer rather than by noticing clamped
  output afterwards. Grid extended to seven levels and the model refit.
- **A scorer's DELETE kept widening, and the third time it destroyed real results.** Three
  scorers write to `backtest_fold_metrics` and each clears its previous output first. The
  point scorer originally deleted the whole table, wiping the quantile scorer's rows.
  Narrowing it to "everything except pinball" fixed that symptom but not the cause — it
  still swept up Phase 6's aggregate-level metrics and the WRMSSE figure, so retraining
  for Phase 7's wider quantile grid **silently destroyed the reconciliation results**. The
  dashboard's MinT panel would have been built on an empty table. Found while extracting
  data for Phase 9: two queries returned 1 row and 0 rows where 8 and 2 were expected. The
  fix, applied to all three: delete an **allowlist of what you write**, never a denylist of
  what you recognise — a denylist can always grow to cover someone else's rows.
  `tests/test_metric_ownership.py` now pins that the three scopes partition cleanly and
  are order-independent.
- **`run-gbm --replace` was deleting Phase 6's reconciled forecasts.** `DELETE ... WHERE
  model_name = 'lgbm'` matched reconciled rows and aggregate-level base rows too, so
  refitting for the wider quantile grid would have silently invalidated the entire
  reconciliation with no error raised. Now scoped to
  `reconciled = FALSE AND level = 'item_store'`.

---

## Cost / latency

_Pending Phase 10._

## Limitations and next steps

- The panel is intermittency-stratified, not volume-representative — see the trade-off
  note above. A volume-weighted re-run would test whether the cost results hold on a
  basket that matches actual revenue mix.
- Scope is one state and one category. Nothing in the pipeline is CA- or FOODS-specific;
  `Scope` in `config.py` is the only thing that would change.
- Currency: M5 is Walmart US data, so all costs are computed and stored in **USD**. The
  dashboard offers a ₹ toggle with the FX rate shown inline rather than baking an
  invented exchange rate into the headline number.

### Limitations of the hosted demo

The Full forecast service is deployed to scale to zero, which costs about $0.32/month
instead of about $15. Three consequences, none of them hidden:

- **An abandoned forecast never finishes.** The API and the arq worker share one Machine,
  because a Fly volume attaches to exactly one Machine and `LocalDiskStorage` hands the
  worker a filesystem path the API wrote. The Machine suspends when traffic stops. While a
  forecast runs the dashboard polls `/forecast/{job_id}` every couple of seconds, so the
  app is never idle and the job completes — but if you close the tab mid-run, polling
  stops, the Machine suspends, and that job's row stays `running` for ever. Nobody is
  waiting on it, and keeping a Machine awake to protect it costs about $12/month. The fix,
  if this were a product, is object storage plus a separate worker process group, which is
  the same conclusion `docker-compose.yml` reaches about running the two on separate hosts.
- **The first request after an idle period is slow.** The Machine suspends rather than
  stops, so it restores from memory in a second or two, and the free-tier Postgres wakes on
  its own schedule after five minutes idle. Expect a few seconds on a cold visit.
- **Free-tier ceilings apply.** 0.5 GB of Postgres storage and 100 compute-hours a month.
  Uploads are destroyed after 30 days regardless, so storage is bounded by usage rather
  than by time, but a burst of traffic would hit the compute ceiling before it hit disk.

Tier 1 is not deployed at all. Every figure it produces is inlined into the dashboard as
JSON at build time, so the live endpoints would serve a 324 MB DuckDB warehouse to nobody.
