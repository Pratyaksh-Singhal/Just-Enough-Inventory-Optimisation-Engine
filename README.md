# Inventory Optimization Engine

Hierarchical demand forecasting and newsvendor inventory optimization on M5 retail data.

**Forecasting accuracy is not the deliverable. Money saved is.** This project forecasts
SKU-level demand, reconciles those forecasts across a store hierarchy so they stay
coherent, converts the forecast *distribution* into optimal stocking quantities via
newsvendor theory, and reports the cost delta against baseline stocking practice.

> **Build status:** Phases 1–6 complete (data foundation, leakage-free features,
> baselines, global GBM, backtest harness, MinT reconciliation). Phases 7–10 in progress.

---

## Headline result

_Pending Phase 7. This section will carry the money table and the total cost delta._

---

## The money table

_Pending Phase 7._

| Policy | Stockout rate | Waste units | Holding cost | Total cost |
|---|---|---|---|---|
| Current (naive: last week's sales) | — | — | — | — |
| Fixed 95% service level | — | — | — | — |
| **Newsvendor + our forecast** | — | — | — | — |

---

## Approach

Baselines → global GBM → hierarchical reconciliation → newsvendor optimization.
Each stage is measured against the one before it, and a baseline that wins is reported
as a win rather than tuned away.

Work is tracked as ten epics in [`docs/BACKLOG.md`](docs/BACKLOG.md). Phases communicate
through named DuckDB tables rather than through code, so each epic states what it consumes
and produces and can be picked up without reading the one before it. Branching follows
[`docs/BRANCHING.md`](docs/BRANCHING.md): `dev` for integration, `main` for phase-complete
work, one tag per phase.

| Phase | Epic | Status |
|---|---|---|
| 1. Data foundation (DuckDB warehouse) | [E1](docs/backlog/epic-01-data-foundation.md) | ✅ Complete |
| 2. Features + leakage test | [E2](docs/backlog/epic-02-features.md) | ✅ Complete |
| 3. Baselines (Seasonal Naive, Croston/TSB, ETS/AutoARIMA) | [E3](docs/backlog/epic-03-baselines.md) | ✅ Complete |
| 4. LightGBM global model + MLflow | [E4](docs/backlog/epic-04-global-model.md) | ✅ Complete |
| 5. Rolling-origin backtest harness | [E5](docs/backlog/epic-05-backtest.md) | ✅ Complete |
| 6. MinT hierarchical reconciliation | [E6](docs/backlog/epic-06-reconciliation.md) | ✅ Complete |
| 7. Newsvendor optimization layer | [E7](docs/backlog/epic-07-optimization.md) | 🔜 Next |
| 8. FastAPI service | [E8](docs/backlog/epic-08-api.md) | ⬜ |
| 9. React dashboard | [E9](docs/backlog/epic-09-dashboard.md) | ⬜ |
| 10. Deploy | [E10](docs/backlog/epic-10-ship.md) | ⬜ |

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

Whether the trade is worth taking is a **cost** question, not an accuracy one: the bottom
level is what orders are placed against, so a 1.5% MASE degradation there has to be weighed
against store- and state-level plans that finally agree with those orders. Phase 7 has the
cost function to settle it.

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

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install -e ".[dev]"
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
pytest                           # 152 tests
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
  deliberately not adopted — see "Stratum-aware routing" above.

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
