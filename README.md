# Inventory Optimization Engine

Hierarchical demand forecasting and newsvendor inventory optimization on M5 retail data.

**Forecasting accuracy is not the deliverable. Money saved is.** This project forecasts
SKU-level demand, reconciles those forecasts across a store hierarchy so they stay
coherent, converts the forecast *distribution* into optimal stocking quantities via
newsvendor theory, and reports the cost delta against baseline stocking practice.

> **Build status:** Phase 1 (data foundation) complete. Phases 2–10 in progress.

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

| Phase | Status |
|---|---|
| 1. Data foundation (DuckDB warehouse) | ✅ Complete |
| 2. Features + leakage test | ⬜ Next |
| 3. Baselines (Seasonal Naive, Croston/TSB, ETS/AutoARIMA) | ⬜ |
| 4. LightGBM global model + MLflow | ⬜ |
| 5. Rolling-origin backtest harness | ⬜ |
| 6. MinT hierarchical reconciliation | ⬜ |
| 7. Newsvendor optimization layer | ⬜ |
| 8. FastAPI service | ⬜ |
| 9. React dashboard | ⬜ |
| 10. Deploy | ⬜ |

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
pytest                           # 20 tests
ruff check . && ruff format --check .
```

Useful flags:

```bash
build-warehouse --all-items                 # every item in scope, 5,748 series
build-warehouse --items-per-dept 100        # bigger budget
build-warehouse --dept FOODS_3 --seed 7     # single dept, different draw
```

---

## Backtest methodology

_Pending Phase 5._ Rolling-origin, 5 folds, 28-day horizon. Never a random split, never a
single holdout. Metrics: WRMSSE, MASE, Bias/ME, pinball loss — reported as mean **and
spread across folds**. MAPE is deliberately excluded; it is undefined on zero-demand days,
which are 61.6% of this panel.

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
