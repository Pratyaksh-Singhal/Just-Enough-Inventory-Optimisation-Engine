# E1 — Data foundation ✅

**Goal:** a scoped, long-format M5 panel in DuckDB that every later phase reads.

**Consumes:** raw Kaggle CSVs in `data/` (`sales_train_evaluation.csv`, `calendar.csv`,
`sell_prices.csv`).

**Produces:** `fact_sales`, `dim_calendar`.

**Definition of done:** `build-warehouse --overwrite` succeeds from a clean checkout,
tests pass, README documents the scoping decision and its trade-off.

---

### E1-S1 — Fail clearly when the data is missing ✅

**As** someone cloning this repo, **I want** the loader to tell me exactly which files
are missing and where to put them, **so that** I am not debugging a `FileNotFoundError`
against a 3 GB Kaggle download.

- [x] `verify_raw_files()` names every missing file, not just the first
- [x] Error message includes the expected directory and the Kaggle URL
- [x] Test asserts the message names the missing files and omits present ones

### E1-S2 — Long-format warehouse ✅

**As** a downstream phase, **I want** one table shaped
`(date, item_id, store_id, dept_id, cat_id, state_id, units, price, snap, event)`,
**so that** I never re-derive the panel shape.

- [x] Pure DuckDB SQL ETL — no pandas in the loading path
- [x] `UNPIVOT` melts d_1..d_1941 with the scope filter pushed into the CSV scan
- [x] Table built from explicit DDL with `NOT NULL` + PK, so the insert asserts against
      join fan-out and duplicate `(date, item, store)` rows
- [x] Whole build is one transaction: a valid warehouse or nothing

### E1-S3 — Calendar retains future days ✅

**As** E2, **I want** `dim_calendar` to extend past the end of sales data, **so that**
"days until next event" is computable for observations at the end of the training window.

- [x] All 1,969 calendar rows loaded, vs 1,941 days of sales
- [x] Test asserts `max(dim_calendar.date) > max(fact_sales.date)`
- [x] Documented as the one legitimate place to read ahead of _t_

### E1-S4 — Intermittency-stratified scope ✅

**As** the project, **I want** the sampled SKU universe to span dense to fully
intermittent demand, **so that** E3's Croston/TSB work addresses a problem that is
actually present rather than one sampling removed.

- [x] 60 items per FOODS dept, split into 3 equal-count zero-day-share bands, 20 drawn each
- [x] Resulting panel: 720 series, 61.6% zero-days, sparse stratum reaching 99–100%
- [x] Trade-off documented: the panel is intermittency-representative, **not**
      volume-representative

### E1-S5 — Scope selection cannot see the evaluation period ✅

**As** a reviewer, **I want** proof that SKU selection did not use backtest-period data,
**so that** the reported results are not quietly inflated by selection leakage.

- [x] Stratification window is 90 days ending `BACKTEST_DAYS` before the panel end
- [x] `test_intermittency_is_measured_before_the_backtest_region` — a synthetic series
      silent through the reference window and dense through the backtest region must
      grade maximally sparse
- [x] Items never listed (no price history in window) excluded before stratification

### E1-S6 — Reproducible sample ✅

**As** anyone rebuilding, **I want** an identical SKU universe, **so that** metrics are
comparable across machines and across time.

- [x] Selection order is `hash(item_id || seed)`, not `random()` — stable across DuckDB
      versions, platforms and thread counts
- [x] Test: two builds produce identical item sets; different seeds produce different sets
- [x] Short departments backfill deterministically rather than silently under-filling
