# E5 — Backtest harness ✅ (WRMSSE deferred to E6)

**Goal:** rolling-origin evaluation that every model in the project is scored by,
identically.

**Consumes:** `forecast`, `fact_sales`, `dim_item_stratum`.

**Produces:** `backtest_fold_metrics`.

**Entrypoint:** `run-backtest`.

**Status:** S1–S5 complete. The fold and metric stories were built during E3 (a baseline
with no folds to run on and no metric to report is not a deliverable); pinball scoring and
the canonical report were completed here once E4 produced quantiles. **WRMSSE is the one
open item** — it weights RMSSE across the M5 aggregation hierarchy, which does not exist
until E6 builds it. Reporting it earlier would mean reporting a number not actually
computed.

---

### E5-S1 — Rolling-origin fold definition ✅

**As** the project, **I want** folds that respect time, **so that** reported accuracy is
achievable in production.

- [x] 5 folds, 28-day horizon, origins stepping backwards from the panel end
- [x] Never a random split; never a single holdout
- [x] Defined once in `backtest/folds.py` and imported by every model and by the scorer.
      If each model derived its own folds, "model A beats model B" could silently become
      "model A was evaluated on easier weeks".
- [x] Test windows contiguous, non-overlapping, and covering exactly `BACKTEST_DAYS` —
      the same region Phase 1's sampler was forbidden from seeing
- [x] Training is expanding, not sliding: the honest production analogue, where you refit
      on all history available at the time
- [x] `assert_no_training_leak` checks no fold's training window reaches its own test
      window, and that `BACKTEST_DAYS == N_FOLDS × HORIZON`
- [x] Exact fold dates pinned by test, since the README quotes them

### E5-S2 — Metrics ✅ (WRMSSE deferred)

- [ ] **WRMSSE** — deferred to E6. WRMSSE is RMSSE weighted across the 12 M5 aggregation
      levels; the per-series RMSSE component exists, but the weighting needs the hierarchy.
      Claiming WRMSSE before then would be claiming a metric not actually computed.
- [x] **MASE** — scale-free, survives zeros
- [x] **RMSSE** — the per-series component of WRMSSE
- [x] **Bias / ME** — signed, because over-forecast (dead stock) and under-forecast
      (stockout) cost different amounts and E7 acts on the sign
- [x] **Pinball loss** — per quantile level, scored on **monotonized** quantiles, because
      that is how E7 reads them; scoring the raw crossed values would report a loss for a
      forecast nothing downstream uses
- [x] **MAPE deliberately excluded**, with the reason recorded: undefined on the 61.6% of
      this panel that is zero-demand, and explosive on the near-zero rows that remain
- [x] Scale denominator measured from each series' first non-zero sale (M5 convention),
      computed once per fold and shared across models so it is provably identical
- [x] Series with no defined scale are excluded **and counted**, never silently dropped

### E5-S3 — Report spread, not just mean ✅

- [x] Mean, std, best and worst fold for every metric
- [x] Per-fold rows persisted, not just the aggregate
- [x] `summarise()` surfaces `min_series_scored` / `max_series_excluded`, so a model scored
      on fewer series than another is visible rather than flattering

### E5-S4 — Breakdowns ✅

- [x] By horizon (1–28 days), so accuracy decay is visible rather than averaged away
- [x] By intermittency stratum, joining `dim_item_stratum` — written once by Phase 1 rather
      than recomputed here, so the bands can't drift from the ones scoping used
- [x] By hierarchy level via the `level` column, ready for E6's before/after comparison

### E5-S5 — Persist to DuckDB ✅

- [x] `backtest_fold_metrics(run_id, model_name, fold, level, stratum, horizon, metric,
      value, n_series, n_excluded)`
- [x] Long format, so a new metric never needs a migration
- [x] E8 and E9 read this table directly — no recomputation at request time
- [x] The point and quantile scorers delete only the metrics they own. A blanket delete
      made the stored result depend on which scorer ran last, which is the kind of bug that
      produces a plausible-looking table with half the rows silently missing.

---

## Pinball loss on monotonized quantiles

| Level | lgbm | hybrid |
|---|---:|---:|
| q0.50 | **0.62031** | 0.63383 |
| q0.90 | **0.39631** | 0.39791 |
| q0.95 | **0.25814** | 0.26252 |
| q0.99 | **0.08276** | 0.09283 |

LightGBM wins at every level. This is the distributional metric E7 consumes, and it is
independent corroboration for carrying plain LightGBM forward rather than the routing
hybrid — see E7-S6.

## Horizon decay is not monotonic

lgbm MASE by forecast week: 1.0104, **1.0334**, 1.0152, 1.0177. Week 2 is the worst, not
week 4. Accuracy does not simply degrade with distance here, which is worth knowing before
anyone assumes a shorter horizon would be easier.
