# Backlog — Inventory Optimization Engine

Ten epics, one per phase of the build. Each epic file states what it **consumes**, what
it **produces**, and its definition of done, so a phase can be picked up without reading
the phase before it.

## Epics

| # | Epic | Goal | Status |
|---|---|---|---|
| [E1](backlog/epic-01-data-foundation.md) | Data foundation | Scoped M5 panel in DuckDB | ✅ Done |
| [E2](backlog/epic-02-features.md) | Features | Leakage-free feature panel | ✅ Done |
| [E3](backlog/epic-03-baselines.md) | Baselines | The honest bar to beat | ✅ Done |
| [E4](backlog/epic-04-global-model.md) | Global model | LightGBM + MLflow | 🔜 Next |
| [E5](backlog/epic-05-backtest.md) | Backtest harness | Rolling-origin, 5 folds | 🔨 S1–S5 pulled forward by E3 |
| [E6](backlog/epic-06-reconciliation.md) | Reconciliation | MinT coherent hierarchy | ⬜ |
| [E7](backlog/epic-07-optimization.md) | Optimization | Newsvendor money table | ⬜ |
| [E8](backlog/epic-08-api.md) | API | FastAPI over precomputed results | ⬜ |
| [E9](backlog/epic-09-dashboard.md) | Dashboard | React analytics UI | ⬜ |
| [E10](backlog/epic-10-ship.md) | Ship | Deploy + recruiter README | ⬜ |

## The phase contract

Phases communicate through **named DuckDB tables**, not through function calls or
in-memory objects. A phase is done when its tables exist and match the schema below.
This is what makes the epics independently pickup-able: E7 does not need to know how E4
produced a quantile, only that `forecast` contains one.

| Table | Written by | Read by |
|---|---|---|
| `fact_sales` | E1 | E2, E3, E5, E7 |
| `dim_calendar` | E1 | E2 |
| `feature_panel` | E2 | E3, E4 |
| `forecast` | E3, E4, E6 | E5, E6, E7, E8 |
| `backtest_fold_metrics` | E5 | E8, E9 |
| `coherence_check` | E6 | E8, E9 |
| `order_policy` | E7 | E8, E9 |
| `cost_comparison` | E7 | E8, E9, E10 |
| `cost_sensitivity` | E7 | E8, E9 |

### `forecast` — the shared spine

One table carries every forecast in the project: baselines, the GBM, point and quantile,
pre- and post-reconciliation. Phases discriminate on columns rather than on table names,
so adding a model never requires a downstream schema change.

```sql
forecast(
    run_id        VARCHAR,   -- MLflow run id, or 'baseline:<name>'
    model_name    VARCHAR,   -- 'seasonal_naive' | 'croston' | 'tsb' | 'ets' | 'lgbm' ...
    fold          INTEGER,   -- 0..4; backtest fold, NULL for production forecasts
    origin_date   DATE,      -- last date of training data for this forecast
    target_date   DATE,      -- date being forecast
    horizon       INTEGER,   -- target_date - origin_date, in days (1..28)
    level         VARCHAR,   -- 'item_store' | 'dept_store' | 'cat_store' | 'store' | 'state'
    item_id       VARCHAR,   -- NULL above item level
    store_id      VARCHAR,   -- NULL above store level
    dept_id       VARCHAR,
    quantile      DOUBLE,    -- NULL for point forecast; 0.5/0.9/0.95/0.99 otherwise
    yhat          DOUBLE,
    reconciled    BOOLEAN    -- FALSE = base forecast, TRUE = post-MinT
)
```

**Why `horizon` is stored explicitly:** every feature in E2 is shifted by the horizon it
serves, and E5 reports metrics by horizon. Deriving it from a date subtraction at read
time invites off-by-one errors in exactly the place they are hardest to notice.

## Conventions

- **Branches map to SDLC stages, not to epics** — `main` (stable, phase-tagged) and `dev`
  (integration), with short-lived `feature/*` off `dev`. See
  [`BRANCHING.md`](BRANCHING.md). The phase contract above is what coordinates epics; a
  branch per epic would duplicate that and cost merge conflicts over the same schema.
- **Story IDs** are `E<epic>-S<story>`, e.g. `E2-S4`. Commit messages reference them.
- **Status**: ⬜ not started · 🔨 in progress · ✅ done · ⏸️ blocked · ❌ dropped (with reason).
- **Definition of done** for every story: code + tests pass + `ruff check` clean +
  docstrings on public functions + the epic's produced tables populated.
- **A baseline beating the model is a result, not a failure.** Stories that compare
  models have an acceptance criterion requiring the honest number be reported either way.

## Ordering note

E3's definition of done is "a metrics table produced by the E5 harness", so E3 cannot
finish before E5-S1 (fold definitions) and E5-S2 (metrics) exist — a baseline with no
folds to run on and no metric to report is not a deliverable. Those two stories, plus the
scorer in E5-S3/S4/S5, were therefore built during E3 rather than deferred.

This is a real dependency the backlog understated, not a scope change: the epic numbering
follows the brief's phase order, which puts the harness after the models it exists to
score. Recorded here rather than silently reordered.

## Cross-cutting rules

These apply to every epic and are not repeated as stories:

1. **No feature at time _t_ may use data from _t_ or later.** `tests/test_no_leakage.py`
   (E2-S6) is the gate; it must pass before any modelling story starts.
2. **Never train on request.** Forecasts are precomputed into DuckDB; the API reads.
3. **Never a random split.** Rolling-origin only, everywhere.
4. **MAPE is rejected.** It is undefined on zero-demand days, which are 61.6% of this
   panel. MASE is the scale-free metric of record.
5. **Cost assumptions are labelled as assumptions** and sensitivity-tested over `Cu/Co`.
6. **Money is computed and stored in USD.** M5 is Walmart US data; the ₹ toggle is a
   presentation concern in E9, with the FX rate shown inline.
