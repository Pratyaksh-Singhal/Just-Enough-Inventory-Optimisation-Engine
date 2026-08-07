# E5 — Backtest harness ⬜

**Goal:** rolling-origin evaluation that every model in the project is scored by,
identically.

**Consumes:** `forecast`, `fact_sales`.

**Produces:** `backtest_fold_metrics`.

**Definition of done:** every model in `forecast` has fold-level metrics persisted, and
the README reports **mean and spread**, never mean alone.

---

### E5-S1 — Rolling-origin fold definition ⬜

**As** the project, **I want** folds that respect time, **so that** reported accuracy is
achievable in production.

- [ ] 5 folds, 28-day horizon, origins stepping backwards from the panel end
- [ ] Never a random split; never a single holdout
- [ ] Fold boundaries defined once, in `config.py`, and shared by every model
- [ ] Test: no training window in any fold contains a date from that fold's test window

### E5-S2 — Metrics ⬜

- [ ] **WRMSSE** — the M5 official metric, with the competition's weighting
- [ ] **MASE** — scale-free and survives zeros
- [ ] **Bias / ME** — signed, because direction is the whole point: over-forecast is dead
      stock, under-forecast is a stockout, and they cost different amounts
- [ ] **Pinball loss** — per quantile level, to score the distribution E7 depends on
- [ ] **MAPE deliberately excluded**, with the reason in the README: it is undefined on
      zero-demand days, which are 61.6% of this panel

### E5-S3 — Report spread, not just mean ⬜

**As** a reviewer, **I want** to see fold variance, **so that** I can tell a robust model
from one that got lucky on a fold.

- [ ] Mean, std, min, max across folds for every metric
- [ ] Per-fold rows persisted, not just the aggregate
- [ ] Any fold where the model underperforms the baseline is called out by name in the
      README's "what broke" section

### E5-S4 — Breakdowns ⬜

- [ ] By horizon (1–28 days) — accuracy decay is expected and should be visible
- [ ] By intermittency stratum — the dense/mid/sparse bands from E1-S4
- [ ] By hierarchy level, ready for E6's before/after comparison

### E5-S5 — Persist to DuckDB ⬜

- [ ] `backtest_fold_metrics(run_id, model_name, fold, level, stratum, horizon, metric, value)`
- [ ] E8 and E9 read this table directly — no recomputation at request time
