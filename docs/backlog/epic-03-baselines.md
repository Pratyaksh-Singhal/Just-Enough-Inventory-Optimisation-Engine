# E3 — Baselines ✅

**Goal:** establish the bar before any ML. A model without a baseline is a number without
meaning.

**Consumes:** `fact_sales`, `feature_panel`, `dim_item_stratum`.

**Produces:** `forecast` rows for `seasonal_naive`, `croston`, `tsb`, `ets`
(`reconciled = FALSE`); `backtest_fold_metrics`.

**Blocked by:** E2-S6 (passed).

**Note on scope:** E3's definition of done is "a metrics table produced by the E5
harness" — a baseline with no folds to run on and no metric to report isn't a
deliverable. E5-S1/S2/S3/S4/S5 were built here rather than deferred; see
[epic-05](epic-05-backtest.md).

---

## Results

5 folds × 28-day horizon × 720 series. `ets` fit via `AutoETS(season_length=7)`,
`croston` via `CrostonOptimized`, `tsb` via `TSB(alpha_d=0.2, alpha_p=0.2)` — the TSB
smoothing parameters are a stated assumption (statsforecast has no optimised TSB variant).

### MASE and RMSSE, mean ± std across folds

| Model | MASE | RMSSE | Bias |
|---|---:|---:|---:|
| **ets** | **1.024** ± 0.026 | **0.763** ± 0.023 | −0.042 ± 0.065 |
| tsb | 1.045 ± 0.026 | 0.778 ± 0.024 | +0.017 ± 0.078 |
| croston | 1.112 ± 0.012 | 0.798 ± 0.009 | +0.111 ± 0.057 |
| seasonal_naive | 1.204 ± 0.034 | 0.989 ± 0.029 | −0.063 ± 0.086 |

All four clear seasonal naive by a wide margin — the honest bar is doing its job. 718/720
series scored on MASE/RMSSE; 2 series had no defined training-window scale (flat or
never-sold) and were excluded and counted, not silently dropped.

### MASE by intermittency stratum

| Stratum | ets | tsb | croston | seasonal_naive |
|---|---:|---:|---:|---:|
| dense | 1.042 | 1.068 | 1.076 | 1.284 |
| mid | 1.037 | 1.056 | 1.134 | 1.221 |
| sparse | 0.993 | 1.009 | **1.127** | 1.107 |

### The finding: ETS beats the intermittent-demand specialists, on their own turf

**Croston loses to plain seasonal naive on the sparse stratum** (1.127 vs 1.107), and
**AutoETS — a classical method with no special handling for zero-inflated demand — beats
both Croston and TSB on every stratum, sparse included.** This is the opposite of the
textbook expectation and is reported as found, not tuned away.

Two things explain most of it, and both are informative rather than embarrassing:

1. **Croston/TSB forecast a flat rate.** Both decompose demand into "probability of a
   sale" × "size when it sells" and produce a constant forecast across the whole 28-day
   horizon. This panel has real day-of-week seasonality (weekend uplift, SNAP-day
   effects) that a flat-rate method cannot represent at all, and that ETS's seasonal
   component captures directly.
2. **TSB earns its inclusion over Croston but doesn't close the gap to ETS.** TSB was
   added specifically because Croston never decays its estimate through a long zero run
   (E3-S2's stated reason for including it) — and the data confirms that reasoning: TSB
   beats Croston on every stratum and every fold. It just isn't enough to also beat a
   method that models the weekly cycle.

**Croston's bias is the more actionable finding for E7.** +0.11 mean bias, worst fold
+0.17, is a *systematic* over-forecast — for fresh food, where overstock cost is close to
100% of unit cost, that bias is spoilage, priced directly into whatever policy uses it.
TSB's bias is much closer to zero (+0.02) and ETS's is slightly negative (−0.04, a mild
under-forecast). Bias sign and magnitude, not just MASE, will matter when E7 picks which
forecast the newsvendor layer trusts.

**Consequence for E4:** the LightGBM global model's bar to clear is AutoETS at
MASE 1.024, not the intermittent-demand baselines. If E4 wins mainly on the sparse
stratum specifically, that is the more interesting story than an aggregate win.

---

### E3-S1 — Seasonal naive ✅

- [x] Forecast for `target_date` = units on the same weekday, one week prior to
      `origin_date`; no smoothing, no parameters
- [x] Implemented in-repo (`baselines.seasonal_naive`) rather than pulled from a library,
      since its whole value is being inspectable and un-tunable
- [x] Written to `forecast` with `model_name = 'seasonal_naive'`
- [x] MASE 1.204 — clears itself trivially (1.0 by definition against its own scale is
      not the test; beating it is what the other three models must do, and do)

### E3-S2 — Croston and TSB ✅

- [x] Both fit via `statsforecast`
- [x] TSB included specifically because Croston does not decay its estimate through long
      zero runs; **confirmed** — TSB beats Croston on every stratum and every fold
- [x] Both reported; Croston is not the winner and that is stated, not hidden
- [x] Metrics broken out by stratum — the finding above (Croston losing to seasonal naive
      on sparse) is exactly what this breakdown exists to surface

### E3-S3 — ETS / AutoARIMA ✅

- [x] AutoETS via `statsforecast`, `season_length=7`
- [x] Fit time recorded: 346s for the three statsforecast models combined across all
      folds and 720 series (`n_jobs=1` — see note below)
- [x] Zero convergence failures; series exclusions (2/720) were scale-related, not fit
      failures

### E3-S4 — Baseline comparison table ✅

- [x] All four scored by the E5 harness on identical folds
- [x] Broken out overall and by stratum
- [x] README states plainly that ETS is hardest to beat, and why

---

## What broke

- **`n_jobs=-1` crashed statsforecast with `BrokenProcessPool`.** Its worker pool
  re-imports scipy/numba per process; at 1.3M training rows that exhausted the Windows
  paging file. Fixed by forcing `n_jobs=1` — the numba JIT keeps it fast enough (346s for
  three models × 5 folds × 720 series), and the failure mode of the parallel version was
  worse than the sequential speedup was worth.
- **Croston and TSB losing to seasonal naive on part of the panel** was not a bug, but it
  is the kind of result the brief explicitly asks to be reported rather than tuned away.
  See "The finding" above.
