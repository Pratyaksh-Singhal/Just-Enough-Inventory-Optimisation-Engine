# E3 — Baselines ⬜

**Goal:** establish the bar before any ML. A model without a baseline is a number without
meaning.

**Consumes:** `fact_sales`, `feature_panel`.

**Produces:** rows in `forecast` with `model_name IN ('seasonal_naive','croston','tsb','ets')`
and `reconciled = FALSE`.

**Blocked by:** E2-S6 must pass.

**Definition of done:** a metrics table for all baselines, produced by the E5 harness, in
the README — **including any baseline that beats the eventual model.**

---

### E3-S1 — Seasonal naive ⬜

**As** the project, **I want** the dumbest defensible forecast, **so that** every later
number has something honest to be compared against.

- [ ] Forecast for `target_date` = units on the same weekday, one week prior to `origin_date`
- [ ] Handles series with gaps without forward-filling across them
- [ ] Written to `forecast` with `model_name = 'seasonal_naive'`

### E3-S2 — Croston and TSB ⬜

**As** an intermittent SKU, **I want** a method built for zero-inflated demand, **so that**
I am not modelled by something that assumes continuous sales.

- [ ] Croston and TSB via `statsforecast`
- [ ] TSB included specifically because Croston does not decay its estimate during long
      zero runs — on a panel that is 61.6% zeros this is not academic
- [ ] Report both; do not silently keep only the winner
- [ ] Metrics broken out **by intermittency stratum** (dense / mid / sparse), since the
      whole point of these methods is the sparse band

### E3-S3 — ETS / AutoARIMA ⬜

- [ ] Classical bar via `statsforecast`
- [ ] Fit time recorded — if it is impractical at 720 series, that is a finding worth
      reporting, not a reason to quietly drop it
- [ ] Failures to converge counted and reported, not swallowed

### E3-S4 — Baseline comparison table ⬜

- [ ] All baselines scored by the E5 harness on identical folds
- [ ] Broken out overall and by intermittency stratum
- [ ] README states plainly which baseline is hardest to beat and why
