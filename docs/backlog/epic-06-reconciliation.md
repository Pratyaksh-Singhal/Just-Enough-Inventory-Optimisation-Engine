# E6 — Hierarchical reconciliation ⬜

**Goal:** SKU forecasts that sum exactly to store and state forecasts. This is the
technical differentiator of the project.

**Consumes:** `forecast` (base, `reconciled = FALSE`).

**Produces:** `forecast` rows with `reconciled = TRUE`; `coherence_check`.

---

### E6-S1 — Build the hierarchy ⬜

- [ ] `State → Store → Category → Dept → Item`, all five levels
- [ ] Summing matrix `S` and the aggregation constraints, validated against `fact_sales`
      actuals: aggregating actuals up the tree must reproduce the aggregate actuals exactly
- [ ] Test asserts the hierarchy is complete — every leaf reachable from the root

### E6-S2 — Base forecasts at every level ⬜

- [ ] Forecasts produced independently at each level, written with the `level` column
- [ ] These are deliberately **incoherent** before reconciliation — that incoherence is
      the thing E6-S4 measures

### E6-S3 — MinT reconciliation ⬜

- [ ] MinT via Nixtla `hierarchicalforecast`
- [ ] Covariance estimator choice (`ols` / `wls_struct` / `mint_shrink`) documented with
      the reason — shrinkage matters when the residual covariance is estimated from few
      observations relative to series count, which is this panel's situation
- [ ] Reconciled forecasts written back with `reconciled = TRUE`, base rows retained so
      before/after is queryable

### E6-S4 — Before/after coherence table ⬜

**As** a reviewer, **I want** proof the hierarchy is now coherent, **so that** the claim
is demonstrated rather than asserted.

- [ ] Coherence gap = |sum of children − parent|, per level, before and after
- [ ] After: gap is zero to floating-point tolerance. **Assert it in a test**, do not
      just print it.
- [ ] Table persisted to `coherence_check` for the dashboard

### E6-S5 — Accuracy change at aggregate levels ⬜

- [ ] E5 metrics rerun on reconciled forecasts, same folds
- [ ] Report the accuracy change **at every level**, not just where it improved
- [ ] MinT usually helps aggregate levels and can slightly hurt the bottom level. If that
      happens here, it goes in the README as the expected trade-off — coherence is
      purchased, and the price should be stated.
