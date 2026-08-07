# E4 — Global model ⬜

**Goal:** one LightGBM model across all series, with series identity as a feature.

**Consumes:** `feature_panel`.

**Produces:** rows in `forecast` with `model_name = 'lgbm'`; MLflow runs; model artifacts.

**Blocked by:** E2-S6 must pass.

---

### E4-S1 — Global LightGBM point model ⬜

**As** the project, **I want** a single model over all 720 series, **so that** sparse
series borrow strength from dense ones instead of each fitting on their own thin history.

- [ ] One model, all series; `item_id` / `store_id` / `dept_id` as categorical features
- [ ] Trains in minutes on this panel, and the runtime is recorded
- [ ] README records **why not deep learning**: at 720 series of daily data a global GBM
      wins on accuracy, trains in minutes, and is explainable. This is a stated engineering
      judgement, not an omission.

### E4-S2 — Quantile models ⬜

**As** E7, **I want** a demand *distribution*, not a point, **so that** the newsvendor
critical ratio has a quantile to select.

- [ ] `objective='quantile'` at α ∈ {0.5, 0.9, 0.95, 0.99}
- [ ] Written to `forecast` with the `quantile` column populated
- [ ] **Monotonicity check:** q0.5 ≤ q0.9 ≤ q0.95 ≤ q0.99 per row. Independently fitted
      quantile models can cross; where they do, the count is reported and the fix
      (sorting, or a monotone re-fit) is documented rather than applied silently.

### E4-S3 — MLflow tracking ⬜

- [ ] Every run logs params, metrics, feature list, git SHA, fold definition
- [ ] Runs are reproducible from the logged config alone
- [ ] `run_id` written into `forecast` so any forecast traces back to its run

### E4-S4 — SHAP explainability ⬜

- [ ] Global feature importance
- [ ] Per-forecast local explanation, to answer "why this forecast?" in E9
- [ ] Persisted for the dashboard — SHAP is too slow to compute per API request

### E4-S5 — Honest comparison against E3 ⬜

- [ ] Same folds, same metrics as the baselines
- [ ] **If a baseline wins, say so in the README** — overall or on a stratum — and
      investigate rather than tuning until the numbers look good
- [ ] Per-stratum breakdown: a global GBM often loses to TSB on the sparsest band, and
      that is worth knowing before the optimization layer trusts it
