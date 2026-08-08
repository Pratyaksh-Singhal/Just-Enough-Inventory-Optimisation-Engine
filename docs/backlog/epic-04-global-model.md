# E4 — Global model ✅

**Goal:** one LightGBM model across all series, with series identity as a feature.

**Consumes:** `feature_panel`.

**Produces:** `forecast` rows for `lgbm` (point + 4 quantile levels); `feature_importance`;
MLflow runs.

**Blocked by:** E2-S6 (passed).

**Carried into E6/E7:** plain `lgbm` with isotonic-sorted quantiles.

---

## Results

5 folds × 28-day horizon × 720 series. 503,600 forecast rows, 756s total training.

| Model | MASE | RMSSE | Bias |
|---|---:|---:|---:|
| **lgbm** | **1.0192** ± 0.0182 | **0.7576** ± 0.0212 | −0.0982 ± 0.0348 |
| ets | 1.0242 ± 0.0255 | 0.7626 ± 0.0227 | −0.0422 ± 0.0650 |
| tsb | 1.0446 ± 0.0260 | 0.7776 ± 0.0237 | +0.0166 ± 0.0776 |
| croston | 1.1122 ± 0.0123 | 0.7976 ± 0.0085 | +0.1106 ± 0.0571 |
| seasonal_naive | 1.2042 ± 0.0341 | 0.9890 ± 0.0294 | −0.0633 ± 0.0856 |

LightGBM wins on both accuracy metrics, but the margin over AutoETS is 0.005 MASE against
a fold-to-fold std of 0.018–0.026. **That is well inside the noise band**, and it should be
read as "the GBM matches a well-tuned classical method", not "the GBM wins".

### Where it loses

**Fold 0:** AutoETS beats LightGBM (1.0026 vs 1.0111). LightGBM wins the other four.

**Sparse stratum:** AutoETS beats LightGBM on MASE (0.9931 vs 1.0090).

| Stratum | lgbm | ets | tsb | croston | seasonal_naive |
|---|---:|---:|---:|---:|---:|
| dense | **1.0155** | 1.0421 | 1.0679 | 1.0757 | 1.2843 |
| mid | **1.0328** | 1.0371 | 1.0564 | 1.1340 | 1.2209 |
| sparse | 1.0090 | **0.9931** | 1.0091 | 1.1270 | 1.1068 |

**Bias:** LightGBM is the *most* under-forecasting model in the panel at −0.098, worse than
seasonal naive. Tweedie's point estimate leans toward the mode on zero-heavy data, and with
61.6% zero-days that pulls predictions down. Consequence for E7: use the quantiles, not the
point forecast — the point estimate carries a systematic understock lean.

Horizon decay is not monotonic: week 1 1.0104, week 2 1.0334, week 3 1.0152, week 4 1.0177.

---

### E4-S1 — Global LightGBM point model ✅

- [x] One model, all 720 series; `item_id`/`store_id`/`dept_id` as native categoricals
- [x] Trains in minutes (756s for 5 folds × 5 models), runtime recorded
- [x] **Why not deep learning** recorded in the module docstring and README: at 720 series
      of daily data a global GBM matches classical methods, trains without a GPU, and is
      directly inspectable — which matters because E7 defends an ordering decision
- [x] Tweedie objective, taken from M5 literature for zero-inflated counts and **labelled
      untuned**. Tuning against the same folds used to report the result would make the
      baseline comparison meaningless.
- [x] No train/predict windowing code needed — E2's horizon shift makes the fold split
      `date <= origin`, correct by the panel's construction

### E4-S2 — Quantile models ✅

- [x] `objective='quantile'` at α ∈ {0.5, 0.9, 0.95, 0.99}
- [x] Written to `forecast` with the `quantile` column populated
- [x] **Monotonicity measured, and the fix applied at read time.** 1.74% of rows crossed
      overall; **1.29% inside the CR 0.5–0.95 band E7 actually selects from**, where a
      crossing would return a *lower* stocking quantity for a *higher* service level.
- [x] Fixed by rearrangement (sorting) in `models/quantiles.py` — crossing rate in the
      CR band drops to **0.0000%**. Stored values stay raw so the rate remains auditable.
- [x] Sorting rather than PAVA isotonic regression: Chernozhukov et al. (2010) prove
      rearrangement weakly reduces pinball loss; PAVA minimises squared deviation from the
      raw predictions with no such guarantee for the loss these are judged by. Verified
      empirically — pinball improves at q0.5/q0.9/q0.99 and is worse by 5e-6 at q0.95,
      which is reported rather than described as a uniform win.

### E4-S3 — MLflow tracking ✅

- [x] Params, fold definition, feature list and git SHA logged per fold
- [x] Runs reproducible from the logged config alone
- [x] `run_id` written into `forecast` so any forecast traces back to its run

### E4-S4 — SHAP explainability ✅

- [x] Global mean-|SHAP| plus gain and split importance, persisted to `feature_importance`
- [x] Precomputed on a 20k-row sample — SHAP over 1M rows is far too slow for E8's request
      path, which is why E9 reads this table instead

Top features (fold 4): `item_id` 0.303, `units_roll_mean_28` 0.293,
`days_since_last_sale` 0.203, `units_roll_mean_7` 0.187, `store_id` 0.095,
`units_roll_std_90` 0.089, then calendar (`wday`, `is_weekend`, `week_of_year`) and price.
Notably **no `units_lag_*` feature reaches the top 15** — the rolling statistics carry the
autoregressive signal, and the specific lags add little on top.

### E4-S5 — Honest comparison against E3 ✅

- [x] Same folds, same metrics, same excluded series (718/720 for all models)
- [x] Baseline wins reported: AutoETS on fold 0 and on the sparse stratum
- [x] Per-stratum breakdown produced, and it drove a real design investigation — see below

---

## The 718 vs 720 series gap — investigated, open

LightGBM produced forecasts for 718 series in fold 0 while the baselines produced 720.

**Root cause** (confirmed by query, not assumed): `FOODS_3_595` at `CA_1` and `CA_3` was
first listed **2016-02-13**, after fold 0's test window closes. The GBM reads
`feature_panel`, which drops pre-listing rows by E2-S5's design; the baselines read raw
`fact_sales`, which still carries genuine `units = 0` rows for that period. It is a
**data-source asymmetry between the two model paths**, not a dead series.

**Impact: provably zero.** Those two series are *exactly* the two already excluded from
MASE/RMSSE for every model (no defined training-window scale — the item never sold before
being listed). Re-scoring fold 0 with them explicitly dropped from all models moves every
model's MASE by exactly `0.000000`.

**Status: open, deliberately.** The asymmetry is structural but numerically inert here. A
future item launching mid-fold with nonzero pre-listing activity would break that
inertness. The fix is to restrict every model to the intersection of series each fold
actually produced; recorded as a known limitation rather than patched on a coincidence.

## Stratum-aware routing — investigated, not adopted

Built and measured in `models/hybrid.py` (retained, not deleted). See the README section
"Stratum-aware routing" and **E7-S6**, which revisits the decision using real cost deltas.
