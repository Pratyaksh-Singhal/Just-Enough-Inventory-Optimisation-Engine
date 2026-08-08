# E6 — Hierarchical reconciliation ✅

**Goal:** SKU forecasts that sum exactly to store and state forecasts. The technical
differentiator of the project.

**Consumes:** `forecast` (base, `reconciled = FALSE`), `fact_sales`.

**Produces:** `forecast` rows with `reconciled = TRUE`; `coherence_check`;
`backtest_fold_metrics` at every level, plus **WRMSSE** (which E5 deferred here).

**Entrypoint:** `run-mint`.

---

## Headline result

**Max coherence gap: 42.76 → 4.55e-13.** Reconciled forecasts sum exactly, to floating-point
tolerance.

**WRMSSE (M5 official): 0.81173 → 0.78411**, a 3.4% improvement.

---

### E6-S1 — Build the hierarchy ✅

- [x] Four-level nested tree over **737 series**: 1 state + 4 stores + 12 store×dept +
      720 item×store
- [x] Summing matrix `S` (737 × 720) via Nixtla `hierarchicalforecast.aggregate`
- [x] **Validated against actuals** — aggregating real units up the tree reproduces each
      parent exactly. If `S` were wrong, MinT would still return coherent-looking output,
      coherent with respect to the wrong structure, so it is checked rather than trusted.
- [x] Every parent asserted to have children

**Deviation from the brief, recorded not hidden.** The brief specifies
`State → Store → Category → Dept → Item`. Phase 1's scope fixes `cat_id = FOODS`, which
makes the Category level **identical** to Store — every store has exactly one category.
Including it would put duplicate rows in `S`, leaving the covariance rank-deficient and
MinT's inverse ill-conditioned: a numerical problem created purely by encoding a level that
carries no information. Dropped, with `test_category_level_is_deliberately_absent` pinning
the decision. On a multi-category scope it restores unchanged.

### E6-S2 — Base forecasts at every level ✅

- [x] Bottom level: production LightGBM forecasts from E4
- [x] Aggregate levels: AutoETS, fit independently
- [x] Deliberately **incoherent** before reconciliation — max gap 42.76 units — which is
      exactly what E6-S4 measures away
- [x] Aggregate base forecasts persisted to `forecast` so the before/after comparison is
      queryable rather than only recomputable

AutoETS at aggregate levels is a considered choice, not expedience: aggregating hundreds of
SKUs washes out the intermittency that made the bottom level hard, leaving dense, strongly
seasonal series a classical method handles well. Building the full E2 feature panel at each
aggregation level would be substantial work to arrive at a worse fit on easier data.

### E6-S3 — MinT reconciliation ✅

- [x] MinT via Nixtla `hierarchicalforecast`
- [x] Covariance estimator documented **with the reason it changed**
- [x] Reconciled rows written with `reconciled = TRUE`, base rows retained so before/after
      is a single query
- [x] Negative reconciled forecasts clipped at zero — MinT is unconstrained and can push a
      low forecast below zero; demand cannot be negative and E7 orders against this number

**The estimator choice needed a correction.** `mint_shrink` (Schäfer–Strimmer shrinkage)
was the original choice and is the textbook default. It cannot be used here: it estimates
the residual covariance from **in-sample** one-step residuals, requiring fitted values for
all 737 series in every fold. E4 persists forecasts, not models, so those residuals do not
exist, and regenerating them would mean refitting the GBM on every training window purely
to obtain them.

`wls_struct` is the honest alternative rather than a silent fallback — it weights each node
by the number of bottom series aggregated into it, structural information from `S` alone,
which is the case Wickramasuriya et al. (2019) propose it for when residuals are
unavailable.

### E6-S4 — Before/after coherence table ✅

| Parent → child | Mean gap before | Max gap before | Max gap after |
|---|---:|---:|---:|
| state → store | 19.7723 | 42.7629 | 4.55e-13 |
| store → store_dept | 4.5730 | 23.0094 | 2.84e-13 |
| store_dept → item_store | 9.0043 | 37.6201 | 8.53e-14 |

- [x] Gap = \|parent − Σ children\|, per level, before and after
- [x] After: zero to floating-point tolerance, and **asserted in tests**, not just printed
- [x] Persisted to `coherence_check` for the dashboard

### E6-S5 — Accuracy change at aggregate levels ✅

**RMSSE**

| Level | base | MinT | delta | |
|---|---:|---:|---:|---|
| state | 0.7144 | 0.6862 | −0.0282 | MinT better |
| store | 0.8026 | 0.7807 | −0.0219 | MinT better |
| store_dept | 0.9263 | 0.8635 | −0.0628 | MinT better |
| item_store | 0.7576 | 0.7615 | **+0.0039** | **MinT worse** |

**MASE**

| Level | base | MinT | delta | |
|---|---:|---:|---:|---|
| state | 0.7696 | 0.7183 | −0.0513 | MinT better |
| store | 0.8121 | 0.7931 | −0.0190 | MinT better |
| store_dept | 0.9809 | 0.9086 | −0.0723 | MinT better |
| item_store | 1.0192 | 1.0348 | **+0.0156** | **MinT worse** |

- [x] E5 metrics rerun on reconciled forecasts, same folds
- [x] Change reported **at every level**, including where it got worse
- [x] **This is the expected trade-off and the price is stated.** MinT improves every
      aggregate level and costs ~1.5% MASE at the bottom. Coherence is purchased, not free.

Whether that trade is worth taking is a **cost** question, not an accuracy one — the bottom
level is what E7 places orders against, so a 1.5% MASE degradation there has to be weighed
against store- and state-level plans that finally agree with those orders. E7 has the cost
function to settle it.

---

## What broke

- **The 718/720 asymmetry from E4 stopped being harmless.** MinT needs a base forecast for
  every column of `S`; a missing series aborts reconciliation outright. `FOODS_3_595` at two
  stores has no LightGBM rows in fold 0 (E2 drops pre-listing rows from the feature panel),
  which was numerically inert for scoring but **structurally fatal** here. Fixed properly:
  unlisted series are filled with zero, which is the correct value rather than a
  placeholder — no shelf presence means no demand, exactly what the raw actuals show.
- **`mint_shrink` failed on missing in-sample residuals** — see E6-S3.
- **I reintroduced the E5 duplicate-metrics bug.** `score_levels` wrote its own
  `(lgbm, item_store)` rows while E5's scorer already owned that key, giving 10 rows where
  5 belong. The two paths agreed numerically — useful independent corroboration of E5's
  numbers — but the duplication would hide any future divergence by averaging it. Fixed by
  having `score_levels` skip the bottom level for the base model.
- **`level_comparison` averaged the per-stratum and per-horizon rows into the headline.**
  Missing a `stratum IS NULL AND horizon IS NULL` filter made bottom-level RMSSE display as
  0.6079 instead of 0.7576 — a number that looked plausible and was wrong. Caught by
  cross-checking against E5's stored value rather than trusting the new report.
