# E7 — Optimization layer ✅

**Goal: this is the point of the project.** Turn the demand distribution into stocking
decisions and report the money.

**Consumes:** `forecast` (base LightGBM quantiles at `item_store`), `fact_sales`.

**Produces:** `order_policy`, `cost_comparison`, `cost_sensitivity`.

**Entrypoint:** `run-optimize`.

---

## THE MONEY TABLE

Cost assumptions: margin 30%, spoilage 60%, holding 2%/day → **CR = 0.4087**.
100,720 (series × day) decisions across 5 folds, priced against **realised** demand.

| Policy | Stockout rate | Waste units | Holding cost | Total cost | vs naive |
|---|---:|---:|---:|---:|---:|
| Current (naive: last week's sales) | 31.1% | 78,106 | $3,198 | $169,172 | — |
| Fixed 95% service level | 6.1% | 326,957 | $13,118 | $415,644 | **+145.7%** |
| **Newsvendor + our forecast** | 48.0% | **20,714** | **$801** | **$120,797** | **−28.6%** |

**Blindly maxing service level costs 2.5× more than doing nothing sophisticated.** The 95%
policy achieves the best stockout rate in the table — 6.1% — and is by far the most
expensive, because hitting it requires 327k units of stock that spoil. That row is the
argument of this project, and it is why it stays in the table.

The newsvendor policy accepts a *higher* stockout rate than naive (48.0% vs 31.1%) and
still costs 28.6% less, because it cuts waste by 73%. On thin-margin perishables an empty
shelf is cheaper than a full bin.

---

### E7-S1 — Cost model, explicit and labelled ✅

- [x] `Cu` = lost gross margin (the retailer keeps the unsold unit); `Co` = spoilage +
      holding on the unit cost
- [x] Every constant in `optimize/costs.py`, each annotated **assumption**
- [x] M5 ships prices but no margins, shelf life or holding costs — so everything below
      price is invented, and the module says so in its first paragraph
- [x] The conclusion depends on the **ratio** Cu/Co rather than either level, which is why
      E7-S4 sweeps it

### E7-S2 — Newsvendor order quantity ✅

- [x] `CR = Cu / (Cu + Co)` = **0.4087**; order the CR-th quantile
- [x] Linear interpolation between fitted quantile levels; **outside the grid it raises**
      rather than extrapolating a tail the model never estimated
- [x] **The counterintuitive result, surfaced as a headline:** optimal service level is
      **41%**, not 95%. For a thin-margin perishable, ordering to a 95th percentile is not
      caution — it is a systematic and expensive error.
- [x] Written to `order_policy`

**The brief's quantile grid could not express its own premise.** It specified
{0.5, 0.9, 0.95, 0.99}, but fresh-food critical ratios land between 0.25 and 0.63 — below
the lowest fitted level. Every order would have been clamped at the median and this entire
finding would have been invisible. The grid was extended to
{0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99} and the model refit.

**Forecast source is an explicit decision.** Base, unreconciled LightGBM quantiles at
`item_store` — *not* the MinT-reconciled ones, even though those were computed more
recently. Orders are placed at `item_store`, so the cost function optimises against the most
accurate forecast at that grain. `USE_RECONCILED = False` is a named constant pinned by
test. Reconciled forecasts serve E9's aggregate planning views, where coherence is the
actual requirement.

### E7-S3 — The money table ✅

- [x] Three policies on identical held-out folds
- [x] **Middle row kept** — it demonstrates that maxing service level costs more
- [x] Costs computed from **realised** demand, never from the forecast that produced the
      order. A test asserts a wildly biased order is still punished.
- [x] Persisted to `cost_comparison`

### E7-S4 — Sensitivity over Cu/Co ✅

| Spoilage | CR | naive | fixed 95% | newsvendor |
|---:|---:|---:|---:|---:|
| 0.0 | 0.955 | 73,225 | 22,094 | **21,756** |
| 0.2 | 0.661 | 105,207 | 153,277 | **86,942** |
| 0.4 | 0.505 | 137,190 | 284,461 | **108,668** |
| 0.6 | 0.409 | 169,172 | 415,644 | **120,797** |
| 0.8 | 0.343 | 201,155 | 546,827 | **128,709** |
| 1.0 | 0.296 | 233,138 | 678,011 | **133,417** |

- [x] Newsvendor wins at **every** spoilage rate — the result is not an artifact of one
      assumption
- [x] **The region where fixed-95% stops working is identified:** it beats naive only at
      spoilage ≈ 0. From 0.2 upward it is worse than naive, and by spoilage 1.0 it costs
      2.9× as much. A flat high service level is only defensible for non-perishables.
- [x] Persisted to `cost_sensitivity`, feeding E9's simulator

### E7-S5 — Savings attribution ✅

| Combination | Total cost | Stockout rate | Waste units |
|---|---:|---:|---:|
| naive forecast + fixed 95% | 800,704 | 18.8% | 603,848 |
| naive forecast + newsvendor CR | 147,688 | 40.8% | 50,875 |
| our forecast + fixed 95% | 415,644 | 6.1% | 326,957 |
| **our forecast + newsvendor CR** | **120,797** | 48.0% | 20,714 |

**The policy is worth far more than the model.** Holding the forecast fixed and switching
policy (fixed 95% → newsvendor CR) cuts cost by **71%**. Holding the policy fixed and
switching forecast (naive → ours) cuts it by **18%**.

That is the honest headline, and it is more interesting than an accuracy win: most of the
value in this project comes from asking *what service level is actually optimal* rather
than from forecasting better. A team with a mediocre forecast and the right cost model
beats a team with a good forecast and a 95% target.

The "naive forecast + service level" rows apply our model's quantile ratio to last week's
actuals, since a naive forecast has no distribution of its own. That is crude by
construction and labelled as such — it exists to separate the two effects, not to be a
serious policy.

### E7-S6 — Stratum-aware routing, decided on cost ✅ — **rejected**

Deferred here from E4 so it could be settled with money rather than a preference between
MASE and RMSSE. The hybrid was rebuilt with the full 7-level quantile grid (AutoETS
prediction intervals, including sub-median bounds) and priced.

| Policy source | Total cost | Stockout rate | Waste units |
|---|---:|---:|---:|
| **plain LightGBM** | **120,797** | 48.0% | 20,714 |
| stratum-routed hybrid | 125,123 | 46.8% | 27,816 |
| | **+4,326 (+3.58%)** | | |

Per stratum, on the band routing was supposed to help:

| Stratum | lgbm cost | hybrid cost | cost share |
|---|---:|---:|---:|
| dense | 60,320 | 60,320 | 49.9% |
| mid | 36,066 | 36,066 | 29.9% |
| **sparse** | **24,411** | **28,737 (+17.7%)** | 20.2% |

**Routing is worse, and worse precisely where it was meant to help.** Two reasons, both
findings in their own right:

1. **AutoETS wins on the point forecast and loses on the distribution.** Its Gaussian
   prediction intervals are badly mis-specified for intermittent count demand — symmetric
   around a near-zero mean, with the lower tail clipped at zero and the upper tail
   overstated. That produces 9,144 waste units on sparse against LightGBM's 2,042, a 4.5×
   difference. The newsvendor consumes the *distribution*, not the point, so a better MASE
   is simply the wrong qualification. This corroborates the pinball-loss evidence from E5.
2. **The premise was wrong about where cost lives.** Routing was motivated by "perishables
   skew sparse". Sparse carries only **20.2%** of total cost; dense carries **49.9%**,
   because dense items move enough volume that their stockouts and spoilage dominate in
   absolute terms. Routing optimises the smallest cost pool.

Decision: **plain LightGBM everywhere.** `models/hybrid.py` is retained and tested as the
record of an investigated alternative.

---

## What broke

- **The brief's quantile grid was too narrow to express its own conclusion** — see E7-S2.
  Caught by computing realistic critical ratios *before* wiring the optimizer, rather than
  discovering clamped orders in the output.
- **`run-gbm --replace` was deleting E6's reconciled forecasts.** `DELETE ... WHERE
  model_name = 'lgbm'` matched reconciled rows and aggregate-level base rows too, so the
  refit would have silently invalidated the whole reconciliation with no error raised. Now
  scoped to `reconciled = FALSE AND level = 'item_store'`.
