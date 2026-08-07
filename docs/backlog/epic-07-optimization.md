# E7 — Optimization layer ⬜

**Goal: this is the point of the project.** Turn the demand distribution into stocking
decisions and report the money.

**Consumes:** `forecast` (quantiles, reconciled), `fact_sales` (prices, actuals).

**Produces:** `order_policy`, `cost_comparison`, `cost_sensitivity`.

---

### E7-S1 — Cost model, explicit and labelled ⬜

**As** a reviewer, **I want** every cost assumption stated as an assumption, **so that**
I can judge the result rather than take it on faith.

- [ ] `Cu` (understock) = lost margin; `Co` (overstock) = spoilage, near 100% of unit cost
      for fresh food
- [ ] Every constant lives in one module, annotated with its source and its uncertainty
- [ ] README labels them **assumptions**, not findings
- [ ] Unit cost derived from M5 `sell_price` and a stated margin assumption — the margin
      is invented, and the README says so

### E7-S2 — Newsvendor order quantity ⬜

- [ ] `CR = Cu / (Cu + Co)`; order the `CR`-th quantile of the demand distribution
- [ ] Interpolate between the fitted α levels {0.5, 0.9, 0.95, 0.99} when `CR` falls
      between them; document the interpolation and its error
- [ ] **Surface the counterintuitive result explicitly:** for high-spoilage thin-margin
      items, optimal service level lands near 0.6, *not* the 0.95 everyone assumes.
      This is a headline finding, not a footnote.
- [ ] Written to `order_policy`

### E7-S3 — The money table ⬜

**The single most important artifact in the repo.**

| Policy | Stockout rate | Waste units | Holding cost | Total cost |
|---|---|---|---|---|
| Current (naive: last week's sales) | | | | |
| Fixed 95% service level | | | | |
| **Newsvendor + our forecast** | | | | **(−X%)** |

- [ ] All three policies simulated on identical held-out folds
- [ ] **The middle row stays in.** It is not filler — it demonstrates that blindly maxing
      service level costs *more* than optimizing, which is the argument of the project.
- [ ] Costs computed from realised demand, not from forecast demand — simulating a policy
      against your own forecast measures nothing
- [ ] Persisted to `cost_comparison`

### E7-S4 — Sensitivity over Cu/Co ⬜

**As** a sceptic, **I want** to know whether the savings survive different cost
assumptions, **so that** I can tell a real result from one tuned to a chosen ratio.

- [ ] Sweep the `Cu/Co` ratio across a wide range; recompute total cost for all policies
- [ ] Identify the region where newsvendor stops winning, if one exists — **report it**
- [ ] Persisted to `cost_sensitivity`, feeding the E9 simulator

### E7-S5 — Savings attribution ⬜

**As** the README, **I want** to know how much of the saving comes from the better
forecast versus from the better *policy*, **so that** the headline number is honest.

- [ ] Decompose: naive forecast + newsvendor policy, vs our forecast + fixed 95%, vs both
- [ ] If most of the gain is the policy rather than the model, say so plainly. It is a
      more interesting finding than a marginal accuracy win, and it is the more likely
      outcome.
