# Feature dictionary

Produced by `inventory_engine.features.build` into the `feature_panel` table.
27 features · 1,080,714 rows · 720 series · builds in ~3.5s.

## The rule

**No feature on a row targeting date _t_ may use any observation from _t_ or later.**

Because this project forecasts 28 days ahead, the rule is stronger than "not today". A
forecast for _t_ is made at origin `t0 = t − horizon`, so the newest actual anyone could
have seen is the one at `t0`. Every units-derived feature is computed on the series
**shifted by `horizon`**.

That shift is not applied per feature and remembered. It is folded into the SQL window
frame by exactly two functions — `_lag_sql` and `_rolling_sql` — and every units-derived
feature is emitted by one of them. There is no other code path in the module that reads
`units`, so a new lag or rolling feature cannot be added without inheriting the shift.

Lags are numbered **from the forecast origin**, not from the target date. `units_lag_7`
means "units 7 days before the origin" = `units[t − horizon − 7]`.

## Units-derived (15) — all horizon-shifted

With `horizon = 28`:

| Feature | Reads | Notes |
|---|---|---|
| `units_lag_7` | `t − 35` | |
| `units_lag_14` | `t − 42` | |
| `units_lag_28` | `t − 56` | |
| `units_lag_365` | `t − 393` | 13.1% null — needs >1 year of history |
| `units_roll_mean_7` | `t−34 … t−28` | |
| `units_roll_mean_28` | `t−55 … t−28` | |
| `units_roll_mean_90` | `t−117 … t−28` | |
| `units_roll_std_7/28/90` | as above | sample stddev; NULL on a 1-observation window |
| `units_roll_max_7/28/90` | as above | |
| `zero_share_90` | `t−117 … t−28` | share of zero-sales days; the intermittency signal |
| `days_since_last_sale` | `≤ t−28` | days from the last non-zero sale **to the origin** |

**Partial windows aggregate what exists** (`min_periods=1` semantics), rather than
returning NULL until the window is full. An early `units_roll_mean_90` is therefore a mean
of fewer than 90 days. This is deliberate — a partial mean is more useful to a GBM than a
NULL — but it means the column name is a maximum, not a guarantee. Pinned by
`test_partial_windows_use_available_data`. A feature is NULL only when its **entire** frame
falls before the series start, which is 0.8% of rows.

## Known in advance (12) — deliberately **not** shifted

This is the one exception to the rule above, and it is fenced rather than assumed.

A retailer genuinely knows next month's calendar and its own future shelf prices at
forecast time. Refusing to use them would model a business that does not exist. But the
same exception is exactly how leakage gets introduced accidentally, so the set is pinned
by `test_known_in_advance_allowlist_is_pinned` against a hard-coded expectation — widening
it forces a reviewer to justify why the retailer really does know that column in advance.

| Feature | Source | Why it is knowable |
|---|---|---|
| `wday`, `month`, `week_of_year`, `is_weekend` | `dim_calendar` | Calendar, published years ahead |
| `event_type` | `dim_calendar` | Holiday schedule; `'none'` when no event |
| `days_since_event`, `days_to_event` | `dim_calendar` | Uses the 28 future calendar days past the sales panel, so this is defined at the end of the window instead of silently NULL |
| `snap` | `fact_sales` | SNAP benefit dates are set by statute |
| `price` | `fact_sales` | The retailer sets its own shelf prices |
| `price_rel_28` | derived | `price ÷` trailing 28-day mean price. Promo proxy: today's price against the recent norm |
| `price_changed` | derived | Price differs from yesterday's |
| `is_listed` | derived | `price IS NOT NULL` — availability, not missing data |

## Panel decisions

**Pre-listing rows are dropped.** 316,806 rows (22.7%) precede the date an item was first
priced at a store. Those are not zero demand; they are the absence of a product. Training
on them teaches the model to forecast shelf gaps. Post-introduction zero-price gaps are
kept, carrying `is_listed = false`.

**`fact_sales` must be rectangular**, and the builder checks it rather than assuming.
Row-based window frames (`ROWS BETWEEN … PRECEDING`) only equal day-based frames on a
dense panel; a single missing day would silently make every frame reach further back in
time than intended, and nothing downstream would notice. Pinned by
`test_non_rectangular_panel_is_rejected`.

**`horizon` is stored as a column.** The panel is valid for any horizon up to the one it
was built with. Deriving it at read time from a date subtraction invites off-by-one errors
in the place they are hardest to see.

## Test coverage

| File | Proves |
|---|---|
| `tests/test_no_leakage.py` | Corrupting actuals at/after `C` leaves every feature targeting `t < C + horizon` bit-identical. Plus a sensitivity check so the gate cannot pass vacuously, and a single-cell probe pinning the frame to exactly `horizon PRECEDING`. |
| `tests/test_features.py` | Every feature's arithmetic against hand-computed expectations on a deterministic series. |
