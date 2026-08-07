# E2 — Features ✅

**Goal:** a feature panel where no row at time _t_ contains information from _t_ or later.

**Consumes:** `fact_sales`, `dim_calendar`.

**Produces:** `feature_panel` — 27 features, 1,080,714 rows, 720 series, ~3.5s build.

**Definition of done:** `tests/test_no_leakage.py` passes. **No story in E3 or E4 starts
until it does.** ✅ Passing — E3 and E4 are unblocked.

Full feature dictionary: [`docs/features.md`](../features.md).

---

### E2-S1 — Feature module skeleton and horizon shifting ✅

**As** every later feature story, **I want** one place that applies the horizon shift,
**so that** the leakage rule is enforced structurally rather than remembered per-feature.

- [x] `features/build.py` exposes `build_features(con, horizon)`
- [x] Every units-derived feature is emitted by `_lag_sql` or `_rolling_sql` — the only
      two code paths in the module that read `units`. A feature cannot be added without
      inheriting the shift.
- [x] Shift is by `horizon`, not by 1: a 28-day-ahead forecast made at _t−28_ cannot use
      actuals from _t−27 … t_
- [x] Partitioned by `(item_id, store_id)`, so frames never bleed across series

### E2-S2 — Lag features ✅

- [x] Lags 7, 14, 28, 365 on `units`
- [x] Each shifted by `horizon`; lags are numbered from the **forecast origin**
- [x] Series shorter than the lag yield NULL, never a forward-filled value
- [x] Exact values asserted against hand-computed expectations in `test_features.py`

### E2-S3 — Rolling statistics ✅

- [x] mean / std / max over 7, 28, 90-day windows
- [x] Frame runs `horizon + window − 1` preceding to `horizon` preceding — the newest row
      it can touch is `t − horizon`
- [x] Test: rolling mean at _t_ equals the manual mean of the correct shifted slice, and
      differs from the mean of the wrongly-unshifted slice
- [x] Partial-window behaviour (`min_periods=1`) pinned by test and documented as a
      deliberate choice

### E2-S4 — Calendar features ✅

- [x] `wday`, `month`, `week_of_year`, `is_weekend`
- [x] `days_to_event` / `days_since_event`, using `dim_calendar`'s future rows so the end
      of the panel is not silently NULL
- [x] `event_type` as well as name — a religious holiday and a sporting event move food
      demand differently
- [x] `snap` flag

### E2-S5 — Price and intermittency features ✅

- [x] `price`; `price_rel_28` (price ÷ 28-day trailing mean, promo proxy); `price_changed`
- [x] `price IS NULL` carries its own `is_listed` indicator rather than being imputed —
      it is an availability signal, not missing data
- [x] `days_since_last_sale`, measured to the forecast origin
- [x] `zero_share_90`
- [x] Price features are **not** horizon-shifted, documented as the deliberate exception:
      a retailer sets its own future shelf prices and genuinely knows them
- [x] Beyond the story: rows before an item's first listing are dropped (316,806 rows).
      They record the absence of a product, not zero demand.

### E2-S6 — `tests/test_no_leakage.py` — the gate ✅

**As** the project, **I want** a test that fails on any feature contaminated by the
future, **so that** leakage is caught by CI rather than by a recruiter.

- [x] **Perturbation test:** corrupt all actuals at _≥ C_, rebuild, assert every feature
      targeting _t < C + horizon_ is bit-identical. Generic — it covers features added
      later by someone who never read the test.
- [x] **Sensitivity check:** past the boundary every units-derived feature must move, so a
      builder emitting constants cannot pass vacuously
- [x] **Exact boundary probe:** a single corrupted cell at _D_ must change a rolling
      feature targeting _D + horizon_ and must not change one targeting _D + horizon − 1_.
      Pins the frame against off-by-one.
- [x] Known-in-advance allowlist pinned against a hard-coded set — the perturbation gate
      corrupts `units` and structurally cannot catch a leak via another source
- [x] Assert no feature is near-perfectly correlated with the target
- [x] Runs on a synthetic panel; fast, no Kaggle download

### E2-S7 — Persist and document the feature panel ✅

- [x] `feature_panel` in DuckDB with a unique index on `(date, item_id, store_id)`
- [x] `build-features` CLI with `--horizon` / `--db-path` / `--table`
- [x] Feature dictionary in [`docs/features.md`](../features.md): name, what it reads,
      shifted or not, and why
- [x] Row count and null profile reported, matching E1's report style
- [x] `fact_sales` rectangularity checked before building — row-based frames only equal
      day-based frames on a dense panel
