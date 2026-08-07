# E2 — Features 🔜

**Goal:** a feature panel where no row at time _t_ contains information from _t_ or later.

**Consumes:** `fact_sales`, `dim_calendar`.

**Produces:** `feature_panel`.

**Definition of done:** `tests/test_no_leakage.py` passes. **No story in E3 or E4 starts
until it does.** Leakage is the single highest-risk failure mode in this project: it does
not crash, it does not look wrong, it just produces backtest numbers that evaporate in
production.

---

### E2-S1 — Feature module skeleton and horizon shifting ⬜

**As** every later feature story, **I want** one place that applies the horizon shift,
**so that** the leakage rule is enforced structurally rather than remembered per-feature.

- [ ] `features/build.py` exposes `build_features(con, horizon: int) -> None`
- [ ] Every lag/rolling feature passes through a single shift helper — a feature cannot
      be added without going through it
- [ ] Shift is `.shift(horizon)`, not `.shift(1)`: a 28-day-ahead forecast made on day
      _t_ cannot use day _t+1..t+27_ actuals
- [ ] Grouped by `(item_id, store_id)` so shifts never bleed across series boundaries

### E2-S2 — Lag features ⬜

- [ ] Lags 7, 14, 28, 365 on `units`
- [ ] Each lag shifted by `horizon` before use
- [ ] Series shorter than the lag yield NULL, not a silently forward-filled value

### E2-S3 — Rolling statistics ⬜

- [ ] mean / std / max over 7, 28, 90-day windows
- [ ] Window computed on already-shifted values — the window must not include _t_
- [ ] Test: rolling mean at _t_ equals the manual mean of the correct shifted slice

### E2-S4 — Calendar features ⬜

- [ ] day-of-week, month, week-of-year
- [ ] days-to-next-event and days-since-last-event (uses `dim_calendar` future rows)
- [ ] Event **type** as well as name — a religious holiday and a sporting event move
      food demand differently
- [ ] SNAP flag

### E2-S5 — Price and intermittency features ⬜

- [ ] Current price; price ÷ 28-day rolling mean price (promo proxy); price-change flag
- [ ] `price IS NULL` handled as an availability signal, not missing data — carries its
      own indicator column rather than being imputed
- [ ] Days since last non-zero sale
- [ ] % zero-days in trailing 90
- [ ] Price features are **not** horizon-shifted, and the reason is documented: future
      prices are set by the retailer and genuinely known at forecast time. This is the
      one deliberate exception to E2-S1 and it must be explicit.

### E2-S6 — `tests/test_no_leakage.py` — the gate ⬜

**As** the project, **I want** a test that fails on any feature contaminated by the
future, **so that** leakage is caught by CI rather than by a recruiter.

- [ ] **Perturbation test:** for each series, corrupt all actuals at _≥ t_, rebuild
      features, assert every feature value at _t_ is unchanged. This catches leakage
      generically — it does not need to know how each feature was computed.
- [ ] Explicit allowlist for the known-in-advance columns (calendar, event, SNAP, price).
      Adding a column to that allowlist must be a deliberate, reviewed edit.
- [ ] Assert no feature column is perfectly correlated with the target
- [ ] Runs on a synthetic panel, so it is fast and needs no Kaggle download

### E2-S7 — Persist and document the feature panel ⬜

- [ ] `feature_panel` written to DuckDB with `(date, item_id, store_id)` PK
- [ ] Feature dictionary in `docs/features.md`: name, definition, shifted?, why
- [ ] Row count and null profile logged, matching the E1 report style
