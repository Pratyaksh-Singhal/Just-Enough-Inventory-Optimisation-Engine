# E8 — API ⬜

**Goal:** FastAPI over precomputed results. **Never train on request.**

**Consumes:** `forecast`, `backtest_fold_metrics`, `coherence_check`, `order_policy`,
`cost_sensitivity`.

**Produces:** a running service the dashboard consumes.

---

### E8-S1 — `POST /forecast` ⬜

- [ ] Request `{sku, store, horizon}` → point + quantile forecasts
- [ ] Reads precomputed rows from `forecast`; no model inference in the request path
- [ ] Pydantic request/response models; unknown SKU returns 404 with a useful message,
      not a 500

### E8-S2 — `POST /optimize` ⬜

- [ ] Request `{sku, Cu, Co}` → recommended order quantity + expected cost
- [ ] Critical ratio computed per request (cheap); quantiles read from `forecast`
- [ ] Must be fast enough to back a dragged slider in E9 — this endpoint sits behind the
      most important screen in the project
- [ ] Validates `Cu, Co > 0` and returns 422 with a clear message otherwise

### E8-S3 — `GET /backtest` and `GET /hierarchy` ⬜

- [ ] `/backtest` returns stored fold metrics with spread, not just means
- [ ] `/hierarchy` returns the drill-down tree for the dashboard
- [ ] Both served from DuckDB reads

### E8-S4 — Nightly precompute job ⬜

- [ ] Batch job refreshes `forecast` and `order_policy` into DuckDB
- [ ] Idempotent and safely re-runnable
- [ ] Failure leaves the previous good results in place rather than a half-written table

### E8-S5 — Latency budget ⬜

- [ ] p95 inference time measured and recorded for E10's README
- [ ] Connection handling appropriate for DuckDB's single-writer model — concurrent
      readers are fine, a writer during a read is not
- [ ] Health endpoint

### E8-S6 — API tests ⬜

- [ ] Contract tests per endpoint against a fixture warehouse
- [ ] Error paths tested, not just happy paths
- [ ] Assert no code path in a request handler trains or fits anything
