# E8 — API ✅

**Goal:** FastAPI over precomputed results. **Never train on request.**

**Consumes:** `forecast`, `backtest_fold_metrics`, `dim_item_stratum` (via `/hierarchy`'s
`fact_sales` query), `cost_comparison`, `cost_sensitivity`.

**Produces:** a running service E9 consumes.

**Entrypoints:** `run-api`, `run-nightly-refresh`, `run-api-bench`.

---

### E8-S1 — `POST /forecast` ✅

- [x] `{sku, store, horizon}` → point + all 7 quantile levels
- [x] Reads precomputed rows only; **checked, not just claimed** —
      `test_no_handler_imports_a_trainer` asserts `app.py` never imports `lightgbm`,
      `statsforecast`, `hierarchicalforecast` or `mlflow`
- [x] Unknown SKU → 404 naming the SKU and pointing at `/hierarchy`; missing horizon → 422

**No production forecast run exists.** Every forecast in this project is a backtest fold
(0–4); there is no live, clock-driven "today's forecast". `/forecast` serves the most
recently completed fold (`max(fold)`) as the closest analogue, and says so in the response
`note` field rather than presenting a backtest artifact as something it isn't. E8-S4's
nightly job is what would eventually replace this with a real production run.

**Bug caught before it shipped: quantiles were being served raw.** E4/E5 established that
~1.3% of fitted quantile rows are crossed inside the CR band and built `monotonize()` as
the read-time fix every consumer must use. The first version of this endpoint queried
`forecast` directly and forgot it — the fixture test deliberately plants a crossed grid
(`q0.75 < q0.5`, `q0.95 < q0.9`) specifically so this can't regress silently.

### E8-S2 — `POST /optimize` ✅

- [x] `{sku, Cu, Co}` → order quantity + expected cost, aggregated across every store
      carrying the SKU (the brief's request shape has no store field)
- [x] Critical ratio and interpolation computed per request — cheap, no training
- [x] `Cu, Co <= 0` → 422 via Pydantic `Field(gt=0)`

**A real structural bug, caught by inspecting row counts rather than eyeballing output.**
The first version fetched every horizon (1–28) for the SKU with no date filter, mixing 28
days of quantile curves per store into one array before interpolating — output that looked
entirely plausible (order quantities in a sane range) over a demand distribution that was
structurally meaningless: duplicated quantile keys from different days, sorted by value
with no coherent ordering. Caught only by checking `784 rows for one SKU` against the
expected `28 days × 7 quantiles × 4 stores` and asking why. Fixed by pinning to
`OPTIMIZE_HORIZON = 1` — "tomorrow" relative to the latest fold's origin, the classic
single-period newsvendor framing, and the only reading that needs no field the brief didn't
ask for.

**A second bug found by the fixture's edge-case test, not the happy path.** A caller-chosen
`Cu`/`Co` can legitimately push `CR` toward 1.0, past the top fitted quantile (0.99).
`interpolate_quantile` correctly refuses to extrapolate — but the handler didn't catch that
`ValueError`, so it propagated as a bare 500. `test_optimize_ratio_outside_fitted_grid_is_422_not_500`
exists because the happy-path test at a moderate ratio would never have exercised this.
Fixed: caught and returned as 422 with the grid bounds in the message.

**`expected_cost` is a distributional estimate, not a backtest result**, and is documented
as such in the response. `quantile_bin_probabilities()` turns the 7 fitted quantile levels
into a discrete probability distribution via the midpoint rule, and expected cost is the
newsvendor cost averaged over that distribution — the only honest option, since a live
request for a future period has no realised demand to price against the way Phase 7's
money table does.

### E8-S3 — `GET /backtest` and `GET /hierarchy` ✅

- [x] `/backtest` returns **per-fold rows**, not a pre-aggregated summary — "spread, not
      just means" means a client needs the individual folds; an average cannot be
      un-averaged back into a spread
- [x] `/hierarchy` returns the nested State → Store → Dept → Item tree via plain
      `GROUP BY`, not `hierarchicalforecast.aggregate` — building the full summing matrix
      for a UI tree with no reconciliation happening would be pure overkill
- [x] Both are DuckDB reads with optional filters (`model_name`, `metric`, `level`)

Two bonus endpoints, not in the original brief but needed by E9 without recomputation:
`/cost-comparison` (the money table as data) and `/cost-sensitivity` (the precomputed
sweep E9-S4's simulator reads instead of round-tripping per slider pixel).

### E8-S4 — Nightly precompute job ✅

**Design driven by E8-S5's constraint, not the other way round.** DuckDB requires a writer
to have the file to itself — no concurrent readers, no concurrent writers. The API opens a
fresh connection per request rather than holding one, so a write window exists between
requests, but "usually available" is not "safe", especially with Windows' file-locking
semantics for open handles. The job never writes to the live file:

1. Copy the live warehouse to a shadow path.
2. Run every refresh step (GBM → MinT → optimizer) against the **shadow copy**.
3. `os.replace()` the shadow over the live path — atomic on the same volume — retrying a
   few times if a reader's connection happens to be open at that exact instant.
4. Any step failing, or every swap attempt failing, discards the shadow and leaves the live
   file untouched.

- [x] **Idempotent**: every run starts from a fresh copy; no accumulated state between runs
- [x] **Safely re-runnable**: `test_refresh_is_idempotent` runs it twice back to back
- [x] **Failure leaves the previous good results in place**: `test_failing_step_leaves_the_live_file_untouched`
      and `test_swap_gives_up_after_max_attempts_and_preserves_the_live_file` assert the
      live file's content is byte-for-byte the pre-refresh content after a failure
- [x] Steps are injected (`RefreshStep` callables), so tests exercise the orchestration
      logic in milliseconds against fake steps rather than the real 13-minute GBM fit

### E8-S5 — Latency budget ✅

Measured in-process via `TestClient` against the real 2.1M-row warehouse (200 requests per
endpoint, no network hop):

| Endpoint | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|
| `GET /health` | 17.9ms | 20.0ms | 22.4ms | 31.1ms |
| `POST /forecast` | 26.3ms | 29.7ms | 31.1ms | 31.2ms |
| `POST /optimize` | 26.1ms | 28.0ms | 29.9ms | 29.9ms |

- [x] p95 recorded for E10's README
- [x] **Connection handling matches DuckDB's single-writer model**: per-request
      `read_only=True` connections (`deps.get_connection`), never a global handle — see
      E8-S4 above for why this specific choice was load-bearing, not incidental
- [x] `GET /health` implemented

The dominant cost is opening a fresh DuckDB connection per request (~15–20ms baseline
visible in `/health`), which is the price of the never-hold-a-connection design. A
persistent connection pool would shave this down but would reintroduce the exact write
contention the design exists to avoid; the trade is accepted and stated rather than hidden.

### E8-S6 — API tests ✅

- [x] Contract tests per endpoint against a synthetic fixture warehouse (19 tests,
      `tests/test_api.py`) — builds all six tables the API reads in a temp DuckDB file, so
      the suite runs in ~2 seconds and never touches the real warehouse
- [x] Error paths tested explicitly: unknown SKU (404), invalid horizon/Cu/Co (422),
      out-of-grid critical ratio (422), missing required fields (422)
- [x] `test_no_handler_imports_a_trainer` — a static check that request-handling code
      cannot import training-time dependencies, so "never train on request" would fail a
      test before it could ship, not just fail a code review
- [x] Nightly refresh orchestration tested separately (9 tests, `tests/test_precompute.py`)
      with fake steps, including simulated lock contention and permanent lock failure

---

## What broke

- **`/forecast` served raw, potentially crossed quantiles.** Fixed by routing through
  `monotonize()`, the same read-time contract E4/E5 established; a fixture test with a
  deliberately crossed grid pins it so it can't regress silently.
- **`/optimize`'s first version mixed 28 days of quantile curves into one array per store**,
  producing plausible-looking but structurally meaningless order quantities. Found by
  checking a row count against what was expected, not by the numbers looking wrong — they
  didn't. Fixed by pinning to a single period (`OPTIMIZE_HORIZON = 1`).
- **An out-of-grid critical ratio 500'd instead of 422'ing.** `interpolate_quantile`'s
  `ValueError` for extreme `Cu`/`Co` ratios wasn't caught in the handler. Found by a test
  written specifically to probe the edge the happy-path test couldn't reach.
