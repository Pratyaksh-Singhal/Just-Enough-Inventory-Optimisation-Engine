# Test Automation Summary

Generated 2026-08-21 for the dashboard rebuild and the tier-2 service.

## Framework

No new dependency. The project is pytest + httpx with no `package.json`, and
`scripts/build_dashboard.py` already treats node as optional. Playwright would have added a
Node toolchain, a browser download and a CI job to a repository that deliberately has none,
so the JavaScript is executed through node when present and the suite skips when it is not
— the same arrangement `tests/test_service_postgres.py` uses for a database.

## Generated tests

### Page behaviour — executes the shipped JavaScript
`tests/test_dashboard_behaviour.py` (25 tests) + `tests/js/page_checks.js`

Lifts `parseCsv`, `dateKey`, `windowTotals` and `quantile` out of the built page and runs
them. This is the only layer that computes anything; every other check reads the page as
text, and a page can be well-formed and still produce the wrong order quantity.

- unreadable units (`N/A`, blank cell, ragged row) are skipped and counted, not read as a
  real zero-sales day
- a written `0` still survives as a real day of no sales
- column matching is exclusive: `sales_date` is a date, `store_name` is a store,
  `unit_price` is not units
- refusals name the missing column
- all five recognised date formats return one YYYYMMDD scale; `week 3` and `banana` return
  null so the file's own order is kept
- day-first dates sort chronologically
- degenerate samples (empty, single, constant) stay finite

### Page contract — structure, no browser
`tests/test_dashboard_contract.py` (15 tests)

Each test exists because the corresponding bug shipped.

- `__DATA__` / `__FONTS__` substituted; DATA parses; four faces embedded, no CDN link
- every queried element id exists; every tab and `data-goto` resolves to a section
- no bare `svg {}` rule (it inflated every icon); every chart opts into `.fluid`
- no SVG stroke or fill from a presentation attribute (it rendered the icon invisible)
- no `padding` shorthand on a `.wrap` selector (it zeroed the vertical padding)
- narrow-screen overrides come **after** the rules they override (the whole tier was inert)
- no function declared twice in one block (a shadowed `renderCurve` survived 39 lines)
- exactly one handler reads `#csvFile` (two racing readers made picking a file
  non-deterministic)
- every emitted class has a rule
- the committed page matches its template

### API contract — the envelope the page renders
`tests/test_service_upload.py` (+4 tests)

- a refusal carries `column_mapping` and `rows_read`
- a refusal names every product and its shortfall
- an unreadable file returns a plain string, not a structure — the page branches on this
  and rendering the wrong shape produced `[object Object]`
- a partial pass is a 201 and still reports what was left out

### Service invariants
`tests/test_service_priors.py` (+5), `tests/test_service_uplift.py` (+1)

- a prior row whose `direction` contradicts its `suggested_multiplier` is refused
- a multiplier of zero or less is refused
- the shipped table obeys both rules
- a partial festival missing from the coverage map degrades instead of raising

## Verification

The new tests were checked against reintroduced bugs, not just run green:

| bug put back | caught by |
|---|---|
| `svg { width: 100% }` | `test_no_bare_svg_rule_can_inflate_the_icons` |
| `Number(cleaned)` without the empty check | 3 behaviour tests |

## Coverage

- **576 tests pass** (was 525 before this session's review)
- API endpoints: upload, health, forecast run/status, datasets read/delete, retention purge
- Dashboard: parsing, dates, degenerate maths, structure, template freshness
- Not covered: real browser rendering, layout at a given viewport, drag-and-drop, the
  polling loop against a live worker

## Next steps

- The narrow-screen assertion checks source order, not computed styles. A real viewport
  test needs a browser; the ordering rule is what actually broke.
- The full-forecast polling loop is still only exercisable by hand against a live service.
- Multi-store grouping, the 2027 calendar expiry and slider-restore desync are known and
  deferred, with no tests yet.
