# What Went Wrong, and What It Changed

A record of decisions this project got wrong the first time, what the wrong version actually
produced, and what replaced it.

It exists because the interesting part of a forecasting system is not the version that
worked. Every one of the failures below was **code working exactly as written**, returning a
confident, plausible, wrong answer. None of them raised an exception. None would have been
caught by asking whether the tests passed — several were found only by running the thing on
real data and reading the output.

Each entry names where to verify it in the source.

---

## The Four That Mattered Most

| # | Symptom | Actual cause | Found by |
| --- | --- | --- | --- |
| 1 | Christmas measured at **0.84x** — demand *falling* at Christmas | One mean averaged rising demand against stores being shut | Running it on real M5 data |
| 2 | An order of **974 units at a 41% service level** on a series averaging 399 | A daily method used at the wrong grain | Running it on real M5 data |
| 3 | "The simple baseline beat the model" reported off a **2% gap** | A true number supporting an unsupported conclusion | Reading the response text |
| 4 | The festival calendar silently lost **Eid in all nine years** | The holiday names depended on the host's locale | CI, on a different OS |

---

## 1. Christmas at 0.84x

**Where:** `src/inventory_engine/service/uplift.py`

The first version measured one mean over `[festival - lead, festival + tail]` and divided by
a baseline. Run against real M5 data it reported Christmas at **0.84x** — demand falling at
Christmas — which is nonsense until you look at the daily totals:

```
24 Dec: 1,168 units.   25 Dec: 0 units.   26 Dec: 1,135 units.
Thanksgiving: 569 against a ~1,100 norm.
```

The stores are shut. A single window mean was averaging stock-up demand rising against
trading hours collapsing, and landing near 1.0 having cancelled two real and opposite
effects.

**What changed.** The measurement was split in two:

- `lead_ratio` — the run-up, **excluding the festival day**. This is what an order is placed
  against; stock has to be on the shelf beforehand.
- `day_ratio` — the day itself and its tail, reported separately so a closure is visible
  rather than quietly averaged into someone's purchase order.

`closed_days` counts days inside the window with zero sales against a non-zero baseline.
That is a supply fact, not a demand fact, and it is excluded from the demand ratios.

**Why it is in this document.** The arithmetic was never wrong. Averaging across a regime
change — open shop, closed shop — was wrong, and the output looked like a modest, plausible
finding rather than an error.

---

## 2. The 996-Unit Order

**Where:** `src/inventory_engine/service/pipeline.py`, `baseline_total`

The first version of the horizon-total baseline used seasonal naive: project the last seven
days four times over, then add the quantiles of its rolling-origin residuals.

`FOODS_3_086-CA_3` happens to end on a week selling 249 units against a 100-unit average
week. Seasonal naive projected **996** units for the next 28 days on a series whose mean
28-day total is **399**. The order quantity came out at **974 units at a 41% service level**
— two and a half times average demand, while claiming to be deliberately under-stocking.

Seasonal naive was not malfunctioning. Carrying the last week forward is its definition, and
it is a reasonable *daily* forecast. But a single anomalous final week should not scale a
month's purchase order.

**What changed.** The horizon total now uses the empirical quantiles of the 28-day totals the
series has actually produced — which is tier 1's rule, verbatim, so the two tiers answer the
same question and stay comparable. One unusual week cannot move that distribution far.

**Stated limitation, not hidden:** the windows overlap, so the totals are autocorrelated and
the tails are slightly tighter than independent sampling would give. Tier 1 has the same
property.

**Why it is in this document.** The failure was a mismatch between a method and a grain. The
service was simultaneously reporting a deliberate 41% service level and ordering as though
it wanted 95% — two internally consistent statements that contradicted each other.

---

## 3. "The Baseline Beat the Model"

**Where:** `src/inventory_engine/service/pipeline.py`, `DECISIVE_MARGIN`

On a real 120-day upload the service reported that the simple baseline had beaten the model.
The evidence was a **2% gap on a single fold**: pinball loss 34.515 against 35.194. Meanwhile
MASE said the model was clearly the better forecast — 0.830 against 1.082.

Both statements were arithmetically true. The conclusion was supported by neither.

**What changed.** `DECISIVE_MARGIN = 0.05`. Below a 5% relative gap the two methods are
called a **draw** rather than a win for either. The simpler method still wins a draw — equal
evidence does not justify the extra machinery — but the response now says "too close to call"
and quotes both metrics, including when they disagree:

> Too close to call: the model and the simple baseline scored within 5% of each other on your
> own data (ordering loss 34.515 vs 35.194 over 1 backtest fold), which is not enough to
> prefer one. The simpler method is used.

**Why it is in this document.** This is the failure mode that worries me most in ML systems:
not a wrong number, but a *true* number presented as a conclusion it cannot carry. Nothing
was broken. The service was over-claiming, and over-claiming reads exactly like confidence.

---

## 4. A Calendar That Depended on the Host's Locale

**Where:** `src/inventory_engine/service/festivals.py`, `LIBRARY_LANGUAGE`

The festival calendar reads its dates from the maintained `holidays` library rather than a
hand-typed table, and matches the library's holiday names as strings.

Those strings are gettext-translated. `holidays` declares India's `default_language` as
`en_IN`, where Eid al-Fitr is `"Id-ul-Fitr"` and Eid al-Adha is `"Id-ul-Zuha (Bakrid)"`.
Under `en_US` the same two are `"Eid al-Fitr"` and `"Eid al-Adha"`.

With no language pinned, the names resolve from **the ambient locale of whatever machine is
running**. The suite passed on the developer machine and on the Linux CI runner, where
gettext landed on `en_US`. It failed on the Windows runner, where it landed on `en_IN`: the
calendar came back with **no Eid in any of the nine years covered**, while every other
festival was fine — because `"Diwali (Deepavali)"` reads identically in both.

Reproduced before fixing:

```
UNPINNED, LANGUAGE=en_IN   ->  58 occurrences,  eid_al_fitr years = []
PINNED en_US               ->  67 occurrences,  eid_al_fitr years = [2019 ... 2027]
```

**What changed.** `LIBRARY_LANGUAGE = "en_US"`, passed explicitly. It is a lookup key, not a
presentation choice — every name a user sees comes from this project's own `Source.name`.
A regression test now runs the whole calendar under `en_IN`, `hi_IN`, `C`, `bn_IN.UTF-8` and
an empty locale.

**How it was caught.** The coverage test — which re-derives the calendar from the library on
every run specifically so that an upgrade cannot silently shrink it — did its job. What it
did *not* do was say what the library offered instead of what was missing, which cost two
wrong diagnoses before the real one. The assertions now print the installed version and the
names actually available.

**Why it is in this document.** A calendar whose contents depend on the host's locale is not
a calendar. The bug was invisible on two of three platforms, and the failure it produced —
one festival absent, everything else correct — looked like a data problem rather than an
environment problem.

---

## Four More, Briefly

| Where | What happened | What changed |
| --- | --- | --- |
| `worker.py`, `FIT_THREADS` | Two fitting threads on the reasoning that LightGBM releases the GIL. True, and beside the point: its training loop is OpenMP-parallel, and concurrent `train()` calls can deadlock in the OMP runtime. Two demo runs finished in 12s; the third wedged at a flat 91 seconds of CPU with the job stuck in `running` forever. A race, so it passed twice before it failed. | `FIT_THREADS = 1`, documented as a correctness constraint rather than a tuning choice. Parallelism moved to processes. |
| `features.py`, `WARMUP_DAYS` | Set to the longest rolling *window* (91) instead of the longest *lag* (28). At 91 it exceeded the training window of every fold on a 120-day upload, so the model was served by walkover with the reason "the model could not be fitted" — on data that could comfortably fit one. | `WARMUP_DAYS = max(LAGS)`, and it now lines up with the fold budget so both agree about what "enough history" means. |
| `.github/workflows/ci.yml` | A `file://` path bug passed 370 tests on Windows and surfaced only in a Linux container: `urlparse().path.lstrip("/")` keeps a Windows drive letter absolute and makes a POSIX root relative. Nothing on the developer machine could have caught it. | Every push runs the suite on Linux and Windows. The Linux job additionally asserts the Postgres tests did **not** skip, because a suite that is green for the wrong reason is worse than a red one. |
| `tests/test_service_postgres.py` | The fixture pointed at the application database and called `Base.metadata.drop_all` in teardown — a test suite that deletes the developer's schema as a side effect of running. It did exactly that: the next end-to-end run failed with `relation "datasets" does not exist`. | Tests create and destroy their own databases. A test may destroy only what it created. |

---

## One That Was Never a Bug

Worth including because the reasoning is the same, applied before the failure rather than
after.

Product names are matched against the festival demand table by keyword. `"Amul Taaza 1L"` is
recognisably milk. `"AT-1L-BLU"` and `"SKU-88213"` are not — and `"Milk Bikis"` is a biscuit
that matches the keyword `milk` perfectly while behaving nothing like dairy at Holi.

That failure would be silent and expensive: double stock on the wrong product, no error
anywhere.

So the match is never allowed to change an order while being invisible. Every match carries
the keyword that produced it — "matched **paneer** → Holi, suggested 1.6x" — which a buyer
can reject at a glance. A test asserts that an unmatched product's order quantity is
byte-identical with the festival feature switched on and off.

---

## The Common Thread

None of these were crashes. Every one produced output that looked like an answer:

- A ratio below 1.0, which is an ordinary thing for a ratio to be.
- An order quantity, in units, with a service level attached.
- A comparison between two methods, with real numbers on both sides.
- A calendar, fully populated, missing exactly one festival.

The defect in each case was not in the arithmetic but in what the arithmetic was being asked
to mean: an average across a regime change, a daily method at a total grain, a true
measurement carrying an unsupported conclusion, a lookup depending on an unstated input.

Three of the four were found by running the system on real data and reading the output.
One was found by CI on a platform the author does not develop on. None would have been found
by a passing test suite alone, which is why the suite now pins each of them.

---

*Every claim here is verifiable in the source. The rationale for each decision lives in the
module that implements it, not only in this document.*
