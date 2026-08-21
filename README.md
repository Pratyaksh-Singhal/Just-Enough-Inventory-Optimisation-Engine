# Just Enough — Inventory Optimisation Engine

Newsvendor ordering for perishables. Forecasting accuracy is not the deliverable; money
saved is.

**Live: https://inventory-optimization-engine.fly.dev/**

The service scales to zero, so the first load takes a moment to wake.

## The finding

On 100,720 stocking decisions from M5 retail data, priced against the demand that actually
arrived:

| Way of deciding | Ran out on | Units wasted | Total cost | vs row one |
|---|---:|---:|---:|---:|
| Order what sold last week | 31.1% | 78,106 | $169,172 | — |
| Always be 95% in stock | 6.1% | 326,957 | $415,644 | +145.7% |
| Order to the cost-optimal target | 48.0% | 20,714 | $120,797 | **−28.6%** |

A flat 95% service level costs **2.5× more** than doing nothing sophisticated at all. With
a 30% margin and 60% spoilage the critical ratio is 0.409, so the cheapest in-stock target
is **41%** — and the winning policy runs out *more often* than current practice while
costing 28.6% less, because it cuts waste by 73%.

Every cost figure is a stated assumption, not a measurement. The conclusion holds at every
spoilage rate from 0.0 to 1.0.

## How it works

Demand is forecast as a **distribution**, not a point — seven quantile levels from a global
LightGBM model, reconciled across the store hierarchy with MinT so the levels stay
coherent. The newsvendor critical ratio `Cu / (Cu + Co)` then picks which quantile to order
to. The policy is worth more than the model: fixing the ordering rule cut cost 71%,
improving the forecast cut it a further 18%.

Two tiers:

- **Tier 1** — precomputed results on M5, served as a static page. The order calculator
  runs entirely in the browser on a CSV you paste; nothing is uploaded.
- **Tier 2** — a live service. Upload your own sales history and it fits a model per
  product, backtests it against a simple baseline, and serves whichever won. It refuses
  rather than guesses when a product has under 90 days of history.

## Stack

Python 3.11 · DuckDB · LightGBM · statsforecast · Nixtla hierarchicalforecast · FastAPI ·
Postgres · Redis + arq · Alembic · Fly.io

## Running it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,models]"
build-warehouse          # M5 -> DuckDB
build-features
run-gbm && run-backtest && run-mint && run-optimize
```

Tier 2 additionally needs Postgres and Redis:

```bash
pip install -e ".[service]"
docker compose up -d db redis
alembic upgrade head
run-service                                            # API
arq inventory_engine.service.worker.WorkerSettings     # worker
```

`pyproject.toml` is the source of truth for dependencies, including the pinned `holidays`
version the festival calendar relies on.

## Your data

Tier 2 stores an uploaded file so the worker can read it. It is deleted **30 days after
upload whether you ask or not**, and one button removes the file, its forecasts and its
stored history immediately. No accounts, no cookies. Analytics count page views and runs —
never file names, product names or figures.

## Documentation

- [`docs/what-went-wrong.md`](docs/what-went-wrong.md) — decisions this project got wrong
  the first time. None crashed; every one returned a confident, plausible, wrong answer.
- [`docs/design-system.md`](docs/design-system.md) — the dashboard's tokens, and the CSS
  rules whose violation each shipped a visible bug.
- [`docs/BACKLOG.md`](docs/BACKLOG.md) — epics and their status.
