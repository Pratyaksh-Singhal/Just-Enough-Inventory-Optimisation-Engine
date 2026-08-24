# One image, two commands (api, worker). See docker-compose.yml for why they must match.
FROM python:3.11-slim

# libgomp is LightGBM's OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependency metadata first, so a source-only change does not reinstall the world.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[api,models,service]"

# Needed by the release command, which runs `alembic upgrade head` before any Machine starts.
COPY alembic.ini ./
COPY migrations ./migrations

# The built dashboard, served at / by the API so the page and its own endpoints share an origin.
COPY dashboard/index.html ./dashboard/index.html
ENV DASHBOARD_DIR=/app/dashboard

RUN mkdir -p /data/uploads
# 8001, not 8000: tier 1's run-api owns 8000 and the two are meant to run side by side.
EXPOSE 8001

CMD ["uvicorn", "inventory_engine.service.app:app", "--host", "0.0.0.0", "--port", "8001"]
