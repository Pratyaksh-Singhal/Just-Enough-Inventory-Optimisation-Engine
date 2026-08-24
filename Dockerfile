FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[api,models,service]"

# Read by release_command (`alembic upgrade head`), which runs in this image.
COPY alembic.ini ./
COPY migrations ./migrations

# Served at / by the API, so the page and its endpoints share an origin.
COPY dashboard/index.html ./dashboard/index.html
ENV DASHBOARD_DIR=/app/dashboard

RUN mkdir -p /data/uploads

# 8001 matches internal_port in fly.toml.
EXPOSE 8001

CMD ["uvicorn", "inventory_engine.service.app:app", "--host", "0.0.0.0", "--port", "8001"]
