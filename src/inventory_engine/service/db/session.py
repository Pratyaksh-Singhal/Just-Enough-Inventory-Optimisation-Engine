"""Engine and session handling."""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from inventory_engine.service.settings import get_settings


@lru_cache(maxsize=8)
def get_engine(database_url: str | None = None) -> Engine:
    """Return the process-wide engine for ``database_url``."""
    url = database_url or get_settings().database_url
    return create_engine(url, pool_pre_ping=True, future=True)


def get_sessionmaker(database_url: str | None = None) -> sessionmaker[Session]:
    """Session factory bound to the engine for ``database_url``."""
    return sessionmaker(bind=get_engine(database_url), expire_on_commit=False, future=True)


def get_session() -> Iterator[Session]:
    """FastAPI dependency: one session per request, rolled back on an exception."""
    factory = get_sessionmaker()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
