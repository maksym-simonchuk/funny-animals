"""Engine and session handling for SQLite (dev) and PostgreSQL (production)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from loguru import logger
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

_engine: Engine | None = None
_Session: sessionmaker[Session] | None = None


def _sqlite_path(url: str) -> Path | None:
    """Return the on-disk path of a sqlite URL, or None for other backends."""
    if not url.startswith("sqlite"):
        return None
    _, _, tail = url.partition(":///")
    return Path(tail) if tail and tail != ":memory:" else None


def init_db(url: str) -> Engine:
    """Create the engine, apply SQLite pragmas and create missing tables."""
    global _engine, _Session

    path = _sqlite_path(url)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(url, future=True)

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record) -> None:  # type: ignore[no-untyped-def]
            cursor = dbapi_conn.cursor()
            # WAL lets the async downloader read while the writer thread commits.
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    Base.metadata.create_all(engine)
    _engine = engine
    _Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    logger.debug("database ready: {}", url)
    return engine


def get_engine() -> Engine:
    if _engine is None:
        raise RuntimeError("database not initialised — call init_db(url) first")
    return _engine


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on error, always close."""
    if _Session is None:
        raise RuntimeError("database not initialised — call init_db(url) first")
    session = _Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
