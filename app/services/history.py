"""Translation history persistence in PostgreSQL (psycopg 3).

Deliberately minimal: one table, plain parameterized SQL, no ORM. The
recorder never raises — a logging failure must not break a translation.
"""

import contextlib
import logging
from dataclasses import dataclass
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS translation_history (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    source_language TEXT        NOT NULL,
    target_language TEXT        NOT NULL,
    source_text     TEXT        NOT NULL,
    translated_text TEXT        NOT NULL,
    latency_ms      REAL        NOT NULL,
    cached          BOOLEAN     NOT NULL DEFAULT FALSE,
    override_hit    BOOLEAN     NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_history_created_at ON translation_history (created_at);
"""

INSERT_SQL = """
INSERT INTO translation_history
    (source_language, target_language, source_text, translated_text,
     latency_ms, cached, override_hit)
VALUES (%s, %s, %s, %s, %s, %s, %s)
"""


@dataclass(frozen=True)
class HistoryRecord:
    """One completed translation, as stored."""

    source_language: str
    target_language: str
    source_text: str
    translated_text: str
    latency_ms: float
    cached: bool = False
    override_hit: bool = False


class HistoryStore:
    """Writes translation records to Postgres; degrades to a no-op store.

    Args:
        connection_factory: zero-arg callable returning a psycopg connection
            (or None). Injectable for tests.
    """

    def __init__(self, connection_factory: Any) -> None:
        self._connection_factory = connection_factory
        self._schema_ready = False

    def _get_connection(self) -> Any | None:
        """Return a live connection or None when the DB is unreachable."""
        try:
            conn = self._connection_factory()
            if conn is None:
                return None
            if not self._schema_ready:
                with conn.cursor() as cur:
                    cur.execute(SCHEMA_SQL)
                conn.commit()
                self._schema_ready = True
            return conn
        except Exception as exc:  # noqa: BLE001 - persistence must not raise
            logger.warning("History store unavailable: %s", exc)
            return None

    def record(self, entry: HistoryRecord) -> bool:
        """Store one translation; returns True when written.

        Never raises: connection failures, SQL errors and lost schema state
        all degrade to a logged warning and a False return.
        """
        conn = self._get_connection()
        if conn is None:
            return False
        try:
            with conn.cursor() as cur:
                cur.execute(
                    INSERT_SQL,
                    (
                        entry.source_language,
                        entry.target_language,
                        entry.source_text,
                        entry.translated_text,
                        entry.latency_ms,
                        entry.cached,
                        entry.override_hit,
                    ),
                )
            conn.commit()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to record translation history: %s", exc)
            with contextlib.suppress(Exception):
                conn.rollback()
            # Allow schema re-bootstrap on the next attempt.
            self._schema_ready = False
            return False

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent records as dicts (newest first)."""
        conn = self._get_connection()
        if conn is None:
            return []
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, created_at, source_language, target_language,
                           source_text, translated_text, latency_ms, cached, override_hit
                    FROM translation_history
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                columns = [d[0] for d in cur.description]
                return [dict(zip(columns, row, strict=False)) for row in cur.fetchall()]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read translation history: %s", exc)
            return []


def make_history_store() -> HistoryStore | None:
    """Build a store from settings; None when the database is disabled."""
    settings = get_settings()
    if not settings.database_enabled:
        logger.info("Database disabled; translation history off")
        return None

    import psycopg

    dsn = settings.database_url
    if dsn.startswith("postgresql+psycopg://"):
        dsn = dsn.replace("postgresql+psycopg://", "postgresql://", 1)
    return HistoryStore(lambda: psycopg.connect(dsn, connect_timeout=3))
