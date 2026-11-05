"""Tests for HistoryStore — mocked connection by default, real Postgres via -m integration."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from app.services.history import HistoryRecord, HistoryStore


def make_store(work: bool = True) -> tuple[HistoryStore, MagicMock]:
    """Store wired to a fake psycopg connection.

    The factory is called per-operation, so the same mock object is returned
    each time while behaving like a fresh connection.
    """
    conn = MagicMock()
    if not work:
        conn.cursor.side_effect = ConnectionError("db down")
    return HistoryStore(lambda: conn), conn


def test_record_success() -> None:
    store, conn = make_store()
    ok = store.record(
        HistoryRecord("eng", "yor", "hi", "Mo tí gbà", 12.5, cached=False, override_hit=True)
    )
    assert ok is True
    # Two commits are expected: the CREATE TABLE bootstrap and the INSERT.
    assert conn.commit.call_count == 2
    sql = conn.cursor.return_value.__enter__.return_value.execute.call_args
    assert "INSERT INTO translation_history" in sql.args[0]
    assert sql.args[1] == ("eng", "yor", "hi", "Mo tí gbà", 12.5, False, True)


def test_record_failure_returns_false_not_raise() -> None:
    store, _ = make_store(work=False)
    ok = store.record(HistoryRecord("eng", "yor", "hi", "x", 1.0))
    assert ok is False  # must never raise


def test_disabled_db_factory_returns_none() -> None:
    store = HistoryStore(lambda: None)
    assert store.record(HistoryRecord("eng", "yor", "hi", "x", 1.0)) is False
    assert store.recent() == []


def test_recent_maps_columns_to_dicts() -> None:
    store, conn = make_store()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.description = [("id", None), ("created_at", None), ("source_language", None)]
    cur.fetchall.return_value = [(1, "now", "eng")]
    rows = store.recent(limit=1)
    assert rows == [{"id": 1, "created_at": "now", "source_language": "eng"}]


def test_record_resets_schema_flag_after_error() -> None:
    """After a mid-flight error the schema must be re-bootstrapped next time."""
    store, conn = make_store()
    cur = conn.cursor.return_value.__enter__.return_value
    # Sequence per attempt: schema DDL, then INSERT.
    # Attempt 1: DDL ok, INSERT fails. Attempt 2: DDL again, INSERT ok.
    cur.execute.side_effect = [None, RuntimeError("insert failed"), None, None]
    assert store.record(HistoryRecord("eng", "yor", "a", "b", 1.0)) is False
    assert store.record(HistoryRecord("eng", "yor", "a", "b", 1.0)) is True


# --- live-Postgres integration (opt-in: make db-up && make test-db) ---

pytestmark_integration = pytest.mark.integration


@pytest.mark.integration
def test_real_postgres_roundtrip() -> None:
    """Skipped automatically unless DATABASE_ENABLED=true and Postgres is up."""
    import os

    if os.environ.get("DATABASE_ENABLED") != "true":
        pytest.skip("DATABASE_ENABLED not set")

    from app.services.history import make_history_store

    store = make_history_store()
    assert store is not None
    record = HistoryRecord(
        source_language="eng",
        target_language="yor",
        source_text="integration test",
        translated_text="ìdánwò ìfọ̀sọ̀wọ́",
        latency_ms=42.0,
        override_hit=False,
    )
    assert store.record(record) is True
    rows: list[dict[str, Any]] = store.recent(limit=1)
    assert rows and rows[0]["source_text"] == "integration test"
