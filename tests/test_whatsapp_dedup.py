"""Tests for WhatsApp message deduplication."""

from unittest.mock import MagicMock

from app.services.whatsapp_dedup import MessageDeduplicator


def make_dedup() -> tuple[MessageDeduplicator, MagicMock]:
    client = MagicMock()
    return MessageDeduplicator(lambda: client), client


def test_first_claim_returns_not_seen() -> None:
    dedup, client = make_dedup()
    client.set.return_value = True  # NX claim succeeded -> we are first
    assert dedup.seen("wamid.A") is False


def test_second_claim_returns_seen() -> None:
    dedup, client = make_dedup()
    client.set.return_value = False  # key existed -> already claimed
    assert dedup.seen("wamid.A") is True


def test_redis_claim_sets_ttl_and_nx() -> None:
    dedup, client = make_dedup()
    client.set.return_value = True
    dedup.seen("wamid.A")
    kwargs = client.set.call_args.kwargs
    assert kwargs["nx"] is True
    assert kwargs["ex"] == 7 * 24 * 3600


def test_redis_failure_falls_back_to_memory() -> None:
    def broken() -> None:
        raise ConnectionError("down")

    dedup = MessageDeduplicator(broken)
    assert dedup.seen("wamid.A") is False
    assert dedup.seen("wamid.A") is True  # in-memory second call is a dup


def test_empty_id_ignored() -> None:
    dedup, _ = make_dedup()
    assert dedup.seen("") is False
