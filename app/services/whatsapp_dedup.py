"""Message deduplication for WhatsApp webhook deliveries.

Meta retries webhook deliveries (e.g. on timeout); without dedup a retried
message would be translated and replied to twice. Storage: Redis SET with a
TTL (multi-worker safe); in-memory set as fallback when Redis is down.
"""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# WhatsApp message ids (wamid.*) are unique per message; 7 days covers any
# realistic retry horizon while bounding memory/redis usage.
DEDUP_TTL_SECONDS = 7 * 24 * 3600


class MessageDeduplicator:
    """Records processed message ids; `seen()` is atomic check-and-set.

    Args:
        redis_getter: callable returning a redis-like client or None.
    """

    def __init__(self, redis_getter: Callable[[], Any]) -> None:
        self._redis_getter = redis_getter
        self._local: set[str] = set()

    def seen(self, message_id: str) -> bool:
        """Return True if this message id was already processed.

        Marks the id as processed regardless of the result (atomic claim):
        first caller wins, later callers are told it was already seen.
        """
        if not message_id:
            return False

        try:
            client = self._redis_getter()
            if client is not None:
                # SET NX = atomic "claim if absent"; True response = we claimed it
                claimed = client.set(f"wa:dedup:{message_id}", "1", ex=DEDUP_TTL_SECONDS, nx=True)
                return not claimed
        except Exception:  # noqa: BLE001 - dedup must not break the pipeline
            logger.warning("Redis dedup unavailable; using in-memory set")

        if message_id in self._local:
            return True
        self._local.add(message_id)
        return False
