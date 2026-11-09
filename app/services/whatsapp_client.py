"""WhatsApp Cloud API client — outbound messages via Graph API.

Kept deliberately thin: one public method (`send_text`) so the webhook
handler stays testable with a mocked client. Auth token comes only from
Settings (never hardcoded).
"""

import logging
from typing import Any, Protocol

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)

GRAPH_API_BASE = "https://graph.facebook.com/v20.0"


def _log_graph_rejection(response: Any) -> None:
    """Log a Graph API rejection with safe, structured detail only.

    Captured: HTTP status, Meta numeric error code/subcode, error type,
    and whether this was a Graph rejection or a transport failure.
    NEVER logged: tokens, phone numbers, message contents, full URLs,
    or raw response bodies (Graph bodies may echo credentials/PII).
    """
    fields: dict[str, object] = {"http_status": response.status_code}
    try:
        err = response.json().get("error", {})
        for key in ("code", "error_subcode", "type", "message"):
            if err.get(key) is not None:
                fields[key] = err[key]
        # error_user_msg is user-facing text; safe to include, no PII
        if err.get("error_user_msg"):
            fields["user_msg"] = err["error_user_msg"]
        fields["kind"] = "graph_api_rejection"
    except Exception:  # noqa: BLE001 - non-JSON or empty body
        fields["kind"] = "graph_api_rejection_unparseable"
    logger.error("Message send rejected: %s", fields)


class HttpPost(Protocol):
    """Minimal surface of httpx.Client used here (easy to mock)."""

    def post(self, url: str, json: dict[str, Any], timeout: float) -> Any: ...


class WhatsAppClient:
    """Sends WhatsApp text messages through the Meta Cloud API.

    Attributes:
        http: HTTP client (httpx.Client in production).
    """

    def __init__(self, http: HttpPost | None = None) -> None:
        self._settings = get_settings()
        self._http = http

    @property
    def configured(self) -> bool:
        """True when token + phone number id are present in settings."""
        return bool(self._settings.whatsapp_token and self._settings.whatsapp_phone_number_id)

    def send_text(self, to_number: str, body: str) -> dict[str, Any]:
        """Send a plain text message.

        Args:
            to_number: recipient phone number in WhatsApp format.
            body: message text (e.g. the translation).

        Returns:
            Graph API response JSON (contains message id on success).

        Raises:
            RuntimeError: if credentials are not configured.
            httpx.HTTPStatusError: on non-2xx Graph API responses.
        """
        if not self.configured:
            raise RuntimeError(
                "WhatsApp credentials not configured (WHATSAPP_TOKEN / WHATSAPP_PHONE_NUMBER_ID)"
            )

        url = f"{GRAPH_API_BASE}/{self._settings.whatsapp_phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_number,
            "type": "text",
            "text": {"preview_url": False, "body": body},
        }
        client = self._http or httpx.Client(
            headers={"Authorization": f"Bearer {self._settings.whatsapp_token}"},
            timeout=10.0,
        )
        try:
            response = client.post(url, json=payload, timeout=10.0)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            # Network-level failure: Meta never answered or the connection broke
            logger.error("Message send failed at transport level: %s", type(exc).__name__)
            raise RuntimeError("whatsapp_send_transport_failure") from exc
        if response.status_code >= 400:
            _log_graph_rejection(response)
        response.raise_for_status()
        logger.info("WhatsApp message sent (status %d)", response.status_code)
        result: dict[str, Any] = response.json()
        return result
