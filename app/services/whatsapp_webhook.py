"""Webhook utilities: Meta signature verification and payload parsing."""

import hashlib
import hmac
from typing import Any


def verify_signature(app_secret: str, request_body: bytes, signature_header: str | None) -> bool:
    """Verify Meta's X-Hub-Signature-256 header against the raw request body.

    Meta signs the body with HMAC-SHA256 using the app secret:
        X-Hub-Signature-256: sha256=<hexdigest>

    The signature MUST be computed over the raw bytes as received —
    re-serializing parsed JSON can change whitespace and break the HMAC.

    Args:
        app_secret: Meta app secret (from settings, never hardcoded).
        request_body: raw request body bytes.
        signature_header: the X-Hub-Signature-256 header value, or None.

    Returns:
        True when the signature is present and valid.
    """
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode("utf-8"), request_body, hashlib.sha256).hexdigest()
    provided = signature_header.removeprefix("sha256=")
    # compare_digest prevents timing attacks.
    return hmac.compare_digest(expected, provided)


def extract_text_messages(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """Pull (sender_number, text) pairs out of a webhook payload.

    Meta's payload nests entries -> changes -> values -> messages. Text
    messages carry a `text.body`; reactions/system messages are skipped.

    Returns:
        List of (wa_id of sender, message text).
    """
    out: list[tuple[str, str]] = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for message in value.get("messages", []):
                if message.get("type") != "text":
                    continue
                body = message.get("text", {}).get("body", "")
                sender = message.get("from", "")
                if body and sender:
                    out.append((sender, body))
    return out


def guess_language(text: str) -> str:
    """Cheap heuristic language detector for eng vs yor.

    Counts Yoruba-specific signals: under-dot letters (ṣ, ẹ, ọ) and the
    distinctive pre-verbal markers / common words. Falls back to 'eng'.
    """
    yoruba_markers = ("ṣ", "ẹ", "ọ", "gb", "ń", "mo ", "ẹ ", "ni ", "kú ")
    score = sum(text.lower().count(m) for m in yoruba_markers)
    return "yor" if score >= 2 else "eng"
