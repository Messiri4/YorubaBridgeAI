"""WhatsApp webhook endpoints (Meta Cloud API conventions).

GET  /webhook  — Meta subscription verification handshake.
POST /webhook  — inbound messages: HMAC-verify (raw body) -> dedup by
                 wamid -> per-sender rate limit -> translate -> reply.

Privacy: message contents and phone numbers are NEVER logged — only
counts and coarse outcomes.
"""

import json
import logging

from fastapi import APIRouter, HTTPException, Request, Response

from app.core.config import get_settings
from app.core.rate_limit import RateLimiter
from app.services.translation import TranslationError
from app.services.whatsapp_client import WhatsAppClient
from app.services.whatsapp_dedup import MessageDeduplicator
from app.services.whatsapp_webhook import extract_text_messages, guess_language, verify_signature

logger = logging.getLogger(__name__)
router = APIRouter(tags=["whatsapp"])


def _split_direction(source: str) -> tuple[str, str]:
    """Map detected source language to (source_code, target_code)."""
    return ("yor", "eng") if source == "yor" else ("eng", "yor")


@router.get("/webhook")
def verify_subscription(request: Request) -> Response:
    """Meta's one-time handshake when you register the webhook URL.

    Echoes hub.challenge only when hub.verify_token matches our secret.
    """
    settings = get_settings()
    params = request.query_params
    if (
        params.get("hub.mode") == "subscribe"
        and params.get("hub.verify_token") == settings.whatsapp_verify_token
        and settings.whatsapp_verify_token
    ):
        return Response(content=params.get("hub.challenge", ""), media_type="text/plain")
    raise HTTPException(status_code=403, detail="Webhook verification failed")


def _get_dedup(request: Request) -> MessageDeduplicator | None:
    """Fetch the deduplicator, building a no-Redis one if lifespan didn't run.

    Guards against state being assembled out of order (e.g. tests that
    patch app.state after app construction): dedup must never be silently
    absent in production.
    """
    dedup = getattr(request.app.state, "wa_dedup", None)
    if dedup is None:
        logger.warning("Deduplicator missing from app.state; building in-memory fallback")
        dedup = MessageDeduplicator(lambda: None)
        request.app.state.wa_dedup = dedup
    return dedup


@router.post("/webhook")
async def receive_message(request: Request) -> dict[str, object]:
    """Receive inbound WhatsApp messages.

    Always returns 200 for well-signed payloads (even when individual
    messages fail) so Meta does not retry the whole delivery.
    """
    settings = get_settings()
    body = await request.body()
    signature = request.headers.get("x-hub-signature-256")

    # Fail closed: without the app secret we cannot verify ANY signature,
    # so unconfigured must return 503 regardless of the header's value.
    if not settings.whatsapp_app_secret:
        raise HTTPException(status_code=503, detail="WhatsApp webhook not configured")

    if not verify_signature(settings.whatsapp_app_secret, body, signature):
        raise HTTPException(status_code=403, detail="Invalid signature")

    payload = json.loads(body)
    messages = extract_text_messages(payload)

    service = getattr(request.app.state, "translation_service", None)
    dedup = _get_dedup(request)
    limiter: RateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Translation service not loaded")

    client = WhatsAppClient()
    received = deduped = replied = failed = 0

    for msg in messages:
        received += 1
        if dedup is not None and dedup.seen(msg["message_id"]):
            deduped += 1
            logger.info("Duplicate delivery ignored (wamid hash recorded)")
            continue

        # Per-sender rate limiting protects the model from floods.
        if limiter is not None:
            try:
                limiter.check(f"wa:{msg['sender']}")
            except HTTPException:
                logger.warning("Rate limit hit for a WhatsApp sender; reply skipped")
                failed += 1
                continue

        source = guess_language(msg["text"])
        src, tgt = _split_direction(source)
        try:
            result = service.translate(msg["text"], src, tgt)
            client.send_text(msg["sender"], result.translated_text)
            replied += 1
        except TranslationError as exc:
            failed += 1
            logger.warning("Translation rejected for a message: %s", exc)
        except Exception:  # noqa: BLE001 - one bad message must not fail the batch
            failed += 1
            logger.exception("Failed to process a WhatsApp message")

    logger.info(
        "Webhook batch: %d received, %d deduped, %d replied, %d failed",
        received,
        deduped,
        replied,
        failed,
    )
    return {"received": received, "deduped": deduped, "replied": replied, "failed": failed}
