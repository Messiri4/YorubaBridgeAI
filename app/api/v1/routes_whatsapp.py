"""WhatsApp webhook endpoints (Meta Cloud API conventions).

GET  /webhook  — Meta subscription verification handshake.
POST /webhook  — inbound messages; HMAC-verified, translated, replied.
"""

import logging

from fastapi import APIRouter, HTTPException, Request, Response

from app.core.config import get_settings
from app.services.translation import TranslationError
from app.services.whatsapp_client import WhatsAppClient
from app.services.whatsapp_webhook import extract_text_messages, guess_language, verify_signature

logger = logging.getLogger(__name__)
router = APIRouter(tags=["whatsapp"])


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


@router.post("/webhook")
async def receive_message(request: Request) -> dict[str, object]:
    """Receive inbound WhatsApp messages.

    Pipeline: verify HMAC signature (raw body) -> extract text messages ->
    detect language -> translate -> send reply via Graph API.

    Returns 200 with a summary so Meta doesn't retry; 403 on bad signature.
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

    import json

    payload = json.loads(body)
    messages = extract_text_messages(payload)
    replied = 0

    service = getattr(request.app.state, "translation_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Translation service not loaded")

    client = WhatsAppClient()
    for sender, text in messages:
        source = guess_language(text)
        target = "eng" if source == "yor" else "yor"
        try:
            result = service.translate(text, source, target)
            client.send_text(sender, result.translated_text)
            replied += 1
        except TranslationError as exc:
            logger.warning("Translation failed for %s: %s", sender, exc)
        except Exception:  # noqa: BLE001 - one bad message must not fail the batch
            logger.exception("Failed to reply to %s", sender)

    return {"received": len(messages), "replied": replied}
