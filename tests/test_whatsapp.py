"""Tests for WhatsApp webhook plumbing — everything mocked, no network."""

import hashlib
import hmac
import json
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.services.whatsapp_webhook import extract_text_messages, guess_language, verify_signature

APP_SECRET = "test-secret"


def _signed(body: bytes) -> str:
    return "sha256=" + hmac.new(APP_SECRET.encode(), body, hashlib.sha256).hexdigest()


# --- signature verification ---


def test_valid_signature_accepted() -> None:
    body = b'{"entry": []}'
    assert verify_signature(APP_SECRET, body, _signed(body))


def test_wrong_secret_rejected() -> None:
    body = b'{"entry": []}'
    assert not verify_signature("other-secret", body, _signed(body))


def test_missing_or_malformed_header_rejected() -> None:
    body = b"{}"
    assert not verify_signature(APP_SECRET, body, None)
    assert not verify_signature(APP_SECRET, body, "sha256=")
    assert not verify_signature(APP_SECRET, body, "md5=abc")


def test_modified_body_rejected() -> None:
    body = b'{"a": 1}'
    sig = _signed(body)
    assert not verify_signature(APP_SECRET, b'{"a": 2}', sig)


# --- payload parsing ---


def test_extract_text_messages() -> None:
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "id": "wamid.ABC",
                                    "from": "234801",
                                    "type": "text",
                                    "text": {"body": "Hello"},
                                },
                                {
                                    "id": "wamid.DEF",
                                    "from": "234802",
                                    "type": "reaction",
                                },  # skipped
                            ]
                        }
                    }
                ]
            }
        ]
    }
    msgs = extract_text_messages(payload)
    assert msgs == [{"message_id": "wamid.ABC", "sender": "234801", "text": "Hello"}]


def test_extract_empty_payload() -> None:
    assert extract_text_messages({}) == []


def test_extract_gracefully_ignores_malformed() -> None:
    payload = {"entry": [{"changes": [{"value": {"messages": [{"id": "wamid.X"}]}}]}]}
    assert extract_text_messages(payload) == []  # no sender/text: skipped


# --- language heuristic ---


def test_guess_language_yoruba() -> None:
    assert guess_language("Mo ń lọ sí ọjà") == "yor"
    assert guess_language("Ẹ kú àárọ̀") == "yor"


def test_guess_language_english() -> None:
    assert guess_language("Good morning, how are you?") == "eng"


# --- full POST pipeline with mocked client ---


def test_webhook_post_rejects_bad_signature() -> None:

    from app.core.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    import os

    os.environ["WHATSAPP_APP_SECRET"] = APP_SECRET
    get_settings.cache_clear()
    try:
        app = create_app(load_model=False)
        client = TestClient(app)
        body = json.dumps({"entry": []}).encode()
        resp = client.post(
            "/api/v1/webhook",
            content=body,
            headers={
                "X-Hub-Signature-256": "sha256=deadbeef",
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 403
    finally:
        os.environ.pop("WHATSAPP_APP_SECRET", None)
        get_settings.cache_clear()


def test_webhook_post_rejects_when_unconfigured() -> None:
    from app.main import create_app

    app = create_app(load_model=False)
    client = TestClient(app)
    resp = client.post(
        "/api/v1/webhook", content=b"{}", headers={"X-Hub-Signature-256": "sha256=x"}
    )
    assert resp.status_code == 503


def test_webhook_verification_handshake() -> None:
    from app.core.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    import os

    os.environ["WHATSAPP_VERIFY_TOKEN"] = "mytoken"
    get_settings.cache_clear()
    try:
        app = create_app(load_model=False)
        client = TestClient(app)
        resp = client.get(
            "/api/v1/webhook",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "mytoken",
                "hub.challenge": "CH123",
            },
        )
        assert resp.status_code == 200
        assert resp.text == "CH123"
    finally:
        os.environ.pop("WHATSAPP_VERIFY_TOKEN", None)
        get_settings.cache_clear()


def test_webhook_post_translates_and_replies() -> None:
    """Full pipeline with mocked translation service + mocked WhatsApp client."""
    from app.main import create_app
    from app.services.translation import TranslationResult

    get_settings.cache_clear()
    import os

    os.environ["WHATSAPP_APP_SECRET"] = APP_SECRET
    get_settings.cache_clear()
    try:
        app = create_app(load_model=False)
        service = MagicMock()
        service.translate.return_value = TranslationResult(
            translated_text="Mo ń lọ sí ọjà",
            source_language="eng",
            target_language="yor",
            latency_ms=1.0,
        )
        app.state.translation_service = service

        sender = MagicMock()
        sender.return_value.json.return_value = {"messages": [{"id": "wamid.X"}]}
        whatsapp = MagicMock()
        whatsapp.configured = True
        whatsapp.send_text.return_value = {"messages": [{"id": "wamid.X"}]}

        body = json.dumps(
            {
                "entry": [
                    {
                        "changes": [
                            {
                                "value": {
                                    "messages": [
                                        {
                                            "id": "wamid.TEST1",
                                            "from": "234801",
                                            "type": "text",
                                            "text": {"body": "I am going to the market"},
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ]
            }
        ).encode()

        from unittest.mock import patch

        with patch("app.api.v1.routes_whatsapp.WhatsAppClient", return_value=whatsapp):
            client = TestClient(app)
            resp = client.post(
                "/api/v1/webhook",
                content=body,
                headers={
                    "X-Hub-Signature-256": _signed(body),
                    "Content-Type": "application/json",
                },
            )
        assert resp.status_code == 200
        assert resp.json() == {"received": 1, "deduped": 0, "replied": 1, "failed": 0}
        service.translate.assert_called_once_with("I am going to the market", "eng", "yor")
        whatsapp.send_text.assert_called_once_with("234801", "Mo ń lọ sí ọjà")
    finally:
        os.environ.pop("WHATSAPP_APP_SECRET", None)
        get_settings.cache_clear()


def test_webhook_dedups_retried_delivery() -> None:
    """Meta retries deliveries; the same wamid must NOT be translated twice."""
    from unittest.mock import patch

    from app.main import create_app
    from app.services.translation import TranslationResult

    get_settings.cache_clear()
    import os

    os.environ["WHATSAPP_APP_SECRET"] = APP_SECRET
    get_settings.cache_clear()
    try:
        app = create_app(load_model=False)
        service = MagicMock()
        service.translate.return_value = TranslationResult(
            translated_text="Mo ń lọ sí ọjà",
            source_language="eng",
            target_language="yor",
            latency_ms=1.0,
        )
        app.state.translation_service = service
        whatsapp = MagicMock()
        whatsapp.configured = True

        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "id": "wamid.DUP1",
                                        "from": "234801",
                                        "type": "text",
                                        "text": {"body": "Hello there"},
                                    }
                                ]
                            }
                        }
                    ]
                }
            ]
        }

        from unittest.mock import patch

        client = TestClient(app)
        for _ in range(2):  # Meta delivers the same message twice
            body = json.dumps(payload).encode()
            with patch("app.api.v1.routes_whatsapp.WhatsAppClient", return_value=whatsapp):
                resp = client.post(
                    "/api/v1/webhook",
                    content=body,
                    headers={"X-Hub-Signature-256": _signed(body)},
                )
            assert resp.status_code == 200

        assert resp.json() == {"received": 1, "deduped": 1, "replied": 0, "failed": 0}
        service.translate.assert_called_once()  # only the first delivery translated
        whatsapp.send_text.assert_called_once()
    finally:
        os.environ.pop("WHATSAPP_APP_SECRET", None)
        get_settings.cache_clear()


def test_webhook_unsupported_types_ignored_gracefully() -> None:
    from unittest.mock import patch

    from app.main import create_app

    get_settings.cache_clear()
    import os

    os.environ["WHATSAPP_APP_SECRET"] = APP_SECRET
    get_settings.cache_clear()
    try:
        app = create_app(load_model=False)
        app.state.translation_service = MagicMock()
        whatsapp = MagicMock()

        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {"id": "wamid.IMG", "from": "234801", "type": "image"},
                                    {"id": "wamid.AUD", "from": "234801", "type": "audio"},
                                ]
                            }
                        }
                    ]
                }
            ]
        }
        body = json.dumps(payload).encode()
        with patch("app.api.v1.routes_whatsapp.WhatsAppClient", return_value=whatsapp):
            resp = TestClient(app).post(
                "/api/v1/webhook",
                content=body,
                headers={"X-Hub-Signature-256": _signed(body)},
            )
        assert resp.status_code == 200
        assert resp.json() == {"received": 0, "deduped": 0, "replied": 0, "failed": 0}
        whatsapp.send_text.assert_not_called()
    finally:
        os.environ.pop("WHATSAPP_APP_SECRET", None)
        get_settings.cache_clear()
