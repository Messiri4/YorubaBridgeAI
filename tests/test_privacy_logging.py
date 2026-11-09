"""Privacy tests: unexpected errors must never leak user content into logs."""

import json
import logging
import os
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings

APP_SECRET = "privacy-test-secret"


def _signed(body: bytes) -> str:
    import hashlib
    import hmac

    return "sha256=" + hmac.new(APP_SECRET.encode(), body, hashlib.sha256).hexdigest()


@pytest.fixture
def crashing_client() -> TestClient:
    """App whose translation service raises an exception CONTAINING user text.

    Real-world equivalent: an exception whose args include the message body.
    If our logging is not privacy-safe, the text leaks into the log records.
    """
    import app.main as main_mod

    get_settings.cache_clear()
    os.environ["WHATSAPP_APP_SECRET"] = APP_SECRET
    get_settings.cache_clear()
    app = main_mod.create_app(load_model=False)

    service = MagicMock()
    # The exception message deliberately contains the secret user content:
    # if any handler logs str(exc) or a traceback, this text appears in logs.
    service.translate.side_effect = RuntimeError(
        "unrecoverable failure while translating user-text-gooseberry-42"
    )
    app.state.translation_service = service

    client = TestClient(app)
    yield client
    os.environ.pop("WHATSAPP_APP_SECRET", None)
    get_settings.cache_clear()


def _payload(text: str, mid: str = "wamid.PRIV1", sender: str = "09900000000") -> bytes:

    return json.dumps(
        {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "id": mid,
                                        "from": sender,
                                        "type": "text",
                                        "text": {"body": text},
                                    }
                                ]
                            }
                        }
                    ]
                }
            ]
        }
    ).encode()


def test_unexpected_error_does_not_log_user_content(
    crashing_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    import hashlib

    text = "user-text-gooseberry-42"
    sender = "09900000000"
    body = _payload(text)
    expected_cid = hashlib.sha256(b"wamid.PRIV1").hexdigest()[:12]

    with caplog.at_level(logging.DEBUG):
        resp = crashing_client.post(
            "/api/v1/webhook",
            content=body,
            headers={"X-Hub-Signature-256": _signed(body)},
        )

    assert resp.status_code == 200  # batch continues, Meta gets 200
    all_logs = " ".join(str(r.getMessage()) for r in caplog.records)
    # The user's content and number must be absent from every log record.
    assert text not in all_logs, "message text leaked into logs"
    assert sender not in all_logs, "sender number leaked into logs"
    # A correlation id derived from the wamid must be present for debugging.
    assert expected_cid in all_logs
    # The exception *class name* is logged (safe) but not its message.
    assert "RuntimeError" in all_logs
    assert "unrecoverable failure" not in all_logs


def test_translation_error_message_is_service_authored(
    crashing_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """TranslationError text is authored by our service (no user content)."""
    from app.services.translation import TranslationError

    service = crashing_client.app.state.translation_service
    service.translate.side_effect = TranslationError(
        "unsupported source_language 'eng'. Supported: ['eng', 'yor']"
    )
    body = _payload("another-private-sentence-xyz")
    with caplog.at_level(logging.INFO):
        resp = crashing_client.post(
            "/api/v1/webhook",
            content=body,
            headers={"X-Hub-Signature-256": _signed(body)},
        )
    assert resp.status_code == 200
    all_logs = " ".join(str(r.getMessage()) for r in caplog.records)
    assert "another-private-sentence-xyz" not in all_logs
