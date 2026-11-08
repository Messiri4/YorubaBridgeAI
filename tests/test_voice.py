"""Tests for the voice pipeline — Whisper mocked; model mocked."""

import io
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.services.speech import WHISPER_TO_FLORES, Transcript


@pytest.fixture
def voice_client() -> TestClient:
    """App with mocked speech + translation services."""
    from app.main import create_app

    app = create_app(load_model=False)
    speech = MagicMock()
    speech.transcribe.return_value = Transcript(
        text="I am going to the market", language="en", flores_code="eng", duration_s=3.2
    )
    translator = MagicMock()
    from app.services.translation import TranslationResult

    translator.translate.return_value = TranslationResult(
        translated_text="Mo ń lọ sí ọjà",
        source_language="eng",
        target_language="yor",
        latency_ms=1.0,
    )
    app.state.speech_service = speech
    app.state.translation_service = translator
    return TestClient(app)


def test_whisper_language_map() -> None:
    assert WHISPER_TO_FLORES == {"en": "eng", "yo": "yor"}


def test_voice_translate_ok(voice_client: TestClient) -> None:
    resp = voice_client.post(
        "/api/v1/voice-translate",
        files={"audio": ("note.ogg", io.BytesIO(b"RIFFfake"), "audio/ogg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["translated_text"] == "Mo ń lọ sí ọjà"
    assert body["source_language"] == "eng"


def test_voice_translate_empty_upload(voice_client: TestClient) -> None:
    resp = voice_client.post(
        "/api/v1/voice-translate",
        files={"audio": ("note.ogg", io.BytesIO(b""), "audio/ogg")},
    )
    assert resp.status_code == 422


def test_voice_translate_unsupported_language(voice_client: TestClient) -> None:
    voice_client.app.state.speech_service.transcribe.side_effect = RuntimeError(
        "Detected language 'fr' is not supported"
    )
    resp = voice_client.post(
        "/api/v1/voice-translate",
        files={"audio": ("note.ogg", io.BytesIO(b"RIFFfake"), "audio/ogg")},
    )
    assert resp.status_code == 422


def test_voice_translate_no_service() -> None:
    from app.main import create_app

    app = create_app(load_model=False)
    del app.state.speech_service
    client = TestClient(app)
    resp = client.post(
        "/api/v1/voice-translate",
        files={"audio": ("note.ogg", io.BytesIO(b"RIFF"), "audio/ogg")},
    )
    assert resp.status_code == 503


@pytest.mark.smoke
def test_whisper_tiny_transcribes_real_english_audio() -> None:
    """Real Whisper tiny + real NLLB on a synthesized spoken sentence.

    Generates a spoken wav via macOS `say` (no downloads of audio datasets),
    transcribes it with real Whisper tiny, then translates with real NLLB.
    Skipped automatically if `say` or ffmpeg is unavailable.
    """
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    if not shutil.which("say") or not shutil.which("ffmpeg"):
        pytest.skip("macOS `say` or ffmpeg not available")

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "hello.wav"
        subprocess.run(
            ["say", "-o", str(wav), "--data-format=LEI16@22050", "Hello, how are you today?"],
            check=True,
            capture_output=True,
        )
        from app.services.model_loader import load_nllb
        from app.services.speech import SpeechService
        from app.services.translation import TranslationService

        transcript = SpeechService().transcribe(str(wav))
        assert transcript.flores_code == "eng"
        assert "hello" in transcript.text.lower()

        result = TranslationService(load_nllb()).translate(transcript.text, "eng", "yor")
        assert result.translated_text.strip()
