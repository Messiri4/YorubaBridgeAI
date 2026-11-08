"""Speech-to-text for WhatsApp voice notes via faster-whisper (CPU).

Model choice: Whisper `tiny` (~75 MB on disk, ~150 MB RAM). Whisper was
trained on 680k hours including Yoruba, and detects language automatically —
a speaker can leave a note in either language and we detect which.

Latency on Intel CPU: roughly real-time or a bit faster for short notes.
"""

import logging
from dataclasses import dataclass
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Whisper -> NLLB language code mapping for the languages we support.
WHISPER_TO_FLORES = {"en": "eng", "yo": "yor"}


@dataclass(frozen=True)
class Transcript:
    """Result of transcribing one audio file."""

    text: str
    language: str  # Whisper language code, e.g. "en"/"yo"
    flores_code: str  # mapped to "eng"/"yor" ("" if unsupported)
    duration_s: float


class SpeechService:
    """Transcribes voice notes and maps them onto translation language codes.

    The whisper model is loaded lazily so importing this module (and every
    test that imports the app) stays cheap.
    """

    def __init__(self) -> None:
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel

            settings = get_settings()
            model_size = settings.speech_model_size
            logger.info("Loading Whisper '%s' on CPU ...", model_size)
            self._model = WhisperModel(
                model_size, device="cpu", compute_type="int8"
            )  # int8: 4x less RAM, faster on CPU, negligible quality loss
        return self._model

    def transcribe(self, audio_path: str) -> Transcript:
        """Transcribe an audio file and auto-detect its language.

        Args:
            audio_path: path to ogg/opus/wav/mp3 (WhatsApp voice notes are
                .ogg/opus; ffmpeg decodes them — faster-whisper bundles it).

        Returns:
            Transcript with text and mapped language codes.

        Raises:
            FileNotFoundError: if the audio file does not exist.
            RuntimeError: if detected language is not eng/yor.
        """
        from pathlib import Path

        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"audio file not found: {audio_path}")

        model = self._get_model()
        segments, info = model.transcribe(str(path), beam_size=1)
        text = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
        flores = WHISPER_TO_FLORES.get(info.language, "")

        if not flores:
            raise RuntimeError(
                f"Detected language '{info.language}' is not supported (need English or Yoruba)"
            )
        return Transcript(
            text=text, language=info.language, flores_code=flores, duration_s=info.duration
        )
