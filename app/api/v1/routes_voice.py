"""POST /api/v1/voice-translate — voice note in, translated text out."""
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile

from app.models.translation import TranslationResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["voice"])

MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB: generous for 2-min opus voice notes


@router.post("/voice-translate", response_model=TranslationResponse)
async def voice_translate(request: Request, audio: UploadFile) -> TranslationResponse:
    """Transcribe an uploaded voice note, then translate it.

    Language is auto-detected by Whisper; translation direction follows the
    detected language (yo→eng or eng→yo).
    """
    service = getattr(request.app.state, "speech_service", None)
    translator = getattr(request.app.state, "translation_service", None)
    if service is None or translator is None:
        raise HTTPException(status_code=503, detail="Voice service not loaded")

    data = await audio.read()
    if not data:
        raise HTTPException(status_code=422, detail="Empty audio upload")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large (max 10 MB)")

    suffix = Path(audio.filename or "note.ogg").suffix or ".ogg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        transcript = service.transcribe(tmp_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:  # unsupported language
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - decode/model errors
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail="Transcription failed") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not transcript.text:
        raise HTTPException(status_code=422, detail="No speech detected in audio")

    target = "eng" if transcript.flores_code == "yor" else "yor"
    result = translator.translate(transcript.text, transcript.flores_code, target)

    response = TranslationResponse(**result.__dict__, cached=False)
    logger.info(
        "Voice note %.1fs -> '%s' (%s) -> '%s'",
        transcript.duration_s, transcript.text, transcript.flores_code, result.translated_text,
    )
    return response
