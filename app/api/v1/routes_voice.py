"""POST /api/v1/voice-translate — voice note in, translated text out."""

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile

from app.models.translation import TranslationResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["voice"])

MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB: generous for 2-min opus voice notes

# Prefix owned by this app in the temp dir. The stale-file sweeper only ever
# deletes files matching this prefix — nothing else in the shared temp dir.
AUDIO_TMP_PREFIX = "ybvoice_"


def purge_stale_audio(max_age_seconds: int = 3600, base_dir: str | None = None) -> int:
    """Delete leftover voice-upload temp files/dirs from abnormal terminations.

    A kill -9 or power loss between TemporaryDirectory creation and its
    cleanup would leave audio (user voice content!) in /tmp indefinitely.
    This sweeper removes ONLY entries with our prefix and older than
    max_age, so it cannot touch unrelated temp files or in-flight uploads.

    Returns: number of entries removed (files AND directories).
    """
    import shutil
    import time

    tmp = Path(base_dir or tempfile.gettempdir())
    now = time.time()
    removed = 0
    for candidate in tmp.glob(f"{AUDIO_TMP_PREFIX}*"):
        try:
            if now - candidate.stat().st_mtime <= max_age_seconds:
                continue  # recent or in-flight: keep it
            if candidate.is_dir():
                shutil.rmtree(candidate)  # stale dir (TemporaryDirectory leftovers)
            else:
                candidate.unlink(missing_ok=True)
            removed += 1
        except OSError:  # vanished mid-scan or locked: skip it
            continue
    if removed:
        logger.info("Purged %d stale voice-upload temp entr(y/ies)", removed)
    return removed


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
    # TemporaryDirectory GUARANTEES cleanup on success, exceptions, and
    # early returns — unlike a manual NamedTemporaryFile + finally, which
    # leaks on interpreter kill. Audio is user voice content: it must
    # never outlive the request.
    with tempfile.TemporaryDirectory(prefix=AUDIO_TMP_PREFIX, dir=tempfile.gettempdir()) as tmp_dir:
        tmp_path = Path(tmp_dir) / f"note{suffix}"
        tmp_path.write_bytes(data)

        try:
            transcript = service.transcribe(str(tmp_path))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:  # unsupported language
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - decode/model errors
            logger.error("Transcription failed: %s", type(exc).__name__)
            raise HTTPException(status_code=500, detail="Transcription failed") from exc

    if not transcript.text:
        raise HTTPException(status_code=422, detail="No speech detected in audio")

    target = "eng" if transcript.flores_code == "yor" else "yor"
    result = translator.translate(transcript.text, transcript.flores_code, target)

    response = TranslationResponse(**result.__dict__, cached=False)
    logger.info(
        "Voice note %.1fs -> '%s' (%s) -> '%s'",
        transcript.duration_s,
        transcript.text,
        transcript.flores_code,
        result.translated_text,
    )
    return response
