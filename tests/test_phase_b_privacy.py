"""Phase B tests: temp-file cleanup, log masking, retention, admin cleanup."""

import logging
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.logging import SensitiveQueryFilter
from app.services.history import RetentionCleanup

# ---------- B1: voice temp files ----------


@pytest.fixture
def isolated_tmpdir(tmp_path: Path) -> Path:
    """Route tempfile.gettempdir() to a per-test directory; auto-undo."""
    mp = pytest.MonkeyPatch()
    mp.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    yield tmp_path
    mp.undo()


def _voice_client_with(crash: bool) -> TestClient:
    from app.main import create_app
    from app.services.speech import Transcript

    app = create_app(load_model=False)
    speech = MagicMock()
    if crash:
        speech.transcribe.side_effect = ValueError("boom")
    else:
        speech.transcribe.return_value = Transcript(
            text="hello", language="en", flores_code="eng", duration_s=1.0
        )
    translator = MagicMock()
    from app.services.translation import TranslationResult

    translator.translate.return_value = TranslationResult(
        translated_text="x", source_language="eng", target_language="yor", latency_ms=0.1
    )
    app.state.speech_service = speech
    app.state.translation_service = translator
    return TestClient(app)


def test_voice_temp_file_removed_on_success(isolated_tmpdir: Path) -> None:
    client = _voice_client_with(crash=False)
    resp = client.post(
        "/api/v1/voice-translate",
        files={"audio": ("n.ogg", b"RIFF", "audio/ogg")},
    )
    assert resp.status_code == 200
    leftovers = list(isolated_tmpdir.glob("ybvoice_*"))
    assert leftovers == [], "temp dir should be empty after success"


def test_voice_temp_file_removed_on_exception(isolated_tmpdir: Path) -> None:
    client = _voice_client_with(crash=True)
    resp = client.post(
        "/api/v1/voice-translate",
        files={"audio": ("n.ogg", b"RIFF", "audio/ogg")},
    )
    assert resp.status_code == 500
    leftovers = list(isolated_tmpdir.glob("ybvoice_*"))
    leftovers_detail = [str(p) for p in leftovers]
    assert leftovers == [], f"temp leftovers after failure: {leftovers_detail}"


def test_purge_stale_audio_removes_only_old_own_files(isolated_tmpdir: Path) -> None:
    from app.api.v1.routes_voice import AUDIO_TMP_PREFIX, purge_stale_audio

    old = isolated_tmpdir / f"{AUDIO_TMP_PREFIX}old.ogg"
    fresh = isolated_tmpdir / f"{AUDIO_TMP_PREFIX}fresh.ogg"
    unrelated = isolated_tmpdir / "someone_else.txt"
    for p in (old, fresh, unrelated):
        p.write_bytes(b"x")
    # make 'old' actually old
    past = time.time() - 7200
    import os

    os.utime(old, (past, past))

    removed = purge_stale_audio(max_age_seconds=3600)

    assert removed == 1
    assert not old.exists(), "stale own file should be purged"
    assert fresh.exists(), "recent own file must be kept"
    assert unrelated.exists(), "foreign temp files must NEVER be touched"


def test_purge_stale_audio_removes_stale_directories(isolated_tmpdir: Path) -> None:
    """Stale TemporaryDirectory leftovers (DIRECTORIES with audio inside)
    must be removed entirely — while recent dirs and foreign files stay."""
    import os

    from app.api.v1.routes_voice import AUDIO_TMP_PREFIX, purge_stale_audio

    stale_dir = isolated_tmpdir / f"{AUDIO_TMP_PREFIX}abc123"
    stale_dir.mkdir()
    (stale_dir / "note.ogg").write_bytes(b"audio-bytes")  # user voice content

    fresh_dir = isolated_tmpdir / f"{AUDIO_TMP_PREFIX}freshdir"
    fresh_dir.mkdir()

    unrelated_dir = isolated_tmpdir / "unrelated_dir"
    unrelated_dir.mkdir()
    (unrelated_dir / "important.txt").write_text("keep me")

    past = time.time() - 7200
    os.utime(stale_dir, (past, past))

    removed = purge_stale_audio(max_age_seconds=3600)

    assert removed == 1
    assert not stale_dir.exists(), "stale dir with audio inside must be purged"
    assert fresh_dir.exists(), "recent dir must be kept (in-flight upload)"
    assert unrelated_dir.exists() and (unrelated_dir / "important.txt").exists(), (
        "foreign dirs must NEVER be touched"
    )
    # audio content is actually gone from disk
    assert not (isolated_tmpdir / f"{AUDIO_TMP_PREFIX}abc123" / "note.ogg").exists()


# ---------- B2: access-log query masking ----------


def test_sensitive_query_filter_redacts_verify_token() -> None:
    record = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/1.1" 200',
        args=(
            "127.0.0.1",
            "GET",
            "/api/v1/webhook?hub.mode=subscribe&hub.verify_token=SUPERSECRET&hub.challenge=CH",
        ),
        exc_info=None,
    )
    f = SensitiveQueryFilter()
    assert f.filter(record) is True  # never blocks records
    out = record.getMessage()
    assert "SUPERSECRET" not in out
    assert "<redacted>" in out


def test_sensitive_query_filter_leaves_normal_lines() -> None:
    record = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="POST /api/v1/translate HTTP/1.1 200",
        args=None,
        exc_info=None,
    )
    SensitiveQueryFilter().filter(record)
    assert record.getMessage() == "POST /api/v1/translate HTTP/1.1 200"


def test_filter_survives_uvicorn_style_reconfiguration(
    capsys: pytest.CaptureFixture,
) -> None:
    """Simulate uvicorn boot: configure ITS log config FIRST, then app import
    runs setup_logging — redaction must still apply on records emitted
    through uvicorn.access's own handler afterwards."""
    import io

    from app.core.logging import setup_logging

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    uv_acc = logging.getLogger("uvicorn.access")
    uv_acc.handlers = [handler]  # uvicorn replaces handlers like fileConfig
    uv_acc.propagate = False
    uv_acc.setLevel(logging.INFO)

    setup_logging("INFO")  # our app import — happens AFTER uvicorn config

    uv_acc.info(
        "GET /api/v1/webhook?hub.mode=subscribe"
        "&hub.verify_token=REALTOKEN123&hub.challenge=C1 HTTP/1.1 200"
    )
    out = stream.getvalue()
    assert "REALTOKEN123" not in out, "verify token leaked through uvicorn's own handler"
    assert "<redacted>" in out


# ---------- B3: retention cleanup ----------


def make_retention(work: bool = True) -> tuple[RetentionCleanup, MagicMock]:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    if work:
        cur.fetchone.return_value = (7,)
    else:
        conn.cursor.side_effect = ConnectionError("down")
    return RetentionCleanup(lambda: conn, retention_days=30), conn


def test_retention_dry_run_counts_without_delete() -> None:
    cleanup, conn = make_retention()
    result = cleanup.cleanup(dry_run=True)
    assert result == {"status": "dry_run", "would_delete": 7}
    sql = conn.cursor.return_value.__enter__.return_value.execute.call_args_list[-1].args[0]
    assert "DELETE" not in sql, "dry run must not emit a DELETE"
    assert "translation_history" in sql


def test_retention_delete_scoped_to_table_and_window() -> None:
    cleanup, conn = make_retention()
    result = cleanup.cleanup(dry_run=False)
    assert result == {"status": "ok", "deleted": 7}
    calls = conn.cursor.return_value.__enter__.return_value.execute.call_args_list
    delete_sql = calls[-1].args[0]
    assert delete_sql.startswith("DELETE FROM translation_history"), "only our table"
    assert "created_at" in delete_sql
    params = calls[-1].args[1]
    assert params == ("30",), "retention days must be parameterized, not interpolated"


def test_retention_never_raises_when_db_down() -> None:
    cleanup, _ = make_retention(work=False)
    result = cleanup.cleanup(dry_run=False)
    assert result["status"] == "unavailable"
    assert result["deleted"] == 0


def test_retention_days_floor_at_one() -> None:
    cleanup, _ = make_retention()
    assert RetentionCleanup(lambda: MagicMock(), retention_days=0).retention_days == 1


# ---------- B4: CLI safety ----------


def test_cli_disabled_database_is_noop(capsys: pytest.CaptureFixture) -> None:
    import sys
    from unittest.mock import MagicMock, patch

    from app.cli.cleanup import main

    with (
        patch.object(sys, "argv", ["cleanup", "--dry-run"]),
        patch("app.cli.cleanup.get_settings") as gs,
    ):
        s = MagicMock()
        s.database_enabled = False
        s.retention_days = 30
        gs.return_value = s
        rc = main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "persistence is off" in out


def test_cli_days_override_cannot_go_below_one(
    capsys: pytest.CaptureFixture,
) -> None:
    """--days 0/negative must never widen the window to 'delete everything'.

    The CLI refuses days < 1 at the boundary (RetentionPolicy also floors).
    """
    import sys
    from unittest.mock import MagicMock, patch

    from app.cli.cleanup import main

    with (
        patch.object(sys, "argv", ["cleanup", "--days", "0", "--yes"]),
        patch("app.cli.cleanup.get_settings") as gs,
    ):
        s = MagicMock()
        s.database_enabled = True
        s.retention_days = 30
        gs.return_value = s
        rc = main()

    out = capsys.readouterr().out
    assert rc == 2
    assert "refusing to run" in out


# ---------- Retention task lifecycle ----------


def test_retention_task_cancelled_cleanly_on_shutdown() -> None:
    """With history enabled, the daily retention task must exist while the
    app runs and be cancelled (not left dangling) at shutdown."""
    import os
    import time
    from unittest.mock import MagicMock, patch

    from app.core.config import get_settings
    from app.main import create_app

    os.environ["DATABASE_ENABLED"] = "true"
    get_settings.cache_clear()
    try:
        with (
            patch(
                "app.services.history.make_history_store",
                side_effect=lambda: MagicMock(_connection_factory=lambda: None),
            ),
            patch(
                "app.services.history.RetentionCleanup.cleanup",
                lambda self, dry_run=False: {"status": "ok", "deleted": 0},
            ),
        ):
            app = create_app(load_model=False)
            with TestClient(app) as c:
                task = getattr(c.app.state, "retention_task", None)
                assert task is not None, "retention task must be scheduled"
                assert not task.done()
                time.sleep(0.2)  # allow the first cleanup to run
            assert task.cancelled() or task.done(), "task must stop at shutdown"
    finally:
        os.environ.pop("DATABASE_ENABLED", None)
        get_settings.cache_clear()


def test_retention_task_absent_when_db_disabled() -> None:
    """Demonstration default: no DB -> no retention task, nothing scheduled."""
    from app.main import create_app

    app = create_app(load_model=False)
    with TestClient(app) as c:
        assert getattr(c.app.state, "retention_task", None) is None
