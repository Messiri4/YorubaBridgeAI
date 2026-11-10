"""Sprint 3 PHASE 2 — optional diagnostic tracing for the protection pipeline.

DIAGNOSTIC ONLY: when disabled (default) this module is never imported at
runtime and adds zero production behavior. When enabled — via the env var
``YB_TRANSLATION_TRACE=1`` (file sink) or ``YB_TRANSLATION_TRACE_ECHO=1``
(stderr echo, for notebooks) — it records each pipeline stage with enough
detail to answer "which stage lost or changed the placeholders?":

    protect               source text, protected text, placeholder map
    generate_protected    raw model output (sentinels as generated)
    restore               ok/failure + restored text
    generate_retry_unprotected  plain regeneration output (if fallback fired)
    fallback_decision     used_fallback flag + failure reason
    residual_scrub        pre-scrub text (if the last-resort scrub acted)
    finish                final returned text

Collected traces are dumped to data/eval/trace_<stamp>.jsonl on process
exit (atexit), one JSON object per translation, and can also be pulled
in-process via ``get_traces()`` (used by tests/diagnostics).
"""
from __future__ import annotations

import atexit
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "eval"

_ENABLED_FILE = None
_ENABLED_ECHO = None


def _enabled_file() -> bool:
    global _ENABLED_FILE
    if _ENABLED_FILE is None:
        _ENABLED_FILE = os.environ.get("YB_TRANSLATION_TRACE", "") not in ("", "0", "false")
    return _ENABLED_FILE


def _enabled_echo() -> bool:
    global _ENABLED_ECHO
    if _ENABLED_ECHO is None:
        _ENABLED_ECHO = os.environ.get("YB_TRANSLATION_TRACE_ECHO", "") not in ("", "0", "false")
    return _ENABLED_ECHO


@dataclass
class Stage:
    name: str
    at_ms: float
    fields: dict = field(default_factory=dict)


@dataclass
class Trace:
    source_text: str
    stages: list[Stage] = field(default_factory=list)
    t0: float = field(default_factory=time.perf_counter)

    def stage(self, name: str, **fields: object) -> None:
        self.stages.append(Stage(name=name, at_ms=(time.perf_counter() - self.t0) * 1000,
                                 fields=dict(fields)))

    def close(self) -> None:
        _TRACES.append(self)

    def to_dict(self) -> dict:
        return {
            "source": self.source_text,
            "stages": [{"name": s.name, "at_ms": round(s.at_ms, 2), **s.fields}
                       for s in self.stages],
        }


_TRACES: list[Trace] = []


def get_traces() -> list[dict]:
    """In-process access for tests/diagnostics (list of dicts)."""
    return [t.to_dict() for t in _TRACES]


def reset_traces() -> None:
    """Clear in-memory traces (test isolation)."""
    _TRACES.clear()


def dump_to_disk() -> str | None:
    """Persist collected traces to data/eval/trace_<stamp>.jsonl (diagnostic)."""
    if not _TRACES:
        return None
    OUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = OUT_DIR / f"trace_{stamp}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for t in _TRACES:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
    return str(path)


def _atexit_dump() -> None:
    if _enabled_file():
        dump_to_disk()


atexit.register(_atexit_dump)


class _NullTrace:
    """Disabled tracer: every stage() call is a no-op."""

    def stage(self, name: str, **fields: object) -> None:  # pragma: no cover
        return None

    def close(self) -> None:  # pragma: no cover
        return None


_NULL = _NullTrace()


def diagnostic_trace(source_text: str = "") -> object:
    """Return an active Trace when diagnostics are enabled, else a no-op.

    The translation service calls this via self._diagnostic_trace() and sets
    ``trace.source_text`` immediately after; the indirection keeps the
    production path free of this module entirely.
    """
    if _enabled_file() or _enabled_echo():
        t = Trace(source_text=source_text)

        if _enabled_echo():
            base_stage = t.stage

            def echo_stage(name: str, **fields: object) -> None:
                print(f"[trace] {name}: {fields}", flush=True)
                base_stage(name, **fields)

            t.stage = echo_stage  # type: ignore[method-assign]
        return t
    return _NULL
