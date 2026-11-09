"""Structured logging configuration."""

import logging
import re
import sys

# Meta's GET handshake carries the verify token as a query parameter; uvicorn
# access logs record full URLs, so the token would otherwise land in logs.
# This filter redacts known-sensitive query values from any log record text.
_SENSITIVE_QUERY = re.compile(
    r"((?:hub\.verify_token|hub\.challenge)=)[^&\s]+",
    re.IGNORECASE,
)


class SensitiveQueryFilter(logging.Filter):
    """Redact sensitive query parameters from log records.

    Replaces values after hub.verify_token= and hub.challenge= with
    <redacted>. Applied to the uvicorn.access logger in setup_logging so
    every access line is filtered regardless of how the server is started.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 - never break logging itself
            return True
        if "verify_token=" in msg or "challenge=" in msg:
            redacted = _SENSITIVE_QUERY.sub(r"\1<redacted>", msg)
            record.msg = redacted
            record.args = None  # already interpolated into msg
        return True


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging with a consistent, readable format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    # Quiet noisy third-party loggers
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.setLevel(logging.INFO)
    access_logger.addFilter(SensitiveQueryFilter())
    # Belt-and-braces: apply the same redaction to EVERY log record (root
    # logger), so the filter survives uvicorn's own logging re-configuration
    # (fileConfig) and covers propagated records from any handler path.
    # NOTE: root-logger filters do NOT apply to child loggers that have
    # propagate=False and their own handlers — that is why uvicorn.access
    # also carries the filter directly above.
    logging.getLogger().addFilter(SensitiveQueryFilter())
