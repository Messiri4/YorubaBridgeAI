"""Local-only administrative cleanup for translation history.

NOT an HTTP endpoint — this is a CLI, so it can never be reached through
the public webhook/HTTP surface. It only ever touches the
translation_history table (single WHERE on created_at).

Usage:
    python -m app.cli.cleanup --dry-run          # report what would go
    python -m app.cli.cleanup                    # delete, after confirm
    python -m app.cli.cleanup --yes              # delete without prompt
"""

import argparse
import sys

from app.core.config import get_settings
from app.services.history import RetentionCleanup, make_history_store


def main() -> int:
    parser = argparse.ArgumentParser(description="YorubaBridgeAI retention cleanup")
    parser.add_argument("--dry-run", action="store_true", help="count expired rows, delete nothing")
    parser.add_argument("--yes", action="store_true", help="skip interactive confirmation")
    parser.add_argument("--days", type=int, default=None, help="override RETENTION_DAYS")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.database_enabled:
        print("DATABASE_ENABLED=false — translation history persistence is off.")
        print("Nothing to clean up. Exiting.")
        return 0

    days = args.days if args.days is not None else settings.retention_days
    # Safety floor at the CLI boundary: --days 0/-1 must never widen the
    # window to 'delete everything'. RetentionCleanup also floors at 1.
    if days < 1:
        print(f"--days must be >= 1 (got {days}); refusing to run.")
        return 2
    store = make_history_store()
    if store is None:  # pragma: no cover - database_enabled checked above
        return 0

    cleanup = RetentionCleanup(
        connection_factory=store._connection_factory,  # noqa: SLF001 - same package
        retention_days=days,
    )

    print(f"Retention window: {days} day(s) — anything older will be deleted.")
    result = cleanup.cleanup(dry_run=True)
    if result["status"] == "unavailable":
        print("Database unavailable; nothing changed.")
        return 1

    would = result.get("would_delete", 0)
    print(f"Dry run: {would} translation record(s) older than {days} day(s) would be deleted.")
    if args.dry_run:
        return 0

    if not args.yes:
        answer = input(f"Type DELETE to permanently remove {would} record(s): ").strip()
        if answer != "DELETE":
            print("Aborted — no records deleted.")
            return 0

    final = cleanup.cleanup(dry_run=False)
    if final["status"] == "ok":
        print(f"Deleted {final['deleted']} record(s).")
        return 0
    print("Cleanup failed; no records deleted.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
