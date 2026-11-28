"""Local SQLite storage for human review of translation-agent cases."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_DEFAULT_DATABASE = Path(__file__).resolve().parents[1] / ".local" / "translation_reviews.sqlite3"
_AGENT_STATUSES = {"needs_review", "no_issue_flagged", "review_incomplete"}
_REVIEW_DECISIONS = {"approve", "correct", "uncertain"}


class ReviewQueueError(Exception):
    """Base error for human-review queue operations."""


class CaseNotFoundError(ReviewQueueError):
    """Raised when a requested case ID does not exist."""


class CaseAlreadyReviewedError(ReviewQueueError):
    """Raised rather than overwriting an existing reviewer decision."""


def default_database_path() -> Path:
    """Return the ignored local database path, optionally overridden by env."""
    override = os.environ.get("YORUBA_REVIEW_DB")
    return Path(override) if override else _DEFAULT_DATABASE


class ReviewQueue:
    """Small SQLite queue whose default database stays under ignored `.local/`."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        self.path = Path(database_path) if database_path is not None else default_database_path()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS review_cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    english_source TEXT NOT NULL,
                    yoruba_candidate TEXT NOT NULL,
                    digit_findings TEXT NOT NULL,
                    agent_status TEXT NOT NULL CHECK (
                        agent_status IN ('needs_review', 'no_issue_flagged', 'review_incomplete')
                    ),
                    agent_reason TEXT NOT NULL,
                    back_translation TEXT,
                    tool_trace TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    audit_unflagged INTEGER NOT NULL DEFAULT 0,
                    reviewer_decision TEXT CHECK (
                        reviewer_decision IS NULL
                        OR reviewer_decision IN ('approve', 'correct', 'uncertain')
                    ),
                    corrected_yoruba TEXT,
                    reviewer_notes TEXT,
                    reviewed_at TEXT,
                    CHECK (
                        (reviewer_decision = 'correct'
                            AND corrected_yoruba IS NOT NULL
                            AND length(trim(corrected_yoruba)) > 0)
                        OR (reviewer_decision IS NULL AND corrected_yoruba IS NULL)
                        OR (reviewer_decision IN ('approve', 'uncertain')
                            AND corrected_yoruba IS NULL)
                    )
                )
                """
            )
            connection.commit()
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def enqueue(
        self,
        *,
        english_source: str,
        yoruba_candidate: str,
        digit_findings: tuple[str, ...],
        agent_status: str,
        agent_reason: str,
        back_translation: str | None,
        tool_trace: tuple[str, ...],
        audit_unflagged: bool = False,
    ) -> int:
        """Insert a review case and return its stable integer ID."""
        if agent_status not in _AGENT_STATUSES:
            raise ValueError(f"Unsupported agent status: {agent_status}")
        if agent_status == "no_issue_flagged" and not audit_unflagged:
            raise ValueError("no_issue_flagged cases require explicit audit selection.")
        if agent_status != "no_issue_flagged" and audit_unflagged:
            raise ValueError("Audit selection only applies to no_issue_flagged cases.")

        created_at = datetime.now(UTC).isoformat(timespec="seconds")
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO review_cases (
                    english_source, yoruba_candidate, digit_findings, agent_status,
                    agent_reason, back_translation, tool_trace, created_at, audit_unflagged
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    english_source,
                    yoruba_candidate,
                    json.dumps(digit_findings, ensure_ascii=False),
                    agent_status,
                    agent_reason,
                    back_translation,
                    json.dumps(tool_trace, ensure_ascii=False),
                    created_at,
                    int(audit_unflagged),
                ),
            )
            if cursor.lastrowid is None:
                raise ReviewQueueError("SQLite did not return an ID for the new case.")
            return cursor.lastrowid

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        case = dict(row)
        case["digit_findings"] = json.loads(case["digit_findings"])
        case["tool_trace"] = json.loads(case["tool_trace"])
        case["audit_unflagged"] = bool(case["audit_unflagged"])
        return case

    def list_pending(self) -> list[dict[str, Any]]:
        """Return queued cases that do not yet have a reviewer decision."""
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM review_cases
                WHERE reviewer_decision IS NULL
                ORDER BY id
                """
            ).fetchall()
        return [self._decode(row) for row in rows]

    def get_case(self, case_id: int) -> dict[str, Any]:
        """Return a case or raise when its ID is unknown."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM review_cases WHERE id = ?", (case_id,)
            ).fetchone()
        if row is None:
            raise CaseNotFoundError(f"Review case {case_id} was not found.")
        return self._decode(row)

    def record_decision(
        self,
        case_id: int,
        decision: str,
        *,
        corrected_yoruba: str | None = None,
        reviewer_notes: str | None = None,
    ) -> None:
        """Record one reviewer decision without allowing silent replacement."""
        if decision not in _REVIEW_DECISIONS:
            raise ValueError(f"Unsupported reviewer decision: {decision}")
        if decision == "correct":
            if corrected_yoruba is None or not corrected_yoruba.strip():
                raise ValueError("A correction requires nonblank corrected Yorùbá text.")
            corrected_yoruba = corrected_yoruba.strip()
        elif corrected_yoruba is not None:
            raise ValueError("Corrected Yorùbá text is only valid for a correct decision.")

        notes = reviewer_notes.strip() if reviewer_notes and reviewer_notes.strip() else None
        reviewed_at = datetime.now(UTC).isoformat(timespec="seconds")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT reviewer_decision FROM review_cases WHERE id = ?", (case_id,)
            ).fetchone()
            if row is None:
                raise CaseNotFoundError(f"Review case {case_id} was not found.")
            if row["reviewer_decision"] is not None:
                raise CaseAlreadyReviewedError(
                    f"Review case {case_id} already has a reviewer decision."
                )
            updated = connection.execute(
                """
                UPDATE review_cases
                SET reviewer_decision = ?, corrected_yoruba = ?, reviewer_notes = ?,
                    reviewed_at = ?
                WHERE id = ? AND reviewer_decision IS NULL
                """,
                (decision, corrected_yoruba, notes, reviewed_at, case_id),
            )
            if updated.rowcount != 1:
                raise CaseAlreadyReviewedError(
                    f"Review case {case_id} already has a reviewer decision."
                )

    def summary(self) -> dict[str, int]:
        """Count reviewed-sample outcomes and pending cases separately."""
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT
                    SUM(CASE WHEN reviewed_at IS NOT NULL THEN 1 ELSE 0 END) AS reviewed,
                    SUM(CASE WHEN reviewed_at IS NOT NULL
                        AND agent_status = 'needs_review'
                        AND reviewer_decision = 'correct' THEN 1 ELSE 0 END) AS caught_errors,
                    SUM(CASE WHEN reviewed_at IS NOT NULL
                        AND agent_status = 'needs_review'
                        AND reviewer_decision = 'approve' THEN 1 ELSE 0 END) AS false_flags,
                    SUM(CASE WHEN reviewed_at IS NOT NULL
                        AND agent_status = 'no_issue_flagged'
                        AND reviewer_decision = 'correct' THEN 1 ELSE 0 END) AS missed_errors,
                    SUM(CASE WHEN reviewed_at IS NOT NULL
                        AND reviewer_decision = 'uncertain' THEN 1 ELSE 0 END) AS uncertain,
                    SUM(CASE WHEN reviewed_at IS NOT NULL
                        AND agent_status = 'review_incomplete' THEN 1 ELSE 0 END) AS incomplete,
                    SUM(CASE WHEN reviewed_at IS NULL THEN 1 ELSE 0 END) AS pending,
                    SUM(CASE WHEN reviewed_at IS NULL
                        AND agent_status = 'review_incomplete' THEN 1 ELSE 0 END)
                        AS pending_incomplete
                FROM review_cases
                """
            ).fetchone()
        fields = (
            "reviewed",
            "caught_errors",
            "false_flags",
            "missed_errors",
            "uncertain",
            "incomplete",
            "pending",
            "pending_incomplete",
        )
        return {key: int(row[key] or 0) for key in fields}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone local queue-management CLI."""
    parser = argparse.ArgumentParser(
        description="Inspect and annotate the local translation-review queue."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def add_database_option(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--db", help="SQLite database path")

    list_parser = commands.add_parser("list", help="list pending human-review cases")
    add_database_option(list_parser)

    show_parser = commands.add_parser("show", help="inspect a case by ID")
    show_parser.add_argument("case_id", type=int)
    add_database_option(show_parser)

    decide_parser = commands.add_parser("decide", help="record one reviewer decision")
    decide_parser.add_argument("case_id", type=int)
    decide_parser.add_argument("decision", choices=("approve", "correct", "uncertain"))
    decide_parser.add_argument("--corrected", help="corrected Yorùbá text (required for correct)")
    decide_parser.add_argument("--notes", help="optional reviewer notes")
    add_database_option(decide_parser)

    summary_parser = commands.add_parser("summary", help="summarize the reviewed sample")
    add_database_option(summary_parser)

    args = parser.parse_args(argv)
    queue = ReviewQueue(args.db)
    try:
        if args.command == "list":
            pending = queue.list_pending()
            if not pending:
                print("No pending review cases.")
                return 0
            for case in pending:
                print(
                    f"#{case['id']} | {case['agent_status']} | {case['created_at']}\n"
                    f"  English: {case['english_source']}\n"
                    f"  Yorùbá: {case['yoruba_candidate']}"
                )
            return 0

        if args.command == "show":
            print(json.dumps(queue.get_case(args.case_id), ensure_ascii=False, indent=2))
            return 0

        if args.command == "decide":
            if args.decision == "correct" and (
                args.corrected is None or not args.corrected.strip()
            ):
                parser.error(
                    "--corrected must contain nonblank Yorùbá text for a correct decision."
                )
            if args.decision != "correct" and args.corrected is not None:
                parser.error("--corrected is only valid with the correct decision.")
            queue.record_decision(
                args.case_id,
                args.decision,
                corrected_yoruba=args.corrected,
                reviewer_notes=args.notes,
            )
            print(f"Recorded {args.decision} for case {args.case_id}.")
            return 0

        summary = queue.summary()
        print("Human review summary — results from reviewed sample")
        print(f"Reviewed sample size: {summary['reviewed']}")
        print(f"Caught errors (flagged, corrected): {summary['caught_errors']}")
        print(f"False flags (flagged, approved): {summary['false_flags']}")
        print(f"Missed errors (audited unflagged, corrected): {summary['missed_errors']}")
        print(f"Uncertain reviewer decisions: {summary['uncertain']}")
        print(f"Agent-incomplete cases in reviewed sample: {summary['incomplete']}")
        print(f"Pending cases (outside reviewed sample): {summary['pending']}")
        print(f"Pending agent-incomplete cases: {summary['pending_incomplete']}")
        return 0
    except (CaseAlreadyReviewedError, CaseNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
