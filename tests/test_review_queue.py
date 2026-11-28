"""SQLite human-review queue tests using temporary databases."""

import pytest

from app.review_queue import (
    CaseAlreadyReviewedError,
    CaseNotFoundError,
    ReviewQueue,
)


def _enqueue(
    queue: ReviewQueue,
    *,
    status: str = "needs_review",
    audit_unflagged: bool = False,
) -> int:
    return queue.enqueue(
        english_source="The driver did not deliver 10 boxes.",
        yoruba_candidate="Awakọ fi apoti 10 ranṣẹ́.",
        digit_findings=("No digit discrepancy found.",),
        agent_status=status,
        agent_reason="The back-translation omitted the negation.",
        back_translation="The driver delivered 10 boxes.",
        tool_trace=("back_translate requested", "back_translate completed"),
        audit_unflagged=audit_unflagged,
    )


def test_case_saves_agent_evidence_and_lists_as_pending(tmp_path) -> None:
    queue = ReviewQueue(tmp_path / "reviews.sqlite3")
    case_id = _enqueue(queue)

    case = queue.get_case(case_id)
    assert case["english_source"] == "The driver did not deliver 10 boxes."
    assert case["yoruba_candidate"] == "Awakọ fi apoti 10 ranṣẹ́."
    assert case["digit_findings"] == ["No digit discrepancy found."]
    assert case["agent_status"] == "needs_review"
    assert case["agent_reason"] == "The back-translation omitted the negation."
    assert case["back_translation"] == "The driver delivered 10 boxes."
    assert case["tool_trace"] == ["back_translate requested", "back_translate completed"]
    assert case["created_at"]
    assert queue.list_pending() == [case]


def test_no_issue_requires_explicit_audit_selection(tmp_path) -> None:
    queue = ReviewQueue(tmp_path / "reviews.sqlite3")

    with pytest.raises(ValueError, match="explicit audit selection"):
        _enqueue(queue, status="no_issue_flagged")

    case_id = _enqueue(queue, status="no_issue_flagged", audit_unflagged=True)
    case = queue.get_case(case_id)
    assert case["agent_status"] == "no_issue_flagged"
    assert case["audit_unflagged"] is True


def test_correction_requires_text_and_preserves_original_candidate(tmp_path) -> None:
    queue = ReviewQueue(tmp_path / "reviews.sqlite3")
    case_id = _enqueue(queue)

    with pytest.raises(ValueError, match="requires nonblank"):
        queue.record_decision(case_id, "correct", corrected_yoruba="  ")

    queue.record_decision(
        case_id,
        "correct",
        corrected_yoruba=" Awakọ kò fi apoti mẹ́wàá ranṣẹ́. ",
        reviewer_notes="Restored negation and spelled the quantity.",
    )
    case = queue.get_case(case_id)

    assert case["yoruba_candidate"] == "Awakọ fi apoti 10 ranṣẹ́."
    assert case["corrected_yoruba"] == "Awakọ kò fi apoti mẹ́wàá ranṣẹ́."
    assert case["reviewer_decision"] == "correct"
    assert case["reviewer_notes"] == "Restored negation and spelled the quantity."
    assert case["reviewed_at"]
    assert queue.list_pending() == []


def test_second_decision_does_not_overwrite_first(tmp_path) -> None:
    queue = ReviewQueue(tmp_path / "reviews.sqlite3")
    case_id = _enqueue(queue)
    queue.record_decision(case_id, "approve")

    with pytest.raises(CaseAlreadyReviewedError, match="already has a reviewer decision"):
        queue.record_decision(
            case_id, "correct", corrected_yoruba="Different Yoruba text."
        )

    assert queue.get_case(case_id)["reviewer_decision"] == "approve"


def test_summary_counts_error_categories_uncertain_and_incomplete(tmp_path) -> None:
    queue = ReviewQueue(tmp_path / "reviews.sqlite3")

    caught_id = _enqueue(queue)
    queue.record_decision(caught_id, "correct", corrected_yoruba="Corrected.")

    false_flag_id = _enqueue(queue)
    queue.record_decision(false_flag_id, "approve")

    missed_id = _enqueue(queue, status="no_issue_flagged", audit_unflagged=True)
    queue.record_decision(missed_id, "correct", corrected_yoruba="Missed correction.")

    uncertain_id = _enqueue(queue)
    queue.record_decision(uncertain_id, "uncertain")

    incomplete_id = _enqueue(queue, status="review_incomplete")
    queue.record_decision(incomplete_id, "uncertain")

    _enqueue(queue, status="review_incomplete")

    summary = queue.summary()
    assert summary == {
        "reviewed": 5,
        "caught_errors": 1,
        "false_flags": 1,
        "missed_errors": 1,
        "uncertain": 2,
        "incomplete": 1,
        "pending": 1,
        "pending_incomplete": 1,
    }


def test_unknown_case_id_is_reported(tmp_path) -> None:
    queue = ReviewQueue(tmp_path / "reviews.sqlite3")

    with pytest.raises(CaseNotFoundError, match="was not found"):
        queue.get_case(99)
