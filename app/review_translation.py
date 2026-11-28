"""Local CLI for reviewing English-to-Yoruba translation candidates."""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.review_agent import OllamaClientProtocol

_NUMBER_PATTERN = re.compile(r"(?<!\w)[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?!\w)")


class TranslationClient(Protocol):
    """Translation service interface used by the review workflow."""

    def translate(
        self, text: str, source_language: str, target_language: str
    ) -> TranslationOutput: ...


class TranslationOutput(Protocol):
    """Translation text returned by the existing bidirectional service."""

    @property
    def translated_text(self) -> str: ...


@dataclass(frozen=True)
class PreparedReview:
    """Source, candidate, and numeric findings before back-translation."""

    english_source: str
    yoruba_candidate: str
    numeric_findings: tuple[str, ...]


@dataclass(frozen=True)
class ReviewResult:
    """Inputs and checks produced by one complete translation review."""

    english_source: str
    yoruba_candidate: str
    numeric_findings: tuple[str, ...]
    english_back_translation: str


def _number_counts(text: str) -> Counter[str]:
    return Counter(match.replace(",", "") for match in _NUMBER_PATTERN.findall(text))


def check_numeric_discrepancies(source: str, candidate: str) -> tuple[str, ...]:
    """Report source/candidate digit differences as possible mismatches."""
    missing = _number_counts(source) - _number_counts(candidate)
    extra = _number_counts(candidate) - _number_counts(source)
    findings = []
    if missing:
        findings.append(
            "Possible mismatch: source digit value(s) missing from the Yoruba candidate: "
            f"{', '.join(missing.elements())}. A number may be spelled out in Yoruba."
        )
    if extra:
        findings.append(
            "Possible mismatch: Yoruba candidate digit value(s) absent from the source: "
            f"{', '.join(extra.elements())}."
        )
    return tuple(findings) or ("No digit discrepancy found.",)


def _load_translation_service() -> TranslationClient:
    """Load the configured NLLB checkpoint only when a review is run."""
    from app.services.model_loader import load_nllb
    from app.services.translation import TranslationService

    return TranslationService(load_nllb())


def prepare_translation_review(
    english_source: str,
    yoruba_candidate: str | None = None,
    service: TranslationClient | None = None,
) -> tuple[PreparedReview, TranslationClient | None]:
    """Generate or accept a candidate and run the digit check."""
    if not english_source.strip():
        raise ValueError("English source must not be empty.")
    if yoruba_candidate is not None and not yoruba_candidate.strip():
        raise ValueError("Yoruba candidate must not be empty.")

    translator = service
    candidate = yoruba_candidate
    if candidate is None:
        translator = translator if translator is not None else _load_translation_service()
        candidate = translator.translate(english_source, "eng", "yor").translated_text
    return (
        PreparedReview(
            english_source=english_source,
            yoruba_candidate=candidate,
            numeric_findings=check_numeric_discrepancies(english_source, candidate),
        ),
        translator,
    )


def back_translate_candidate(
    candidate: str, service: TranslationClient | None = None
) -> str:
    """Back-translate a Yoruba candidate through the existing service."""
    translator = service if service is not None else _load_translation_service()
    return translator.translate(candidate, "yor", "eng").translated_text


def review_translation(
    english_source: str,
    yoruba_candidate: str | None = None,
    service: TranslationClient | None = None,
) -> ReviewResult:
    """Review a supplied candidate or generate one, then back-translate it."""
    prepared, translator = prepare_translation_review(english_source, yoruba_candidate, service)
    return ReviewResult(
        english_source=prepared.english_source,
        yoruba_candidate=prepared.yoruba_candidate,
        numeric_findings=prepared.numeric_findings,
        english_back_translation=back_translate_candidate(prepared.yoruba_candidate, translator),
    )


def main(
    argv: Sequence[str] | None = None,
    service: TranslationClient | None = None,
    ollama_client: OllamaClientProtocol | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Review an English-to-Yoruba translation and its English back-translation."
    )
    parser.add_argument("english_source", help="English source sentence")
    parser.add_argument(
        "--candidate",
        help="saved Yoruba candidate; when supplied, skips forward translation",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="ask a local Ollama model to review the candidate",
    )
    parser.add_argument(
        "--ollama-model",
        default=os.environ.get("OLLAMA_MODEL", "llama3.2"),
        help="installed local Ollama model (default: OLLAMA_MODEL or llama3.2)",
    )
    parser.add_argument(
        "--audit-no-issue",
        action="store_true",
        help="queue a no_issue_flagged result for a human audit",
    )
    parser.add_argument("--queue-db", help="SQLite review-queue path")
    args = parser.parse_args(argv)
    if args.audit_no_issue and not args.agent:
        parser.error("--audit-no-issue requires --agent.")

    if args.agent:
        from app.review_agent import OllamaClient, run_agent_review
        from app.review_queue import ReviewQueue

        agent_result = run_agent_review(
            args.english_source,
            args.candidate,
            service=service,
            ollama=ollama_client if ollama_client is not None else OllamaClient(),
            model=args.ollama_model,
        )
        print(f"English source: {agent_result.english_source}")
        print(f"Yoruba candidate: {agent_result.yoruba_candidate or '(unavailable)'}")
        print("Numeric-check findings:")
        for finding in agent_result.numeric_findings:
            print(f"- {finding}")
        if not agent_result.numeric_findings:
            print("- (not run: no Yoruba candidate)")
        print(
            "English back-translation: "
            f"{agent_result.english_back_translation or '(unavailable)'}"
        )
        print(f"Decision: {agent_result.decision}")
        print(f"Reason: {agent_result.reason}")
        print("Tool-call trace:")
        for entry in agent_result.tool_call_trace:
            print(f"- {entry}")
        if not agent_result.tool_call_trace:
            print("- (none)")
        print(f"Review latency: {agent_result.review_latency_ms:.1f} ms")
        should_queue = agent_result.decision in {"needs_review", "review_incomplete"} or (
            agent_result.decision == "no_issue_flagged" and args.audit_no_issue
        )
        if should_queue:
            case_id = ReviewQueue(args.queue_db).enqueue(
                english_source=agent_result.english_source,
                yoruba_candidate=agent_result.yoruba_candidate,
                digit_findings=agent_result.numeric_findings,
                agent_status=agent_result.decision,
                agent_reason=agent_result.reason,
                back_translation=agent_result.english_back_translation,
                tool_trace=agent_result.tool_call_trace,
                audit_unflagged=args.audit_no_issue
                and agent_result.decision == "no_issue_flagged",
            )
            label = (
                "Queued audit case ID"
                if agent_result.decision == "no_issue_flagged"
                else "Queued case ID"
            )
            print(f"{label}: {case_id}")
        return 2 if agent_result.decision == "review_incomplete" else 0

    result = review_translation(args.english_source, args.candidate, service)
    print(f"English source: {result.english_source}")
    print(f"Yoruba candidate: {result.yoruba_candidate}")
    print("Numeric-check findings:")
    for finding in result.numeric_findings:
        print(f"- {finding}")
    print(f"English back-translation: {result.english_back_translation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
