"""Tests for tool-free, evidence-first translation review and its queue."""

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

from app.review_agent import OllamaClient, OllamaError, run_agent_review
from app.review_queue import ReviewQueue
from app.review_queue import main as review_queue_main
from app.review_translation import main


@dataclass
class FakeResult:
    translated_text: str


class FakeTranslator:
    def __init__(
        self,
        back_translation: str = "The bus arrives at 8.",
        forward_translation: str = "Ó ra ìwé 3.",
    ) -> None:
        self.back_translation = back_translation
        self.forward_translation = forward_translation
        self.calls: list[tuple[str, str, str]] = []

    def translate(self, text: str, source_language: str, target_language: str) -> FakeResult:
        self.calls.append((text, source_language, target_language))
        translated = (
            self.forward_translation
            if (source_language, target_language) == ("eng", "yor")
            else self.back_translation
        )
        return FakeResult(translated)


class FakeOllama:
    def __init__(
        self, responses: list[dict[str, Any]], models: set[str] | None = None
    ) -> None:
        self.responses = responses
        self.models = models if models is not None else {"llama3.2:latest"}
        self.calls: list[dict[str, Any]] = []

    def list_model_names(self) -> set[str]:
        return self.models

    def chat(self, model: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        self.calls.append({"model": model, "messages": messages})
        return self.responses.pop(0)


def _decision(decision: str, reason: str) -> dict[str, Any]:
    content = json.dumps({"decision": decision, "reason": reason})
    return {"message": {"role": "assistant", "content": content}}


def test_model_list_uses_get_without_body_and_other_requests_remain_post() -> None:
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"models":[{"name":"gemma3:1b"}]}'
        return response

    client = OllamaClient()
    with patch("app.review_agent.urlopen", side_effect=fake_urlopen):
        assert client.list_model_names() == {"gemma3:1b"}
        client._post("/api/show", {"model": "gemma3:1b"})

    assert requests[0].get_method() == "GET"
    assert requests[0].data is None
    assert requests[1].get_method() == "POST"


def test_connection_reset_while_listing_models_returns_incomplete() -> None:
    client = OllamaClient()
    with patch(
        "app.review_agent.urlopen",
        side_effect=ConnectionResetError(10054, "Connection reset by peer"),
    ):
        result = run_agent_review(
            "The driver did not deliver 10 boxes.",
            "Awakọ fi apoti 10 ranṣẹ́.",
            service=FakeTranslator(),
            ollama=client,
        )

    assert result.decision == "review_incomplete"
    assert "Connection reset while contacting local Ollama" in result.reason
    assert "list_model_names" not in result.reason


def test_backtranslation_is_sent_before_ollama_can_return_a_decision() -> None:
    translator = FakeTranslator("The driver delivered 10 boxes.")

    class EvidenceCheckingOllama(FakeOllama):
        def chat(self, model: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
            assert translator.calls == [
                ("Awakọ fi apoti 10 ranṣẹ́.", "yor", "eng")
            ]
            evidence_text = messages[1]["content"].removeprefix(
                "Review this translation evidence: "
            )
            evidence = json.loads(evidence_text)
            assert evidence == {
                "english_source": "The driver did not deliver 10 boxes.",
                "english_back_translation": "The driver delivered 10 boxes.",
                "digit_findings": ["No digit discrepancy found."],
            }
            return super().chat(model, messages)

    ollama = EvidenceCheckingOllama(
        [
            _decision(
                "needs_review",
                "The source denies delivery, but the back-translation affirms delivery.",
            )
        ]
    )
    result = run_agent_review(
        "The driver did not deliver 10 boxes.",
        "Awakọ fi apoti 10 ranṣẹ́.",
        service=translator,
        ollama=ollama,
    )

    assert len(ollama.calls) == 1
    assert result.english_back_translation == "The driver delivered 10 boxes."
    assert result.tool_call_trace == (
        "back_translate started by Python",
        "back_translate completed",
    )
    assert result.decision == "needs_review"
    assert "source denies delivery" in result.reason
    assert "back-translation affirms delivery" in result.reason


def test_faithful_translation_can_be_cleared_after_evidence() -> None:
    translator = FakeTranslator("The bus arrives at 8.")
    ollama = FakeOllama([_decision("no_issue_flagged", "The timing and quantity match.")])

    result = run_agent_review(
        "The bus arrives at 8.",
        "Bọ́ọ̀sì náà dé ní 8.",
        service=translator,
        ollama=ollama,
    )

    assert result.decision == "no_issue_flagged"
    assert result.english_back_translation == "The bus arrives at 8."
    assert translator.calls == [("Bọ́ọ̀sì náà dé ní 8.", "yor", "eng")]
    assert len(ollama.calls) == 1


def test_valid_paraphrase_receives_all_evidence_in_single_json_decision() -> None:
    translator = FakeTranslator("I'll phone you the next day.")
    ollama = FakeOllama(
        [_decision("no_issue_flagged", "Calling the next day preserves tomorrow.")]
    )

    result = run_agent_review(
        "I will call you tomorrow.",
        "Màá pè ọ lọ́la.",
        service=translator,
        ollama=ollama,
    )

    evidence = json.loads(ollama.calls[0]["messages"][1]["content"].removeprefix(
        "Review this translation evidence: "
    ))
    assert evidence["english_source"] == "I will call you tomorrow."
    assert evidence["english_back_translation"] == "I'll phone you the next day."
    assert evidence["digit_findings"] == ["No digit discrepancy found."]
    assert result.decision == "no_issue_flagged"
    assert len(ollama.calls) == 1


def test_backtranslation_failure_prevents_ollama_decision() -> None:
    class BrokenTranslator(FakeTranslator):
        def translate(self, text: str, source_language: str, target_language: str) -> FakeResult:
            self.calls.append((text, source_language, target_language))
            raise RuntimeError("local translation failed")

    ollama = FakeOllama([_decision("needs_review", "Unused response.")])
    result = run_agent_review(
        "The driver did not deliver 10 boxes.",
        "Awakọ fi apoti 10 ranṣẹ́.",
        service=BrokenTranslator(),
        ollama=ollama,
    )

    assert result.decision == "review_incomplete"
    assert "Back-translation failed" in result.reason
    assert result.tool_call_trace == (
        "back_translate started by Python",
        "back_translate failed",
    )
    assert ollama.calls == []


def test_unavailable_ollama_model_returns_incomplete_without_backtranslation() -> None:
    translator = FakeTranslator()
    result = run_agent_review(
        "Source sentence.",
        "Yoruba candidate.",
        service=translator,
        ollama=FakeOllama([], models=set()),
    )

    assert result.decision == "review_incomplete"
    assert "not installed locally" in result.reason
    assert result.english_back_translation is None
    assert translator.calls == []


def test_duplicate_action_reply_is_rejected_by_strict_decision_schema() -> None:
    duplicate_action = (
        '{"action":"final","decision":"needs_review","reason":"provisional",'
        '"action":"back_translate"}'
    )
    ollama = FakeOllama([{"message": {"role": "assistant", "content": duplicate_action}}])
    translator = FakeTranslator("The driver delivered 10 boxes.")

    result = run_agent_review(
        "The driver did not deliver 10 boxes.",
        "Awakọ fi apoti 10 ranṣẹ́.",
        service=translator,
        ollama=ollama,
    )

    assert result.decision == "review_incomplete"
    assert "duplicate JSON field" in result.reason
    assert result.english_back_translation == "The driver delivered 10 boxes."
    assert len(ollama.calls) == 1
    assert len(translator.calls) == 1


def test_invalid_decision_and_native_tool_call_are_rejected() -> None:
    invalid = run_agent_review(
        "Source.",
        "Candidate.",
        service=FakeTranslator(),
        ollama=FakeOllama([_decision("maybe", "Unclear.")]),
    )
    assert invalid.decision == "review_incomplete"
    assert "unsupported decision" in invalid.reason

    tool_call = run_agent_review(
        "Source.",
        "Candidate.",
        service=FakeTranslator(),
        ollama=FakeOllama(
            [{"message": {"role": "assistant", "content": "", "tool_calls": []}}]
        ),
    )
    assert tool_call.decision == "review_incomplete"
    assert "unexpected tool call" in tool_call.reason


def test_agent_generates_candidate_then_backtranslates_once() -> None:
    translator = FakeTranslator("I bought 3 books.")
    result = run_agent_review(
        "I bought 3 books.",
        service=translator,
        ollama=FakeOllama([_decision("no_issue_flagged", "Meaning matches.")]),
    )

    assert result.yoruba_candidate == "Ó ra ìwé 3."
    assert translator.calls == [
        ("I bought 3 books.", "eng", "yor"),
        ("Ó ra ìwé 3.", "yor", "eng"),
    ]


def test_cli_agent_mode_prints_review_and_queues_flagged_case(tmp_path, capsys) -> None:
    database = tmp_path / "reviews.sqlite3"
    ollama = FakeOllama(
        [
            _decision(
                "needs_review",
                "The source denies delivery, but the back-translation affirms delivery.",
            )
        ]
    )
    assert (
        main(
            [
                "The driver did not deliver 10 boxes.",
                "--candidate",
                "Awakọ fi apoti 10 ranṣẹ́.",
                "--agent",
                "--queue-db",
                str(database),
            ],
            service=FakeTranslator("The driver delivered 10 boxes."),
            ollama_client=ollama,
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Decision: needs_review" in output
    assert "English back-translation: The driver delivered 10 boxes." in output
    case_id = int(output.split("Queued case ID: ")[1].strip())
    case = ReviewQueue(database).get_case(case_id)
    assert case["agent_status"] == "needs_review"

    assert (
        review_queue_main(
            [
                "decide",
                str(case_id),
                "correct",
                "--corrected",
                "Awakọ kò fi apoti mẹ́wàá ranṣẹ́.",
                "--db",
                str(database),
            ]
        )
        == 0
    )
    assert ReviewQueue(database).get_case(case_id)["yoruba_candidate"] == "Awakọ fi apoti 10 ranṣẹ́."
    assert review_queue_main(["summary", "--db", str(database)]) == 0
    assert "Caught errors (flagged, corrected): 1" in capsys.readouterr().out


def test_audited_unflagged_case_is_queued_only_when_requested(tmp_path, capsys) -> None:
    database = tmp_path / "reviews.sqlite3"
    response = _decision("no_issue_flagged", "The source and back-translation agree.")
    args = ["Source.", "--candidate", "Candidate.", "--agent", "--queue-db", str(database)]
    assert (
        main(
            args,
            service=FakeTranslator("Source."),
            ollama_client=FakeOllama([response]),
        )
        == 0
    )
    assert not database.exists()

    assert (
        main(
            args + ["--audit-no-issue"],
            service=FakeTranslator("Source."),
            ollama_client=FakeOllama([response]),
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Queued audit case ID: 1" in output
    assert ReviewQueue(database).get_case(1)["audit_unflagged"] is True


def test_ollama_unavailable_result_remains_incomplete_and_queued(tmp_path, capsys) -> None:
    class UnavailableOllama(FakeOllama):
        def list_model_names(self) -> set[str]:
            raise OllamaError("Could not reach localhost Ollama.")

    database = tmp_path / "reviews.sqlite3"
    assert (
        main(
            [
                "Source.",
                "--candidate",
                "Candidate.",
                "--agent",
                "--queue-db",
                str(database),
            ],
            service=FakeTranslator(),
            ollama_client=UnavailableOllama([]),
        )
        == 2
    )
    output = capsys.readouterr().out
    assert "Decision: review_incomplete" in output
    assert "Queued case ID: 1" in output
    assert ReviewQueue(database).get_case(1)["agent_status"] == "review_incomplete"
