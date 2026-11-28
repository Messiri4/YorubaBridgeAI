"""Tests for paired phrase-override evaluation and blind output handling."""

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from app.phrase_override_eval import (
    PairedInput,
    _input_cases,
    create_blind_packet,
    review_paired_outputs,
    run_paired_translations,
    score_reference_backed_subset,
)
from app.review_agent import OllamaClientProtocol
from app.services.translation import PhraseOverrides, TranslationResult

_OVERRIDES = {
    "good morning": "Ẹ kú àárọ̀.",
    "how are you": "Báwo ni?",
    "thank you very much": "O ṣé púpọ̀.",
    "where is the market": "Nibo ni ọjà wà?",
    "how much is this": "Eèló ni èyí?",
}


class FakeOverrides:
    def lookup(self, text: str, source_language: str, target_language: str) -> str | None:
        if source_language != "eng" or target_language != "yor":
            return None
        return _OVERRIDES.get(PhraseOverrides._normalize(text))


class FakeTranslationService:
    def __init__(self) -> None:
        self.overrides = FakeOverrides()
        self.calls: list[tuple[str, str, str, dict[str, Any]]] = []

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        *,
        num_beams: int | None = None,
        max_new_tokens: int | None = None,
        length_penalty: float | None = None,
        use_phrase_overrides: bool = True,
    ) -> TranslationResult:
        options = {
            "num_beams": num_beams,
            "max_new_tokens": max_new_tokens,
            "length_penalty": length_penalty,
            "use_phrase_overrides": use_phrase_overrides,
        }
        self.calls.append((text, source_language, target_language, options))
        if target_language == "eng":
            return TranslationResult(
                f"Back translation of {text}",
                source_language,
                target_language,
                1.0,
            )
        key = PhraseOverrides._normalize(text)
        override = _OVERRIDES.get(key) if use_phrase_overrides else None
        if text == "Control identical.":
            output = "Same candidate."
        else:
            output = override or f"NLLB output for {text}"
        return TranslationResult(
            output,
            source_language,
            target_language,
            1.0,
            override_key=key if override is not None else None,
        )


class FakeOllama:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.messages: list[list[dict[str, Any]]] = []

    def list_model_names(self) -> set[str]:
        return {"gemma3:4b"}

    def chat(self, model: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        self.messages.append(messages)
        return self.responses.pop(0)


def _record_for(source: str) -> PairedInput:
    return PairedInput("test-case", "control", source, False, "No full key expected.")


def test_paired_runs_only_toggle_overrides_and_records_actual_match() -> None:
    service = FakeTranslationService()
    inputs = [
        PairedInput("exact", "exact", "Good morning.", True, "Exact table key."),
        PairedInput("variation", "variation", "I said good morning today.", False, "Longer text."),
        PairedInput("control", "control", "The bus arrives at 8:30.", False, "Unrelated control."),
    ]

    records = run_paired_translations(service, inputs)

    assert len(service.calls) == 6
    for off_call, on_call in zip(service.calls[::2], service.calls[1::2], strict=True):
        assert off_call[:3] == on_call[:3]
        assert off_call[3] == {
            "num_beams": 4,
            "max_new_tokens": 256,
            "length_penalty": 1.1,
            "use_phrase_overrides": False,
        }
        assert on_call[3] == {
            **off_call[3],
            "use_phrase_overrides": True,
        }

    exact, variation, control = records
    assert exact["on"]["override_fired"] is True
    assert exact["on"]["matched_entry"] == "good morning"
    assert exact["outputs_changed"] is True
    assert variation["on"]["override_fired"] is False
    assert variation["on"]["translation_path"] == "nllb"
    assert control["on"]["override_fired"] is False
    assert control["on"]["translation_path"] == "nllb"
    assert all(not row["unexpected_match"] for row in records)


def test_blind_packet_randomizes_outputs_and_marks_identical_as_tie(tmp_path: Path) -> None:
    service = FakeTranslationService()
    records = run_paired_translations(
        service,
        [
            PairedInput("phrase", "exact", "Good morning.", True, "Exact key."),
            _record_for("Control identical."),
        ],
    )

    packet_metadata = create_blind_packet(records, tmp_path, seed=441)
    with (tmp_path / "human_review_blind.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        packet = list(csv.DictReader(handle))
    key = json.loads((tmp_path / "answer_key_private.json").read_text(encoding="utf-8"))
    public_text = (tmp_path / "human_review_blind.csv").read_text(encoding="utf-8")

    identical = next(row for row in packet if row["English source"] == "Control identical.")
    assert identical["identical_outputs"] == "yes — count as tie"
    assert identical["Reviewer choice (A/B/tie)"] == "Tie (identical outputs)"
    assert "condition_by_side" not in public_text
    assert packet_metadata["rows"] == 2
    assert packet_metadata["identical_ties"] == 1
    assert packet_metadata["packet_path"] == "human_review_blind.csv"
    assert packet_metadata["private_key_path"] == "answer_key_private.json"
    assert not Path(packet_metadata["packet_path"]).is_absolute()
    assert not Path(packet_metadata["private_key_path"]).is_absolute()
    assert len(key["assignment"]) == 2
    assert all("condition_by_side" in row for row in key["assignment"])


def test_reference_scores_use_existing_verified_references_and_sacrebleu_metrics() -> None:
    service = FakeTranslationService()
    records = run_paired_translations(service)
    results = score_reference_backed_subset(
        records, Path("data/eval/en_yor_eval.json")
    )

    assert results["reference_count"] == 5
    assert results["reference_ids"] == [
        "greet-001",
        "greet-002",
        "expr-001",
        "q-001",
        "q-002",
    ]
    assert set(results["scores"]["off"]) >= {"bleu", "chrf++", "ter"}
    assert results["scores"]["on"]["bleu"] == pytest.approx(100.0)
    assert results["scores"]["on"]["chrf++"] == pytest.approx(100.0)
    assert results["scores"]["on"]["ter"] == pytest.approx(0.0)
    assert results["meteor"]["reported"] is False


def test_gemma_review_records_incomplete_separately_and_hides_condition() -> None:
    service = FakeTranslationService()
    records = run_paired_translations(service, [_input_cases()[0]])
    responses = [
        {"message": {"role": "assistant", "content": "not json"}},
        {
            "message": {
                "role": "assistant",
                "content": json.dumps(
                    {"decision": "needs_review", "reason": "The evidence may differ."}
                ),
            }
        },
    ]
    ollama: OllamaClientProtocol = FakeOllama(responses)

    reviews = review_paired_outputs(
        records,
        service=service,
        ollama=ollama,
        model="gemma3:4b",
    )

    assert [row["decision"] for row in reviews] == ["review_incomplete", "needs_review"]
    assert all("condition" not in json.dumps(messages) for messages in ollama.messages)
    assert all(row["english_back_translation"] for row in reviews)
