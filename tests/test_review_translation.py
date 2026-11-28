"""Tests for the local translation-review workflow; no model is loaded."""

from dataclasses import dataclass

import pytest

from app.review_translation import main, review_translation


@dataclass
class FakeResult:
    translated_text: str


class FakeTranslator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def translate(self, text: str, source_language: str, target_language: str) -> FakeResult:
        self.calls.append((text, source_language, target_language))
        translated = {
            ("eng", "yor"): "Ó san 3 àpùkọ́.",
            ("yor", "eng"): "He bought 3 books.",
        }[(source_language, target_language)]
        return FakeResult(translated)


def test_saved_candidate_skips_forward_translation_and_prints_review(capsys) -> None:
    translator = FakeTranslator()

    assert main(
        ["I bought 2 books.", "--candidate", "Ó ra ìwé 3."], service=translator
    ) == 0

    output = capsys.readouterr().out
    assert translator.calls == [("Ó ra ìwé 3.", "yor", "eng")]
    assert "English source: I bought 2 books." in output
    assert "Yoruba candidate: Ó ra ìwé 3." in output
    assert "Possible mismatch: source digit value(s) missing" in output
    assert "number may be spelled out in Yoruba" in output
    assert "English back-translation: He bought 3 books." in output


def test_generated_candidate_uses_both_translation_directions() -> None:
    translator = FakeTranslator()

    result = review_translation("I have 3 books.", service=translator)

    assert result.yoruba_candidate == "Ó san 3 àpùkọ́."
    assert result.english_back_translation == "He bought 3 books."
    assert result.numeric_findings == ("No digit discrepancy found.",)
    assert translator.calls == [
        ("I have 3 books.", "eng", "yor"),
        ("Ó san 3 àpùkọ́.", "yor", "eng"),
    ]


def test_grouped_numbers_are_compared_as_the_same_value() -> None:
    result = review_translation(
        "The population is 1,000.", "Olùgbé jẹ́ 1000.", service=FakeTranslator()
    )

    assert result.numeric_findings == ("No digit discrepancy found.",)


def test_empty_saved_candidate_is_rejected_without_translation() -> None:
    translator = FakeTranslator()

    with pytest.raises(ValueError, match="Yoruba candidate must not be empty"):
        review_translation("Source", "  ", service=translator)
    assert translator.calls == []


def test_queue_is_still_valid_english_source_for_baseline_cli(capsys) -> None:
    assert main(["queue"], service=FakeTranslator()) == 0

    output = capsys.readouterr().out
    assert "English source: queue" in output
    assert "Yoruba candidate: Ó san 3 àpùkọ́." in output
