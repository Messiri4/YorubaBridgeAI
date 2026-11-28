"""Batch input alignment, candidate labels, and persisted incomplete cases."""

import csv
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.review_batch import (
    FloresAlignment,
    ReviewCase,
    audit_flores_alignment,
    load_blind_cases,
    load_flores_cases,
    run_batch,
    select_verified_flores_ids,
)
from app.review_queue import ReviewQueue


@dataclass
class FakeTranslation:
    translated_text: str


class FakeTranslator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, dict[str, Any]]] = []

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        **options: Any,
    ) -> FakeTranslation:
        self.calls.append((text, source_language, target_language, options))
        return FakeTranslation("A back-translation.")


class FakeOllama:
    def __init__(self, responses: dict[str, Any] | list[dict[str, Any]]) -> None:
        self.responses = responses if isinstance(responses, list) else [responses]
        self.messages: list[list[dict[str, Any]]] = []

    def list_model_names(self) -> set[str]:
        return {"gemma3:4b"}

    def chat(self, model: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        self.messages.append(messages)
        return self.responses.pop(0)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_flores_loader_uses_verified_ids_and_fresh_raw_generation(tmp_path) -> None:
    source_path = tmp_path / "devtest.eng_Latn"
    reference_path = tmp_path / "devtest.yor_Latn"
    official_source_path = tmp_path / "eng_Latn.devtest"
    official_reference_path = tmp_path / "yor_Latn.devtest"
    predictions_path = tmp_path / "predictions.csv"
    metadata_path = tmp_path / "metadata.json"
    provenance_path = tmp_path / "README.md"
    sources = [f"Official English {i}." for i in range(1, 1013)]
    references = [f"Yorùbá {i}." for i in range(1, 1013)]
    sources[1] = "English two."
    references[1] = "Yorùbá two."
    sources[2] = "English three."
    references[2] = "Yorùbá three."
    official_source_path.write_text("\n".join(sources) + "\n", encoding="utf-8")
    official_reference_path.write_text("\n".join(references) + "\n", encoding="utf-8")
    source_path.write_text("\n".join(sources[:3]) + "\n", encoding="utf-8")
    reference_path.write_text(
        "\n".join([references[0], unicodedata.normalize("NFD", references[1]), references[2]])
        + "\n",
        encoding="utf-8",
    )
    provenance_path.write_text("official local subset provenance", encoding="utf-8")
    _write_csv(
        predictions_path,
        ["flores_id", "english_reference", "yoruba_reference", "nllb_en_to_yo"],
        [
            {
                "flores_id": f"devtest-{i:04d}",
                "english_reference": sources[i - 1],
                "yoruba_reference": references[i - 1],
                "nllb_en_to_yo": f"Model Yoruba {i}.",
            }
            for i in range(1, 1013)
        ],
    )
    metadata = {
        "split": "devtest",
        "expected_pairs": 1012,
        "model": "facebook/nllb-200-distilled-600M",
        "status": "partial",
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    translator = FakeTranslator()

    alignment = audit_flores_alignment(
        source_path=source_path,
        reference_path=reference_path,
        official_source_path=official_source_path,
        official_reference_path=official_reference_path,
        predictions_path=predictions_path,
        metadata_path=metadata_path,
        provenance_path=provenance_path,
    )
    cases, details = load_flores_cases(
        alignment=alignment,
        row_ids=["devtest-0002", "devtest-0003"],
        translator=translator,
        generation_settings={
            "num_beams": 4,
            "max_new_tokens": 256,
            "length_penalty": 1.1,
        },
    )

    assert alignment.report["counts"]["local_pairs"] == 3
    assert alignment.report["counts"]["local_yoruba_mismatches_csv_exact"] == 1
    assert (
        alignment.report["counts"]["local_yoruba_mismatches_csv_resolved_by_nfc_whitespace"]
        == 1
    )
    assert alignment.report["unresolved_rows"] == []
    assert alignment.report["files"]["local_english"]["path"] == source_path.name
    assert not Path(alignment.report["files"]["local_english"]["path"]).is_absolute()
    assert details["row_ids"] == ["devtest-0002", "devtest-0003"]
    assert [(case.row_id, case.candidate_label) for case in cases] == [
        ("devtest-0002", "flores_reference"),
        ("devtest-0002", "nllb_generated"),
        ("devtest-0003", "flores_reference"),
        ("devtest-0003", "nllb_generated"),
    ]
    assert cases[0].source == "English two."
    assert cases[0].candidate == "Yorùbá two."
    assert cases[1].candidate == "A back-translation."
    assert cases[1].candidate_provenance["kind"] == "fresh_local_raw_nllb"
    assert [call[3] for call in translator.calls] == [
        {
            "num_beams": 4,
            "max_new_tokens": 256,
            "length_penalty": 1.1,
            "use_phrase_overrides": False,
        },
        {
            "num_beams": 4,
            "max_new_tokens": 256,
            "length_penalty": 1.1,
            "use_phrase_overrides": False,
        },
    ]


def test_unresolved_flores_ids_are_replaced_only_with_verified_rows() -> None:
    alignment = FloresAlignment(
        report={},
        verified_rows=(
            {"id": "devtest-0021", "source": "Source 21.", "reference": "Yorùbá 21."},
            {"id": "devtest-0024", "source": "Source 24.", "reference": "Yorùbá 24."},
        ),
    )

    selected, unresolved, replacements = select_verified_flores_ids(
        alignment, ["devtest-0021", "devtest-0023"]
    )

    assert selected == ["devtest-0021", "devtest-0024"]
    assert unresolved == ["devtest-0023"]
    assert replacements == ["devtest-0024"]


def test_blind_loader_labels_sides_without_loading_answer_key(tmp_path) -> None:
    blind_csv = tmp_path / "blind.csv"
    _write_csv(
        blind_csv,
        ["ID", "direction", "source", "Translation A", "Translation B"],
        [
            {
                "ID": f"BR-{i:03d}",
                "direction": "ENG->YO",
                "source": f"Source {i}.",
                "Translation A": f"Candidate A {i}.",
                "Translation B": f"Candidate B {i}.",
            }
            for i in range(1, 4)
        ],
    )

    cases, details = load_blind_cases(blind_csv, sample_size=2, seed=5)

    assert details["row_ids"] == ["BR-002", "BR-003"]
    assert [(case.row_id, case.candidate_label) for case in cases] == [
        ("BR-002", "A"),
        ("BR-002", "B"),
        ("BR-003", "A"),
        ("BR-003", "B"),
    ]
    assert [case.candidate for case in cases[:2]] == ["Candidate A 2.", "Candidate B 2."]
    assert details["private_answer_key_in_prompt"] is False
    assert details["source_file"] == blind_csv.name
    assert not Path(details["source_file"]).is_absolute()


def test_incomplete_candidate_is_saved_to_jsonl_and_sqlite_queue(tmp_path) -> None:
    output_dir = tmp_path / "fresh-output"
    case = ReviewCase(
        dataset="blind_review",
        row_id="BR-001",
        candidate_label="A",
        source="The source.",
        candidate="The candidate.",
    )
    ollama = FakeOllama({"message": {"role": "assistant", "content": "not json"}})

    records, summary = run_batch(
        [case],
        service=FakeTranslator(),
        ollama=ollama,
        model="gemma3:4b",
        output_dir=output_dir,
        audit_no_issue=True,
        run_metadata={"private_answer_key_in_prompt": False},
    )

    assert summary["attempted"] == 1
    assert summary["completed"] == 0
    assert summary["incomplete"] == 1
    assert records[0]["candidate_label"] == "A"
    assert records[0]["decision"] == "review_incomplete"
    assert records[0]["queue_id"] == 1
    saved = json.loads((output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert saved["decision"] == "review_incomplete"
    queued = ReviewQueue(output_dir / "reviews.sqlite3").get_case(1)
    assert queued["agent_status"] == "review_incomplete"
    assert queued["yoruba_candidate"] == "The candidate."


def test_all_result_labels_are_outside_model_payload(tmp_path) -> None:
    ollama = FakeOllama(
        {
            "message": {
                "role": "assistant",
                "content": json.dumps(
                    {"decision": "needs_review", "reason": "The evidence differs."}
                ),
            }
        }
    )
    case = ReviewCase(
        dataset="blind_review",
        row_id="BR-010",
        candidate_label="B",
        source="Source text.",
        candidate="Candidate text.",
    )

    run_batch(
        [case],
        service=FakeTranslator(),
        ollama=ollama,
        model="gemma3:4b",
        output_dir=tmp_path / "label-output",
        audit_no_issue=False,
        run_metadata={},
    )

    model_payload = json.dumps(ollama.messages, ensure_ascii=False)
    assert "BR-010" not in model_payload
    assert '"candidate_label"' not in model_payload
    assert "Candidate text." not in model_payload


def test_private_key_produces_aggregate_only_after_results_are_saved(tmp_path) -> None:
    answer_key = tmp_path / "answer-key.json"
    answer_key.write_text(
        json.dumps({"rows": [{"ID": "BR-001", "candidate_side": "B"}]}),
        encoding="utf-8",
    )
    decision = {
        "message": {
            "role": "assistant",
            "content": json.dumps({"decision": "no_issue_flagged", "reason": "Meaning matches."}),
        }
    }
    output_dir = tmp_path / "aggregate-output"
    cases = [
        ReviewCase("blind_review", "BR-001", "A", "Source.", "Candidate A."),
        ReviewCase("blind_review", "BR-001", "B", "Source.", "Candidate B."),
    ]

    records, summary = run_batch(
        cases,
        service=FakeTranslator(),
        ollama=FakeOllama([decision, decision]),
        model="gemma3:4b",
        output_dir=output_dir,
        audit_no_issue=True,
        run_metadata={},
        blind_answer_key=answer_key,
    )

    assert (output_dir / "results.jsonl").is_file()
    assert [record["candidate_label"] for record in records] == ["A", "B"]
    assert summary["blind_protection_aggregate"]["protection_on"]["no_issue_flagged"] == 1
    assert summary["blind_protection_aggregate"]["protection_off"]["no_issue_flagged"] == 1
    saved_results = (output_dir / "results.jsonl").read_text(encoding="utf-8")
    assert "protection_on" not in saved_results
    assert "candidate_side" not in saved_results
