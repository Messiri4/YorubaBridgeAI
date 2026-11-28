"""Reproducible local batch evaluation for the translation-review agent."""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import os
import random
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from app.review_agent import AgentReviewResult, OllamaClient, OllamaClientProtocol, run_agent_review
from app.review_queue import ReviewQueue
from app.review_translation import TranslationClient, TranslationOutput

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLORES_SOURCE = ROOT / "data" / "flores" / "devtest.eng_Latn"
DEFAULT_FLORES_REFERENCE = ROOT / "data" / "flores" / "devtest.yor_Latn"
DEFAULT_FLORES_PREDICTIONS = (
    ROOT / "data" / "eval" / "flores_devtest_dataset"
    / "flores_devtest_1012_predictions.csv"
)
DEFAULT_FLORES_METADATA = (
    ROOT / "data" / "eval" / "flores_devtest_dataset"
    / "flores_devtest_dataset_metadata.json"
)
DEFAULT_FLORES_OFFICIAL_DIR = ROOT / "data" / "flores" / "official"
DEFAULT_FLORES_OFFICIAL_SOURCE = DEFAULT_FLORES_OFFICIAL_DIR / "eng_Latn.devtest"
DEFAULT_FLORES_OFFICIAL_REFERENCE = DEFAULT_FLORES_OFFICIAL_DIR / "yor_Latn.devtest"
DEFAULT_BLIND_CSV = ROOT / "data" / "eval" / "blind_review" / "blind_review_30.csv"
DEFAULT_FLORES_ROW_IDS = (
    "devtest-0021",
    "devtest-0023",
    "devtest-0064",
    "devtest-0072",
    "devtest-0074",
)
_RAW_GENERATION_OPTIONS = {
    "num_beams": 4,
    "max_new_tokens": 256,
    "length_penalty": 1.1,
    "use_phrase_overrides": False,
}


class RawNllbTranslator(Protocol):
    """Translation interface for reproducible, unprotected NLLB generation."""

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        *,
        num_beams: int,
        max_new_tokens: int,
        length_penalty: float,
        use_phrase_overrides: bool,
    ) -> TranslationOutput: ...


@dataclass(frozen=True)
class ReviewCase:
    """One unlabeled candidate review request."""

    dataset: str
    row_id: str
    candidate_label: str
    source: str
    candidate: str
    reference: str | None = None
    candidate_provenance: dict[str, Any] | None = None


@dataclass(frozen=True)
class FloresAlignment:
    """Audited local subset joined to official FLORES IDs and references."""

    report: dict[str, Any]
    verified_rows: tuple[dict[str, str], ...]


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _sample_rows(rows: list[dict[str, str]], sample_size: int, seed: int) -> list[dict[str, str]]:
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")
    if sample_size > len(rows):
        raise ValueError(f"Requested {sample_size} rows but only {len(rows)} are available.")
    return sorted(random.Random(seed).sample(rows, sample_size), key=lambda row: row["id"])


def _normalize_reference(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _raw_text_diff(left: str, right: str) -> list[dict[str, str]]:
    matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)
    return [
        {
            "local_text": left[a_start:a_end],
            "comparison_text": right[b_start:b_end],
        }
        for operation, a_start, a_end, b_start, b_end in matcher.get_opcodes()
        if operation != "equal"
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_flores_alignment(
    *,
    source_path: Path,
    reference_path: Path,
    official_source_path: Path,
    official_reference_path: Path,
    predictions_path: Path,
    metadata_path: Path,
    provenance_path: Path,
) -> FloresAlignment:
    """Join local references to official FLORES IDs and audit all text variants."""
    local_sources = _read_lines(source_path)
    local_references = _read_lines(reference_path)
    official_sources = _read_lines(official_source_path)
    official_references = _read_lines(official_reference_path)
    if len(local_sources) != len(local_references):
        raise ValueError(
            f"Local FLORES line counts differ: {len(local_sources)} vs {len(local_references)}."
        )
    if len(official_sources) != len(official_references):
        raise ValueError(
            "Official FLORES English/Yorùbá line counts differ: "
            f"{len(official_sources)} vs {len(official_references)}."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("split") != "devtest":
        raise ValueError("Prediction metadata does not identify the FLORES devtest split.")
    with predictions_path.open(encoding="utf-8-sig", newline="") as handle:
        prediction_rows = list(csv.DictReader(handle))
    predictions: dict[str, dict[str, str]] = {}
    for prediction_row in prediction_rows:
        row_id = prediction_row.get("flores_id", "")
        if not row_id or row_id in predictions:
            raise ValueError(
                f"Prediction CSV contains a missing or duplicate FLORES ID: {row_id!r}."
            )
        predictions[row_id] = prediction_row

    expected_count = metadata.get("expected_pairs")
    if (
        expected_count != len(official_sources)
        or expected_count != len(predictions)
        or len(official_sources) != 1012
    ):
        raise ValueError(
            "Official files, prediction CSV, and devtest metadata do not agree on 1,012 rows."
        )

    official_id_by_english: dict[str, list[str]] = {}
    official_row_by_id: dict[str, dict[str, str]] = {}
    for index, (english, yoruba) in enumerate(
        zip(official_sources, official_references, strict=True), start=1
    ):
        row_id = f"devtest-{index:04d}"
        official_id_by_english.setdefault(english, []).append(row_id)
        official_row_by_id[row_id] = {"source": english, "reference": yoruba}

    csv_source_mismatches = []
    csv_reference_mismatches = []
    for row_id, official_row in official_row_by_id.items():
        prediction = predictions.get(row_id)
        if prediction is None:
            csv_source_mismatches.append({"id": row_id, "reason": "ID missing from CSV"})
            continue
        if prediction.get("english_reference") != official_row["source"]:
            csv_source_mismatches.append(
                {"id": row_id, "csv": prediction.get("english_reference", ""),
                 "official": official_row["source"]}
            )
        if prediction.get("yoruba_reference") != official_row["reference"]:
            csv_reference_mismatches.append(
                {
                    "id": row_id,
                    "exact_match": False,
                    "nfc_whitespace_match": _normalize_reference(
                        prediction.get("yoruba_reference", "")
                    ) == _normalize_reference(official_row["reference"]),
                    "csv_reference": prediction.get("yoruba_reference", ""),
                    "official_reference": official_row["reference"],
                    "remaining_text_differences": _raw_text_diff(
                        prediction.get("yoruba_reference", ""), official_row["reference"]
                    )
                    if _normalize_reference(prediction.get("yoruba_reference", ""))
                    != _normalize_reference(official_row["reference"])
                    else [],
                }
            )

    verified_rows: list[dict[str, str]] = []
    local_csv_mismatches = []
    local_official_mismatches = []
    unresolved_rows = []
    local_rows = []
    local_source_exact_id_matches = 0
    for line_number, (source, local_reference) in enumerate(
        zip(local_sources, local_references, strict=True), start=1
    ):
        ids = official_id_by_english.get(source, [])
        row_id = ids[0] if len(ids) == 1 else None
        row: dict[str, Any] = {
            "local_line_number": line_number,
            "id": row_id,
            "source_exactly_matches_official": row_id is not None,
            "id_match_count": len(ids),
        }
        if row_id is None:
            unresolved_rows.append(
                {
                    "local_line_number": line_number,
                    "reason": "English text has no unique official ID",
                }
            )
            local_rows.append(row)
            continue

        official = official_row_by_id[row_id]
        prediction = predictions.get(row_id)
        if prediction is None or prediction.get("english_reference") != source:
            unresolved_rows.append(
                {
                    "local_line_number": line_number,
                    "id": row_id,
                    "reason": "Prediction CSV source does not verify the official ID",
                }
            )
            local_rows.append(row)
            continue
        local_source_exact_id_matches += 1

        local_csv_exact = local_reference == prediction.get("yoruba_reference", "")
        local_csv_normalized = _normalize_reference(local_reference) == _normalize_reference(
            prediction.get("yoruba_reference", "")
        )
        local_official_exact = local_reference == official["reference"]
        local_official_normalized = _normalize_reference(local_reference) == _normalize_reference(
            official["reference"]
        )
        row.update(
            {
                "local_reference_exactly_matches_csv": local_csv_exact,
                "local_reference_matches_csv_after_nfc_whitespace": local_csv_normalized,
                "local_reference_exactly_matches_official": local_official_exact,
                "local_reference_matches_official_after_nfc_whitespace": local_official_normalized,
            }
        )
        if not local_csv_exact:
            local_csv_mismatches.append(
                {
                    "id": row_id,
                    "disappears_after_nfc_whitespace_normalization": local_csv_normalized,
                    "remaining_text_differences": _raw_text_diff(
                        local_reference, prediction.get("yoruba_reference", "")
                    )
                    if not local_csv_normalized
                    else [],
                    "local_reference": local_reference,
                    "csv_reference": prediction.get("yoruba_reference", ""),
                }
            )
        if not local_official_exact:
            local_official_mismatches.append(
                {
                    "id": row_id,
                    "disappears_after_nfc_whitespace_normalization": local_official_normalized,
                    "remaining_text_differences": _raw_text_diff(
                        local_reference, official["reference"]
                    )
                    if not local_official_normalized
                    else [],
                    "local_reference": local_reference,
                    "official_reference": official["reference"],
                }
            )
        if not local_official_normalized:
            unresolved_rows.append(
                {
                    "local_line_number": line_number,
                    "id": row_id,
                    "reason": "Yorùbá reference differs from official text after normalization",
                }
            )
        else:
            verified_rows.append(
                {"id": row_id, "source": official["source"], "reference": official["reference"]}
            )
        local_rows.append(row)

    canonical_file_info = {}
    for label, path in (
        ("local_english", source_path),
        ("local_yoruba", reference_path),
        ("official_english", official_source_path),
        ("official_yoruba", official_reference_path),
        ("prediction_csv", predictions_path),
        ("prediction_metadata", metadata_path),
        ("provenance_document", provenance_path),
    ):
        canonical_file_info[label] = {
            "path": path.name,
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }

    report = {
        "dataset": "FLORES-200",
        "split": "devtest",
        "official_line_to_id_rule": (
            "The original official files are ordered aligned sentence lists. "
            "scripts/export_flores_devtest.py enumerates their lines from 1 and "
            "constructs devtest-{line:04d}; local English text was matched exactly "
            "and uniquely to official English text before assigning an ID."
        ),
        "authority": (
            "Official Meta FLORES-200 files in the local archive are authoritative; "
            "the project README identifies the checked-in pair files as the first 80 "
            "lines of those files. The prediction CSV is a comparison source, not "
            "treated as authoritative."
        ),
        "metadata": {
            "expected_pairs": expected_count,
            "csv_rows": len(predictions),
            "official_english_rows": len(official_sources),
            "official_yoruba_rows": len(official_references),
            "model": metadata.get("model"),
            "split": metadata.get("split"),
            "status": metadata.get("status"),
        },
        "counts": {
            "local_pairs": len(local_sources),
            "official_pairs": len(official_sources),
            "local_english_exactly_joined_to_official_and_csv": local_source_exact_id_matches,
            "local_yoruba_exact_matches_csv": len(local_sources) - len(local_csv_mismatches),
            "local_yoruba_mismatches_csv_exact": len(local_csv_mismatches),
            "local_yoruba_mismatches_csv_resolved_by_nfc_whitespace": sum(
                1
                for entry in local_csv_mismatches
                if entry["disappears_after_nfc_whitespace_normalization"] is True
            ),
            "local_yoruba_mismatches_official_exact": len(local_sources)
            - len(local_official_mismatches),
            "local_yoruba_mismatches_official_resolved_by_nfc_whitespace": sum(
                1
                for entry in local_official_mismatches
                if entry["disappears_after_nfc_whitespace_normalization"] is True
            ),
            "csv_yoruba_mismatches_official_exact": len(csv_reference_mismatches),
            "csv_yoruba_mismatches_official_resolved_by_nfc_whitespace": sum(
                1
                for entry in csv_reference_mismatches
                if entry["nfc_whitespace_match"] is True
            ),
            "csv_source_mismatches_official_exact": len(csv_source_mismatches),
            "verified_local_rows": len(verified_rows),
            "unresolved_local_rows": len(unresolved_rows),
        },
        "files": canonical_file_info,
        "local_rows": local_rows,
        "local_vs_csv_yoruba_mismatches": local_csv_mismatches,
        "local_vs_official_yoruba_mismatches": local_official_mismatches,
        "csv_vs_official_yoruba_mismatches": csv_reference_mismatches,
        "csv_vs_official_english_mismatches": csv_source_mismatches,
        "remaining_text_differences_after_nfc_whitespace": [
            {
                "id": item["id"],
                "side": side,
                "differences": item["remaining_text_differences"],
            }
            for side, entries in (
                ("local_vs_csv", local_csv_mismatches),
                ("local_vs_official", local_official_mismatches),
                ("csv_vs_official", csv_reference_mismatches),
            )
            for item in entries
            if (
                item.get("disappears_after_nfc_whitespace_normalization") is False
                or item.get("nfc_whitespace_match") is False
            )
        ],
        "unresolved_rows": unresolved_rows,
        "verified_rows": verified_rows,
    }
    return FloresAlignment(report=report, verified_rows=tuple(verified_rows))


def load_flores_cases(
    *,
    alignment: FloresAlignment,
    row_ids: list[str],
    translator: RawNllbTranslator,
    generation_settings: dict[str, Any],
) -> tuple[list[ReviewCase], dict[str, Any]]:
    """Generate fresh raw NLLB candidates for specifically verified FLORES IDs."""
    verified_by_id = {row["id"]: row for row in alignment.verified_rows}
    missing_ids = [row_id for row_id in row_ids if row_id not in verified_by_id]
    if missing_ids:
        raise ValueError(f"Cannot run unresolved FLORES IDs: {', '.join(missing_ids)}.")

    cases: list[ReviewCase] = []
    for row_id in row_ids:
        row = verified_by_id[row_id]
        hypothesis = translator.translate(
            row["source"],
            "eng",
            "yor",
            num_beams=generation_settings["num_beams"],
            max_new_tokens=generation_settings["max_new_tokens"],
            length_penalty=generation_settings["length_penalty"],
            use_phrase_overrides=False,
        ).translated_text
        provenance = {
            "kind": "fresh_local_raw_nllb",
            "source_row_verified": True,
            "generation_settings": generation_settings,
            "inference_call_verified": True,
        }
        cases.extend(
            (
                ReviewCase(
                    dataset="flores_devtest",
                    row_id=row_id,
                    candidate_label="flores_reference",
                    source=row["source"],
                    candidate=row["reference"],
                    reference=row["reference"],
                    candidate_provenance={
                        "kind": "official_flores_reference",
                        "source_row_verified": True,
                    },
                ),
                ReviewCase(
                    dataset="flores_devtest",
                    row_id=row_id,
                    candidate_label="nllb_generated",
                    source=row["source"],
                    candidate=hypothesis,
                    reference=row["reference"],
                    candidate_provenance=provenance,
                ),
            )
        )

    return cases, {
        "dataset": "FLORES-200",
        "split": "devtest",
        "row_ids": list(row_ids),
        "candidate_groups": ["flores_reference", "nllb_generated"],
        "verified_row_ids": list(row_ids),
        "generated_nllb_row_ids": list(row_ids),
        "reused_nllb_row_ids": [],
        "candidate_generation": generation_settings,
        "alignment_audit_counts": alignment.report["counts"],
    }


def select_verified_flores_ids(
    alignment: FloresAlignment, requested_ids: list[str]
) -> tuple[list[str], list[str], list[str]]:
    """Keep requested verified IDs and replace unresolved ones within devtest."""
    verified_ids = {row["id"] for row in alignment.verified_rows}
    unique_requested = list(dict.fromkeys(requested_ids))
    selected = [row_id for row_id in unique_requested if row_id in verified_ids]
    unresolved = [row_id for row_id in unique_requested if row_id not in verified_ids]
    replacements = [
        row["id"] for row in alignment.verified_rows if row["id"] not in unique_requested
    ][: len(unresolved)]
    return selected + replacements, unresolved, replacements


def load_blind_cases(
    path: Path, sample_size: int, seed: int
) -> tuple[list[ReviewCase], dict[str, Any]]:
    """Read public blind packet fields only; never load the private answer key here."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("direction") == "ENG->YO"
            and row.get("ID")
            and row.get("source")
            and row.get("Translation A") is not None
            and row.get("Translation B") is not None
        ]
    selected = _sample_rows(
        [
            {
                "id": row["ID"],
                "source": row["source"],
                "A": row["Translation A"],
                "B": row["Translation B"],
            }
            for row in rows
        ],
        sample_size,
        seed,
    )
    cases = [
        ReviewCase(
            dataset="blind_review",
            row_id=row["id"],
            candidate_label=side,
            source=row["source"],
            candidate=row[side],
        )
        for row in selected
        for side in ("A", "B")
    ]
    return cases, {
        "dataset": "blind_review_30",
        "split": "blind_review",
        "sample_seed": seed,
        "row_ids": [row["id"] for row in selected],
        "candidate_groups": ["A", "B"],
        "source_file": path.name,
        "source_file_sha256": _sha256(path),
        "private_answer_key_in_prompt": False,
    }


def _result_record(
    case: ReviewCase, result: AgentReviewResult, queue_id: int | None
) -> dict[str, Any]:
    return {
        "dataset": case.dataset,
        "row_id": case.row_id,
        "candidate_label": case.candidate_label,
        "source": case.source,
        "reference": case.reference,
        "candidate_text": case.candidate,
        "candidate_provenance": case.candidate_provenance,
        "digit_findings": list(result.numeric_findings),
        "english_back_translation": result.english_back_translation,
        "decision": result.decision,
        "reason": result.reason,
        "tool_trace": list(result.tool_call_trace),
        "review_latency_ms": result.review_latency_ms,
        "queue_id": queue_id,
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, int]] = {}
    for record in records:
        group_key = f"{record['dataset']}:{record['candidate_label']}"
        group = groups.setdefault(
            group_key,
            {
                "attempted": 0,
                "completed": 0,
                "incomplete": 0,
                "needs_review": 0,
                "no_issue_flagged": 0,
            },
        )
        group["attempted"] += 1
        decision = record["decision"]
        if decision == "review_incomplete":
            group["incomplete"] += 1
        else:
            group["completed"] += 1
            group[decision] += 1
    return {
        "attempted": len(records),
        "completed": sum(record["decision"] != "review_incomplete" for record in records),
        "incomplete": sum(record["decision"] == "review_incomplete" for record in records),
        "by_candidate_group": groups,
    }


def _protection_aggregate(
    records: list[dict[str, Any]], answer_key_path: Path
) -> dict[str, dict[str, int]]:
    """Use the private mapping only to derive aggregate counts after results persist."""
    key = json.loads(answer_key_path.read_text(encoding="utf-8"))
    mapping = {
        row["ID"]: row["candidate_side"]
        for row in key["rows"]
        if row.get("candidate_side") in {"A", "B"}
    }
    aggregate = {
        mode: {
            "attempted": 0,
            "completed": 0,
            "incomplete": 0,
            "needs_review": 0,
            "no_issue_flagged": 0,
        }
        for mode in ("protection_on", "protection_off")
    }
    for record in records:
        if record["dataset"] != "blind_review":
            continue
        protected_side = mapping.get(record["row_id"])
        if protected_side is None:
            continue
        mode = "protection_on" if record["candidate_label"] == protected_side else "protection_off"
        bucket = aggregate[mode]
        bucket["attempted"] += 1
        decision = record["decision"]
        if decision == "review_incomplete":
            bucket["incomplete"] += 1
        else:
            bucket["completed"] += 1
            bucket[decision] += 1
    return aggregate


def run_batch(
    cases: list[ReviewCase],
    *,
    service: TranslationClient,
    ollama: OllamaClientProtocol,
    model: str,
    output_dir: Path,
    audit_no_issue: bool,
    run_metadata: dict[str, Any],
    blind_answer_key: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run and persist every review, its queue row, and aggregate summaries."""
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory must be fresh: {output_dir}")
    output_dir.mkdir(parents=True)
    result_path = output_dir / "results.jsonl"
    database_path = output_dir / "reviews.sqlite3"
    queue = ReviewQueue(database_path)
    records: list[dict[str, Any]] = []

    with result_path.open("w", encoding="utf-8", newline="\n") as result_file:
        for case in cases:
            result = run_agent_review(
                case.source,
                case.candidate,
                service=service,
                ollama=ollama,
                model=model,
            )
            queue_id = None
            if result.decision in {"needs_review", "review_incomplete"} or (
                result.decision == "no_issue_flagged" and audit_no_issue
            ):
                queue_id = queue.enqueue(
                    english_source=result.english_source,
                    yoruba_candidate=result.yoruba_candidate,
                    digit_findings=result.numeric_findings,
                    agent_status=result.decision,
                    agent_reason=result.reason,
                    back_translation=result.english_back_translation,
                    tool_trace=result.tool_call_trace,
                    audit_unflagged=result.decision == "no_issue_flagged" and audit_no_issue,
                )
            record = _result_record(case, result, queue_id)
            records.append(record)
            result_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            result_file.flush()

    summary = _summary(records)
    if blind_answer_key is not None:
        if not blind_answer_key.is_file():
            raise FileNotFoundError(f"Private blind answer key not found: {blind_answer_key}")
        summary["blind_protection_aggregate"] = _protection_aggregate(records, blind_answer_key)
        summary["blind_protection_aggregate_note"] = (
            "Computed from private key after all result rows were saved; assignments omitted."
        )

    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return records, summary


def _local_output_dir(value: str | None) -> Path:
    local_root = (ROOT / ".local").resolve()
    requested = (
        Path(value)
        if value is not None
        else local_root / f"review-batch-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    if not requested.is_absolute():
        requested = ROOT / requested
    requested = requested.resolve()
    if not requested.is_relative_to(local_root):
        raise ValueError(f"Output directory must be under Git-ignored {local_root}.")
    if requested.exists():
        raise FileExistsError(f"Output directory must be fresh: {requested}")
    return requested


def _generation_settings(checkpoint: Path) -> dict[str, Any]:
    from app.core.config import get_settings

    settings = get_settings()
    return {
        "model": "facebook/nllb-200-distilled-600M",
        "model_local_path": checkpoint.name,
        **_RAW_GENERATION_OPTIONS,
        "max_input_chars": settings.translation_max_input_chars,
        "phrase_overrides": "disabled; bypassed at inference call",
        "numeric_protection": "disabled; no numeric substitutions are applied",
    }


def _back_translation_settings() -> dict[str, Any]:
    from app.core.config import get_settings

    settings = get_settings()
    phrase_override_path = Path(settings.phrase_overrides_path)
    return {
        "num_beams": settings.translation_num_beams,
        "max_new_tokens": settings.translation_max_new_tokens,
        "length_penalty": settings.translation_length_penalty,
        "max_input_chars": settings.translation_max_input_chars,
        "phrase_overrides": "enabled" if phrase_override_path.is_file() else "unavailable",
        "numeric_protection": "disabled",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a local review-agent batch pilot.")
    parser.add_argument("--dataset", choices=("all", "flores", "blind"), default="all")
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--output-dir", help="fresh output directory under .local")
    parser.add_argument(
        "--flores-source", type=Path, default=DEFAULT_FLORES_SOURCE, help="FLORES English lines"
    )
    parser.add_argument(
        "--flores-reference",
        type=Path,
        default=DEFAULT_FLORES_REFERENCE,
        help="aligned FLORES Yorùbá lines",
    )
    parser.add_argument(
        "--flores-predictions",
        type=Path,
        default=DEFAULT_FLORES_PREDICTIONS,
        help="existing NLLB predictions CSV",
    )
    parser.add_argument(
        "--flores-metadata",
        type=Path,
        default=DEFAULT_FLORES_METADATA,
        help="metadata for existing NLLB predictions",
    )
    parser.add_argument(
        "--flores-official-source",
        type=Path,
        default=DEFAULT_FLORES_OFFICIAL_SOURCE,
        help="original full FLORES devtest English file",
    )
    parser.add_argument(
        "--flores-official-reference",
        type=Path,
        default=DEFAULT_FLORES_OFFICIAL_REFERENCE,
        help="original full FLORES devtest Yorùbá file",
    )
    parser.add_argument(
        "--flores-provenance",
        type=Path,
        default=DEFAULT_FLORES_SOURCE.parent / "README.md",
        help="documentation describing the local FLORES subset",
    )
    parser.add_argument(
        "--flores-row-ids",
        nargs="+",
        default=list(DEFAULT_FLORES_ROW_IDS),
        help="verified FLORES IDs to review (default: previous pilot IDs)",
    )
    parser.add_argument("--blind-csv", type=Path, default=DEFAULT_BLIND_CSV)
    parser.add_argument(
        "--blind-answer-key",
        type=Path,
        help="private key used only for aggregate results after all decisions are saved",
    )
    parser.add_argument("--ollama-model", default=os.environ.get("OLLAMA_MODEL", "gemma3:4b"))
    parser.add_argument("--audit-no-issue", action="store_true")
    args = parser.parse_args(argv)

    try:
        output_dir = _local_output_dir(args.output_dir)
        checkpoint = Path(os.environ.get("TRANSLATION_MODEL_ID", ""))
        if not checkpoint.is_dir():
            raise ValueError(
                "TRANSLATION_MODEL_ID must point to an existing local NLLB checkpoint directory."
            )
        generation_settings = _generation_settings(checkpoint)
        output_dir.mkdir(parents=True)

        datasets: list[dict[str, Any]] = []
        cases: list[ReviewCase] = []
        alignment: FloresAlignment | None = None
        selected_ids: list[str] = []
        if args.dataset in {"all", "flores"}:
            alignment = audit_flores_alignment(
                source_path=args.flores_source,
                reference_path=args.flores_reference,
                official_source_path=args.flores_official_source,
                official_reference_path=args.flores_official_reference,
                predictions_path=args.flores_predictions,
                metadata_path=args.flores_metadata,
                provenance_path=args.flores_provenance,
            )
            requested_ids = list(dict.fromkeys(args.flores_row_ids))
            selected_ids, unresolved_requested, replacements = select_verified_flores_ids(
                alignment, requested_ids
            )
            alignment.report["requested_row_ids"] = requested_ids
            alignment.report["selected_row_ids"] = selected_ids
            alignment.report["unresolved_requested_row_ids"] = unresolved_requested
            alignment.report["replacement_row_ids"] = replacements
            (output_dir / "alignment_audit.json").write_text(
                json.dumps(alignment.report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        from app.services.model_loader import load_nllb
        from app.services.translation import TranslationService

        service = TranslationService(load_nllb(str(checkpoint)))
        if args.dataset in {"all", "flores"}:
            if alignment is None:
                raise ValueError("FLORES alignment was not audited.")
            flores_cases, flores_meta = load_flores_cases(
                alignment=alignment,
                row_ids=selected_ids,
                translator=service,
                generation_settings=generation_settings,
            )
            cases.extend(flores_cases)
            datasets.append(flores_meta)
        if args.dataset in {"all", "blind"}:
            blind_cases, blind_meta = load_blind_cases(
                args.blind_csv, args.sample_size, args.seed
            )
            cases.extend(blind_cases)
            datasets.append(blind_meta)

        metadata = {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "sample_size_per_dataset": args.sample_size,
            "seed": args.seed,
            "review_model": args.ollama_model,
            "translation_model_id": checkpoint.name,
            "translation_generation_settings": generation_settings,
            "back_translation_settings": _back_translation_settings(),
            "audit_no_issue": args.audit_no_issue,
            "datasets": datasets,
            "private_answer_key_in_prompt": False,
        }
        records, summary = run_batch(
            cases,
            service=service,
            ollama=OllamaClient(),
            model=args.ollama_model,
            output_dir=output_dir / "reviews",
            audit_no_issue=args.audit_no_issue,
            run_metadata=metadata,
            blind_answer_key=args.blind_answer_key,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    print(f"Output directory: {output_dir}")
    if args.dataset in {"all", "flores"}:
        print(f"Alignment audit: {output_dir / 'alignment_audit.json'}")
    print(f"Results: {output_dir / 'reviews' / 'results.jsonl'}")
    print(f"Queue database: {output_dir / 'reviews' / 'reviews.sqlite3'}")
    print(f"Summary: {output_dir / 'reviews' / 'summary.json'}")
    print(f"Attempted: {summary['attempted']}")
    print(f"Completed: {summary['completed']}")
    print(f"Incomplete: {summary['incomplete']}")
    for group, counts in summary["by_candidate_group"].items():
        print(
            f"{group}: attempted={counts['attempted']} completed={counts['completed']} "
            f"incomplete={counts['incomplete']} needs_review={counts['needs_review']} "
            f"no_issue_flagged={counts['no_issue_flagged']}"
        )
    if "blind_protection_aggregate" in summary:
        print("Blind protection aggregate (after decisions saved; key assignments withheld):")
        for mode, counts in summary["blind_protection_aggregate"].items():
            print(
                f"{mode}: attempted={counts['attempted']} completed={counts['completed']} "
                f"incomplete={counts['incomplete']} needs_review={counts['needs_review']} "
                f"no_issue_flagged={counts['no_issue_flagged']}"
            )
    for record in records:
        if record["decision"] == "needs_review":
            print(
                f"FLAG {record['dataset']} {record['row_id']} {record['candidate_label']}: "
                f"{record['reason']} | back-translation={record['english_back_translation']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
