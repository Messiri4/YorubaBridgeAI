"""Run a local paired numeric-protection evaluation against a saved baseline CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from unittest.mock import patch

if TYPE_CHECKING:
    from app.services.protection import ProtectionResult
    from app.services.translation import TranslationResult

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "eval" / "beam_eval_dev_beams4_max400_lp11.csv"
EXPECTED_IDS = tuple(f"dev-{index:04d}" for index in range(1, 31))
_DECODING = {
    "num_beams": 4,
    "max_new_tokens": 400,
    "length_penalty": 1.1,
    "max_input_chars": 2000,
}


class NumericTranslator(Protocol):
    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        protect_numerics: bool | None = None,
    ) -> TranslationResult: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_input_rows(path: Path, row_ids: Sequence[str] | None = None) -> list[dict[str, str]]:
    """Load verified EN→YO rows and optionally select a stable ID sequence."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sentence_id", "direction", "source", "reference", "prediction"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Input CSV must contain columns: {', '.join(sorted(required))}.")
        rows = [
            {
                "row_id": row["sentence_id"],
                "source": row["source"],
                "reference": row["reference"],
                "baseline_prediction": row["prediction"],
            }
            for row in reader
            if row.get("direction") == "eng_yor"
        ]

    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        row_id = row["row_id"]
        if not row_id or row_id in by_id:
            raise ValueError(
                f"Input CSV contains a missing or duplicate EN-to-YO ID: {row_id!r}."
            )
        if not row["source"].strip() or not row["reference"].strip():
            raise ValueError(f"Input CSV has an empty source or reference for {row_id}.")
        if not row["baseline_prediction"].strip():
            raise ValueError(f"Input CSV has no saved baseline prediction for {row_id}.")
        by_id[row_id] = row

    selected_ids = list(row_ids) if row_ids is not None else list(EXPECTED_IDS)
    missing = [row_id for row_id in selected_ids if row_id not in by_id]
    if missing:
        raise ValueError(f"Input CSV is missing requested IDs: {', '.join(missing)}.")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Requested row IDs must be unique.")
    return [by_id[row_id] for row_id in selected_ids]


def run_paired_rows(
    rows: Sequence[dict[str, str]],
    translator: NumericTranslator,
    span_detector: Callable[[str], Sequence[str]],
    restoration_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Translate each source with protection off and on, preserving paired inputs."""
    records: list[dict[str, Any]] = []
    for row in rows:
        spans = list(span_detector(row["source"]))
        off = translator.translate(
            row["source"], "eng", "yor", protect_numerics=False
        ).translated_text
        event_index = len(restoration_events)
        on = translator.translate(
            row["source"], "eng", "yor", protect_numerics=True
        ).translated_text
        restoration = (
            restoration_events[event_index]
            if len(restoration_events) > event_index
            else None
        )
        records.append(
            {
                **row,
                "numeric_protection_off": off,
                "numeric_protection_on": on,
                "eligible_spans": spans,
                "off_matches_saved_baseline": off == row["baseline_prediction"],
                "output_changed": on != off,
                "restoration": restoration,
            }
        )
    return records


def score_outputs(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from sacrebleu.metrics.bleu import BLEU
    from sacrebleu.metrics.chrf import CHRF

    references = [unicodedata.normalize("NFC", row["reference"]) for row in records]
    scores: dict[str, Any] = {}
    bleu_metric = BLEU()
    chrf_metric = CHRF(word_order=2)
    for condition in ("numeric_protection_off", "numeric_protection_on"):
        hypotheses = [
            unicodedata.normalize("NFC", row[condition]) for row in records
        ]
        bleu = bleu_metric.corpus_score(hypotheses, [references])
        chrf = chrf_metric.corpus_score(hypotheses, [references])
        scores[condition] = {
            "n": len(records),
            "bleu": bleu.score,
            "bleu_signature": bleu_metric.get_signature().format(),
            "chrf++": chrf.score,
            "chrf++_signature": chrf_metric.get_signature().format(),
        }
    scores["on_minus_off"] = {
        metric: scores["numeric_protection_on"][metric]
        - scores["numeric_protection_off"][metric]
        for metric in ("bleu", "chrf++")
    }
    return scores


def _new_output_dir(value: str | None) -> Path:
    local_root = (ROOT / ".local").resolve()
    default_name = f"numeric-protection-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    requested = Path(value) if value else local_root / default_name
    if not requested.is_absolute():
        requested = ROOT / requested
    requested = requested.resolve()
    if not requested.is_relative_to(local_root):
        raise ValueError(f"Output directory must be under ignored {local_root}.")
    if requested.exists():
        raise FileExistsError(f"Output directory must be fresh: {requested}")
    requested.mkdir(parents=True)
    return requested


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    fields = (
        "row_id",
        "source",
        "reference",
        "baseline_prediction",
        "numeric_protection_off",
        "numeric_protection_on",
        "eligible_spans",
        "off_matches_saved_baseline",
        "output_changed",
        "restoration",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **record,
                    "eligible_spans": json.dumps(record["eligible_spans"], ensure_ascii=False),
                    "restoration": json.dumps(record["restoration"], ensure_ascii=False),
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", help="fresh output directory under .local")
    parser.add_argument(
        "--row-ids",
        nargs="+",
        help="EN-to-YO row IDs to run; default is dev-0001 through dev-0030",
    )
    args = parser.parse_args(argv)
    try:
        input_path = args.input_csv.resolve()
        rows = load_input_rows(input_path, args.row_ids)
        checkpoint = Path(os.environ.get("TRANSLATION_MODEL_ID", "")).expanduser()
        if not checkpoint.is_dir():
            raise ValueError(
                "TRANSLATION_MODEL_ID must point to an existing local NLLB checkpoint directory."
            )
        output_dir = _new_output_dir(args.output_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TRANSLATION_MODEL_ID"] = str(checkpoint)
    os.environ["TRANSLATION_NUM_BEAMS"] = str(_DECODING["num_beams"])
    os.environ["TRANSLATION_MAX_NEW_TOKENS"] = str(_DECODING["max_new_tokens"])
    os.environ["TRANSLATION_LENGTH_PENALTY"] = str(_DECODING["length_penalty"])
    os.environ["TRANSLATION_MAX_INPUT_CHARS"] = str(_DECODING["max_input_chars"])
    os.environ["NUMERIC_PROTECTION_ENABLED"] = "true"
    os.environ["GLOSSARY_ENABLED"] = "false"
    override_path = output_dir / "phrase_overrides_disabled.json"
    _write_json(override_path, {"eng_yor": {}, "yor_eng": {}})
    os.environ["PHRASE_OVERRIDES_PATH"] = str(override_path)

    import sacrebleu

    from app.core.config import get_settings
    from app.services import protection
    from app.services.model_loader import load_nllb
    from app.services.translation import TranslationService

    get_settings.cache_clear()
    settings = get_settings()
    if (
        settings.translation_num_beams != _DECODING["num_beams"]
        or settings.translation_max_new_tokens != _DECODING["max_new_tokens"]
        or settings.translation_length_penalty != _DECODING["length_penalty"]
        or settings.translation_max_input_chars != _DECODING["max_input_chars"]
        or settings.glossary_enabled
    ):
        raise ValueError("Loaded translation settings differ from the requested evaluation.")

    loaded = load_nllb(str(checkpoint))
    generate_calls: Counter[str] = Counter()
    original_generate = loaded.model.generate

    def record_generate(*positional: Any, **kwargs: Any) -> Any:
        signature = json.dumps(
            {
                key: kwargs.get(key)
                for key in ("forced_bos_token_id", "num_beams", "max_new_tokens", "length_penalty")
            },
            sort_keys=True,
        )
        generate_calls[signature] += 1
        return original_generate(*positional, **kwargs)

    loaded.model.generate = record_generate
    restoration_events: list[dict[str, Any]] = []
    original_restore = protection.restore

    def record_restore(
        model_output: str, protected: ProtectionResult
    ) -> tuple[str, bool, str]:
        restored, ok, reason = original_restore(model_output, protected)
        restoration_events.append({"ok": ok, "reason": reason})
        return restored, ok, reason

    service = TranslationService(loaded)
    with patch.object(protection, "restore", side_effect=record_restore):
        records = run_paired_rows(
            rows,
            service,
            lambda source: list(protection.protect(source).placeholders.values()),
            restoration_events,
        )

    scores = score_outputs(records)
    spans = [span for record in records for span in record["eligible_spans"]]
    digit_spans = [span for span in spans if any(char.isdigit() for char in span)]
    numeric_summary = {
        "rows": len(records),
        "numeric_protection_off_calls": len(records),
        "numeric_protection_on_calls": len(records),
        "inputs_with_eligible_spans": sum(bool(row["eligible_spans"]) for row in records),
        "eligible_span_count": len(spans),
        "inputs_with_digit_bearing_spans": sum(
            any(any(char.isdigit() for char in span) for span in row["eligible_spans"])
            for row in records
        ),
        "digit_bearing_span_count": len(digit_spans),
        "non_digit_match_count": len(spans) - len(digit_spans),
        "restoration_fallbacks": sum(
            bool(row["restoration"] and not row["restoration"]["ok"]) for row in records
        ),
        "outputs_changed_vs_off": sum(row["output_changed"] for row in records),
        "off_outputs_matching_saved_baseline": sum(
            row["off_matches_saved_baseline"] for row in records
        ),
    }
    scores["numeric_protection_summary"] = numeric_summary
    _write_json(output_dir / "per_row_results.json", {"records": records})
    _write_csv(output_dir / "per_row_results.csv", records)
    _write_json(output_dir / "scores.json", scores)
    _write_json(
        output_dir / "run_metadata.json",
        {
            "dataset": "FLORES-200",
            "split": "dev",
            "direction": "English-to-Yorùbá",
            "row_ids": [row["row_id"] for row in records],
            "input_file": input_path.name,
            "input_sha256": _sha256(input_path),
            "checkpoint": checkpoint.name,
            "checkpoint_source": "local path configured through TRANSLATION_MODEL_ID",
            "offline": True,
            "generation": _DECODING,
            "phrase_overrides": "disabled with an empty local table",
            "glossary": "disabled",
            "numeric_protection": {
                "off_call": False,
                "on_call": True,
            },
            "cache": "bypassed; direct TranslationService calls",
            "actual_generate_calls": sum(generate_calls.values()),
            "actual_generate_call_settings": dict(generate_calls),
            "inference_settings_verified_at_model_generate": True,
            "scoring": {
                "sacrebleu_version": sacrebleu.__version__,
                "bleu": "corpus BLEU, default 13a",
                "chrf++": "corpus chrF++, word_order=2",
                "unicode_preprocessing": "NFC",
                "sample_count_per_condition": len(records),
            },
            "numeric_protection_summary": numeric_summary,
        },
    )
    print(f"Output directory: {output_dir}")
    print(
        f"Rows: {len(records)}; OFF baseline matches: "
        f"{numeric_summary['off_outputs_matching_saved_baseline']}"
    )
    print(f"Scores: {output_dir / 'scores.json'}")
    print(f"Per-row results: {output_dir / 'per_row_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
