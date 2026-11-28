"""Paired local evaluation of English-to-Yorùbá phrase overrides."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, TypedDict

from app.review_agent import (
    AgentReviewResult,
    OllamaClient,
    OllamaClientProtocol,
    OllamaError,
    run_agent_review,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DATA = ROOT / "data" / "eval" / "en_yor_eval.json"
_NUM_BEAMS = 4
_MAX_NEW_TOKENS = 256
_LENGTH_PENALTY = 1.1


class _InputDefinition(TypedDict):
    id: str
    group: str
    source: str
    expected_match: bool
    why: str


_OVERRIDE_INPUTS: tuple[_InputDefinition, ...] = (
    {
        "id": "good-morning-exact",
        "group": "exact",
        "source": "Good morning.",
        "expected_match": True,
        "why": "The full phrase exists in the eng_yor table.",
    },
    {
        "id": "good-morning-context",
        "group": "variation",
        "source": "I said good morning to my neighbor.",
        "expected_match": False,
        "why": "The extra words make the complete normalized input differ from the key.",
    },
    {
        "id": "how-are-you-exact",
        "group": "exact",
        "source": "How are you?",
        "expected_match": True,
        "why": "The full phrase exists in the eng_yor table.",
    },
    {
        "id": "how-are-you-context",
        "group": "variation",
        "source": "How are you doing today?",
        "expected_match": False,
        "why": "The added words prevent a whole-input exact match.",
    },
    {
        "id": "thank-you-exact",
        "group": "exact",
        "source": "Thank you very much.",
        "expected_match": True,
        "why": "The full phrase exists in the eng_yor table.",
    },
    {
        "id": "thank-you-context",
        "group": "variation",
        "source": "Thank you very much for helping me today.",
        "expected_match": False,
        "why": "The full sentence differs from the shorter exact key.",
    },
    {
        "id": "where-market-exact",
        "group": "exact",
        "source": "Where is the market?",
        "expected_match": True,
        "why": "The full phrase exists in the eng_yor table.",
    },
    {
        "id": "where-market-context",
        "group": "variation",
        "source": "Could you tell me where the market is?",
        "expected_match": False,
        "why": "The longer question is not an exact normalized table key.",
    },
    {
        "id": "how-much-exact",
        "group": "exact",
        "source": "How much is this?",
        "expected_match": True,
        "why": "The full phrase exists in the eng_yor table.",
    },
    {
        "id": "how-much-context",
        "group": "variation",
        "source": "Please tell me how much this costs.",
        "expected_match": False,
        "why": "The wording differs from the complete exact-match key.",
    },
    {
        "id": "control-window",
        "group": "control",
        "source": "Close the window before leaving.",
        "expected_match": False,
        "why": "Unrelated control; it shares no complete table phrase.",
    },
    {
        "id": "control-bus",
        "group": "control",
        "source": "The bus arrives at 8:30.",
        "expected_match": False,
        "why": "Unrelated control; it shares no complete table phrase.",
    },
    {
        "id": "control-airport",
        "group": "control",
        "source": "I need directions to the airport.",
        "expected_match": False,
        "why": "Unrelated control; it shares no complete table phrase.",
    },
)


class PhraseTable(Protocol):
    """Lookup surface exposed by the translation service."""

    def lookup(self, text: str, source_language: str, target_language: str) -> str | None: ...


class TranslationOutput(Protocol):
    @property
    def translated_text(self) -> str: ...

    @property
    def latency_ms(self) -> float: ...

    @property
    def override_key(self) -> str | None: ...


class PairedTranslationService(Protocol):
    """Service operations needed by this paired experiment."""

    @property
    def overrides(self) -> PhraseTable: ...

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
    ) -> TranslationOutput: ...


@dataclass(frozen=True)
class PairedInput:
    """One source reviewed with phrase overrides disabled and enabled."""

    input_id: str
    group: str
    source: str
    expected_match: bool
    expected_match_reason: str


def _input_cases() -> list[PairedInput]:
    return [
        PairedInput(
            input_id=item["id"],
            group=item["group"],
            source=item["source"],
            expected_match=item["expected_match"],
            expected_match_reason=item["why"],
        )
        for item in _OVERRIDE_INPUTS
    ]


def run_paired_translations(
    service: PairedTranslationService,
    inputs: Sequence[PairedInput] | None = None,
) -> list[dict[str, Any]]:
    """Translate each input twice, changing only the override switch."""
    records: list[dict[str, Any]] = []
    for item in inputs or _input_cases():
        table_value = service.overrides.lookup(item.source, "eng", "yor")
        observed_table_match = table_value is not None
        off = service.translate(
            item.source,
            "eng",
            "yor",
            num_beams=_NUM_BEAMS,
            max_new_tokens=_MAX_NEW_TOKENS,
            length_penalty=_LENGTH_PENALTY,
            use_phrase_overrides=False,
        )
        on = service.translate(
            item.source,
            "eng",
            "yor",
            num_beams=_NUM_BEAMS,
            max_new_tokens=_MAX_NEW_TOKENS,
            length_penalty=_LENGTH_PENALTY,
            use_phrase_overrides=True,
        )
        off_fired = off.override_key is not None
        on_fired = on.override_key is not None
        records.append(
            {
                "input_id": item.input_id,
                "group": item.group,
                "source": item.source,
                "expected_match": item.expected_match,
                "expected_match_reason": item.expected_match_reason,
                "observed_table_match_before_translation": observed_table_match,
                "expected_lookup_matches_table": observed_table_match == item.expected_match,
                "off": {
                    "output": off.translated_text,
                    "latency_ms": off.latency_ms,
                    "override_fired": off_fired,
                    "matched_entry": off.override_key,
                    "translation_path": "phrase_override" if off_fired else "nllb",
                },
                "on": {
                    "output": on.translated_text,
                    "latency_ms": on.latency_ms,
                    "override_fired": on_fired,
                    "matched_entry": on.override_key,
                    "translation_path": "phrase_override" if on_fired else "nllb",
                },
                "outputs_changed": off.translated_text != on.translated_text,
                "unexpected_match": on_fired and not item.expected_match,
                "missed_expected_match": item.expected_match and not on_fired,
                "unexpected_off_match": off_fired,
                "numeric_protection": "disabled; no numeric substitutions are used",
                "cache": "bypassed; direct TranslationService calls do not use API/Redis cache",
            }
        )
    return records


def score_reference_backed_subset(
    records: list[dict[str, Any]], dataset_path: Path
) -> dict[str, Any]:
    """Score only inputs with pre-existing, verified bilingual references."""
    import sacrebleu
    from sacrebleu.metrics.bleu import BLEU
    from sacrebleu.metrics.chrf import CHRF
    from sacrebleu.metrics.ter import TER

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    verified = {
        entry["source"]: entry
        for entry in dataset["entries"]
        if entry.get("direction") == "eng_yor" and entry.get("verified") is True
    }
    subset = [
        (record, verified[record["source"]])
        for record in records
        if record["source"] in verified
    ]
    if not subset:
        return {
            "reference_count": 0,
            "reference_ids": [],
            "scores": None,
            "meteor": {
                "reported": False,
                "reason": "No verified Yorùbá references are available for this input set.",
            },
        }

    references = [entry["reference"] for _, entry in subset]
    reference_ids = [entry["id"] for _, entry in subset]
    scorer_settings = {
        "sacrebleu_version": sacrebleu.__version__,
        "bleu": {
            "tokenize": "13a",
            "lowercase": False,
            "effective_order": False,
        },
        "chrf++": {
            "char_order": 6,
            "word_order": 2,
            "beta": 2,
            "lowercase": False,
            "whitespace": False,
            "eps_smoothing": False,
        },
        "ter": {
            "normalized": False,
            "no_punct": False,
            "asian_support": False,
            "case_sensitive": False,
        },
    }
    scores: dict[str, dict[str, Any]] = {}
    for condition in ("off", "on"):
        hypotheses = [record[condition]["output"] for record, _ in subset]
        bleu = BLEU(tokenize="13a", lowercase=False, effective_order=False)
        chrf = CHRF(
            char_order=6,
            word_order=2,
            beta=2,
            lowercase=False,
            whitespace=False,
            eps_smoothing=False,
        )
        ter = TER(
            normalized=False,
            no_punct=False,
            asian_support=False,
            case_sensitive=False,
        )
        scores[condition] = {
            "bleu": bleu.corpus_score(hypotheses, [references]).score,
            "bleu_signature": bleu.get_signature().format(),
            "chrf++": chrf.corpus_score(hypotheses, [references]).score,
            "chrf++_signature": chrf.get_signature().format(),
            "ter": ter.corpus_score(hypotheses, [references]).score,
            "ter_signature": ter.get_signature().format(),
        }
    scores["on_minus_off"] = {
        metric: scores["on"][metric] - scores["off"][metric]
        for metric in ("bleu", "chrf++", "ter")
    }
    return {
        "reference_count": len(subset),
        "reference_ids": reference_ids,
        "reference_source": dataset_path.name,
        "reference_provenance": (
            "Pre-existing eval entries marked verified; committed before the phrase table. "
            "The project documentation notes deliberate domain overlap, so this is not "
            "held-out evidence and may reflect references later used to curate the table."
        ),
        "scorer_settings": scorer_settings,
        "scores": scores,
        "meteor": {
            "reported": False,
            "reason": (
                "No vetted Yorùbá METEOR implementation or Yorùbá lexical resources are "
                "configured; an English-WordNet-oriented scorer would not be appropriate."
            ),
        },
    }


def create_blind_packet(
    records: list[dict[str, Any]], output_dir: Path, seed: int
) -> dict[str, Any]:
    """Write a randomized A/B packet and a separate private assignment key."""
    rng = random.Random(seed)
    shuffled_records = list(records)
    rng.shuffle(shuffled_records)
    public_rows: list[dict[str, str]] = []
    private_rows: list[dict[str, Any]] = []
    for index, record in enumerate(shuffled_records, start=1):
        on_is_a = bool(rng.getrandbits(1))
        outputs = {
            "A": record["on"]["output"] if on_is_a else record["off"]["output"],
            "B": record["off"]["output"] if on_is_a else record["on"]["output"],
        }
        identical = outputs["A"] == outputs["B"]
        case_id = f"R{index:03d}"
        public_rows.append(
            {
                "case_id": case_id,
                "English source": record["source"],
                "Translation A": outputs["A"],
                "Translation B": outputs["B"],
                "identical_outputs": "yes — count as tie" if identical else "no",
                "Reviewer choice (A/B/tie)": "Tie (identical outputs)" if identical else "",
                "Reviewer notes": "",
            }
        )
        private_rows.append(
            {
                "case_id": case_id,
                "input_id": record["input_id"],
                "input_group": record["group"],
                "expected_match": record["expected_match"],
                "expected_match_reason": record["expected_match_reason"],
                "matched_override_entry": record["on"]["matched_entry"],
                "condition_by_side": {
                    "A": "on" if on_is_a else "off",
                    "B": "off" if on_is_a else "on",
                },
                "identical_outputs": identical,
                "off_output": record["off"]["output"],
                "on_output": record["on"]["output"],
            }
        )

    packet_path = output_dir / "human_review_blind.csv"
    with packet_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(public_rows[0]))
        writer.writeheader()
        writer.writerows(public_rows)
    (output_dir / "human_review_instructions.txt").write_text(
        "For each English source, compare Translation A and B without trying to infer "
        "which system produced either one. Judge accuracy, fluency, and cultural "
        "appropriateness, including idiomatic phrasing and Yoruba tone marks/underdots. "
        "Choose A, B, or tie, and add notes where useful. Identical outputs are marked "
        "as ties for the comparison summary.\n",
        encoding="utf-8",
    )
    key_path = output_dir / "answer_key_private.json"
    key_path.write_text(
        json.dumps(
            {
                "private": True,
                "seed": seed,
                "assignment": private_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "packet_path": packet_path.name,
        "private_key_path": key_path.name,
        "seed": seed,
        "rows": len(public_rows),
        "identical_ties": sum(
            row["identical_outputs"] == "yes — count as tie" for row in public_rows
        ),
    }


def _agent_review_record(
    record: dict[str, Any],
    condition: str,
    result: AgentReviewResult,
) -> dict[str, Any]:
    return {
        "input_id": record["input_id"],
        "group": record["group"],
        "condition": condition,
        "source": record["source"],
        "candidate": record[condition]["output"],
        "decision": result.decision,
        "reason": result.reason,
        "english_back_translation": result.english_back_translation,
        "digit_findings": list(result.numeric_findings),
        "tool_trace": list(result.tool_call_trace),
        "review_latency_ms": result.review_latency_ms,
    }


def review_paired_outputs(
    records: list[dict[str, Any]],
    *,
    service: PairedTranslationService,
    ollama: OllamaClientProtocol,
    model: str,
) -> list[dict[str, Any]]:
    """Run the existing reviewer independently on each OFF/ON candidate."""
    reviews: list[dict[str, Any]] = []
    for record in records:
        for condition in ("off", "on"):
            result = run_agent_review(
                record["source"],
                record[condition]["output"],
                service=service,
                ollama=ollama,
                model=model,
            )
            reviews.append(_agent_review_record(record, condition, result))
    return reviews


def unavailable_agent_reviews(records: list[dict[str, Any]], reason: str) -> list[dict[str, Any]]:
    """Keep review outcomes explicit when the local reviewer cannot run."""
    return [
        {
            "input_id": record["input_id"],
            "group": record["group"],
            "condition": condition,
            "source": record["source"],
            "candidate": record[condition]["output"],
            "decision": "review_incomplete",
            "reason": reason,
            "english_back_translation": None,
            "digit_findings": [],
            "tool_trace": [],
            "review_latency_ms": 0.0,
        }
        for record in records
        for condition in ("off", "on")
    ]


def _agent_summary(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, dict[str, int]] = defaultdict(
        lambda: {"attempted": 0, "needs_review": 0, "no_issue_flagged": 0, "incomplete": 0}
    )
    for review in reviews:
        stats = by_condition[review["condition"]]
        stats["attempted"] += 1
        decision = review["decision"]
        if decision == "review_incomplete":
            stats["incomplete"] += 1
        else:
            stats[decision] += 1
    return {"by_condition": dict(by_condition)}


def _group_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for record in records:
        group = summary.setdefault(
            record["group"],
            {
                "inputs": 0,
                "expected_matches": 0,
                "table_matches": 0,
                "actual_on_hits": 0,
                "output_changes": 0,
                "unexpected_matches": 0,
                "missed_expected_matches": 0,
                "unexpected_off_hits": 0,
            },
        )
        group["inputs"] += 1
        group["expected_matches"] += int(record["expected_match"])
        group["table_matches"] += int(record["observed_table_match_before_translation"])
        group["actual_on_hits"] += int(record["on"]["override_fired"])
        group["output_changes"] += int(record["outputs_changed"])
        group["unexpected_matches"] += int(record["unexpected_match"])
        group["missed_expected_matches"] += int(record["missed_expected_match"])
        group["unexpected_off_hits"] += int(record["unexpected_off_match"])
    return summary


def _fresh_output_dir(value: str | None) -> Path:
    local_root = (ROOT / ".local").resolve()
    requested = (
        Path(value)
        if value is not None
        else local_root / f"phrase-override-eval-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    if not requested.is_absolute():
        requested = ROOT / requested
    requested = requested.resolve()
    if not requested.is_relative_to(local_root):
        raise ValueError(f"Output directory must be under Git-ignored {local_root}.")
    if requested.exists():
        raise FileExistsError(f"Output directory must be fresh: {requested}")
    return requested


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a local phrase-override A/B evaluation.")
    parser.add_argument("--output-dir", help="fresh output directory under .local")
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--eval-data", type=Path, default=DEFAULT_EVAL_DATA)
    parser.add_argument("--ollama-model", default=os.environ.get("OLLAMA_MODEL", "gemma3:4b"))
    args = parser.parse_args(argv)

    try:
        checkpoint = Path(os.environ.get("TRANSLATION_MODEL_ID", ""))
        if not checkpoint.is_dir():
            raise ValueError(
                "TRANSLATION_MODEL_ID must point to an existing local NLLB checkpoint directory."
            )
        output_dir = _fresh_output_dir(args.output_dir)
        output_dir.mkdir(parents=True)

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from app.core.config import get_settings
        from app.services.model_loader import load_nllb
        from app.services.translation import TranslationService

        service = TranslationService(load_nllb(str(checkpoint)))
        records = run_paired_translations(service)
        quality = score_reference_backed_subset(records, args.eval_data)
        packet = create_blind_packet(records, output_dir, args.seed)
        settings = {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "dataset": "curated_override_cases",
            "translation_model_id": checkpoint.name,
            "translation_language": "eng->yor",
            "generation_settings": {
                "num_beams": _NUM_BEAMS,
                "max_new_tokens": _MAX_NEW_TOKENS,
                "length_penalty": _LENGTH_PENALTY,
                "phrase_overrides_off": False,
                "phrase_overrides_on": True,
                "numeric_protection": "disabled",
                "cache": "bypassed; direct TranslationService calls, not API/Redis",
            },
            "override_table": Path(get_settings().phrase_overrides_path).name,
            "override_match_rules": (
                "Direction-specific whole-input exact match after NFC, punctuation removal "
                "for .,!?:;, whitespace collapse, and lowercase; tone marks remain significant."
            ),
            "inputs": [item.__dict__ for item in _input_cases()],
            "blind_packet": packet,
            "gemma_model": args.ollama_model,
            "reviewer_prompt": "Existing app.review_agent prompt and decision logic; unchanged.",
            "review_method": (
                "Each OFF and ON output is independently reviewed against its English source; "
                "condition labels and private key are not passed to Ollama."
            ),
            "reference_quality": quality,
        }
        (output_dir / "paired_results.json").write_text(
            json.dumps(
                {
                    "input_count": len(records),
                    "condition_runs": len(records) * 2,
                    "group_summary": _group_summary(records),
                    "reference_quality": quality,
                    "records": records,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (output_dir / "run_settings.json").write_text(
            json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        ollama = OllamaClient()
        ollama_probe_started = time.perf_counter()
        ollama_probe_error = None
        try:
            installed = ollama.list_model_names()
            if args.ollama_model not in installed:
                ollama_probe_error = f"Ollama model '{args.ollama_model}' is not installed locally."
        except OllamaError as exc:
            ollama_probe_error = str(exc)
        ollama_probe_ms = round((time.perf_counter() - ollama_probe_started) * 1000, 1)

        reviews = (
            review_paired_outputs(
                records,
                service=service,
                ollama=ollama,
                model=args.ollama_model,
            )
            if ollama_probe_error is None
            else unavailable_agent_reviews(records, ollama_probe_error)
        )
        with (output_dir / "gemma_reviews.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as review_file:
            for review in reviews:
                review_file.write(json.dumps(review, ensure_ascii=False) + "\n")
        summary: dict[str, Any] = {
            "input_count": len(records),
            "translation_runs": len(records) * 2,
            "override_groups": _group_summary(records),
            "reference_quality": quality,
            "gemma_model_installed": ollama_probe_error is None,
            "gemma_probe_error": ollama_probe_error,
            "gemma_probe_latency_ms": ollama_probe_ms,
            "gemma_review_summary": _agent_summary(reviews),
            "human_review_packet": packet,
            "human_judgments_received": False,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    print(f"Output directory: {output_dir}")
    print(f"Paired results: {output_dir / 'paired_results.json'}")
    print(f"Blind packet: {output_dir / 'human_review_blind.csv'}")
    print(f"Private key: {output_dir / 'answer_key_private.json'}")
    print(f"Gemma reviews: {output_dir / 'gemma_reviews.jsonl'}")
    print(f"Summary: {output_dir / 'summary.json'}")
    print(f"Inputs: {summary['input_count']}; translation runs: {summary['translation_runs']}")
    for group, counts in summary["override_groups"].items():
        print(
            f"{group}: inputs={counts['inputs']} expected_matches={counts['expected_matches']} "
            f"actual_on_hits={counts['actual_on_hits']} output_changes={counts['output_changes']} "
            f"unexpected_matches={counts['unexpected_matches']}"
        )
    print(f"Gemma reviews: {summary['gemma_review_summary']['by_condition']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
