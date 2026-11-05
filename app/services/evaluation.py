"""Translation quality evaluation: chrF and BLEU via sacrebleu.

chrF is the primary metric here: character n-gram F-score is much more
stable than BLEU on small test sets and morphologically rich languages
like Yoruba (BLEU's 4-gram precision is meaningless on ~10 sentences).
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import sacrebleu

from app.services.model_loader import SUPPORTED_LANGUAGES, load_nllb
from app.services.translation import TranslationService

# Direction keys used in the eval dataset.
ENG_YOR = "eng_yor"
YOR_ENG = "yor_eng"


@dataclass(frozen=True)
class EntryResult:
    """Outcome of translating one eval entry."""

    id: str
    direction: str
    source: str
    reference: str
    hypothesis: str
    verified: bool
    latency_ms: float


@dataclass(frozen=True)
class DirectionScores:
    """Aggregate metric values for one translation direction."""

    direction: str
    chrf: float
    bleu: float
    n_entries: int
    n_verified: int


def score_direction(results: list[EntryResult]) -> DirectionScores:
    """Compute chrF and BLEU for one direction's results.

    Args:
        results: EntryResults for a single direction.

    Returns:
        DirectionScores with corpus-level chrF (0-100) and BLEU (0-100).
    """
    refs = [[r.reference for r in results]]
    hyps = [r.hypothesis for r in results]
    chrf = sacrebleu.corpus_chrf(hyps, refs)
    bleu = sacrebleu.corpus_bleu(hyps, refs)
    return DirectionScores(
        direction=results[0].direction,
        chrf=chrf.score,
        bleu=bleu.score,
        n_entries=len(results),
        n_verified=sum(1 for r in results if r.verified),
    )


def run_evaluation(
    dataset_path: Path,
    output_path: Path | None = None,
    service: TranslationService | None = None,
) -> tuple[list[EntryResult], list[DirectionScores]]:
    """Translate the eval set and score both directions.

    Args:
        dataset_path: path to en_yor_eval.json.
        output_path: if given, write per-entry results + scores as JSON.
        service: inject a TranslationService (tests pass a mock); when None
            the real NLLB model is loaded.

    Returns:
        (per-entry results, per-direction aggregate scores).
    """
    if service is None:
        service = TranslationService(load_nllb())

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    entries = dataset["entries"]

    all_results: list[EntryResult] = []
    for entry in entries:
        direction = entry["direction"]
        src_code, tgt_code = direction.split("_")  # "eng_yor" -> eng, yor
        assert src_code in SUPPORTED_LANGUAGES and tgt_code in SUPPORTED_LANGUAGES

        translated = service.translate(entry["source"], src_code, tgt_code)
        latency = translated.latency_ms

        all_results.append(
            EntryResult(
                id=entry["id"],
                direction=direction,
                source=entry["source"],
                reference=entry["reference"],
                hypothesis=translated.translated_text,
                verified=bool(entry.get("verified", False)),
                latency_ms=latency,
            )
        )

    scores: list[DirectionScores] = []
    for direction in (ENG_YOR, YOR_ENG):
        subset = [r for r in all_results if r.direction == direction]
        if subset:
            scores.append(score_direction(subset))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": "facebook/nllb-200-distilled-600M (local safetensors)",
            "results": [asdict(r) for r in all_results],
            "scores": [asdict(s) for s in scores],
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return all_results, scores
