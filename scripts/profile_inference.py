"""CLI inference profiler for YorubaBridgeAI (Sprint 2B).

Deliberately opt-in: importing or running this script with no flags does
NOT load NLLB and executes no inference. Real inference requires an
explicit ``--real-inference`` flag plus a local model checkpoint.

Usage:
    # Dry-run (no model, prints what would be measured)
    .venv/bin/python scripts/profile_inference.py

    # Mock pipeline timing (validates instrumentation, no model needed)
    .venv/bin/python scripts/profile_inference.py --mock --runs 5

    # Real inference (explicit consent; uses the local checkpoint)
    .venv/bin/python scripts/profile_inference.py \
        --real-inference --runs 5 --direction eng_yor \
        --samples data/flores_eval/dev.30.en.txt --limit 3

Outputs a JSON profile report (stdout, or --output PATH). Nothing is
written under data/eval unless --output points there.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.profiling import (  # noqa: E402
    TranslationProfiler,
    collect_system_context,
    maybe_collect_gc,
)


def build_mock_stages() -> dict:
    """Synthetic pipeline stages standing in for NLLB (test/dry-run only)."""

    class _FakeInputs:
        shape = (1, 12)

    class _FakeGenerated:
        shape = (1, 30)

    return {
        "tokenize": lambda text: (_FakeInputs(),),
        "generate": lambda inputs: _FakeGenerated(),
        "decode_output": lambda generated: "mock translation output",
    }


def run_real_stages(text: str) -> dict:
    """Build real pipeline stages bound to one hot-loaded model."""
    from app.core.config import get_settings
    from app.services.model_loader import SUPPORTED_LANGUAGES, load_nllb

    loaded = load_nllb()
    tokenizer, model = loaded.tokenizer, loaded.model

    def tokenize(sentence: str) -> object:
        tokenizer.src_lang = "eng_Latn"
        return tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=get_settings().translation_max_input_chars,
        )

    forced_bos = tokenizer.convert_tokens_to_ids(SUPPORTED_LANGUAGES["yor_Latn"])

    def generate(inputs: object) -> object:
        settings = get_settings()
        return model.generate(
            **inputs,  # type: ignore[arg-type]
            forced_bos_token_id=forced_bos,
            max_new_tokens=settings.translation_max_new_tokens,
            num_beams=settings.translation_num_beams,
            length_penalty=settings.translation_length_penalty,
        )

    return {
        "tokenize": tokenize,
        "generate": generate,
        "decode_output": lambda generated: tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="number of timed passes (default 3)")
    parser.add_argument(
        "--direction",
        choices=("eng_yor", "yor_eng"),
        default="eng_yor",
    )
    parser.add_argument("--samples", type=Path, help="flores txt file of source sentences")
    parser.add_argument("--limit", type=int, default=3, help="how many sentences to profile")
    parser.add_argument(
        "--real-inference", action="store_true", help="REQUIRED to touch the real NLLB model"
    )
    parser.add_argument("--mock", action="store_true", help="profile the synthetic pipeline")
    parser.add_argument("--output", type=Path, help="write the JSON report here")
    args = parser.parse_args()

    if not (args.real_inference or args.mock):
        print(
            "Dry run only: nothing was loaded or timed.\n"
            "Pass --real-inference (real NLLB, opt-in) or --mock (synthetic).",
            file=sys.stderr,
        )
        print(json.dumps({"dry_run": True, "context": collect_system_context()}, indent=2))
        return 0

    sources: list[str] = ["MOCK SENTENCE"]
    if args.samples and args.samples.exists():
        sources = args.samples.read_text(encoding="utf-8").splitlines()
        sources = sources[: max(1, args.limit)]

    profiler = TranslationProfiler()
    maybe_collect_gc()

    first = True
    for sentence in sources:
        stages = run_real_stages(sentence) if args.real_inference else build_mock_stages()
        for i in range(args.runs):
            cold = first and i == 0

            def make_tokenize(stages: dict, sentence: str) -> Callable[[], object]:
                return lambda: stages["tokenize"](sentence)

            t = profiler.profile_run(
                model_init=(lambda: None) if args.real_inference and cold else None,
                tokenize=make_tokenize(stages, sentence),
                generate=stages["generate"],
                decode_output=stages["decode_output"],
                cold_start=cold,
            )
            first = False
            print(
                f"{'cold' if t.cold_start else 'warm'} run: "
                f"tokenize={t.tokenize_s:.4f}s generate={t.generate_s:.4f}s "
                f"decode={t.decode_output_s:.4f}s total={t.total_s:.4f}s",
                flush=True,
            )

    report_json = profiler.report.to_json()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_json, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(report_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
