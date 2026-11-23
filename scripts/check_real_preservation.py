from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ORIGINAL = ROOT.parent / "YorubaBridgeAI"
MODEL_ID = ORIGINAL / "models" / "nllb-200-distilled-600M-safetensors"

from app.services.model_loader import load_nllb
from app.services.translation import TranslationService


TEXTS = [
    "Adebayo sent $2.3 billion for order AK-4410 on 2026-11-23 to +234 803 123 4567.",
    "The repeated values are 3 and 3.",
    "Visit https://example.com/report?q=1 for details.",
]
YORUBA_TEXTS = [
    "Adebayo fi $2.3 billion ránṣẹ́ fún àṣẹ AK-4410 ní ọjọ́ 2026-11-23 sí +234 803 123 4567.",
    "Àwọn iye tí a tún sọ ni 3 àti 3.",
    "Ṣabẹwo sí https://example.com/report?q=1 fún ìkúnlẹ̀kọ̀ọ́.",
]
EXPECTED_BY_TEXT = [
    ["Adebayo", "$2.3 billion", "AK-4410", "2026-11-23", "+234 803 123 4567"],
    ["3"],
    ["https://example.com/report?q=1"],
]


def main() -> None:
    service = TranslationService(load_nllb(str(MODEL_ID)))
    report: dict[str, object] = {}

    for direction, source_language, target_language, texts in (
        ("eng_yor", "eng", "yor", TEXTS),
        ("yor_eng", "yor", "eng", YORUBA_TEXTS),
    ):
        rows = []
        for source, expected in zip(texts, EXPECTED_BY_TEXT):
            result = service.translate(
                source,
                source_language,
                target_language,
                protect_numerics=True,
            )
            output = result.translated_text
            spans = {}
            for span in expected:
                spans[span] = output.count(span)
            rows.append({
                "source": source,
                "output": output,
                "expected_counts": spans,
                "no_placeholders": "@@" not in output,
            })
        report[direction] = rows

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
