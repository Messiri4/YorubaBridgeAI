from __future__ import annotations

import csv
import json
import random
import sys
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.model_loader import load_nllb
from app.services.translation import TranslationService


ORIGINAL = ROOT.parent / "YorubaBridgeAI"
MODEL_ID = str(ORIGINAL / "models" / "nllb-200-distilled-600M-safetensors")
PUBLIC_DIR = ROOT / "data" / "eval" / "blind_review"
PRIVATE_KEY = Path(
    r"C:\Users\User\.copilot\session-state\f878f44d-c9e3-47e7-b91a-91547d3f095c"
    r"\files\blind_review_answer_key_private.json"
)
SEED = 20261123

EXAMPLES = [
    ("eng", "Adebayo paid $2.3 billion on 2026-11-23."),
    ("eng", "Call +234 803 123 4567 about order AK-4410."),
    ("eng", "Visit https://example.com/report?q=1 before noon."),
    ("eng", "The repeated value is 3 and 3."),
    ("eng", "Ada wrote to ada@example.com from @Adebayo."),
    ("eng", "The meeting starts at 11:29 pm."),
    ("eng", "Maria scored 7-2 in match 2026."),
    ("eng", "Install version 3.12.0 for customer CUST-77."),
    ("eng", "Email @NewsDesk at news@example.org by 04.06.2026."),
    ("eng", "The package weighs 12.5 kg and costs 98.6 USD."),
    ("eng", "Good morning."),
    ("eng", "I am going to the market."),
    ("eng", "Please speak more slowly."),
    ("eng", "The rain is falling."),
    ("eng", "Where is the market?"),
    ("yor", "Adebayo san ní ọjọ́ 2026-11-23."),
    ("yor", "Pè +234 803 123 4567 nípa àṣẹ AK-4410."),
    ("yor", "Ṣabẹwo sí https://example.com/report?q=1 kí àárọ̀ tó kọjá."),
    ("yor", "Ọlọ́pọ̀ iye náà jẹ́ 3 àti 3."),
    ("yor", "Ada fi imeeli ránṣẹ́ sí ada@example.com láti @Adebayo."),
    ("yor", "Ìpàdé náà bẹ̀rẹ̀ ní 11:29 pm."),
    ("yor", "Maria gba 7-2 ní ìdíje 2026."),
    ("yor", "Fi àtúnṣe 3.12.0 sori ẹrọ fún oníbàárà CUST-77."),
    ("yor", "Ẹ kí @NewsDesk ní símeeli news@example.org kí 04.06.2026 tó dé."),
    ("yor", "Aṣọ náà wú 12.5 kg, ó sì jẹ́ 98.6 USD."),
    ("yor", "Ẹ kú àárọ̀."),
    ("yor", "Mo ń lọ sí ọjà."),
    ("yor", "Jọ̀wọ́, sọ̀rọ̀ díẹ̀."),
    ("yor", "Òjò ń rọ̀."),
    ("yor", "Níbo ni ọjà wà?"),
]


def main() -> None:
    service = TranslationService(load_nllb(MODEL_ID))
    randomizer = random.Random(SEED)
    rows = []
    key = []

    for index, (direction, source) in enumerate(EXAMPLES, start=1):
        source_language, target_language = (
            ("eng", "yor") if direction == "eng" else ("yor", "eng")
        )
        off = unicodedata.normalize(
            "NFC",
            service.translate(
                source, source_language, target_language, protect_numerics=False
            ).translated_text,
        )
        on = unicodedata.normalize(
            "NFC",
            service.translate(
                source, source_language, target_language, protect_numerics=True
            ).translated_text,
        )
        candidate_is_a = randomizer.choice((True, False))
        sentence_id = f"BR-{index:03d}"
        rows.append({
            "ID": sentence_id,
            "direction": "ENG->YO" if direction == "eng" else "YO->ENG",
            "source": unicodedata.normalize("NFC", source),
            "Translation A": on if candidate_is_a else off,
            "Translation B": off if candidate_is_a else on,
            "preference": "",
            "meaning error in A or B": "",
            "missing or altered detail in A or B": "",
            "notes": "",
            "suggested correction": "",
        })
        key.append({
            "ID": sentence_id,
            "candidate_side": "A" if candidate_is_a else "B",
            "candidate": on,
            "control": off,
        })

    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    public_csv = PUBLIC_DIR / "blind_review_30.csv"
    fieldnames = list(rows[0].keys())
    with public_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    PRIVATE_KEY.parent.mkdir(parents=True, exist_ok=True)
    PRIVATE_KEY.write_text(
        json.dumps({
            "seed": SEED,
            "model": MODEL_ID,
            "num_beams": 4,
            "length_penalty": 1.1,
            "max_new_tokens": 400,
            "rows": key,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "rows": len(rows),
        "seed": SEED,
        "public_csv": str(public_csv),
        "private_key": str(PRIVATE_KEY),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
