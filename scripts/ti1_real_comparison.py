"""Sprint TI-1 targeted real-model comparison: baseline vs protection+glossary.

Runs ONE model instance sequentially over a small set of known failure
cases (Phase-2 verified errors + new numeric probes), twice per sentence:
  A) legacy path (protect_numerics=False, glossary off via settings)
  B) improved path (protection + glossary enabled)
Writes data/eval/ti1_improvement_comparison.json. No historical artifacts touched.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.services.model_loader import LoadedModel
from app.services.translation import TranslationService
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

CASES = [
    # (direction, source, why)
    ("eng", "The head-to-head record is 7–2.", "verified: score corrupted to 72"),
    ("eng", "The show starts at 11:29.", "time garbling"),
    ("eng", "About 9,000 people attended the fair.", "9,000 dropped/changed"),
    ("eng", "Email support@example.com for a refund of $1,000.", "email+currency"),
    ("yor", "Ẹ kú ìrọ̀lẹ́, ọ̀rẹ́ mi.", "greeting idiom (regression canary)"),
    ("yor", "Òjò ń rò ní ìlú Kano.", "simple sentence (regression canary)"),
    ("yor", "Ọgbẹ́n àtògbẹ́ ń pọ̀ ní àárín ìlú náà.", "glossary term àtògbẹ→diabetes"),
    ("yor", "Ó ń kọ́ ẹ̀kọ́ nípa àìsàn àtògbẹ́.", "glossary term"),
]


def main() -> None:
    settings = get_settings()
    local = ".model-local/nllb-600M-fixed"
    tok = AutoTokenizer.from_pretrained(local)
    model = AutoModelForSeq2SeqLM.from_pretrained(local)
    model.eval()
    loaded = LoadedModel(model=model, tokenizer=tok, device="cpu")
    svc = TranslationService(loaded)

    rows = []
    for direction, src, why in CASES:
        src_lang, tgt_lang = ("eng", "yor") if direction == "eng" else ("yor", "eng")
        # A: legacy
        t0 = time.perf_counter()
        a = svc.translate(src, src_lang, tgt_lang, protect_numerics=False)
        a_ms = (time.perf_counter() - t0) * 1000
        # B: improved
        t0 = time.perf_counter()
        b = svc.translate(src, src_lang, tgt_lang, protect_numerics=True)
        b_ms = (time.perf_counter() - t0) * 1000
        rows.append({
            "direction": direction, "source": src, "why": why,
            "baseline": a.translated_text, "baseline_ms": round(a_ms, 1),
            "improved": b.translated_text, "improved_ms": round(b_ms, 1),
        })
        print(f"[{direction}] {src[:50]}")
        print(f"   base: {a.translated_text[:90]}")
        print(f"   impr: {b.translated_text[:90]}\n", flush=True)

    payload = {
        "sprint": "TI-1",
        "model": "facebook/nllb-200-distilled-600M (local safetensors snapshot)",
        "n_cases": len(rows),
        "numeric_protection": settings.numeric_protection_enabled,
        "glossary_enabled": settings.glossary_enabled,
        "rows": rows,
        "mean_latency_baseline_ms": round(statistics.mean(r["baseline_ms"] for r in rows), 1),
        "mean_latency_improved_ms": round(statistics.mean(r["improved_ms"] for r in rows), 1),
    }
    out = ROOT / "data" / "eval" / "ti1_improvement_comparison.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", out.name)


if __name__ == "__main__":
    main()
