"""Build the prioritized 20-pair human-review subset for Sprint 2A.

Selects from the 100-pair worksheet (deterministic, seed 20261110):
6 meaning-changing candidates, 4 numeric, 3 entity, 3 omission/short,
2 orthographic, 2 unflagged controls. Fully traceable: every row records
selection reason + provenance indices; judgments stay BLANK.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "eval"
SEED = 20261110

# Verified meaning-changing candidates from Phase-2 manual reading, mapped to
# the A/B arms (direction, idx) they came from. These carry the strongest
# prior evidence of real semantic damage.
MEANING_CHANGING = [
    ("yor_eng", 2, "office-title inversion: alákòso ìsúra (treasurer) rendered 'prime minister' in both beams"),
    ("yor_eng", 6, "polarity inversion: 'economically viable' rendered 'economically disadvantaged' in both beams"),
    ("eng_yor", 3, "score corruption: '7–2' rendered '72' (beams=2) and '7 2' (beams=4)"),
    ("yor_eng", 22, "role reversal: 'bosses had been reined in by Provenzano' rendered as Provenzano leading/overthrowing"),
    ("eng_yor", 13, "measurement drift: '9,000-year lease' dropped/transformed differently per beam"),
    ("yor_eng", 4, "actant flip: banks helping clients HIDE money became investors INVESTING money"),
]

PLAN = [
    ("meaning_change", 6, MEANING_CHANGING),
    ("numeric", 4, None),       # picked from audit NUM_* findings deterministically
    ("entity", 3, None),
    ("omission", 3, None),
    ("orthographic", 2, None),
    ("control", 2, None),
]


def dev_index_for_ab(direction: str, ab_idx: int) -> int:
    """Reproduce the seeded A/B sampling to map (direction, ab_idx) -> dev idx."""
    en = (ROOT / "data" / "flores_eval" / "dev.997.en.txt").read_text("utf-8").splitlines()
    idx_list = list(range(len(en)))
    random.Random(42).shuffle(idx_list)
    return idx_list[ab_idx]


def main() -> None:
    worksheet = list(csv.DictReader((OUT / "flores_reference_review.csv").open(encoding="utf-8")))
    by_idx = {int(r["pair_idx"]): r for r in worksheet}
    audit = list(csv.DictReader((OUT / "flores_reference_audit.csv").open(encoding="utf-8")))

    # The six verified meaning-changing pairs live in the beam-A/B 30-pair
    # sample, not necessarily in the 100-pair worksheet. Extend the sheet:
    # pull their full dev rows + A/B predictions so every reviewed pair is
    # traceable and self-contained.
    ab_rows = list(csv.DictReader((OUT / "beams_ab_20261109T231413Z.csv").open(encoding="utf-8")))
    ab_by = {(r["direction"], int(r["idx"])): r for r in ab_rows}
    en997 = (ROOT / "data" / "flores_eval" / "dev.997.en.txt").read_text("utf-8").splitlines()
    yo997 = (ROOT / "data" / "flores_eval" / "dev.997.yo.txt").read_text("utf-8").splitlines()

    def extend_worksheet(direction: str, ab_idx: int) -> int:
        dev_idx = dev_index_for_ab(direction, ab_idx)
        if dev_idx not in by_idx:
            src = ab_by[(direction, ab_idx)]
            by_idx[dev_idx] = {
                "pair_idx": str(dev_idx),
                "sampling_reasons": "extended:meaning-change-priority",
                "english_source": en997[dev_idx],
                "yoruba_reference": yo997[dev_idx],
                "nllb_beams2_eng_yor": src["prediction"] if src["direction"] == "eng_yor" and src["config"] == "beams=2" else "",
                "nllb_beams4_eng_yor": src["prediction"] if src["direction"] == "eng_yor" and src["config"] == "beams=4" else "",
                "nllb_beams2_yor_eng": src["prediction"] if src["direction"] == "yor_eng" and src["config"] == "beams=2" else "",
                "nllb_beams4_yor_eng": src["prediction"] if src["direction"] == "yor_eng" and src["config"] == "beams=4" else "",
            }
        return dev_idx
    by_cat: dict[str, list[int]] = {}
    for r in audit:
        by_cat.setdefault(r["category"], []).append(int(r["pair_idx"]))
    rng = random.Random(SEED)

    chosen: dict[int, list[str]] = {}
    rows: list[dict] = []
    priority_rank = 0

    def take(dev_idx: int, reason: str, extra: str = "") -> None:
        nonlocal priority_rank
        priority_rank += 1
        chosen[dev_idx] = [f"rank={priority_rank}", reason]
        if extra:
            chosen[dev_idx].append(extra)

    # 1. meaning-changing (highest priority; from A/B verdicts)
    for direction, ab_idx, why in MEANING_CHANGING:
        dev_idx = extend_worksheet(direction, ab_idx)
        take(dev_idx, f"meaning_change ({direction} ab_idx={ab_idx})", why)

    # 2-5. deterministic picks from audit categories — restricted to worksheet
    # pairs, since every review row must be traceable to the 100-pair sheet.
    ws_idx = set(by_idx)
    picks = [
        ("numeric", ["NUM_DIGIT_DIFF", "NUM_DIGITS_DROPPED"], 4),
        ("entity", ["ENTITY_NOT_CARRIED"], 3),
        ("omission", ["SUSPICIOUSLY_SHORT", "LOW_TONE_MARK_COVERAGE"], 3),
        ("orthographic", ["MIXED_UNICODE_FORMS", "REFERENCE_UNACCENTED",
                          "MISSING_UNDERDOT_CANDIDATE"], 2),
    ]
    for label, cats, k in picks:
        pool: list[int] = []
        for c in cats:
            pool += by_cat.get(c, [])
        pool = sorted(set(pool) & ws_idx)
        rng.shuffle(pool)
        got = 0
        for idx in pool:
            if got >= k:
                break
            if idx in chosen:
                continue
            take(idx, f"{label}:{'+'.join(cats)}")
            got += 1

    # 6. controls: unflagged worksheet rows (only ones actually IN the worksheet)
    flagged = {int(r["pair_idx"]) for r in audit}
    controls = [i for i in sorted(by_idx) if i not in flagged]
    rng.shuffle(controls)
    for idx in controls[: PLAN_COUNT_CONTROL]:
        take(idx, "control:unflagged")
    # If the meaning-changing list referenced a dev index not present in the
    # worksheet (possible: the worksheet stratification may have skipped it),
    # substitute the nearest unclaimed worksheet row with an audit finding.
    if len(chosen) < sum(k for _, k, _ in PLAN):
        spare = [i for i in sorted(by_idx) if i not in chosen]
        rng.shuffle(spare)
        for idx in spare:
            if len(chosen) >= sum(k for _, k, _ in PLAN):
                break
            take(idx, "substitute:worksheet-row-available")

    # assemble output rows (judgments blank; predictions from worksheet)
    for dev_idx in sorted(chosen):
        if dev_idx not in by_idx:
            continue  # cannot produce a worksheet row for a dev idx outside the 100-pair sheet
        w = by_idx[dev_idx]
        rows.append({
            "priority": chosen[dev_idx][0],
            "selection_reasons": ";".join(chosen[dev_idx][1:]),
            "pair_idx": dev_idx,
            "english_source": w["english_source"],
            "yoruba_reference": w["yoruba_reference"],
            "nllb_beams2_eng_yor": w["nllb_beams2_eng_yor"],
            "nllb_beams4_eng_yor": w["nllb_beams4_eng_yor"],
            "nllb_beams2_yor_eng": w["nllb_beams2_yor_eng"],
            "nllb_beams4_yor_eng": w["nllb_beams4_yor_eng"],
            # judgment fields — deliberately blank per rubric
            "reference_accuracy": "", "meaning_preservation": "",
            "grammar_naturalness": "", "diacritic_correctness": "",
            "numerical_consistency": "", "named_entity_consistency": "",
            "reviewer_comments": "", "confidence": "", "final_review_status": "",
        })

    out = OUT / "flores_priority_review_20.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)
    (OUT / "flores_priority_review_20.meta.json").write_text(json.dumps({
        "seed": SEED, "n": len(rows),
        "plan": [{"bucket": b, "n": k, "categories": c} for b, k, c in PLAN],
        "trace": "pair_idx -> dev.997 line; worksheet -> flores_reference_review.csv",
    }, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} pairs -> {out.name}")


PLAN_COUNT_CONTROL = 2

if __name__ == "__main__":
    main()
