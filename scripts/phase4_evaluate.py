"""Sprint 3 PHASE 4: meaning-reliability evaluation over the saved
devtest integration comparison (160 pairs x 2 arms) — NO inference.

Computes per arm:
  chrF++ / BLEU (identical inputs, canonical config)
  numerical accuracy (exact-digit preservation rate + spelled-out verified)
  critical semantic-error counts (negation flips, degeneration loops,
    omissions of numbers/entities)
  restoration-relevant evidence (where the artifact permits)

All semantic counts are heuristic-warning counts; nothing is claimed as a
verified error rate. Writes docs/phase3_sprint3_evaluation.md + a JSON sidecar.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import sacrebleu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/eval/integration_comparison_20261110T102741Z.csv"
OUT_MD = ROOT / "docs/phase3_sprint3_evaluation.md"
OUT_JSON = ROOT / "data/eval/phase3_sprint3_evaluation.json"
CHRF_KW = dict(char_order=6, word_order=2, beta=2.0)

from scripts.meaning_diagnostics import (  # noqa: E402
    detect_degeneration,
    detect_negation_flip,
    detect_omissions,
)
from scripts.sprint2_failure_analysis import classify_row  # noqa: E402


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    by_key: dict[tuple[str, int], dict[str, dict]] = {}
    for r in rows:
        by_key.setdefault((r["direction"], int(r["idx"])), {})[r["arm"]] = r

    stats: dict[str, Counter] = {"baseline": Counter(), "integrated": Counter()}
    lat: dict[str, list[float]] = {"baseline": [], "integrated": []}
    hyps: dict[str, list[str]] = {"baseline": [], "integrated": []}
    refs: dict[str, list[str]] = {"baseline": [], "integrated": []}
    examples: list[dict] = []

    for (direction, idx), arms in sorted(by_key.items()):
        for arm in ("baseline", "integrated"):
            r = arms[arm]
            src, pred, ref = r["source"], r["prediction"], r["reference"]
            hyps[arm].append(pred)
            refs[arm].append(ref)
            if "latency_s" in r and r["latency_s"]:
                try:
                    lat[arm].append(float(r["latency_s"]))
                except ValueError:
                    pass
            s = stats[arm]
            # numerics
            st, dropped = classify_row(src, pred)
            s[f"numeric_{st}"] += 1
            # meaning
            if detect_negation_flip(src, pred) == "negation_flip_suspected":
                s["negation_flip_suspected"] += 1
                if len(examples) < 24:
                    examples.append({"direction": direction, "idx": idx, "arm": arm,
                                     "kind": "negation_flip", "source": src,
                                     "prediction": pred, "reference": ref})
            if detect_degeneration(pred) == "repetition_loop":
                s["degeneration_loop"] += 1
                if len(examples) < 24:
                    examples.append({"direction": direction, "idx": idx, "arm": arm,
                                     "kind": "degeneration", "source": src,
                                     "prediction": pred[:200], "reference": ref})
            for om in detect_omissions(src, pred, ref):
                s["omission_" + om.split(":")[0]] += 1

    result_rows = []
    for arm in ("baseline", "integrated"):
        bleu = sacrebleu.corpus_bleu(hyps[arm], [refs[arm]]).score
        chrf = sacrebleu.corpus_chrf(hyps[arm], [refs[arm]], **CHRF_KW).score
        n = len(hyps[arm])
        c = stats[arm]
        num_checked = c["numeric_preserved"] + c["numeric_rendered_as_words"] + \
            c["numeric_dropped"] + c["numeric_reordered_or_changed"]
        result_rows.append({
            "arm": arm, "n": n, "bleu": round(bleu, 2), "chrfpp": round(chrf, 2),
            "numeric_preserved": c["numeric_preserved"],
            "numeric_rendered_as_words": c["numeric_rendered_as_words"],
            "numeric_dropped": c["numeric_dropped"],
            "numeric_accuracy_rate": (round(c["numeric_preserved"] / num_checked, 4)
                                      if num_checked else None),
            "negation_flip_warnings": c["negation_flip_suspected"],
            "degeneration_loops": c["degeneration_loop"],
            "omission_number_warnings": c["omission_omitted_number"],
            "omission_entity_warnings": c["omission_omitted_entity"],
            "omission_paren_warnings": c["omission_omitted_parenthetical"],
            "mean_latency_s": (round(statistics.mean(lat[arm]), 2) if lat[arm] else None),
        })

    OUT_JSON.write_text(json.dumps({
        "artifact": CSV_PATH.name, "sacrebleu": sacrebleu.__version__,
        "chrf_kwargs": CHRF_KW, "arms": result_rows,
        "examples": examples[:24],
        "disclaimer": "semantic counts are heuristic warnings, not verified error rates",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    doc = [
        "# Sprint 3 PHASE 4 — Meaning-Reliability Evaluation (devtest artifact, no inference)",
        "",
        f"Artifact: `{CSV_PATH.name}` · 160 aligned pairs · sacrebleu {sacrebleu.__version__} ·",
        f"chrF++ kwargs {CHRF_KW} · semantic counts are HEURISTIC WARNINGS, not verified error rates.",
        "",
        "| arm | n | BLEU | chrF++ | numeric exact rate | numeric dropped | negation flips | degeneration | omission(num/ent/paren) | mean lat s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in result_rows:
        doc.append(
            f"| {r['arm']} | {r['n']} | {r['bleu']} | {r['chrfpp']} | "
            f"{r['numeric_accuracy_rate']} | {r['numeric_dropped']} | "
            f"{r['negation_flip_warnings']} | {r['degeneration_loops']} | "
            f"{r['omission_number_warnings']}/{r['omission_entity_warnings']}/"
            f"{r['omission_paren_warnings']} | {r['mean_latency_s']} |")
    doc += ["", "## Representative examples (heuristic findings)", ""]
    for e in examples[:12]:
        doc += [
            f"- **{e['kind']}** ({e['arm']}, {e['direction']} #{e['idx']})",
            f"  - src: {e['source'][:120]}",
            f"  - pred: {e['prediction'][:120]}",
            f"  - ref: {e['reference'][:120]}", "",
        ]
    doc += [
        "## Benchmark limitations",
        "- Same 80 devtest pairs as the Sprint-2 artifact (public FLORES data);",
        "  no new inference, so decoding/latency improvements are not re-measured here.",
        "- Heuristic omissions/negation flips need human adjudication before",
        "  being called errors; counts are upper bounds on true error counts.",
        "- Numeric 'exact rate' counts only digit preservation; word-verified",
        "  renderings are excluded by design (see sprint2_failure_analysis).",
    ]
    OUT_MD.write_text("\n".join(doc) + "\n", encoding="utf-8")
    for r in result_rows:
        print(r["arm"], r["bleu"], r["chrfpp"], r["numeric_accuracy_rate"],
              r["negation_flip_warnings"], r["degeneration_loops"])
    print("wrote", OUT_MD.name)


if __name__ == "__main__":
    main()
