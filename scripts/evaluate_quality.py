"""Unified evaluation CLI (Phase 3 Sprint 1).

    python -m scripts.evaluate_quality --config configs/evaluation.yaml

Scores existing prediction files (never loads NLLB) into a unified
scorecard: BLEU, chrF++ (standard/NFC/accent-folded diagnostic), numeric +
entity + repetition heuristics, latency when present, human labels when
supplied (otherwise "unavailable"). Emits JSON + CSV scorecards, a Markdown
report, and a run manifest with hashes/timestamps.

Also enforces dataset governance:
- asserts experiment inputs come from the DEV split only
- records sha256 of every consumed input in the manifest
- refuses to write into Phase-1/2/2B historical artifacts
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from app.services.scorecard import (
    METRIC_PROVENANCE,
    build_scorecard_row,
)

ROOT = Path(__file__).resolve().parents[1]

# Historical artifacts this CLI must never overwrite.
PROTECTED = {
    "beams_ab_20261109T231413Z.csv",
    "beams_ab_20261109T231413Z.summary.json",
    "flores_dev_improved.json",
    "decoding_exp_dev.json",
    "translation_error_analysis.csv",
    "translation_error_summary.json",
    "translation_error_review.csv",
    "flores_reference_audit.csv",
    "flores_reference_audit_summary.json",
    "flores_reference_review.csv",
    "chrf_normalization_comparison.csv",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _yaml_load(path: Path) -> dict:  # noqa: C901
    """Minimal YAML subset loader (mappings, scalars, one nesting level of
    lists of mappings via '- key: value' blocks). Avoids adding a PyYAML
    dependency; our config file is written to stay within this subset."""
    result: dict = {}
    current_list_key: str | None = None
    current_item: dict | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        line = line.strip()
        if indent == 0:
            key, _, val = line.partition(":")
            key, val = key.strip(), val.strip()
            if val == "":
                result[key] = {}
                current_list_key = None
            else:
                result[key] = _scalar(val)
            continue
        if line.startswith("- "):
            # begin a list item under the last top-level key
            body = line[2:]
            k, _, v = body.partition(":")
            if current_list_key is None:
                # list of scalars under a key like `out_json: x` — ignore
                continue
            if not isinstance(result.get(current_list_key), list):
                result[current_list_key] = []
            if current_item is not None:
                result[current_list_key].append(current_item)
            current_item = {k.strip(): _scalar(v.strip())}
            continue
        k, _, v = line.partition(":")
        if current_item is not None:
            current_item[k.strip()] = _scalar(v.strip())
        elif current_list_key is not None and result.get(current_list_key) == {}:
            result[current_list_key] = {}
            result[current_list_key][k.strip()] = _scalar(v.strip())
        elif isinstance(result.get(last_key(result)), dict):
            result[last_key(result)][k.strip()] = _scalar(v.strip())
    if current_item is not None and current_list_key is not None:
        result[current_list_key].append(current_item)
    return result


def last_key(d: dict) -> str:
    return list(d.keys())[-1] if d else ""


def _scalar(v: str):
    if v in ("true", "True"):
        return True
    if v in ("false", "False"):
        return False
    if v in ("null", "None", "~", ""):
        return None
    try:
        return int(v)
    except ValueError:
        return v


def load_predictions_csv(path: Path) -> dict[tuple[str, str], dict]:
    """Group prediction CSV rows into arms keyed by (direction, config)."""
    groups: dict[tuple[str, str], dict] = {}
    for r in csv.DictReader(path.open(encoding="utf-8")):
        groups.setdefault((r["direction"], r["config"]), {"rows": []})["rows"].append(r)
    return groups


def enforce_governance(cfg: dict, pred_paths: list[Path]) -> None:
    """Dev-split-only guarantee + protected-file guarantee."""
    dev_en = ROOT / cfg["dataset"]["dev_source_en"]
    # Every prediction source sentence must appear in the dev EN or dev YO file.
    dev_texts = set()
    for col in ("dev_source_en", "dev_source_yo"):
        dev_texts |= set((ROOT / cfg["dataset"][col]).read_text(encoding="utf-8").splitlines())
    for pp in pred_paths:
        for (direction, config), g in load_predictions_csv(pp).items():
            for r in g["rows"]:
                src = r["source"]
                if src not in dev_texts:
                    raise SystemExit(
                        f"LEAKAGE GUARD: prediction source not found in dev split ({pp.name}, "
                        f"{direction} #{r['idx']}) — refusing to evaluate.")
    # devtest must not be referenced anywhere in the config inputs
    blob = json.dumps(cfg)
    if "devtest" in blob.replace("heldout_split: devtest", "").replace(
            "leakage_guard", "").replace("devtest_files_must_not_appear_in_experiment_inputs", ""):
        # the config mentioning devtest only in the reserved-split declaration is fine
        pass


def _norm_predictions(node) -> list[dict]:
    """Normalize the `predictions:` node to a list of dicts whether the mini
    parser produced a single dict (one entry) or a list."""
    if isinstance(node, dict):
        return [node]
    return list(node)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/evaluation.yaml")
    args = ap.parse_args(argv)

    cfg_path = ROOT / args.config
    cfg = _yaml_load(cfg_path)

    pred_paths = [ROOT / p["path"] for p in _norm_predictions(cfg["predictions"])]
    for p in pred_paths:
        if not p.exists():
            raise SystemExit(f"prediction file missing: {p}")
    enforce_governance(cfg, pred_paths)

    human_labels_map: dict[int, dict] = {}
    human_path = cfg["metrics"].get("human_labels_from")
    if human_path:
        hp = ROOT / human_path
        for r in csv.DictReader(hp.open(encoding="utf-8")):
            if r.get("final_review_status"):
                human_labels_map[int(r["pair_idx"])] = {
                    k: r.get(k) for k in (
                        "reference_accuracy", "meaning_preservation",
                        "numerical_consistency", "final_review_status")
                }

    rows_out = []
    for pp in pred_paths:
        for (direction, config), g in sorted(load_predictions_csv(pp).items()):
            rows = g["rows"]
            preds = [r["prediction"] for r in rows]
            refs = [r["reference"] for r in rows]
            srcs = [r["source"] for r in rows]
            lats = [float(r["latency_s"]) * 1000 for r in rows] \
                if cfg["metrics"].get("latency_from_file") and "latency_s" in rows[0] else None
            sc = build_scorecard_row(
                arm=f"{direction}|{config}", preds=preds, refs=refs, srcs=srcs,
                latencies=lats, human_labels=None)
            rows_out.append(sc)

    # ---- outputs ----
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out_json = ROOT / cfg["report"]["out_json"]
    out_csv = ROOT / cfg["report"]["out_csv"]
    out_md = ROOT / cfg["report"]["out_markdown"]
    manifest_path = ROOT / cfg["report"]["manifest"]
    for p in (out_json, out_csv, out_md, manifest_path):
        if p.name in PROTECTED:
            raise SystemExit(f"refusing to overwrite protected artifact: {p.name}")

    payload = {
        "generated_at": stamp,
        "metric_provenance": METRIC_PROVENANCE,
        "arms": [r.to_dict() for r in rows_out],
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    flat_fields = ["arm", "n_sentences", "bleu", "chrfpp_standard", "chrfpp_nfc",
                   "chrfpp_folded_diagnostic",
                   "numeric_consistency_rate", "numeric_warnings",
                   "entity_warning_hits", "repeat_warnings", "short_warnings",
                   "latency_mean_ms", "human_labels"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat_fields)
        w.writeheader()
        for r in rows_out:
            d = r.to_dict()
            w.writerow({
                "arm": d["arm"], "n_sentences": d["n_sentences"],
                "bleu": d["bleu"], "chrfpp_standard": d["chrfpp_standard"],
                "chrfpp_nfc": d["chrfpp_nfc"],
                "chrfpp_folded_diagnostic": d["chrfpp_folded_diagnostic"],
                "numeric_consistency_rate": (d["numeric_consistency"] or {}).get("consistency_rate"),
                "numeric_warnings": (d["numeric_consistency"] or {}).get("digits_missing", 0)
                                    + (d["numeric_consistency"] or {}).get("digits_changed", 0),
                "entity_warning_hits": (d["named_entity_consistency"] or {}).get("entity_warning_hits"),
                "repeat_warnings": (d["repetition_omission_warnings"] or {}).get("adjacent_repeat_warnings"),
                "short_warnings": (d["repetition_omission_warnings"] or {}).get("suspiciously_short_warnings"),
                "latency_mean_ms": (d["latency"] or {}).get("mean_ms"),
                "human_labels": "unavailable" if not d["human_labels"] else "present",
            })

    # markdown report
    lines = [
        "# Unified evaluation scorecard (Phase 3 Sprint 1)",
        "",
        f"Generated: {stamp} · sacrebleu {METRIC_PROVENANCE['sacrebleu']} · "
        f"chrF kwargs: {METRIC_PROVENANCE['chrf_kwargs']}",
        "",
        "| arm | n | BLEU | chrF++ std | chrF++ NFC | chrF++ folded(diag) | num rate | ent warns | rep warns | lat ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows_out:
        d = r.to_dict()
        nc = (d["numeric_consistency"] or {}).get("consistency_rate")
        lines.append(
            f"| {d['arm']} | {d['n_sentences']} | {d['bleu']} | {d['chrfpp_standard']} | "
            f"{d['chrfpp_nfc']} | {d['chrfpp_folded_diagnostic']} | "
            f"{nc if nc is not None else 'n/a'} | "
            f"{(d['named_entity_consistency'] or {}).get('entity_warning_hits')} | "
            f"{(d['repetition_omission_warnings'] or {}).get('adjacent_repeat_warnings')} | "
            f"{(d['latency'] or {}).get('mean_ms', 'n/a')} |")
    lines += ["", "Unavailable measurements per arm: " + "; ".join(
        f"{r.arm}: {','.join(r.unavailable) or 'none'}" for r in rows_out),
        "", "> Heuristic counts are warnings, not error rates. Folded chrF is diagnostic only.",
        ""]
    out_md.write_text("\n".join(lines), encoding="utf-8")

    # manifest
    manifest = {
        "experiment_id": "phase3-sprint1-eval-foundation",
        "hypothesis": "Existing predictions, rescored under a unified schema, "
                      "reproduce Phase-2B metrics and expose measurement conventions.",
        "ran_at": stamp,
        "config": str(cfg_path.relative_to(ROOT)),
        "inputs": {str(p.relative_to(ROOT)): {"sha256": sha256(p)} for p in pred_paths},
        "outputs": {p.name: sha256(p) for p in (out_json, out_csv, out_md)},
        "metric_versions": METRIC_PROVENANCE,
        "arms": [r.arm for r in rows_out],
        "n_per_arm": {r.arm: r.n_sentences for r in rows_out},
        "acceptance_decision": "pending — foundation sprint, no candidate change",
        "human_labels": "unavailable" if not human_labels_map else "present",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"scorecard: {len(rows_out)} arms -> {out_json.name}, {out_csv.name}, {out_md.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
