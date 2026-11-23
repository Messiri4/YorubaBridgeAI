from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "eval" / "mac_vs_windows_quality"
OUTPUT.mkdir(parents=True, exist_ok=True)

ROWS = [
    {"comparison": "historical_mac_context", "direction": "EN→YO", "sample": 40,
     "metric_variant": "plain_chrf", "beam_setting": "2", "bleu": 4.7,
     "chrf": 25.1, "hardware": "Mac historical", "matched": "no"},
    {"comparison": "historical_mac_context", "direction": "YO→EN", "sample": 40,
     "metric_variant": "plain_chrf", "beam_setting": 2, "bleu": 20.7,
     "chrf": 44.4, "hardware": "Mac historical", "matched": "no"},
    {"comparison": "matched_windows_30_dev", "direction": "EN→YO", "sample": 30,
     "metric_variant": "chrf++", "beam_setting": 2, "bleu": 5.205,
     "chrf": 28.498, "hardware": "Windows CPU", "matched": "yes"},
    {"comparison": "matched_windows_30_dev", "direction": "EN→YO", "sample": 30,
     "metric_variant": "chrf++", "beam_setting": 4, "bleu": 5.683,
     "chrf": 28.872, "hardware": "Windows CPU", "matched": "yes"},
    {"comparison": "matched_windows_30_dev", "direction": "YO→EN", "sample": 30,
     "metric_variant": "chrf++", "beam_setting": 2, "bleu": 20.698,
     "chrf": 44.382, "hardware": "Windows CPU", "matched": "yes"},
    {"comparison": "matched_windows_30_dev", "direction": "YO→EN", "sample": 30,
     "metric_variant": "chrf++", "beam_setting": 4, "bleu": 20.391,
     "chrf": 44.274, "hardware": "Windows CPU", "matched": "yes"},
]


def main() -> None:
    csv_path = OUTPUT / "quality_comparison_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROWS[0].keys()))
        writer.writeheader()
        writer.writerows(ROWS)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for axis, metric in zip(axes, ("bleu", "chrf")):
        grouped = {}
        for row in ROWS:
            key = (row["direction"], row["beam_setting"])
            if row["comparison"] == "matched_windows_30_dev":
                grouped.setdefault(row["direction"], {})[row["beam_setting"]] = row[metric]
        labels = ["EN→YO", "YO→EN"]
        beam2 = [grouped[d].get(2, float("nan")) for d in labels]
        beam4 = [grouped[d].get(4, float("nan")) for d in labels]
        x = range(len(labels))
        axis.bar([v - 0.15 for v in x], beam2, width=0.3, label="Beams 2")
        axis.bar([v + 0.15 for v in x], beam4, width=0.3, label="Beams 4")
        axis.set_xticks(list(x), labels)
        axis.set_title("BLEU" if metric == "bleu" else "chrF++")
        axis.set_ylabel("Score")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.suptitle("Matched Windows 30-pair FLORES dev beam comparison")
    fig.text(
        0.5, 0.01,
        "Mac figures are historical context only: 40-pair plain chrF/BLEU, not matched.",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    png_path = OUTPUT / "mac_vs_windows_quality.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(json.dumps({"csv": str(csv_path), "png": str(png_path)}))


if __name__ == "__main__":
    main()
