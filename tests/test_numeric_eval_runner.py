import csv

from scripts.run_numeric_protection_comparison import (
    load_input_rows,
    run_paired_rows,
    score_outputs,
)


def _write_input(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sentence_id", "direction", "source", "reference", "prediction"),
        )
        writer.writeheader()
        writer.writerows(rows)


def test_input_rows_are_selected_by_id_and_keep_yoruba_text(tmp_path):
    input_path = tmp_path / "baseline.csv"
    _write_input(
        input_path,
        [
            {
                "sentence_id": "dev-0002",
                "direction": "eng_yor",
                "source": "Source two.",
                "reference": "Yorùbá méjì.",
                "prediction": "Baseline two.",
            },
            {
                "sentence_id": "dev-0001",
                "direction": "eng_yor",
                "source": "Source one.",
                "reference": "Yorùbá kan.",
                "prediction": "Baseline one.",
            },
            {
                "sentence_id": "dev-en-0001",
                "direction": "yor_eng",
                "source": "Not selected.",
                "reference": "Not selected.",
                "prediction": "Not selected.",
            },
        ],
    )

    rows = load_input_rows(input_path, ["dev-0001", "dev-0002"])

    assert [row["row_id"] for row in rows] == ["dev-0001", "dev-0002"]
    assert rows[0]["reference"] == "Yorùbá kan."
    assert rows[1]["baseline_prediction"] == "Baseline two."


def test_paired_runner_toggles_protection_once_per_candidate():
    class FakeTranslator:
        def __init__(self):
            self.calls = []
            self.restoration_events = []

        def translate(
            self, text, source_language, target_language, protect_numerics=None
        ):
            self.calls.append((text, source_language, target_language, protect_numerics))
            if protect_numerics:
                self.restoration_events.append({"ok": True, "reason": "restored"})
            label = "protected" if protect_numerics else "raw"
            return type("Result", (), {"translated_text": f"{label}: {text}"})()

    translator = FakeTranslator()
    rows = [
        {
            "row_id": "dev-0001",
            "source": "The number is 10.",
            "reference": "Nọ́mbà náà jẹ́ 10.",
            "baseline_prediction": "raw: The number is 10.",
        }
    ]

    records = run_paired_rows(
        rows,
        translator,
        lambda source: ["10"] if "10" in source else [],
        translator.restoration_events,
    )

    assert [call[3] for call in translator.calls] == [False, True]
    assert records[0]["eligible_spans"] == ["10"]
    assert records[0]["off_matches_saved_baseline"] is True
    assert records[0]["output_changed"] is True
    assert records[0]["restoration"] == {"ok": True, "reason": "restored"}


def test_scorer_includes_all_rows_and_uses_chrf_plus_plus():
    rows = [
        {
            "reference": "Yorùbá output one.",
            "numeric_protection_off": "Different output.",
            "numeric_protection_on": "Yorùbá output one.",
        },
        {
            "reference": "Yorùbá output two.",
            "numeric_protection_off": "Another output.",
            "numeric_protection_on": "Yorùbá output two.",
        },
    ]

    scores = score_outputs(rows)

    assert scores["numeric_protection_off"]["n"] == 2
    assert scores["numeric_protection_on"]["n"] == 2
    assert scores["numeric_protection_on"]["chrf++"] == 100
    assert scores["on_minus_off"]["chrf++"] > 0
