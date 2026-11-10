# Human-Review Annotation Rubric — YorubaBridgeAI evaluation worksheets

Applies to `data/eval/flores_reference_review.csv` (100 FLORES dev pairs)
and future review worksheets. **All judgment columns stay blank until a
fluent Yoruba reviewer fills them. Automated scores/predictions must never
be copied into judgment fields** — the reviewer should assess the reference
against the English source BEFORE looking at the NLLB prediction fields.

## Protocol (order matters)

1. Read the `english_source` only; write down what you expect a good Yoruba
   translation to say (optionally in `reviewer_comments`).
2. Read `yoruba_reference`. Judge items 1–6 below for the **reference**.
3. Only afterwards consult the `nllb_beams*` fields and judge whether the
   model output preserves meaning — never let model output influence the
   reference judgments.

## Judgment columns and scales

| Column                     | Scale | Definition                                                                                                                                                                          |
| -------------------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `reference_accuracy`       | 0/1/2 | 2 = faithful; 1 = partially wrong (one fact off, e.g. a number/name); 0 = wrong meaning or major omission                                                                           |
| `meaning_preservation`     | 0/1/2 | semantic completeness vs the English: 2 = nothing added/dropped; 1 = minor omission/addition that doesn't change the message; 0 = meaning altered                                   |
| `grammar_naturalness`      | 0/1/2 | 2 = fluent native Yoruba; 1 = understandable but awkward/anglicized; 0 = ungrammatical                                                                                              |
| `diacritic_correctness`    | 0/1/2 | 2 = full, correct tone marks + underdots; 1 = mostly correct, some missing (incl. consistent accent-dropped style — note which in comments); 0 = systematic loss or wrong placement |
| `numerical_consistency`    | 0/1/2 | 2 = all numbers/times/measures match (digits or correct Yoruba number words); 1 = one mismatch/omission; 0 = multiple or value-changing errors                                      |
| `named_entity_consistency` | 0/1/2 | 2 = all entities present (standard transliteration OK); 1 = one entity missing/altered; 0 = several missing/invented                                                                |
| `confidence`               | L/M/H | reviewer's confidence in their own judgments for this pair                                                                                                                          |
| `final_review_status`      | enum  | `accepted` / `reference-issue` / `model-issue` / `both-issue` / `needs-second-opinion`                                                                                              |

`reviewer_comments` is free text; please cite the exact word(s) concerned.

## Aggregation rules (used by the scorecard)

- A pair's reference is "reference-issue" if `reference_accuracy ≤ 1` OR
  `diacritic_correctness = 0`.
- Model findings from Phase 2 are "confirmed" ONLY when a reviewer marks
  `model-issue` or `both-issue` on the corresponding row. Heuristic counts
  never become confirmed errors by themselves.
- Inter-reviewer disagreement: rows reviewed twice must record both passes;
  disagreements on numerical/named-entity items are adjudicated by a third reviewer.

## Leakage rules

- Never enter NLLB text into judgment columns, and never derive judgment
  values from chrF/BLEU or detector output. Heuristics may _select_ which
  rows to review; they may not _judge_ them.
