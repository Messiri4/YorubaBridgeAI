# Unified evaluation scorecard (Phase 3 Sprint 1)

Generated: 2026-11-10T02:38:26Z · sacrebleu 2.6.0 · chrF kwargs: {'char_order': 6, 'word_order': 2, 'beta': 2.0}

| arm | n | BLEU | chrF++ std | chrF++ NFC | chrF++ folded(diag) | num rate | ent warns | rep warns | lat ms |
|---|---|---|---|---|---|---|---|---|---|
| eng_yor|beams=2 | 30 | 3.86 | 26.07 | 29.3 | 37.42 | 0.5455 | 2 | 5 | 25666.1 |
| eng_yor|beams=4 | 30 | 4.57 | 26.75 | 30.04 | 37.91 | 0.6364 | 2 | 5 | 30452.6 |
| yor_eng|beams=2 | 30 | 19.21 | 43.56 | 43.56 | 43.56 | 0.8333 | 2 | 0 | 13480.4 |
| yor_eng|beams=4 | 30 | 18.55 | 43.01 | 43.01 | 43.01 | 0.8333 | 2 | 0 | 17490.2 |

Unavailable measurements per arm: eng_yor|beams=2: human_review_labels; eng_yor|beams=4: human_review_labels; yor_eng|beams=2: human_review_labels; yor_eng|beams=4: human_review_labels

> Heuristic counts are warnings, not error rates. Folded chrF is diagnostic only.
