# Sprint 2A verified baseline (canonical chrF++)

sacrebleu 2.6.0 · char_order=6, word_order=2, beta=2.0 · BLEU tokenizer 13a
Input: `beams_ab_20261109T231413Z.csv` (sha256 ed88ebd53663361c… in `phase3_manifest_sprint2a.json`) · n=30/arm · no inference

| arm              | BLEU  | chrF++ std | NFC   | folded (diag) | plain chrF (historical) | numeric rate | rep warns | mean lat ms |
| ---------------- | ----- | ---------- | ----- | ------------- | ----------------------- | ------------ | --------- | ----------- |
| eng_yor\|beams=2 | 3.86  | 26.07      | 29.30 | 37.42         | 28.12                   | 0.545        | 0         | 25 666      |
| eng_yor\|beams=4 | 4.57  | 26.75      | 30.04 | 37.91         | 28.76                   | 0.636        | 0         | 30 453      |
| yor_eng\|beams=2 | 19.21 | 43.56      | 43.56 | 43.56         | 45.99                   | 0.833        | 0         | 13 480      |
| yor_eng\|beams=4 | 18.55 | 43.01      | 43.01 | 43.01         | 45.51                   | 0.833        | 0         | 17 490      |

- Historical plain chrF values are preserved (word_order=0) and reproduced
  exactly by test `test_plain_chrf_reproduces_historical_scores`.
- Human-review labels: unavailable. Folded chrF: diagnostic only.
