# Sprint 3 PHASE 4 — Meaning-Reliability Evaluation (devtest artifact, no inference)

Artifact: `integration_comparison_20261110T102741Z.csv` · 160 aligned pairs · sacrebleu 2.6.0 ·
chrF++ kwargs {'char_order': 6, 'word_order': 2, 'beta': 2.0} · semantic counts are HEURISTIC WARNINGS, not verified error rates.

| arm | n | BLEU | chrF++ | numeric exact rate | numeric dropped | negation flips | degeneration | omission(num/ent/paren) | mean lat s |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 160 | 13.24 | 35.12 | 0.6944 | 7 | 59 | 1 | 7/168/1 | 17.53 |
| integrated | 160 | 13.14 | 34.85 | 0.7778 | 4 | 58 | 1 | 3/168/1 | 20.71 |

## Representative examples (heuristic findings)

- **negation_flip** (baseline, eng_yor #0)
  - src: "We now have 4-month-old mice that are non-diabetic that used to be diabetic," he added.
  - pred: Ó fi kún un pé: "A ní àwọn ọ̀dọ́ ọ̀dọ́ tó ti pé oṣù mẹ́rin báyìí tí kò ní àrùn àtọ̀gbẹ, àmọ́ tí wọ́n ti ní àrùn àtọ̀gbẹ 
  - ref: “Ní báyìí a ní àwọn ẹku olóṣù mẹ́rin tí kò ní dayabẹ́tìsì tẹ́lẹ̀tẹ́lẹ̀ tó ti ní ayabẹ́tìsì̀,” o ṣe àfikún.

- **negation_flip** (integrated, eng_yor #0)
  - src: "We now have 4-month-old mice that are non-diabetic that used to be diabetic," he added.
  - pred: Ó fi kún un pé: "A ní àwọn ọ̀dọ́ ọ̀dọ́ tó ti pé oṣù mẹ́rin báyìí tí kò ní àrùn àtọ̀gbẹ, àmọ́ tí wọ́n ti ní àrùn àtọ̀gbẹ 
  - ref: “Ní báyìí a ní àwọn ẹku olóṣù mẹ́rin tí kò ní dayabẹ́tìsì tẹ́lẹ̀tẹ́lẹ̀ tó ti ní ayabẹ́tìsì̀,” o ṣe àfikún.

- **negation_flip** (baseline, eng_yor #2)
  - src: Like some other experts, he is skeptical about whether diabetes can be cured, noting that these findings have no relevan
  - pred: Bíi ti àwọn ògbógi mìíràn, ó máa ń ṣiyèméjì bóyá a lè wo àrùn àtọ̀gbẹ sàn, ó sì sọ pé àwọn àbá yìí kò kan àwọn tó ti ní 
  - ref: Bí àwọn àkọ́ṣẹ́mọṣẹ́, ó ṣiyèméjì bóyá àrùn ìtọ ṣúgá lè ní ìwòsàn, tí ó sì mọ̀ pé kò wúlò fún àwọn tó bá ní àrùn ìtọ ṣúgá

- **negation_flip** (baseline, eng_yor #3)
  - src: On Monday, Sara Danius, permanent secretary of the Nobel Committee for Literature at the Swedish Academy, publicly annou
  - pred: Ní ọjọ́ Monday, Sara Danius, akẹ́kọ̀ọ́ alákòókò kíkún ti Ìgbìmọ̀ Nobel fún Èdè ní Swedish Academy, kéde ní gbangba nígbà
  - ref: Ní ọjọ́ Ajé, Sara Danius, akọ̀wé àgbà ti ẹgbẹ́ Nobel fún ìtàn àròsọ ní ilé ẹ̀ka Swedin, tí wọ́n kéde ní ètò kan lóri rẹ́

- **negation_flip** (integrated, eng_yor #3)
  - src: On Monday, Sara Danius, permanent secretary of the Nobel Committee for Literature at the Swedish Academy, publicly annou
  - pred: Ní ọjọ́ Monday, Sara Danius, akẹ́kọ̀ọ́ alákòókò kíkún ti Ìgbìmọ̀ Nobel fún Èdè ní Swedish Academy, kéde ní gbangba nígbà
  - ref: Ní ọjọ́ Ajé, Sara Danius, akọ̀wé àgbà ti ẹgbẹ́ Nobel fún ìtàn àròsọ ní ilé ẹ̀ka Swedin, tí wọ́n kéde ní ètò kan lóri rẹ́

- **negation_flip** (baseline, eng_yor #4)
  - src: Danius said, "Right now we are doing nothing. I have called and sent emails to his closest collaborator and received ver
  - pred: Danius sọ pé, "Ní báyìí, a kò ṣe nǹkan kan. Mo ti pe, mo sì ti fi imeeli ránṣẹ́ sí òṣìṣẹ́ tó sún mọ́ mi jù lọ, mo sì ti 
  - ref: Daniusi wipe, “Nisin a o se nkankan. Mo ti meeli ranse mo de ti pe si awon ajo sise timo timo re won de fun mi lesi to l

- **negation_flip** (integrated, eng_yor #4)
  - src: Danius said, "Right now we are doing nothing. I have called and sent emails to his closest collaborator and received ver
  - pred: Danius sọ pé, "Ní báyìí, a kò ṣe nǹkan kan. Mo ti pe, mo sì ti fi imeeli ránṣẹ́ sí òṣìṣẹ́ tó sún mọ́ mi jù lọ, mo sì ti 
  - ref: Daniusi wipe, “Nisin a o se nkankan. Mo ti meeli ranse mo de ti pe si awon ajo sise timo timo re won de fun mi lesi to l

- **negation_flip** (baseline, eng_yor #5)
  - src: Previously, Ring's CEO, Jamie Siminoff, remarked the company started when his doorbell wasn't audible from his shop in h
  - pred: Ṣáájú ìgbà yẹn, Jamie Siminoff tó jẹ́ ọ̀gá àgbà fún Ring sọ pé ilé iṣẹ́ náà bẹ̀rẹ̀ nígbà tí a kò gbọ́ sí i nínú ilé ìtaj
  - ref: Tẹ́lẹ̀rí, olùdarí àgbà iléeṣẹ́ Ring, Jamie Siminoff ní iléeṣẹ́ náà bẹ̀rẹ̀ iṣẹ́ nígbà tí aago ẹnu ọ̀nà òhun kò dún sókè l

- **negation_flip** (integrated, eng_yor #5)
  - src: Previously, Ring's CEO, Jamie Siminoff, remarked the company started when his doorbell wasn't audible from his shop in h
  - pred: Ṣáájú ìgbà yẹn, Jamie Siminoff tó jẹ́ ọ̀gá àgbà fún Ring sọ pé ilé iṣẹ́ náà bẹ̀rẹ̀ nígbà tí a kò gbọ́ sí i nínú ilé ìtaj
  - ref: Tẹ́lẹ̀rí, olùdarí àgbà iléeṣẹ́ Ring, Jamie Siminoff ní iléeṣẹ́ náà bẹ̀rẹ̀ iṣẹ́ nígbà tí aago ẹnu ọ̀nà òhun kò dún sókè l

- **negation_flip** (baseline, eng_yor #6)
  - src: He built a WiFi door bell, he said.
  - pred: Ó sọ pé ó kọ́ àtẹ́lẹ́kùn WiFi kan.
  - ref: O ko ago ilekun WiFi, O wi.

- **negation_flip** (integrated, eng_yor #6)
  - src: He built a WiFi door bell, he said.
  - pred: Ó sọ pé ó kọ́ àtẹ́lẹ́kùn WiFi kan.
  - ref: O ko ago ilekun WiFi, O wi.

- **negation_flip** (baseline, eng_yor #7)
  - src: Siminoff said sales boosted after his 2013 appearance in a Shark Tank episode where the show panel declined funding the 
  - pred: Siminoff sọ pe awọn tita pọ si lẹhin ifarahan rẹ ni ọdun 2013 ninu iṣẹlẹ Shark Tank nibiti igbimọ ifihan naa kọ lati ṣe 
  - ref: Siminoff sọ pé ọjà lọ sókè lẹ́yìn tí òun hàn nínú eré Shark Tank lọ́dún 2013, ní ibi tí àwọn aláṣẹ eré ti ṣẹ́

## Known heuristic inflation (measured, disclosed)

The negation-flip detector's count-delta fires on *legitimate* translations
of inherently negated English adjectives: `non-diabetic` → `kò ní àrùn
àtògbẹ` adds one Yoruba negation marker for a source-side `non-` prefix the
marker counter does not see (the 59/58 warnings include such cases). The
missing-marker EN side (`non-` prefix) is the cause. Consequence: these
counts are UPPER BOUNDS on true polarity errors and every flagged sentence
requires human adjudication before being called an error. Values are still
useful for arm comparison (58 vs 59 — parity), not for absolute error rates.
Entity-omission warnings (168 per arm, identical in both arms) are similarly
conservative: transcribed or re-spelled proper names trigger them.

## Benchmark limitations
- Same 80 devtest pairs as the Sprint-2 artifact (public FLORES data);
  no new inference, so decoding/latency improvements are not re-measured here.
- Heuristic omissions/negation flips need human adjudication before
  being called errors; counts are upper bounds on true error counts.
- Numeric 'exact rate' counts only digit preservation; word-verified
  renderings are excluded by design (see sprint2_failure_analysis).
